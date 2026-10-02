"""One duplex turn: listen, Fish ASR, LLM, then isolated TTS."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from fish_audio_suite_kit import (
    DEFAULT_SEED_EXCHANGE,
    DEFAULT_SYSTEM_PROMPT,
    FishHttpError,
    LatencySnapshot,
    elapsed_ms,
    ensure_lead_cue,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_tts_junk,
    make_traceparent,
    normalize_cues,
    same_utterance,
    scrub_tts,
    trace_id_of,
)
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.asr import asr_client, fish_asr
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.debug import (
    clear_turn,
    console_print,
    conversation,
    debug,
    end_reply_line,
    env_debug,
    mark_turn,
    trace,
    warn,
    write_reply_token,
)
from fish_audio_suite_voice.listen import record_utterance
from fish_audio_suite_voice.live import IsolatedFishTts, IsolatedResult, is_cancel_noise
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.playback import PortAudioMissingError, make_sink
from fish_audio_suite_voice.signals import DuplexSession

EXIT_OK = 0
EXIT_FATAL = 2
_KEEP_SYSTEM = 1
_ROLES_PER_TURN = 2
_FATAL_FISH = frozenset({401, 402, 403})
# watch() polls the mic for 0.2 s. The join has to cover that poll so the
# stream is closed before the next listen opens the device.
_BARGE_JOIN_S = 1.0
# The TTS side reads the token queue on its own loop. A short poll keeps the
# hand-off under one frame without a cross-loop wake-up.
_PIPE_POLL_S = 0.005


class _TokenPipe:
    """Hand LLM tokens to the TTS thread's event loop.

    The two sides run on different loops and threads, so the queue is a plain
    thread-safe one and the reader polls it.
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()

    def push(self, token: str) -> None:
        """Add one token."""
        self._queue.put(token)

    def close(self) -> None:
        """Mark the end of the reply."""
        self._queue.put(None)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._drain()

    async def _drain(self) -> AsyncIterator[str]:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(_PIPE_POLL_S)
                continue
            if item is None:
                return
            yield item


@dataclass
class _Loop:
    config: VoiceCliConfig
    tts: IsolatedFishTts
    device: str | int | None
    backend: ChatBackend
    session: DuplexSession
    asr_http: httpx.AsyncClient
    history: list[dict[str, str]]
    barge_prefix: bytes = b""
    pinned: int = _KEEP_SYSTEM


async def _collect_reply(
    loop: _Loop,
    *,
    llm_cancel: asyncio.Event,
    trace_id: str | None,
    started: float,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, float | None]:
    parts: list[str] = []
    ttft_ms: float | None = None
    # Debug lines print while the reply streams. Buffering the reply keeps it
    # on one line instead of split around them.
    live = not env_debug()
    try:
        async for tok in loop.backend.stream(loop.history, cancel=llm_cancel, trace_id=trace_id):
            if ttft_ms is None:
                ttft_ms = elapsed_ms(started)
                debug("llm.first_token {:.0f}ms", ttft_ms)
                if live:
                    write_reply_token("llm \u25b8 ")
            parts.append(tok)
            if live:
                write_reply_token(tok)
            if on_token is not None:
                on_token(tok)
    except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError) as e:
        if not is_cancel_noise(e, cancelled=llm_cancel.is_set()):
            warn(f"[llm] {e}")
    finally:
        end_reply_line()
    reply = "".join(parts).strip()
    if reply and not live:
        conversation("llm", reply)
    return reply, ttft_ms


def _after_speech(
    loop: _Loop,
    snapshot: LatencySnapshot,
    result: IsolatedResult,
    *,
    started: float,
    first_audio_at: float | None = None,
) -> tuple[LatencySnapshot, int | None]:
    c = loop.config
    history = loop.history
    notes = ["cancelled" if result.cancelled else "", "no audio" if not result.got_audio else ""]
    if result.error_status is not None or result.error_message:
        notes.append(f"error={result.error_status} {result.error_message or ''}".strip())
    debug(
        "tts.done spoken={} chars, played {} kB {}",
        len(result.spoken_so_far),
        result.bytes_played // 1000,
        " ".join(note for note in notes if note),
    )
    if result.error_status in _FATAL_FISH:
        return snapshot, EXIT_FATAL
    if not result.got_audio and result.cancelled:
        console_print("  [tts cancelled before audio]", flush=True)
    elif not result.got_audio and result.error_status is None and not result.error_message:
        console_print(
            f"  [tts silent] voice={c.fish_voice_id} model={c.tts_model}",
            flush=True,
        )
    spoken = result.spoken_so_far
    # The last history row is the user line this reply answers. Saving an
    # exact copy teaches the next turn to repeat them again.
    last_user = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
    if (
        spoken
        and (result.got_audio or not result.cancelled)
        and not same_utterance(spoken, last_user)
    ):
        history.append({"role": "assistant", "content": spoken})
    return (
        replace(
            snapshot,
            llm_ttfs=result.llm_ttfs_ms,
            ttfa=result.ttfa_ms,
            # From the start of ASR, so it covers the whole wait after you stop talking.
            first_audio=None if first_audio_at is None else (first_audio_at - started) * 1000,
            voice_to_voice=elapsed_ms(started),
        ),
        None,
    )


async def _speak_reply(
    loop: _Loop,
    reply: str,
    cancel: threading.Event,
    snapshot: LatencySnapshot,
    *,
    started: float,
    trace_id: str | None,
) -> tuple[LatencySnapshot, int | None]:
    c = loop.config
    scrubbed = ensure_lead_cue(normalize_cues(scrub_tts(reply), lead=c.mood_lead))
    if is_tts_junk(scrubbed, drop_narration=c.drop_narration):
        console_print("  (skip junk TTS)", flush=True)
        return snapshot, None
    barge = BargeGate(device=loop.device, tune=c.barge, aec=loop.session.aec)
    thread = barge.start_after_bleed(cancel)
    sink = make_sink(
        c.playback,
        path=None,
        sample_rate=c.sample_rate,
        device=loop.device,
        cancel=cancel,
        aec=loop.session.aec,
    )
    loop.tts.trace_headers = {"traceparent": make_traceparent(trace_id=trace_id)}
    try:
        try:
            first_audio: list[float] = []
            result = await asyncio.to_thread(
                loop.tts.speak_isolated,
                scrubbed,
                sink,
                cancel,
                lambda: first_audio.append(time.perf_counter()),
            )
        except PortAudioMissingError as exc:
            warn(str(exc))
            return snapshot, EXIT_FATAL
        if result.cancelled and barge.captured:
            loop.barge_prefix = barge.captured
        return _after_speech(
            loop,
            snapshot,
            result,
            started=started,
            first_audio_at=first_audio[0] if first_audio else None,
        )
    finally:
        # watch() holds the mic until this event is set. The next listen
        # opens the same device as soon as this function returns.
        cancel.set()
        thread.join(timeout=_BARGE_JOIN_S)


def _end_llm_when_tts_stops(
    task: asyncio.Future[IsolatedResult], llm_cancel: asyncio.Event
) -> None:
    # A barge-in or a dead sink ends the reply, so the model should stop too.
    # A Fish failure before any audio leaves the model running. The finished
    # reply is spoken afterwards on the whole-string path.
    if (
        task.cancelled()
        or task.exception() is not None
        or task.result().cancelled
        or task.result().error_status in _FATAL_FISH
    ):
        llm_cancel.set()


async def _stream_turn(
    loop: _Loop,
    heard: _HeardLine,
    cancel: threading.Event,
    llm_cancel: asyncio.Event,
) -> tuple[LatencySnapshot, int | None]:
    """Speak the reply while the model is still writing it.

    Parameters
    ----------
    loop : _Loop
        Session state.
    heard : _HeardLine
        The accepted user line.
    cancel : threading.Event
        Stops the TTS turn.
    llm_cancel : asyncio.Event
        Stops the model.

    Returns
    -------
    tuple of LatencySnapshot and int or None
        Timings and an exit code when the turn hit a fatal error.

    Notes
    -----
    Tokens go to the TTS thread as they arrive. The barge-in gate arms at the
    first audio chunk, not at the start, so it does not listen to a silent
    speaker. A Fish failure before any audio falls back to speaking the
    finished reply.
    """
    c = loop.config
    pipe = _TokenPipe()
    barge = BargeGate(device=loop.device, tune=c.barge, aec=loop.session.aec)
    barge_threads: list[threading.Thread] = []
    first_audio: list[float] = []

    def on_first_audio() -> None:
        first_audio.append(time.perf_counter())
        barge_threads.append(barge.start_after_bleed(cancel))

    sink = make_sink(
        c.playback,
        path=None,
        sample_rate=c.sample_rate,
        device=loop.device,
        cancel=cancel,
        aec=loop.session.aec,
    )
    loop.tts.trace_headers = {"traceparent": make_traceparent(trace_id=heard.trace_id)}
    snapshot = LatencySnapshot(asr_ms=heard.asr_ms, trace_id=heard.trace_id)
    tts_task = asyncio.create_task(
        asyncio.to_thread(loop.tts.speak_stream_isolated, pipe, sink, cancel, on_first_audio)
    )
    tts_task.add_done_callback(lambda task: _end_llm_when_tts_stops(task, llm_cancel))
    reply = ""
    try:
        try:
            reply, ttft_ms = await _collect_reply(
                loop,
                llm_cancel=llm_cancel,
                trace_id=heard.trace_id,
                started=time.perf_counter(),
                on_token=pipe.push,
            )
        finally:
            pipe.close()
        snapshot = replace(snapshot, llm_ttft=ttft_ms)
        if not reply.strip():
            # Stop the turn before Fish is asked to flush nothing.
            cancel.set()
        try:
            result = await tts_task
        except PortAudioMissingError as exc:
            warn(str(exc))
            return snapshot, EXIT_FATAL
        if result.cancelled and barge.captured:
            loop.barge_prefix = barge.captured
    finally:
        cancel.set()
        for thread in barge_threads:
            thread.join(timeout=_BARGE_JOIN_S)
    if not reply.strip():
        # Nothing to say. The non-streaming path never opens a Fish turn for an
        # empty reply, so skip the "silent" report and the history entry too.
        debug("tts.stream empty reply, nothing to speak")
        return snapshot, None
    failed_early = (
        not result.got_audio
        and not result.cancelled
        and (result.error_status is not None or bool(result.error_message))
        and result.error_status not in _FATAL_FISH
    )
    if failed_early and reply and not loop.session.stop.is_set():
        debug("tts.stream failed before audio, speaking the finished reply")
        retry_cancel = threading.Event()
        loop.session.turn.bind(retry_cancel, llm_cancel)
        if loop.session.stop.is_set():
            # Quit landed between the check above and the rebind, so the old
            # cancel flag no longer reaches this turn. Stop is sticky.
            retry_cancel.set()
            return snapshot, None
        return await _speak_reply(
            loop, reply, retry_cancel, snapshot, started=heard.started, trace_id=heard.trace_id
        )
    return _after_speech(
        loop,
        snapshot,
        result,
        started=heard.started,
        first_audio_at=first_audio[0] if first_audio else None,
    )


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def _turn_summary(snapshot: LatencySnapshot) -> str:
    """One human line for a finished turn. The first number is the wait that matters."""
    fields = [
        ("first audio", snapshot.first_audio),
        ("asr", snapshot.asr_ms),
        ("llm first token", snapshot.llm_ttft),
        ("tts first audio", snapshot.ttfa),
        ("total", snapshot.voice_to_voice),
    ]
    shown = [f"{name} {_seconds(value)}" for name, value in fields if value is not None]
    return "  \u21b3 " + " \u00b7 ".join(shown) if shown else ""


def _skip_asr(reason: str, text: str) -> Literal["skip"]:
    debug("asr.skip {} ({} chars)", reason, len(text.strip()))
    return "skip"


def _accept_asr(
    text: str,
    last_user: str,
    *,
    stale: bool = True,
) -> Literal["skip", "quit", "ok"]:
    if is_backchannel(text):
        return _skip_asr("backchannel", text)
    if is_asr_hallucination(text):
        return _skip_asr("hallucination", text)
    if is_quit_utterance(text):
        return "quit"
    # Only a clip that ended almost as the mic opened can be a stale copy. A
    # later repeat is the user saying it again on purpose.
    if stale and same_utterance(text, last_user):
        return "skip"
    return "ok"


@dataclass(frozen=True)
class _HeardLine:
    kind: Literal["bye", "again", "fatal", "line"]
    text: str = ""
    asr_ms: float = 0.0
    started: float = 0.0
    trace_id: str | None = None
    code: int = 0


async def _recognize(
    loop: _Loop,
    wav: bytes,
    last_user: str,
    *,
    stale: bool = True,
) -> _HeardLine:
    c = loop.config
    started = time.perf_counter()
    asr_parent = make_traceparent()
    try:
        text = await fish_asr(
            wav,
            c.fish_api_key,
            base=c.fish_base,
            language=c.fish_asr_language,
            model=c.asr_model,
            extra_headers={"traceparent": asr_parent},
            cancel=loop.session.stop,
            client=loop.asr_http,
        )
    except FishHttpError as e:
        if loop.session.stop.is_set():
            return _HeardLine("bye")
        warn(f"[asr] {e.status} {e.message}")
        if e.status in _FATAL_FISH:
            return _HeardLine("fatal", code=EXIT_FATAL)
        return _HeardLine("again")
    except Exception as e:
        if loop.session.stop.is_set():
            return _HeardLine("bye")
        warn(f"[asr] {e}")
        return _HeardLine("again")
    if loop.session.stop.is_set():
        return _HeardLine("bye")
    asr_ms = elapsed_ms(started)
    decision = _accept_asr(text, last_user, stale=stale)
    if decision == "quit":
        return _HeardLine("bye")
    if decision == "skip":
        return _HeardLine("again")
    conversation("you", text)
    return _HeardLine(
        "line",
        text=text,
        asr_ms=asr_ms,
        started=started,
        trace_id=trace_id_of(asr_parent),
    )


async def _hear_line(loop: _Loop, last_user: str) -> _HeardLine:
    if env_debug():
        debug("listen.waiting for you")
    else:
        console_print("listening…")
    clear_turn()
    trace("listen.waiting device={}", loop.device)
    try:
        prefix = loop.barge_prefix
        loop.barge_prefix = b""
        opened = time.monotonic()
        wav = await asyncio.to_thread(
            record_utterance,
            loop.device,
            loop.session.stop,
            prefix=prefix,
            tune=loop.config.listen,
            aec=loop.session.aec,
        )
    except PortAudioMissingError as e:
        warn(str(e))
        return _HeardLine("fatal", code=EXIT_FATAL)
    if loop.session.stop.is_set():
        return _HeardLine("bye")
    if not wav:
        debug("listen.dropped (too short or none)")
        return _HeardLine("again")
    mark_turn()
    # A barge-in clip carries fresh speech, so it is never a stale copy.
    window = loop.config.repeat_window_s
    stale = not prefix and time.monotonic() - opened < window
    heard = await _recognize(loop, wav, last_user, stale=stale)
    # Quit during the Fish request used to come back as a normal line, so
    # the LLM still answered after Ctrl+C.
    if loop.session.stop.is_set():
        return _HeardLine("bye")
    return heard


def bye() -> int:
    """Print the quit line and return success.

    Returns
    -------
    int
        ``EXIT_OK`` (0). Fatal Fish and PortAudio failures use 2 instead.
    """
    console_print("\nbye")
    return EXIT_OK


def _opening_history(system_prompt: str) -> tuple[list[dict[str, str]], int]:
    """Build the starting history and say how many leading messages stay pinned.

    Parameters
    ----------
    system_prompt : str
        The system prompt for this session.

    Returns
    -------
    tuple of list and int
        The history and the count of messages that trimming never drops. With
        the default prompt, one opening exchange with several cues is pinned
        after the system message, because the model copies the pattern of the
        replies it sees. A custom prompt gets no seed, so it stays in control
        of how the model replies.
    """
    history = [{"role": "system", "content": system_prompt}]
    if system_prompt == DEFAULT_SYSTEM_PROMPT:
        for user, assistant in DEFAULT_SEED_EXCHANGE:
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": assistant})
    return history, len(history)


def _trim_history(
    history: list[dict[str, str]],
    turns: int,
    pinned: int = _KEEP_SYSTEM,
) -> None:
    cap = pinned + turns * _ROLES_PER_TURN
    while len(history) > cap:
        # Drop the oldest user and assistant together. Popping one message
        # leaves that assistant answering the next user.
        paired = (
            len(history) > pinned + 1
            and history[pinned]["role"] == "user"
            and history[pinned + 1]["role"] == "assistant"
        )
        if paired:
            del history[pinned : pinned + _ROLES_PER_TURN]
            continue
        del history[pinned]


def _remember_user(
    history: list[dict[str, str]],
    text: str,
    turns: int,
    pinned: int = _KEEP_SYSTEM,
) -> None:
    history.append({"role": "user", "content": text})
    _trim_history(history, turns, pinned)


async def _answer_line(loop: _Loop, heard: _HeardLine) -> int | None:
    cancel = threading.Event()
    llm_cancel = asyncio.Event()
    loop.session.turn.bind(cancel, llm_cancel)
    if loop.config.stream_tts:
        snapshot, fatal = await _stream_turn(loop, heard, cancel, llm_cancel)
        if fatal is not None:
            return fatal
    else:
        reply, ttft_ms = await _collect_reply(
            loop,
            llm_cancel=llm_cancel,
            trace_id=heard.trace_id,
            started=time.perf_counter(),
        )
        snapshot = LatencySnapshot(asr_ms=heard.asr_ms, llm_ttft=ttft_ms, trace_id=heard.trace_id)
        if loop.session.stop.is_set():
            return bye()
        if reply:
            snapshot, fatal = await _speak_reply(
                loop,
                reply,
                cancel,
                snapshot,
                started=heard.started,
                trace_id=heard.trace_id,
            )
            if fatal is not None:
                return fatal
    summary = _turn_summary(snapshot)
    if env_debug():
        debug("turn.summary {}", summary.strip().removeprefix("\u21b3 "))
        trace("turn.timing {}", snapshot.log_line())
    else:
        console_print(summary, flush=True)
    stop = loop.session.stop
    loop.session.turn.fire()
    loop.session.turn.clear()
    barged = bool(loop.barge_prefix)
    if stop.is_set() or (
        not barged and await asyncio.to_thread(stop.wait, loop.config.barge.cooldown_s)
    ):
        return bye()
    return None


async def duplex_turns(
    c: VoiceCliConfig,
    tts: IsolatedFishTts,
    device: str | int | None,
    backend: ChatBackend,
    session: DuplexSession | None = None,
) -> int:
    """Mic, Fish ASR, LLM, then one isolated TTS turn, until quit.

    Parameters
    ----------
    c : VoiceCliConfig
        Env-backed duplex settings.
    tts : IsolatedFishTts
        Live TTS client. Each reply uses ``speak_isolated``.
    device : str or int or None
        Mic and speaker device, or None for the host default.
    backend : ChatBackend
        Open chat backend, usually from ``open_chat_backend``.
    session : DuplexSession or None, optional
        Cancel flags and the echo canceller. A new one is created when omitted.
        Pass your own to wire ``request_quit`` to a signal handler.

    Returns
    -------
    int
        ``0`` on a normal bye. ``2`` when Fish returns 401, 402, or 403, or
        PortAudio is missing.

    Notes
    -----
    One sampled trace id is shared by ASR and the TTS websocket for that turn.
    Barge-in keeps the audio that tripped the gate and skips the post-speak
    cooldown. ``session.request_quit`` cancels the in-flight reply and ends
    the loop.
    """
    session = session or DuplexSession(aec=EchoCanceller(c.aec))
    async with AsyncExitStack() as stack:
        asr_http = await stack.enter_async_context(asr_client())
        history, pinned = _opening_history(c.system_prompt)
        loop = _Loop(
            config=c,
            tts=tts,
            device=device,
            backend=backend,
            session=session,
            asr_http=asr_http,
            history=history,
            pinned=pinned,
        )
        last_user = ""
        while True:
            heard = await _hear_line(loop, last_user)
            if heard.kind == "bye":
                return bye()
            if heard.kind == "fatal":
                return heard.code
            if heard.kind == "again":
                continue
            last_user = heard.text
            _remember_user(loop.history, heard.text, c.history_turns, loop.pinned)
            code = await _answer_line(loop, heard)
            if code is not None:
                return code
