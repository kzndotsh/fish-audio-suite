"""One duplex turn: listen, Fish ASR, LLM, then isolated TTS."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import Literal

import httpx

from fish_audio_suite_kit import (
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
    console_print,
    debug,
    end_reply_line,
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


async def _collect_reply(
    loop: _Loop,
    *,
    llm_cancel: asyncio.Event,
    trace_id: str | None,
    started: float,
) -> tuple[str, float | None]:
    parts: list[str] = []
    ttft_ms: float | None = None
    try:
        async for tok in loop.backend.stream(loop.history, cancel=llm_cancel, trace_id=trace_id):
            if ttft_ms is None:
                ttft_ms = elapsed_ms(started)
                console_print(f"  [llm ttft {ttft_ms:.0f}ms]", flush=True)
                write_reply_token("llm: ")
            parts.append(tok)
            write_reply_token(tok)
    except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError) as e:
        if not is_cancel_noise(e, cancelled=llm_cancel.is_set()):
            warn(f"[llm] {e}")
    finally:
        end_reply_line()
    reply = "".join(parts).strip()
    console_print(f"  [got {len(reply)} chars]", flush=True)
    return reply, ttft_ms


def _after_speech(
    loop: _Loop,
    snapshot: LatencySnapshot,
    result: IsolatedResult,
    *,
    started: float,
) -> tuple[LatencySnapshot, int | None]:
    c = loop.config
    history = loop.history
    debug(
        "tts.done cancelled={} audio={} bytes={} spoken_chars={} err={} {}",
        result.cancelled,
        result.got_audio,
        result.bytes_played,
        len(result.spoken_so_far),
        result.error_status,
        result.error_message or "",
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
            result = await asyncio.to_thread(loop.tts.speak_isolated, scrubbed, sink, cancel)
        except PortAudioMissingError as exc:
            warn(str(exc))
            return snapshot, EXIT_FATAL
        if result.cancelled and barge.captured:
            loop.barge_prefix = barge.captured
        return _after_speech(loop, snapshot, result, started=started)
    finally:
        # watch() holds the mic until this event is set. The next listen
        # opens the same device as soon as this function returns.
        cancel.set()
        thread.join(timeout=_BARGE_JOIN_S)


def _skip_asr(reason: str, text: str) -> Literal["skip"]:
    debug("asr skip {} chars={}", reason, len(text.strip()))
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
        warn("[asr skip hallucination]")
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
    console_print(f"you: {text}  [asr {asr_ms:.0f}ms]")
    return _HeardLine(
        "line",
        text=text,
        asr_ms=asr_ms,
        started=started,
        trace_id=trace_id_of(asr_parent),
    )


async def _hear_line(loop: _Loop, last_user: str) -> _HeardLine:
    console_print("listening…")
    debug("listen.waiting device={}", loop.device)
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
    debug("listen.wav bytes={}", len(wav))
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


def _trim_history(history: list[dict[str, str]], turns: int) -> None:
    cap = _KEEP_SYSTEM + turns * _ROLES_PER_TURN
    while len(history) > cap:
        # Drop the oldest user and assistant together. Popping one message
        # leaves that assistant answering the next user.
        paired = (
            len(history) > _KEEP_SYSTEM + 1
            and history[_KEEP_SYSTEM]["role"] == "user"
            and history[_KEEP_SYSTEM + 1]["role"] == "assistant"
        )
        if paired:
            del history[_KEEP_SYSTEM : _KEEP_SYSTEM + _ROLES_PER_TURN]
            continue
        del history[_KEEP_SYSTEM]


def _remember_user(history: list[dict[str, str]], text: str, turns: int) -> None:
    history.append({"role": "user", "content": text})
    _trim_history(history, turns)


async def _answer_line(loop: _Loop, heard: _HeardLine) -> int | None:
    cancel = threading.Event()
    llm_cancel = asyncio.Event()
    loop.session.turn.bind(cancel, llm_cancel)
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
    console_print(f"  {snapshot.log_line()}", flush=True)
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
        loop = _Loop(
            config=c,
            tts=tts,
            device=device,
            backend=backend,
            session=session,
            asr_http=asr_http,
            history=[{"role": "system", "content": c.system_prompt}],
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
            _remember_user(loop.history, heard.text, c.history_turns)
            code = await _answer_line(loop, heard)
            if code is not None:
                return code
