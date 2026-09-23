"""One duplex turn: listen, Fish ASR, LLM, then isolated TTS."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal

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
    scrub_tts,
    trace_id_of,
)
from fish_audio_suite_voice.asr import fish_asr
from fish_audio_suite_voice.barge import BargeGate, post_speak_cooldown_s
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.listen import record_utterance
from fish_audio_suite_voice.live import IsolatedFishTts, IsolatedResult, is_cancel_noise
from fish_audio_suite_voice.llm import llm_token_stream
from fish_audio_suite_voice.playback import PortAudioMissingError, make_sink
from fish_audio_suite_voice.signals import HISTORY_TURNS, STOP_RECORD, TURN

EXIT_OK = 0
EXIT_FATAL = 2
_KEEP_SYSTEM = 1
_ROLES_PER_TURN = 2
_FATAL_FISH = frozenset({401, 402, 403})


@dataclass
class _Loop:
    config: VoiceCliConfig
    tts: IsolatedFishTts
    device: str | int | None
    or_client: Any | None
    history: list[dict[str, str]]
    llm_session: str


async def _collect_reply(
    loop: _Loop,
    *,
    llm_cancel: asyncio.Event,
    trace_id: str | None,
    started: float,
) -> tuple[str, float | None]:
    parts: list[str] = []
    ttft_ms: float | None = None
    c = loop.config
    try:
        async for tok in llm_token_stream(
            loop.history,
            base=c.llm_base,
            key=c.llm_key,
            model=c.llm_model,
            cancel=llm_cancel,
            client=loop.or_client,
            session_id=loop.llm_session,
            trace_id=trace_id,
        ):
            if ttft_ms is None:
                ttft_ms = elapsed_ms(started)
                print(f"  [llm ttft {ttft_ms:.0f}ms]", flush=True)
            parts.append(tok)
            print(tok, end="", flush=True)
        print()
    except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError) as e:
        if not is_cancel_noise(e):
            print(f"[llm] {e}", file=sys.stderr)
    reply = "".join(parts).strip()
    print(f"  [got {len(reply)} chars]", flush=True)
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
        print("  [tts cancelled before audio]", flush=True)
    elif not result.got_audio and result.error_status is None:
        print(
            f"  [tts silent] voice={c.fish_voice_id} model={c.fish_tts_model}",
            flush=True,
        )
    spoken = result.spoken_so_far
    if spoken and (result.got_audio or not result.cancelled):
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
    scrubbed = ensure_lead_cue(normalize_cues(scrub_tts(reply)))
    if is_tts_junk(scrubbed):
        print("  (skip junk TTS)", flush=True)
        return snapshot, None
    c = loop.config
    barge = BargeGate(device=loop.device)
    barge.start_after_bleed(cancel)
    sink = make_sink(
        c.playback,
        path=None,
        sample_rate=c.fish_sample_rate,
        device=loop.device,
        cancel=cancel,
    )
    loop.tts.trace_headers = {"traceparent": make_traceparent(trace_id=trace_id)}
    result = await asyncio.to_thread(loop.tts.speak_isolated, scrubbed, sink, cancel)
    return _after_speech(loop, snapshot, result, started=started)


def _skip_asr(reason: str, text: str) -> Literal["skip"]:
    debug("asr skip {} chars={}", reason, len(text.strip()))
    return "skip"


def _accept_asr(text: str, last_user: str) -> Literal["skip", "quit", "ok"]:
    if is_backchannel(text):
        return _skip_asr("backchannel", text)
    if is_asr_hallucination(text):
        print("[asr skip hallucination]", file=sys.stderr)
        return _skip_asr("hallucination", text)
    if is_quit_utterance(text):
        return "quit"
    if text.strip() == last_user.strip():
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


async def _recognize(loop: _Loop, wav: bytes, last_user: str) -> _HeardLine:
    c = loop.config
    started = time.perf_counter()
    asr_parent = make_traceparent()
    try:
        text = await fish_asr(
            wav,
            c.fish_api_key,
            base=c.fish_base,
            language=c.fish_asr_language,
            extra_headers={"traceparent": asr_parent},
        )
    except FishHttpError as e:
        print(f"[asr] {e.status} {e.message}", file=sys.stderr)
        if e.status in _FATAL_FISH:
            return _HeardLine("fatal", code=EXIT_FATAL)
        return _HeardLine("again")
    except Exception as e:
        print(f"[asr] {e}", file=sys.stderr)
        return _HeardLine("again")
    asr_ms = elapsed_ms(started)
    decision = _accept_asr(text, last_user)
    if decision == "quit":
        return _HeardLine("bye")
    if decision == "skip":
        return _HeardLine("again")
    print(f"you: {text}  [asr {asr_ms:.0f}ms]")
    return _HeardLine(
        "line",
        text=text,
        asr_ms=asr_ms,
        started=started,
        trace_id=trace_id_of(asr_parent),
    )


async def _hear_line(loop: _Loop, last_user: str) -> _HeardLine:
    print("listening…")
    debug("listen.waiting device={}", loop.device)
    try:
        wav = await asyncio.to_thread(record_utterance, loop.device, STOP_RECORD)
    except PortAudioMissingError as e:
        print(e, file=sys.stderr)
        return _HeardLine("fatal", code=EXIT_FATAL)
    if STOP_RECORD.is_set():
        return _HeardLine("bye")
    if not wav:
        debug("listen.dropped (too short or none)")
        return _HeardLine("again")
    debug("listen.wav bytes={}", len(wav))
    return await _recognize(loop, wav, last_user)


def bye() -> int:
    print("\nbye")
    return EXIT_OK


def _remember_user(history: list[dict[str, str]], text: str) -> None:
    history.append({"role": "user", "content": text})
    while len(history) > _KEEP_SYSTEM + HISTORY_TURNS * _ROLES_PER_TURN:
        history.pop(_KEEP_SYSTEM)


async def _answer_line(loop: _Loop, heard: _HeardLine) -> int | None:
    cancel = threading.Event()
    llm_cancel = asyncio.Event()
    TURN.bind(cancel, llm_cancel)
    reply, ttft_ms = await _collect_reply(
        loop,
        llm_cancel=llm_cancel,
        trace_id=heard.trace_id,
        started=time.perf_counter(),
    )
    snapshot = LatencySnapshot(srt=heard.asr_ms, llm_ttft=ttft_ms, trace_id=heard.trace_id)
    if STOP_RECORD.is_set():
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
    print(f"  {snapshot.log_line()}", flush=True)
    TURN.fire()
    TURN.clear()
    if STOP_RECORD.is_set() or await asyncio.to_thread(STOP_RECORD.wait, post_speak_cooldown_s()):
        return bye()
    return None


async def duplex_turns(
    c: VoiceCliConfig,
    tts: IsolatedFishTts,
    device: str | int | None,
    or_client: Any | None,
) -> int:
    loop = _Loop(
        config=c,
        tts=tts,
        device=device,
        or_client=or_client,
        history=[{"role": "system", "content": c.system_prompt}],
        llm_session=uuid.uuid4().hex,
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
        _remember_user(loop.history, heard.text)
        code = await _answer_line(loop, heard)
        if code is not None:
            return code
