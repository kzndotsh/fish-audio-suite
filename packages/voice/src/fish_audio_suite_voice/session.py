"""Fish websocket turn: retry before the first audio byte, on a private loop."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from typing import Any

import httpx
from fishaudio import AsyncFishAudio

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    ensure_trace_headers,
    fish_backoff_seconds,
)
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.wire import (
    EventAcc,
    Heard,
    IsolatedResult,
    TurnRun,
    TurnSpec,
    as_async,
    is_cancel_noise,
    isolated_result,
    quiet_shutdown,
    send_turn,
    text_events,
    turn_failure,
)

_WS_TIMEOUT_S = 240.0

__all__ = [
    "IsolatedResult",
    "TurnSpec",
    "as_async",
    "is_cancel_noise",
    "run_isolated",
    "run_turn",
    "text_events",
]


class _HeldClient:
    def __init__(self) -> None:
        self.client: AsyncFishAudio | None = None

    def open(self, spec: TurnSpec, headers: dict[str, str]) -> AsyncFishAudio:
        http = httpx.AsyncClient(
            base_url=spec.base_url,
            headers=headers,
            timeout=httpx.Timeout(_WS_TIMEOUT_S),
            http2=False,
        )
        self.client = AsyncFishAudio(
            api_key=spec.api_key,
            base_url=spec.base_url,
            httpx_client=http,
        )
        return self.client

    async def close(self) -> None:
        client = self.client
        if client is None:
            return
        self.client = None
        with contextlib.suppress(Exception):
            await client.close()


@dataclass
class _Turn:
    run: TurnRun
    held: _HeldClient
    headers: dict[str, str]


async def _one_attempt(turn: _Turn, events: AsyncIterator[Any], attempt: int) -> bool:
    """True when the turn should stop. A false result is a retry before any audio."""
    run = turn.run
    client = turn.held.open(run.spec, turn.headers)
    try:
        await send_turn(client, events, run, close_client=turn.held.close)
    except (asyncio.CancelledError, GeneratorExit) as exc:
        await turn.held.close()
        if run.cancel.is_set() or is_cancel_noise(exc):
            return True
        raise
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        fate = turn_failure(
            exc,
            attempt=attempt,
            sent_text=run.sent_text,
            got_audio=run.audio.got_audio,
            cancel=run.cancel,
        )
        await turn.held.close()
        if not fate.retry:
            run.err_status = fate.err_status
            run.err_message = fate.err_message
            return True
        await asyncio.sleep(fish_backoff_seconds(attempt))
        return False
    return True


def _turn_headers(spec: TurnSpec, sent_text: str) -> dict[str, str]:
    extra = ensure_trace_headers(spec.trace_headers)
    debug(
        "tts.start voice={} model={} format={} sr={} latency={} speed={} chars={} trace={}",
        spec.voice_id,
        spec.model,
        spec.audio_format,
        spec.sample_rate,
        spec.latency,
        spec.speed,
        len(sent_text),
        extra.get("traceparent", ""),
    )
    return extra


async def run_turn(
    spec: TurnSpec,
    events: AsyncIterator[Any],
    sink: PlaybackSink,
    cancel: threading.Event,
    *,
    sent_text: str,
) -> IsolatedResult:
    run = TurnRun(
        spec=spec,
        sink=sink,
        cancel=cancel,
        sent_text=sent_text,
        acc=EventAcc(),
        t0=time.perf_counter(),
        audio=Heard(),
    )
    turn = _Turn(run=run, held=_HeldClient(), headers=_turn_headers(spec, sent_text))
    sink.start()

    try:
        for attempt in range(FISH_RETRY_ATTEMPTS):
            if await _one_attempt(turn, events, attempt):
                break
    finally:
        sink.finish(kill=cancel.is_set())
        await turn.held.close()

    return isolated_result(run)


def run_isolated(coro: Coroutine[Any, Any, IsolatedResult]) -> IsolatedResult:
    """Fresh loop for Fish WS. Close the client on cancel; do not aclose the iterator."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        if not loop.is_closed() and not loop.is_running():
            with contextlib.suppress(BaseException):
                loop.run_until_complete(quiet_shutdown(loop))
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)
