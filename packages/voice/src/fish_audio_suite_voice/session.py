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
from fishaudio.exceptions import AuthenticationError

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
_RETRY_POLL_S = 0.05

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
        # The SDK copies the key into Authorization. A newline makes the
        # websocket client raise before Fish can answer, so every later
        # turn fails the same way instead of stopping on 401.
        token = spec.api_key.strip()
        if not token or any(ord(ch) < 32 or ord(ch) >= 127 for ch in token):
            raise AuthenticationError(401, "Invalid Token")
        http = httpx.AsyncClient(
            base_url=spec.base_url,
            headers=headers,
            timeout=httpx.Timeout(_WS_TIMEOUT_S),
            http2=False,
        )
        self.client = AsyncFishAudio(
            api_key=token,
            base_url=spec.base_url,
            httpx_client=http,
        )
        return self.client

    async def close(self) -> None:
        client = self.client
        if client is None:
            return
        self.client = None
        # A task group of CancelledError is a BaseException, so Exception
        # does not catch it. That used to escape after the audio had already
        # played, and the next turn repeated the line.
        with contextlib.suppress(Exception, BaseExceptionGroup):
            await client.close()


@dataclass
class _Turn:
    run: TurnRun
    held: _HeldClient
    headers: dict[str, str]


async def _retry_pause(cancel: threading.Event, seconds: float) -> bool:
    """Wait out a Fish backoff. True means the turn was cancelled."""
    # One sleep ignores barge-in. The next attempt would speak after the
    # user already interrupted, or Ctrl+C would wait out the full backoff.
    left = seconds
    while left > 0:
        if cancel.is_set():
            return True
        step = min(_RETRY_POLL_S, left)
        await asyncio.sleep(step)
        left -= step
    return cancel.is_set()


async def _one_attempt(turn: _Turn, events: AsyncIterator[Any], attempt: int) -> bool:
    """Return whether this attempt should stop the turn. False means retry before audio."""
    run = turn.run
    if run.cancel.is_set():
        return True
    try:
        client = turn.held.open(run.spec, turn.headers)
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
        return await _retry_pause(run.cancel, fish_backoff_seconds(attempt))
    return True


def _turn_headers(spec: TurnSpec, sent_text: str) -> dict[str, str]:
    extra = ensure_trace_headers(spec.trace_headers)
    debug(
        "tts.start voice={} model={} format={} sr={} latency={} speed={} chars={} trace={} text={}",
        spec.voice_id,
        spec.model,
        spec.audio_format,
        spec.sample_rate,
        spec.latency,
        spec.speed,
        len(sent_text),
        extra.get("traceparent", ""),
        sent_text,
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
    """Play one Fish turn, retrying 429 and 5xx only before the first audio byte.

    Parameters
    ----------
    spec : TurnSpec
        Voice, model, and format.
    events : AsyncIterator
        Text and flush events. Replayed from ``sent_text`` when that string
        is non-empty.
    sink : PlaybackSink
        Started here and finished in ``finally``, including on cancel.
    cancel : threading.Event
        Barge-in or Ctrl+C.
    sent_text : str
        Full text to replay. Empty means the original ``events`` iterator is
        the only source, so a failed attempt cannot be repeated.

    Returns
    -------
    IsolatedResult
        Spoken prefix derived from bytes actually played.

    Notes
    -----
    The httpx client is closed here. The websocket iterator is not
    ``aclose()``'d; closing the client ends the socket.
    """
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

    try:
        # start() can open the DAC and then raise. finish() still has to
        # close that stream, or the device stays busy for the next turn.
        sink.start()
        for attempt in range(FISH_RETRY_ATTEMPTS):
            if await _one_attempt(turn, events, attempt):
                break
    finally:
        try:
            sink.finish(kill=cancel.is_set())
        finally:
            # A failed file write must not skip the client close. The socket
            # would stay open until the process exits.
            await turn.held.close()

    return isolated_result(run)


def run_isolated(coro: Coroutine[Any, Any, IsolatedResult]) -> IsolatedResult:
    """Run ``coro`` on a new event loop and close that loop.

    Parameters
    ----------
    coro : Coroutine
        Usually ``IsolatedFishTts.speak``. It must close its own httpx client.

    Returns
    -------
    IsolatedResult
        Whatever ``coro`` returns.

    Notes
    -----
    ``asyncio.wait_for`` must not wrap the websocket read. A timeout cancels
    the generator, and ``aclose()`` on that iterator raises. Poll with
    ``asyncio.wait`` and close the client instead.
    """
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
