"""Fish websocket turn: retry before the first audio byte, on a private loop.

``run_turn`` plays one turn through the shared Fish retry loop (``pause.fish_retries``), and
``turn_failure`` decides which failure of an attempt is worth another one.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Final, cast

import httpx
from fishaudio import AsyncFishAudio
from fishaudio.exceptions import APIError, AuthenticationError, ValidationError, WebSocketError
from fishaudio.types import Model

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    describe_request_error,
    ensure_trace_headers,
    fish_attempt_exhausted,
    is_header_safe,
    should_retry_fish_status,
)
from fish_audio_suite_voice.cancel import first_real_error, is_cancel_noise, quiet_shutdown
from fish_audio_suite_voice.debug import debug, warn, with_detail
from fish_audio_suite_voice.pause import Retry, fish_retries
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.tts_text import WholeText
from fish_audio_suite_voice.wire import (
    AudioArrival,
    FishSocket,
    SentText,
    SinkError,
    TtsResult,
    TurnRun,
    TurnSpec,
    send_turn,
    tts_result,
)

_WS_TIMEOUT_S: Final = 240.0

__all__ = [
    "SdkSocket",
    "TtsFailure",
    "run_isolated",
    "run_turn",
    "turn_failure",
]


class SdkSocket:
    """The Fish SDK's realtime websocket, with a fresh client for each attempt.

    Parameters
    ----------
    headers : dict of str to str
        Sent on the websocket upgrade, such as the trace headers.
    """

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.client: AsyncFishAudio | None = None

    def stream(self, events: AsyncIterator[Any], spec: TurnSpec) -> AsyncIterator[bytes]:
        """Open a client and its websocket for one attempt. See ``FishSocket.stream``.

        Parameters
        ----------
        events : AsyncIterator
            Text and flush events for this attempt.
        spec : TurnSpec
            Voice, model, format and key.

        Returns
        -------
        AsyncIterator
            Audio chunks from the SDK.

        Raises
        ------
        AuthenticationError
            The key is empty or holds a character a header cannot carry.
        """
        # The SDK copies the key into Authorization. A newline makes the
        # websocket client raise before Fish can answer, so every later
        # turn fails the same way instead of stopping on 401.
        token = spec.api_key.strip()
        if not token or not is_header_safe(token):
            raise AuthenticationError(401, "Invalid Token")
        http = httpx.AsyncClient(
            base_url=spec.base_url,
            headers=self.headers,
            timeout=httpx.Timeout(_WS_TIMEOUT_S),
            http2=False,
        )
        self.client = AsyncFishAudio(
            api_key=token,
            base_url=spec.base_url,
            httpx_client=http,
        )
        # TTSConfig has no features. Quality-guard stays on the proxy HTTP path. The format,
        # latency and speed travel in the config only: the SDK's keyword overrides would
        # send the same values again.
        return self.client.tts.stream_websocket(
            events,
            reference_id=spec.voice_id,
            config=spec.config,
            model=cast(Model, spec.model),
        )

    async def close(self) -> None:
        """Close the attempt's client, which ends its websocket."""
        client = self.client
        if client is None:
            return
        self.client = None
        # A task group of CancelledError is a BaseException, so Exception
        # does not catch it. Escaping after the audio has played, it would
        # make the next turn repeat the line.
        with contextlib.suppress(Exception, BaseExceptionGroup):
            await client.close()


@dataclass(slots=True)
class _Turn:
    run: TurnRun
    socket: FishSocket
    source: WholeText | AsyncIterator[Any]

    def events(self) -> AsyncIterator[Any]:
        # A whole reply is cut afresh for each attempt. A stream can be read only once.
        if isinstance(self.source, WholeText):
            return self.source.events(self.run.cancel)
        return self.source


async def _one_attempt(turn: _Turn, attempt: int) -> Retry | None:
    """Play one attempt. ``Retry`` asks for another, before any audio; None ends the turn."""
    run = turn.run
    try:
        await send_turn(turn.socket, turn.events(), run)
    except (asyncio.CancelledError, GeneratorExit) as exc:
        await turn.socket.close()
        if run.cancel.is_set() or is_cancel_noise(exc):
            return None
        raise
    except SinkError as exc:
        await turn.socket.close()
        # A sink that fails is the caller's problem, not Fish's: its own error goes up,
        # as a sink that fails to open does. Torn down by a cancel, it is only noise.
        if run.cancel.is_set():
            return None
        raise exc.error from None
    except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001 - turn boundary: classified by turn_failure and reported in the result
        fate = turn_failure(
            exc,
            attempt=attempt,
            sent_text=run.sent_text,
            got_audio=run.audio.got_audio,
            cancel=run.cancel,
        )
        await turn.socket.close()
        if fate.retry:
            # The SDK's APIError carries no headers, so a 429's Retry-After cannot be read.
            return Retry()
        run.error_status = fate.error_status
        run.error_message = fate.error_message
    return None


def _turn_headers(spec: TurnSpec, sent_text: str) -> dict[str, str]:
    extra = ensure_trace_headers(spec.trace_headers)
    debug(
        "tts.start voice={} model={} {} {}Hz latency={} speed={}{}",
        spec.voice_id[:8],
        spec.model,
        spec.audio_format,
        spec.sample_rate,
        spec.latency,
        spec.speed,
        f" chars={len(sent_text)}" if sent_text else " streaming",
    )
    return extra


async def run_turn(
    spec: TurnSpec,
    source: WholeText | AsyncIterator[Any],
    sink: PlaybackSink,
    cancel: threading.Event,
    *,
    on_first_audio: Callable[[], None] | None = None,
    socket: FishSocket | None = None,
) -> TtsResult:
    """Play one Fish turn, retrying a retryable failure only before the first audio byte.

    Parameters
    ----------
    spec : TurnSpec
        Voice, model, and format.
    source : WholeText or AsyncIterator
        A whole shaped reply, which a failed attempt can send again, or a stream of
        text and flush events, which cannot be repeated.
    sink : PlaybackSink
        Started here and finished in ``finally``, including on cancel.
    cancel : threading.Event
        Barge-in or Ctrl+C.
    on_first_audio : Callable or None, optional
        Called once, on the websocket's thread, when the first audio chunk
        arrives. It must not block.
    socket : FishSocket or None, optional
        Where each attempt streams. The Fish SDK when omitted.

    Returns
    -------
    TtsResult
        Spoken prefix derived from bytes actually played.

    Notes
    -----
    ``turn_failure`` decides what is retryable. The socket is closed here. The
    websocket iterator is not ``aclose()``'d; closing the socket ends it.
    """
    sent_text = source.text if isinstance(source, WholeText) else ""
    run = TurnRun(
        spec=spec,
        sink=sink,
        cancel=cancel,
        sent_text=sent_text,
        acc=SentText(),
        t0=time.perf_counter(),
        audio=AudioArrival(),
        on_first_audio=on_first_audio,
    )
    headers = _turn_headers(spec, sent_text)
    turn = _Turn(run=run, socket=SdkSocket(headers) if socket is None else socket, source=source)

    try:
        # start() can open the DAC and then raise. finish() still has to
        # close that stream, or the device stays busy for the next turn.
        sink.start()
        await fish_retries(lambda attempt: _one_attempt(turn, attempt), cancel.is_set)
    finally:
        try:
            sink.finish(kill=cancel.is_set())
        finally:
            # A failed file write must not skip the client close. The socket
            # would stay open until the process exits.
            await turn.socket.close()

    return tts_result(run)


def run_isolated(coro: Coroutine[Any, Any, TtsResult]) -> TtsResult:
    """Run ``coro`` on a new event loop and close that loop.

    Parameters
    ----------
    coro : Coroutine
        One turn, as ``FishSpeaker`` builds it. It must close its own httpx client.

    Returns
    -------
    TtsResult
        Whatever ``coro`` returns.

    Notes
    -----
    The loop is built and torn down by hand, not with ``asyncio.Runner`` or
    ``asyncio.run``. Both finish with ``shutdown_asyncgens``, which calls
    ``aclose()`` on the Fish websocket iterator. That iterator holds an anyio
    task group, so closing it from another task raises "cancel scope in a
    different task" and logs an error on every cancelled turn.

    ``asyncio.wait_for`` must not wrap the websocket read. A timeout cancels
    the generator, and ``aclose()`` on that iterator raises. Poll with
    ``asyncio.wait`` and close the client instead.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            if not loop.is_closed() and not loop.is_running():
                # KeyboardInterrupt and SystemExit are not suppressed here, but
                # the loop is still closed below.
                with contextlib.suppress(Exception, BaseExceptionGroup, asyncio.CancelledError):
                    loop.run_until_complete(quiet_shutdown(loop))
        finally:
            if not loop.is_closed():
                loop.close()
            asyncio.set_event_loop(None)


@dataclass(frozen=True, slots=True)
class TtsFailure:
    """How one failed Fish attempt should be handled.

    Attributes
    ----------
    retry : bool
        Whether the turn should be tried again.
    error_status : int or None
        Fish HTTP status, when the failure had one.
    error_message : str or None
        Text to report. None when the failure was a barge-in.
    """

    retry: bool
    error_status: int | None = None
    error_message: str | None = None


def turn_failure(
    exc: BaseException,
    *,
    attempt: int,
    sent_text: str,
    got_audio: bool,
    cancel: threading.Event,
) -> TtsFailure:
    """Decide whether a Fish exception can be retried.

    Parameters
    ----------
    exc : BaseException
        Error from one attempt. Exception groups are unwrapped to the first real error.
    attempt : int
        Zero-based attempt. ``FISH_RETRY_ATTEMPTS - 1`` is the last.
    sent_text : str
        Text available to replay. Empty blocks a retry.
    got_audio : bool
        True after the first audio byte. Retries stop there so the listener
        does not hear the sentence twice.
    cancel : threading.Event
        When set, the failure is a barge-in, not a retry.

    Returns
    -------
    TtsFailure
        ``retry`` is True only for a 429, a 5xx, a websocket error or a failed
        connection, before audio, with text to replay and attempts left.
    """
    root = first_real_error(exc)
    retry, status, message = _classify_fish_exc(root)
    if cancel.is_set() or (is_cancel_noise(root) and not got_audio):
        return TtsFailure(retry=False)
    # The socket died after audio started, and the turn was not cancelled.
    # Recording the unplayed tail makes the next turn assume it was heard.
    if is_cancel_noise(root):
        error_message = str(root)
        warn(f"[tts] {error_message}")
        return TtsFailure(retry=False, error_message=error_message)
    last = fish_attempt_exhausted(attempt)
    can_replay = bool(sent_text) and not got_audio
    if retry and not last and can_replay:
        warn(f"[tts] retry status={status} attempt={attempt + 1}/{FISH_RETRY_ATTEMPTS}")
        return TtsFailure(retry=True)
    error_message = message or str(root)
    warn(f"[tts] {status} {error_message}" if status is not None else f"[tts] {error_message}")
    return TtsFailure(retry=False, error_status=status, error_message=error_message)


def _classify_fish_exc(exc: BaseException) -> tuple[bool, int | None, str]:
    if is_cancel_noise(exc):
        return False, None, ""
    if isinstance(exc, APIError):
        return should_retry_fish_status(exc.status), exc.status, exc.message
    if isinstance(exc, ValidationError):
        return False, 400, str(exc)
    if isinstance(exc, WebSocketError):
        return True, None, str(exc)
    if isinstance(exc, httpx.RequestError):
        status, message = describe_request_error(exc, httpx.TimeoutException)
        return True, status, with_detail(message, exc)
    return False, None, str(exc)
