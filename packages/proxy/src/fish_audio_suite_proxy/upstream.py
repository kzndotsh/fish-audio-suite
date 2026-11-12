"""Send one request to Fish with a bounded, polite retry, and the headers every call carries."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import httpx

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    FishErrorBody,
    FishHttpError,
    ensure_trace_headers,
    fish_backoff_s,
    parse_fish_error,
    retry_after_s,
)
from fish_audio_suite_proxy.errors import FishProviderError, ProxyError

__all__ = ["RetryPolicy", "fish_send", "traced_model_headers"]

log: Final[logging.Logger] = logging.getLogger("fish-audio-suite-proxy")
_BODY_UNREADABLE: Final = "Fish answered with status {code} but the error body could not be read"

# Only a failure before the request reached Fish is safe to repeat. A read
# timeout may mean Fish already made, and billed, the audio.
_RETRY_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_CLIENT_CLOSED = 499


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to try and how long to keep trying.

    Attributes
    ----------
    attempts : int
        Total tries, including the first.
    deadline_s : float
        Wall-clock budget for all tries and pauses. 0 means no deadline.
    """

    attempts: int = FISH_RETRY_ATTEMPTS
    deadline_s: float = 0.0


type FishFiles = Mapping[str, tuple[str, bytes, str]]
"""Multipart files for a Fish request: field name to (filename, bytes, media type)."""


class FishHttp(Protocol):
    """The part of ``httpx.AsyncClient`` that ``fish_send`` uses."""

    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: bytes | None = None,
        data: Mapping[str, str] | None = None,
        files: FishFiles | None = None,
        json: object | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Request:
        """Build a request without sending it."""
        ...

    async def send(self, request: httpx.Request, *, stream: bool = False) -> httpx.Response:
        """Send a request, keeping the body open when ``stream`` is True."""
        ...


_DEADLINE_STATUS = 504
_DEADLINE_MESSAGE = "Fish did not answer before the retry deadline"


def _fixed(err: FishHttpError) -> ProxyError:
    # Fish never answered, so this is the proxy's own error, not a provider error.
    return ProxyError(err.status, err.message)


def _transport_error(exc: httpx.RequestError) -> ProxyError:
    """Classify a failed Fish request.

    Parameters
    ----------
    exc : httpx.RequestError
        The transport failure.

    Returns
    -------
    ProxyError
        504 with the kit's fixed timeout message for a timeout, 502 with the fixed
        unreachable message otherwise. The exception's own text can name hosts,
        addresses or errno values, so it is logged here and never sent to the client.
    """
    if isinstance(exc, httpx.TimeoutException):
        return _fixed(FishHttpError.for_timeout())
    log.warning("fish transport error: %s", exc)
    return _fixed(FishHttpError.for_unreachable())


def _error_ids(detail: FishErrorBody) -> dict[str, str]:
    """Return Fish's error ``code`` and ``request_id`` as the client's error metadata.

    ``transcribe-1-pro`` errors are ``{status, message, code, request_id}``, and
    the request id is also in ``x-request-id``. Fish support asks for it.
    """
    ids: dict[str, str] = {}
    if detail.code:
        ids["provider_code"] = detail.code
    if detail.request_id:
        ids["request_id"] = detail.request_id
    return ids


def _log_fish_error(status: int, ids: Mapping[str, str]) -> None:
    log.warning(
        "fish error status=%s code=%s request_id=%s",
        status,
        ids.get("provider_code", "-"),
        ids.get("request_id", "-"),
    )


async def _closed_error(upstream: httpx.Response) -> tuple[FishProviderError, FishHttpError]:
    """Read and close a failed Fish response.

    Parameters
    ----------
    upstream : httpx.Response
        A non-2xx response whose body is still open.

    Returns
    -------
    tuple of FishProviderError and FishHttpError
        The error for the client and the typed failure the retry loop reads
        (``retryable`` and ``retry_after``).
    """
    # The body read can fail on a dropped connection. The response is closed
    # either way, and the failure is classified like any other transport error.
    try:
        body = await upstream.aread()
    except httpx.RequestError as exc:
        # Fish already sent a status, and it keeps meaning what it meant: a 401
        # is still a key problem, a 429 still asks the client to back off, and
        # neither is retried or reported as a gateway fault. Only the body is
        # lost, so the message says that and the exception text stays in the log.
        code = upstream.status_code
        log.warning("fish error body unreadable status=%s: %s", code, exc)
        ids = _error_ids(parse_fish_error(code, b"", upstream.headers))
        _log_fish_error(code, ids)
        failure = FishHttpError(
            code,
            _BODY_UNREADABLE.format(code=code),
            retry_after=retry_after_s(upstream.headers),
        )
        return FishProviderError(code, failure.message, metadata=ids), failure
    finally:
        await upstream.aclose()
    detail = parse_fish_error(upstream.status_code, body, upstream.headers)
    ids = _error_ids(detail)
    _log_fish_error(upstream.status_code, ids)
    # The HTTP status decides whether to retry, even when the body names another.
    failure = FishHttpError(
        upstream.status_code,
        detail.message,
        retry_after=retry_after_s(upstream.headers),
    )
    return FishProviderError(detail.status, detail.message, metadata=ids), failure


def _set_read_timeout(request: httpx.Request, seconds: float) -> None:
    """Give one request its own read and write timeout, keeping connect and pool."""
    # httpx reads the per-request timeout from this extension when it sends.
    timeout = dict(request.extensions.get("timeout") or {})
    timeout.update(read=seconds, write=seconds)
    request.extensions["timeout"] = timeout


def _remaining(policy: RetryPolicy, started: float) -> float | None:
    """Return the seconds left in the deadline, or None when there is none."""
    if not policy.deadline_s:
        return None
    return max(0.001, policy.deadline_s - (time.monotonic() - started))


async def fish_send(
    client: FishHttp,
    *,
    stream: bool,
    method: str,
    url: str,
    headers: Mapping[str, str] | None = None,
    content: bytes | None = None,
    json: object | None = None,
    data: Mapping[str, str] | None = None,
    files: FishFiles | None = None,
    policy: RetryPolicy | None = None,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    read_timeout_s: float | None = None,
) -> httpx.Response:
    """Send a request to Fish, retrying only what is safe to repeat.

    Parameters
    ----------
    client : FishHttp
        An httpx client with ``build_request`` and ``send``.
    stream : bool
        Keep the response body open for streaming.
    method : str
        HTTP method.
    url : str
        Path or URL, resolved against the client's base URL.
    headers : Mapping or None, optional
        Request headers.
    content : bytes or None, optional
        Raw request body, such as MessagePack.
    json : object or None, optional
        JSON request body.
    data : Mapping or None, optional
        Form fields for a multipart request.
    files : FishFiles or None, optional
        Multipart files.
    policy : RetryPolicy or None, optional
        Attempt count and deadline. Default is five tries and no deadline.
    is_disconnected : Callable or None, optional
        Awaited before each pause. When it returns True the loop stops, so a
        caller that hung up does not keep spending Fish requests.
    read_timeout_s : float or None, optional
        Read and write timeout for this request, in seconds, in place of the
        client's. None keeps the client's. A long transcription needs more than
        a speech request does.

    Returns
    -------
    httpx.Response
        A 2xx response.

    Raises
    ------
    FishProviderError
        Fish's own error for the last failure, with its status and ids.
    ProxyError
        502 or 504 with a fixed message when Fish was unreachable or slow, 504
        at the retry deadline, and 499 when the client hung up during a pause.

    Notes
    -----
    429 and 5xx are retried, honoring ``Retry-After``, with jitter from the
    kit. Transport errors are retried only when the connection never opened.
    A read timeout is returned as a 504 without a second request. The
    deadline applies to each send and each error-body read, so a stalled
    request is cut off at the deadline, not at the read timeout.
    """
    policy = policy or RetryPolicy()
    started = time.monotonic()
    last_error: ProxyError | None = None
    for attempt in range(policy.attempts):
        retry_after: float | None = None
        try:
            # The deadline bounds a stalled request too, not only the pauses.
            async with asyncio.timeout(_remaining(policy, started)):
                request = client.build_request(
                    method, url, content=content, data=data, files=files, json=json, headers=headers
                )
                if read_timeout_s is not None:
                    _set_read_timeout(request, read_timeout_s)
                upstream = await client.send(request, stream=stream)
        except TimeoutError as exc:
            # Fish may already be working on it, so it is not sent again.
            raise ProxyError(_DEADLINE_STATUS, _DEADLINE_MESSAGE) from exc
        except httpx.RequestError as exc:
            last_error = _transport_error(exc)
            if not isinstance(exc, _RETRY_TRANSPORT):
                raise last_error from exc
        else:
            # Only 2xx is audio. A 3xx redirect body streamed through would
            # reach the client as a 200 file with no speech in it.
            if 200 <= upstream.status_code < 300:
                return upstream
            try:
                async with asyncio.timeout(_remaining(policy, started)):
                    last_error, failure = await _closed_error(upstream)
            except TimeoutError as exc:
                raise ProxyError(_DEADLINE_STATUS, _DEADLINE_MESSAGE) from exc
            if not failure.retryable:
                raise last_error
            retry_after = failure.retry_after
            log.warning(
                "fish retry status=%s attempt=%s/%s",
                upstream.status_code,
                attempt + 1,
                policy.attempts,
            )
        if attempt + 1 >= policy.attempts:
            break
        pause = fish_backoff_s(attempt, retry_after=retry_after)
        if policy.deadline_s and time.monotonic() - started + pause > policy.deadline_s:
            break
        if is_disconnected is not None and await is_disconnected():
            raise ProxyError(_CLIENT_CLOSED, "client closed the request")
        await asyncio.sleep(pause)
    raise last_error or _fixed(FishHttpError.for_unreachable())


def traced_model_headers(model: str, incoming: Mapping[str, str]) -> dict[str, str]:
    """Fish ``model`` header plus the trace headers for this request.

    Parameters
    ----------
    model : str
        Already resolved Fish model id.
    incoming : Mapping
        Client headers. A valid ``traceparent`` is forwarded; otherwise one is minted.

    Returns
    -------
    dict
        Sent on TTS and ASR upstream calls.
    """
    return {"model": model, **ensure_trace_headers(incoming)}
