"""Send one request to Fish with a bounded, polite retry."""

from __future__ import annotations

import asyncio
import json as jsonlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    FishErrorBody,
    FishHttpError,
    fish_backoff_s,
    parse_fish_error,
    retry_after_s,
    utf8_text,
)
from fish_audio_suite_proxy.errors import json_error, json_from_call_failure, provider_json_error

__all__ = ["FishFiles", "FishHttp", "RetryPolicy", "fish_send"]

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


def _transport_error(exc: httpx.RequestError) -> FishHttpError:
    """Classify a failed Fish request.

    Parameters
    ----------
    exc : httpx.RequestError
        The transport failure.

    Returns
    -------
    FishHttpError
        ``FishTimeoutError`` (504) for a timeout and ``FishUpstreamError`` (502)
        with the fixed unreachable message otherwise. The exception's own text
        can name hosts, addresses or errno values, so it is logged here and
        never sent to the client.
    """
    if isinstance(exc, httpx.TimeoutException):
        return FishHttpError.for_timeout()
    log.warning("fish transport error: %s", exc)
    return FishHttpError.for_unreachable()


# Fish ids are short. A long or multi-line value is cut so it cannot fill a log line.
_ID_MAX_CHARS: Final = 128


def _short_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    text = str(value).strip()
    cut = next((i for i, ch in enumerate(text) if ord(ch) < 32), len(text))
    return utf8_text(text[:cut][:_ID_MAX_CHARS].strip())


def _fish_error_ids(body: bytes, headers: Mapping[str, str]) -> dict[str, str]:
    """Return Fish's error ``code`` and ``request_id``, from the body or the header.

    ``transcribe-1-pro`` errors are ``{status, message, code, request_id}``, and
    the request id is also in ``x-request-id``. Fish support asks for it.
    """
    try:
        data: object = jsonlib.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        data = None
    found = data if isinstance(data, dict) else {}
    ids: dict[str, str] = {}
    if code := _short_id(found.get("code")):
        ids["provider_code"] = code
    if request_id := _short_id(found.get("request_id")) or _short_id(headers.get("x-request-id")):
        ids["request_id"] = request_id
    return ids


def _log_fish_error(status: int, ids: Mapping[str, str]) -> None:
    log.warning(
        "fish error status=%s code=%s request_id=%s",
        status,
        ids.get("provider_code", "-"),
        ids.get("request_id", "-"),
    )


async def _closed_error(upstream: httpx.Response) -> tuple[JSONResponse, FishHttpError]:
    """Read and close a failed Fish response.

    Parameters
    ----------
    upstream : httpx.Response
        A non-2xx response whose body is still open.

    Returns
    -------
    tuple of JSONResponse and FishHttpError
        The OpenAI error for the client and the typed failure the retry loop
        reads (``retryable`` and ``retry_after``).
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
        ids = _fish_error_ids(b"", upstream.headers)
        _log_fish_error(code, ids)
        failure = FishHttpError.from_status(
            code,
            _BODY_UNREADABLE.format(code=code),
            retry_after=retry_after_s(upstream.headers),
        )
        return provider_json_error(FishErrorBody(code, failure.message), metadata=ids), failure
    finally:
        await upstream.aclose()
    detail = parse_fish_error(upstream.status_code, body)
    ids = _fish_error_ids(body, upstream.headers)
    _log_fish_error(upstream.status_code, ids)
    # The HTTP status decides whether to retry, even when the body names another.
    failure = FishHttpError.from_status(
        upstream.status_code,
        detail.message,
        retry_after=retry_after_s(upstream.headers),
    )
    return provider_json_error(detail, metadata=ids), failure


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
) -> httpx.Response | JSONResponse:
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
    httpx.Response or JSONResponse
        A 2xx response, or the OpenAI error for the last failure.

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
    last_error: JSONResponse | None = None
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
        except TimeoutError:
            # Fish may already be working on it, so it is not sent again.
            return json_error(_DEADLINE_STATUS, _DEADLINE_MESSAGE)
        except httpx.RequestError as exc:
            last_error = json_from_call_failure(_transport_error(exc))
            if not isinstance(exc, _RETRY_TRANSPORT):
                return last_error
        else:
            # 3xx is not audio. A redirect body was streamed to the client
            # as a 200 file, so the reply was never spoken.
            if 200 <= upstream.status_code < 300:
                return upstream
            try:
                async with asyncio.timeout(_remaining(policy, started)):
                    last_error, failure = await _closed_error(upstream)
            except TimeoutError:
                return json_error(_DEADLINE_STATUS, _DEADLINE_MESSAGE)
            if not failure.retryable:
                return last_error
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
            return json_error(_CLIENT_CLOSED, "client closed the request")
        await asyncio.sleep(pause)
    return last_error or json_from_call_failure(FishHttpError.for_unreachable())
