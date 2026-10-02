"""Send one request to Fish with a bounded, polite retry."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    fish_backoff_s,
    fish_request_error,
    fish_unreachable,
    should_retry_fish_status,
)
from fish_audio_suite_proxy.errors import json_error, json_from_upstream

log = logging.getLogger("fish-audio-suite-proxy")

# Only a failure before the request reached Fish is safe to repeat. A read
# timeout may mean Fish already made, and billed, the audio.
_RETRY_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_CLIENT_CLOSED = 499


@dataclass(frozen=True)
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


class _FishHttp(Protocol):
    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        headers: Any = None,
    ) -> Any: ...

    async def send(self, request: Any, *, stream: bool = False) -> Any: ...


def retry_after_s(headers: Any) -> float | None:
    """Read a ``Retry-After`` value given in seconds.

    Parameters
    ----------
    headers : Any
        Response headers with a ``get`` method, or None.

    Returns
    -------
    float or None
        The seconds, or None when the header is absent, an HTTP date, or junk.
    """
    getter = getattr(headers, "get", None)
    raw = getter("retry-after") if callable(getter) else None
    if not isinstance(raw, str):
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


_DEADLINE_STATUS = 504
_DEADLINE_MESSAGE = "Fish did not answer before the retry deadline"


async def _closed_error(upstream: httpx.Response) -> JSONResponse:
    # The body read can fail on a dropped connection. The response is closed
    # either way, and the failure is classified like any other transport error.
    try:
        body = await upstream.aread()
    except httpx.RequestError as exc:
        status, message = fish_request_error(exc, httpx.TimeoutException)
        return json_error(status, message)
    finally:
        await upstream.aclose()
    return json_from_upstream(upstream.status_code, body)


def _remaining(policy: RetryPolicy, started: float) -> float | None:
    """Return the seconds left in the deadline, or None when there is none."""
    if not policy.deadline_s:
        return None
    return max(0.001, policy.deadline_s - (time.monotonic() - started))


async def fish_send(
    client: _FishHttp,
    *,
    stream: bool,
    policy: RetryPolicy | None = None,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    **request_kwargs: Any,
) -> httpx.Response | JSONResponse:
    """Send a request to Fish, retrying only what is safe to repeat.

    Parameters
    ----------
    client : _FishHttp
        An httpx client with ``build_request`` and ``send``.
    stream : bool
        Keep the response body open for streaming.
    policy : RetryPolicy or None, optional
        Attempt count and deadline. Default is five tries and no deadline.
    is_disconnected : Callable or None, optional
        Awaited before each pause. When it returns True the loop stops, so a
        caller that hung up does not keep spending Fish requests.
    **request_kwargs : Any
        Passed to ``build_request``.

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
                upstream = await client.send(client.build_request(**request_kwargs), stream=stream)
        except TimeoutError:
            # Fish may already be working on it, so it is not sent again.
            return json_error(_DEADLINE_STATUS, _DEADLINE_MESSAGE)
        except httpx.RequestError as exc:
            status, message = fish_request_error(exc, httpx.TimeoutException)
            last_error = json_error(status, message)
            if not isinstance(exc, _RETRY_TRANSPORT):
                return last_error
        else:
            # 3xx is not audio. A redirect body was streamed to the client
            # as a 200 file, so the reply was never spoken.
            if 200 <= upstream.status_code < 300:
                return upstream
            retry_after = retry_after_s(getattr(upstream, "headers", None))
            try:
                async with asyncio.timeout(_remaining(policy, started)):
                    last_error = await _closed_error(upstream)
            except TimeoutError:
                return json_error(_DEADLINE_STATUS, _DEADLINE_MESSAGE)
            if not should_retry_fish_status(upstream.status_code):
                return last_error
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
    status, message = fish_unreachable()
    return last_error or json_error(status, message)
