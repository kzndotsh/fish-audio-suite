"""Fish HTTP error shape and retry policy. No network."""

from __future__ import annotations

import asyncio
import json
import math
import random
from typing import Any, cast

from fish_audio_suite_kit._charsets import utf8_text
from fish_audio_suite_kit.defaults import number_or

FISH_TTS_PATH = "/v1/tts"
FISH_ASR_PATH = "/v1/asr"
FISH_RETRY_ATTEMPTS = 5
FISH_TIMEOUT_STATUS = 504
FISH_TIMEOUT_MESSAGE = "Fish request timed out"
FISH_UNREACHABLE_STATUS = 502
FISH_UNREACHABLE_MESSAGE = "Fish upstream unreachable"
FISH_NON_JSON_MESSAGE = "Fish returned a non-JSON body"
FISH_NON_OBJECT_MESSAGE = "Fish returned a non-object body"
# Longest single wait between retries, and the longest Retry-After we honor.
FISH_BACKOFF_CAP_S = 30.0
FISH_RETRY_AFTER_CAP_S = 60.0


class FishHttpError(Exception):
    """REST/WS-adjacent Fish failure after retries or a non-retryable status."""

    def __init__(self, status: int, message: str) -> None:
        """Store a Fish status and a UTF-8 message.

        Parameters
        ----------
        status : int
            HTTP status. Coerced with ``int``.
        message : str
            Human-readable Fish or transport message. Invalid UTF-8 is replaced.
        """
        self.status = int(status)
        self.message = utf8_text(str(message))
        super().__init__(f"HTTP {self.status}: {self.message}")

    @classmethod
    def unreachable(cls) -> FishHttpError:
        """Build the 502 for a retry loop that ended with no Fish response."""
        return cls(FISH_UNREACHABLE_STATUS, FISH_UNREACHABLE_MESSAGE)

    @classmethod
    def timed_out(cls) -> FishHttpError:
        """Build the 504 for a Fish request that timed out."""
        return cls(FISH_TIMEOUT_STATUS, FISH_TIMEOUT_MESSAGE)

    @classmethod
    def non_json(cls) -> FishHttpError:
        """Build the 502 for a Fish body that is not JSON."""
        return cls(FISH_UNREACHABLE_STATUS, FISH_NON_JSON_MESSAGE)

    @classmethod
    def non_object(cls) -> FishHttpError:
        """Build the 502 for JSON that is not an object, or a non-string ASR text."""
        return cls(FISH_UNREACHABLE_STATUS, FISH_NON_OBJECT_MESSAGE)


def should_retry_fish_status(status: int) -> bool:
    """Retry 429 and 5xx only. Other 4xx need a different request."""
    return status == 429 or status >= 500


def fish_attempt_exhausted(attempt: int) -> bool:
    """Return whether this zero-based attempt is the last try.

    Parameters
    ----------
    attempt : int
        Index into ``FISH_RETRY_ATTEMPTS`` (5), so attempt 4 is the last.

    Returns
    -------
    bool
        True when no further retry should be scheduled.
    """
    return attempt + 1 >= FISH_RETRY_ATTEMPTS


def fish_backoff_seconds(attempt: int) -> float:
    """Return the documented base wait, ``2 ** attempt`` seconds, with no jitter."""
    return float(2 ** max(0, attempt))


def fish_backoff_s(
    attempt: int,
    *,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Return how long to wait before the next try.

    Parameters
    ----------
    attempt : int
        Zero-based attempt that just failed.
    retry_after : float or None, optional
        Seconds from a ``Retry-After`` header. A usable value raises the wait
        to at least that long, capped at ``FISH_RETRY_AFTER_CAP_S``. Negative,
        non-finite, or None is ignored.
    rng : random.Random or None, optional
        Source of jitter. Default is the module-level ``random``.

    Returns
    -------
    float
        Exponential base (``2 ** attempt``, capped at ``FISH_BACKOFF_CAP_S``)
        with equal jitter: a uniform draw between half the base and the base.
    """
    base = min(FISH_BACKOFF_CAP_S, fish_backoff_seconds(attempt))
    draw = (rng or random).random()
    wait = base / 2 + draw * base / 2
    if retry_after is not None and math.isfinite(retry_after) and retry_after >= 0:
        wait = max(wait, min(retry_after, FISH_RETRY_AFTER_CAP_S))
    return wait


async def fish_sleep_before_retry(
    attempt: int,
    *,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> bool:
    """Sleep with jitter before another try.

    Parameters
    ----------
    attempt : int
        Zero-based attempt that just failed.
    retry_after : float or None, optional
        Seconds from a ``Retry-After`` header. See ``fish_backoff_s``.
    rng : random.Random or None, optional
        Source of jitter.

    Returns
    -------
    bool
        True after sleeping, so the caller should try again. False when this
        was the last attempt and nothing was slept.
    """
    if fish_attempt_exhausted(attempt):
        return False
    await asyncio.sleep(fish_backoff_s(attempt, retry_after=retry_after, rng=rng))
    return True


async def fish_retry_pause(attempt: int) -> bool:
    """Sleep ``2 ** attempt`` seconds, or report that this attempt is the last.

    Prefer ``fish_sleep_before_retry``. Its return value is the natural way
    round, and it adds jitter and ``Retry-After``.

    Parameters
    ----------
    attempt : int
        Zero-based attempt that just failed.

    Returns
    -------
    bool
        True when the caller must stop. False after sleeping ``2 ** attempt``
        seconds. The return value is the stop flag, not "pause succeeded".

    Notes
    -----
    Callers that treat True as "paused" will exit the retry loop one try early.
    """
    if fish_attempt_exhausted(attempt):
        return True
    await asyncio.sleep(fish_backoff_seconds(attempt))
    return False


def bearer(key: str) -> str:
    """Build an ``Authorization`` header value.

    Parameters
    ----------
    key : str
        Fish API key. The function always adds the prefix, so pass the raw key.

    Returns
    -------
    str
        ``Bearer`` plus the key.
    """
    token = key.strip()
    # A newline would split the header, and a non-ASCII character cannot be
    # encoded into it. Send an empty token so the upstream answers 401.
    if any(ord(ch) < 32 or ord(ch) >= 127 for ch in token):
        token = ""
    return f"Bearer {token}"


def fish_unreachable() -> tuple[int, str]:
    """Status and message when a retry loop ends with no Fish response.

    Returns
    -------
    tuple of int and str
        ``(502, "Fish upstream unreachable")``.
    """
    return FISH_UNREACHABLE_STATUS, FISH_UNREACHABLE_MESSAGE


def fish_non_json() -> tuple[int, str]:
    """Status and message when Fish's body is not JSON.

    Returns
    -------
    tuple of int and str
        ``(502, "Fish returned a non-JSON body")``.
    """
    return FISH_UNREACHABLE_STATUS, FISH_NON_JSON_MESSAGE


def fish_non_object() -> tuple[int, str]:
    """Status and message when JSON is not an object, or ASR ``text`` is not a string.

    Returns
    -------
    tuple of int and str
        ``(502, "Fish returned a non-object body")``.
    """
    return FISH_UNREACHABLE_STATUS, FISH_NON_OBJECT_MESSAGE


def parse_asr_body(body: object) -> tuple[dict[str, Any], str]:
    """ASR JSON object plus its text. Missing text is empty. A bad shape is 502."""
    data = _as_dict(body)
    if data is None:
        raise FishHttpError.non_object()
    raw = data.get("text")
    if raw is None:
        return data, ""
    if not isinstance(raw, str):
        raise FishHttpError.non_object()
    return data, raw


def fish_transport_error(exc: BaseException | None, *, timed_out: bool) -> tuple[int, str]:
    """Map a transport failure to a status and message.

    Parameters
    ----------
    exc : BaseException or None
        The caught error. A blank string falls back to ``fish_unreachable``.
    timed_out : bool
        True when the caller already classified ``exc`` as its timeout type.

    Returns
    -------
    tuple of int and str
        ``(504, "Fish request timed out")`` on timeout, otherwise 502 plus
        ``str(exc)`` or the unreachable message.
    """
    if timed_out:
        return FISH_TIMEOUT_STATUS, FISH_TIMEOUT_MESSAGE
    text = str(exc).strip() if exc is not None else ""
    if text:
        return FISH_UNREACHABLE_STATUS, text
    return fish_unreachable()


def fish_request_error(exc: BaseException, timeout_type: type[BaseException]) -> tuple[int, str]:
    """Classify ``exc`` using the caller's timeout class.

    Parameters
    ----------
    exc : BaseException
        ``httpx.RequestError`` or a subclass.
    timeout_type : type of BaseException
        Usually ``httpx.TimeoutException``. Passed in so kit does not import httpx.

    Returns
    -------
    tuple of int and str
        See ``fish_transport_error``.
    """
    return fish_transport_error(exc, timed_out=isinstance(exc, timeout_type))


def fish_error_body(status: int, message: str) -> dict[str, str | int]:
    """Fish-shaped error object.

    Parameters
    ----------
    status : int
        HTTP status stored under ``status``.
    message : str
        Stored under ``message`` after a UTF-8 round trip.

    Returns
    -------
    dict
        ``{"message": ..., "status": ...}``.
    """
    return {"message": utf8_text(str(message)), "status": int(status)}


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return None


def _nested_message(value: Any) -> Any:
    data = _as_dict(value)
    if data is None:
        return value
    return data.get("message")


def _read_message(value: Any) -> tuple[str | None, bool]:
    # A blank string or an object with no text carries no detail. Keep looking
    # in the other fields before falling back to "error" or "HTTP <status>".
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, False
    if isinstance(value, str):
        return value.strip(), False
    if isinstance(value, dict):
        inner = _nested_message(value)
        # A nested list is a 422 payload: report its messages, joined, not the dict.
        if inner is not None and inner is not value:
            found, _saw = _read_message(inner)
            if found:
                return found, True
        # One validation object is a one-item list. Never str() a dict: the
        # client would see a Python repr instead of the field message.
        summary = _validation_msgs([value])
        if summary:
            return summary, True
        return None, True
    if isinstance(value, list):
        items = cast(list[Any], value)
        summary = _validation_msgs(items)
        if summary:
            return summary, True
        parts = [item.strip() for item in items if isinstance(item, str) and item.strip()]
        if parts:
            return "; ".join(parts), True
        return None, True
    text = str(value).strip()
    return (text or None), False


def _message_of(data: dict[str, Any]) -> Any:
    saw_object = False
    for key in ("message", "detail"):
        found, saw = _read_message(data.get(key))
        saw_object = saw_object or saw
        if found:
            return found
    found, saw = _read_message(_nested_message(data.get("error")))
    if found:
        return found
    if saw_object or saw:
        return "error"
    return None


def _decoded(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="replace")
    if raw is None:
        return ""
    return str(raw)


def _fallback_message(text: str, status: int) -> str:
    return text or f"HTTP {status}"


def _validation_msgs(items: list[Any]) -> str | None:
    # Fish 422 is a list of {loc, msg}. The client should see the messages,
    # not the whole validation array.
    parts: list[str] = []
    for item in items:
        # An entry without a message is skipped; the others still report.
        if not isinstance(item, dict):
            continue
        entry = cast(dict[str, Any], item)
        msg = entry.get("msg")
        if not isinstance(msg, str) or not msg.strip():
            # Some 422 objects use "message" instead of "msg".
            msg = entry.get("message")
        if isinstance(msg, str) and msg.strip():
            parts.append(msg.strip())
    if not parts:
        return None
    return "; ".join(parts)


def parse_fish_error(status: int, raw: Any) -> dict[str, str | int]:
    """Normalize Fish `{message, status}` or a plain-text parse error."""
    data = _as_dict(raw)
    if data is not None:
        msg = _message_of(data)
        text = str(msg).strip() if msg is not None else ""
        code = number_or(data.get("status", status), status, int)
        # 200 or 0 in the body must not turn an HTTP 500 into a success.
        if code < 400 or code > 599:
            code = status
        return fish_error_body(code, _fallback_message(text, status))

    if isinstance(raw, list):
        summary = _validation_msgs(cast(list[Any], raw))
        if summary:
            return fish_error_body(status, summary)

    stripped = _decoded(raw).strip()
    if stripped:
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return fish_error_body(status, stripped)
        if isinstance(loaded, list):
            summary = _validation_msgs(cast(list[Any], loaded))
            if summary:
                return fish_error_body(status, summary)
        elif _as_dict(loaded) is not None:
            return parse_fish_error(status, loaded)
    return fish_error_body(status, _fallback_message(stripped, status))
