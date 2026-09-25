"""Fish HTTP error shape and retry policy. No network."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from fish_audio_suite_kit.defaults import number_or
from fish_audio_suite_kit.text_filters import utf8_text

FISH_TTS_PATH = "/v1/tts"
FISH_ASR_PATH = "/v1/asr"
FISH_RETRY_ATTEMPTS = 5
FISH_TIMEOUT_STATUS = 504
FISH_TIMEOUT_MESSAGE = "Fish request timed out"
FISH_UNREACHABLE_STATUS = 502
FISH_UNREACHABLE_MESSAGE = "Fish upstream unreachable"


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
    """Docs: `2 ** attempt` for attempts 0..4."""
    return float(2 ** max(0, attempt))


async def fish_retry_pause(attempt: int) -> bool:
    """Sleep before another try, or report that this attempt is the last.

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
    # A newline would split the header. A non-ASCII character cannot be
    # encoded into it, so the client raises before the request is sent.
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
    return FISH_UNREACHABLE_STATUS, "Fish returned a non-JSON body"


def fish_non_object() -> tuple[int, str]:
    """Status and message when JSON is not an object, or ASR ``text`` is not a string.

    Returns
    -------
    tuple of int and str
        ``(502, "Fish returned a non-object body")``.
    """
    return FISH_UNREACHABLE_STATUS, "Fish returned a non-object body"


def parse_asr_body(body: object) -> tuple[dict[str, Any], str]:
    """ASR JSON object plus its text. Missing text is empty. A bad shape is 502."""
    data = _as_dict(body)
    if data is None:
        raise FishHttpError(*fish_non_object())
    raw = data.get("text")
    if raw is None:
        return data, ""
    if not isinstance(raw, str):
        raise FishHttpError(*fish_non_object())
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
    # A blank string and a dict with no text are not the error. Stopping
    # on them hid detail, so the client saw "error" or "HTTP 400".
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, False
    if isinstance(value, str):
        return value.strip(), False
    if isinstance(value, dict):
        inner = _nested_message(value)
        # A nested list is the same 422 payload. Returning the dict as
        # "error" hid "field required". A string list joined the same way.
        if inner is not None and inner is not value:
            found, _saw = _read_message(inner)
            if found:
                return found, True
        # One validation object is the same payload as a one-item list.
        # str(dict) is the Python repr, so the field message never showed.
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
        # One entry without a message used to hide the rest of the 422.
        if not isinstance(item, dict):
            continue
        entry = cast(dict[str, Any], item)
        msg = entry.get("msg")
        if not isinstance(msg, str) or not msg.strip():
            # Some 422 objects use message instead of msg. Stopping on the
            # missing msg hid the field text and the client saw "error".
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
