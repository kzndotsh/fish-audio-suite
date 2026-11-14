"""Fish HTTP error shape and retry policy. No network."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Self, cast

from fish_audio_suite_kit.config._values import parse_number
from fish_audio_suite_kit.text._charsets import cut_at_control, is_header_safe, utf8_text
from fish_audio_suite_kit.types.payloads import AsrBody

__all__ = [
    "FISH_ASR_PATH",
    "FISH_RETRY_ATTEMPTS",
    "FISH_TTS_PATH",
    "FishAuthError",
    "FishErrorBody",
    "FishHttpError",
    "FishRateLimitError",
    "FishTimeoutError",
    "FishUpstreamError",
    "bearer",
    "fish_attempt_exhausted",
    "fish_backoff_s",
    "fish_request_id",
    "parse_asr_body",
    "parse_fish_error",
    "retry_after_s",
    "should_retry_fish_status",
]

FISH_TTS_PATH: Final = "/v1/tts"
FISH_ASR_PATH: Final = "/v1/asr"
FISH_RETRY_ATTEMPTS: Final = 5
FISH_TIMEOUT_STATUS: Final = 504
FISH_TIMEOUT_MESSAGE: Final = "Fish request timed out"
FISH_UNREACHABLE_STATUS: Final = 502
FISH_UNREACHABLE_MESSAGE: Final = "Fish upstream unreachable"
FISH_NON_JSON_MESSAGE: Final = "Fish returned a non-JSON body"
FISH_NON_OBJECT_MESSAGE: Final = "Fish returned a non-object body"
# Longest single wait between retries, and the longest Retry-After we honor.
FISH_BACKOFF_CAP_S: Final = 30.0
FISH_RETRY_AFTER_CAP_S: Final = 60.0
# A Retry-After longer than a day is not a retry hint, so the parser drops it.
_RETRY_AFTER_MAX_S: Final = 86_400.0
_AUTH_STATUSES: Final = frozenset({401, 402, 403})
# Whole or decimal seconds only. The HTTP-date form is not a wait we can use.
_SECONDS_RE: Final = re.compile(r"[0-9]+(?:\.[0-9]+)?")


class FishHttpError(Exception):
    """A Fish failure after retries, or a status that should not be retried.

    Attributes
    ----------
    status : int
        HTTP status of the failure.
    message : str
        Human-readable Fish or transport message.
    retry_after : float or None
        Seconds Fish asked the caller to wait, when it sent a usable hint.
    """

    status: int
    message: str
    retry_after: float | None

    def __new__(cls, status: int, message: str, *, retry_after: float | None = None) -> Self:
        """Pick the subclass that matches the status when called on the base class.

        Parameters
        ----------
        status : int
            HTTP status. ``FishHttpError(401, ...)`` is a ``FishAuthError``, so code
            that catches by type sees the class that matches the status: ``FishAuthError``
            for 401, 402 and 403, ``FishRateLimitError`` for 429, ``FishTimeoutError``
            for 504, ``FishUpstreamError`` for any other 5xx, and ``FishHttpError`` for
            the rest.
        message : str
            Not used here. Stored by ``__init__``.
        retry_after : float or None, optional
            Not used here. Stored by ``__init__``.

        Returns
        -------
        Self
            An instance of the matching subclass, or of ``cls`` for a subclass call.
        """
        del message, retry_after
        target = _class_for_status(int(status)) if cls is FishHttpError else cls
        # The subclass is a FishHttpError, and Self is at least that.
        return super().__new__(cast("type[Self]", target))

    def __init__(self, status: int, message: str, *, retry_after: float | None = None) -> None:
        """Store a Fish status and a UTF-8 message.

        Parameters
        ----------
        status : int
            HTTP status. Coerced with ``int``.
        message : str
            Human-readable Fish or transport message. Invalid UTF-8 is replaced.
        retry_after : float or None, optional
            Seconds from a ``Retry-After`` header. A negative or non-finite value is
            stored as None.
        """
        self.status = int(status)
        self.message = utf8_text(str(message))
        usable = (
            retry_after is not None
            and not isinstance(retry_after, bool)
            and math.isfinite(retry_after)
            and retry_after >= 0
        )
        self.retry_after = float(retry_after) if usable and retry_after is not None else None
        super().__init__(f"HTTP {self.status}: {self.message}")

    def __reduce__(self) -> tuple[Any, ...]:
        """Support copy and pickle, which the default exception reduce cannot.

        Returns
        -------
        tuple
            The class, the constructor arguments and the ``retry_after`` state.
        """
        return (self.__class__, (self.status, self.message), {"retry_after": self.retry_after})

    @property
    def retryable(self) -> bool:
        """Whether the same request can be tried again (429 and 5xx)."""
        return should_retry_fish_status(self.status)

    @classmethod
    def for_unreachable(cls) -> FishUpstreamError:
        """Build the 502 for a retry loop that ended with no Fish response."""
        return FishUpstreamError(FISH_UNREACHABLE_STATUS, FISH_UNREACHABLE_MESSAGE)

    @classmethod
    def for_timeout(cls) -> FishTimeoutError:
        """Build the 504 for a Fish request that timed out."""
        return FishTimeoutError(FISH_TIMEOUT_STATUS, FISH_TIMEOUT_MESSAGE)

    @classmethod
    def for_non_json(cls) -> FishUpstreamError:
        """Build the 502 for a Fish body that is not JSON."""
        return FishUpstreamError(FISH_UNREACHABLE_STATUS, FISH_NON_JSON_MESSAGE)

    @classmethod
    def for_non_object(cls) -> FishUpstreamError:
        """Build the 502 for JSON that is not an object, or a non-string ASR text."""
        return FishUpstreamError(FISH_UNREACHABLE_STATUS, FISH_NON_OBJECT_MESSAGE)


class FishAuthError(FishHttpError):
    """401, 402 or 403: the key is missing, wrong or out of credit. Never retryable."""


class FishRateLimitError(FishHttpError):
    """429: Fish asked the caller to slow down. ``retry_after`` carries its hint."""


class FishUpstreamError(FishHttpError):
    """A 5xx other than 504, or a Fish that could not be reached or understood."""


class FishTimeoutError(FishHttpError):
    """504: the request to Fish timed out."""


def _class_for_status(code: int) -> type[FishHttpError]:
    if code in _AUTH_STATUSES:
        return FishAuthError
    if code == 429:
        return FishRateLimitError
    if code == FISH_TIMEOUT_STATUS:
        return FishTimeoutError
    if code >= 500:
        return FishUpstreamError
    return FishHttpError


def should_retry_fish_status(status: int) -> bool:
    """Say whether a failed request is worth sending again.

    Parameters
    ----------
    status : int
        HTTP status of the failed response.

    Returns
    -------
    bool
        True for 429 and any 5xx. Other 4xx statuses need a different request.
    """
    return status == 429 or status >= 500


def fish_attempt_exhausted(attempt: int) -> bool:
    """Return whether this zero-based attempt is the last try.

    Parameters
    ----------
    attempt : int
        Zero-based attempt number. ``FISH_RETRY_ATTEMPTS`` is 5, so attempt 4
        is the last.

    Returns
    -------
    bool
        True when no further retry should be scheduled.
    """
    return attempt + 1 >= FISH_RETRY_ATTEMPTS


def _base_wait(attempt: int) -> float:
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
    base = min(FISH_BACKOFF_CAP_S, _base_wait(attempt))
    draw = (rng or random).random()
    wait = base / 2 + draw * base / 2
    if retry_after is not None and math.isfinite(retry_after) and retry_after >= 0:
        wait = max(wait, min(retry_after, FISH_RETRY_AFTER_CAP_S))
    return wait


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
    return f"Bearer {token if is_header_safe(token) else ''}"


def parse_asr_body(body: object) -> tuple[AsrBody, str]:
    """Read a Fish ASR response and its transcript text.

    Parameters
    ----------
    body : object
        The decoded JSON from ``/v1/asr``.

    Returns
    -------
    tuple of AsrBody and str
        The response object and its ``text``. Missing or null text is the empty
        string. Only ``text`` is checked: the other keys keep whatever Fish sent,
        so ``AsrBody`` describes the expected shape and does not enforce it.

    Raises
    ------
    FishHttpError
        A 502 when the body is not a JSON object or ``text`` is not a string.

    Examples
    --------
    >>> data, text = parse_asr_body({"text": "hello", "duration": 1.5})
    >>> text, data["duration"]
    ('hello', 1.5)
    >>> parse_asr_body({})[1]
    ''
    """
    data = _as_dict(body)
    if data is None:
        raise FishHttpError.for_non_object()
    raw = data.get("text")
    if raw is None:
        return cast(AsrBody, data), ""
    if not isinstance(raw, str):
        raise FishHttpError.for_non_object()
    return cast(AsrBody, data), raw


def retry_after_s(headers: Mapping[str, str] | None) -> float | None:
    """Read a usable ``Retry-After`` wait, in seconds, from response headers.

    Parameters
    ----------
    headers : Mapping of str to str, or None
        Response headers. The name is matched without regard to case, so an
        ``httpx.Headers`` object works as well as a plain dict.

    Returns
    -------
    float or None
        Whole or decimal seconds, or None when the header is absent or cannot
        be used: negative, ``nan``, ``inf``, an exponent form, an HTTP date, or
        longer than a day.

    Examples
    --------
    >>> retry_after_s({"Retry-After": "7"})
    7.0
    >>> retry_after_s({"retry-after": "-1"}) is None
    True
    >>> retry_after_s({"Retry-After": "nan"}) is None
    True
    >>> retry_after_s(None) is None
    True
    """
    if not headers:
        return None
    raw = ""
    for key, value in headers.items():
        if str(key).lower() == "retry-after":
            raw = str(value).strip()
            break
    if not _SECONDS_RE.fullmatch(raw):
        return None
    seconds = float(raw)
    if not math.isfinite(seconds) or seconds > _RETRY_AFTER_MAX_S:
        return None
    return seconds


@dataclass(frozen=True, slots=True)
class FishErrorBody:
    """A Fish ``{message, status, code, request_id}`` error, with the text already made safe.

    Attributes
    ----------
    status : int
        HTTP status of the error.
    message : str
        Error text after a UTF-8 round trip.
    code : str
        Fish's own error code, such as ``audio_too_long`` (``transcribe-1-pro`` sends one),
        cut like a request id. Empty when the body has none.
    request_id : str
        The id Fish support traces the request by, from the body or ``x-request-id``.
        Empty when neither has one.
    """

    status: int
    message: str
    code: str = ""
    request_id: str = ""

    @classmethod
    def of(cls, status: int, message: str) -> FishErrorBody:
        """Build a body from raw values.

        Parameters
        ----------
        status : int
            HTTP status. Coerced with ``int``.
        message : str
            Error text. Coerced with ``str`` and cleaned of invalid UTF-8.

        Returns
        -------
        FishErrorBody
            The normalized body, with no code and no request id.
        """
        return cls(int(status), utf8_text(str(message)))


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return None


def _nested_message(value: Any) -> Any:
    data = _as_dict(value)
    if data is None:
        return value
    return data.get("message")


# Fish nests "message" one or two levels. A body nested deeper is junk, and reading
# it to the bottom would raise RecursionError.
_MESSAGE_DEPTH: Final = 16


def _read_message(value: Any, depth: int = 0) -> tuple[str | None, bool]:
    # A blank string or an object with no text carries no detail. Keep looking
    # in the other fields before falling back to "error" or "HTTP <status>".
    if depth > _MESSAGE_DEPTH:
        return None, True
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, False
    if isinstance(value, str):
        return value.strip(), False
    if isinstance(value, dict):
        inner = _nested_message(value)
        # A nested list is a 422 payload: report its messages, joined, not the dict.
        if inner is not None and inner is not value:
            found, _saw = _read_message(inner, depth + 1)
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


# Fish ids are short. A longer value is cut so it cannot fill a log line.
_REQUEST_ID_MAX_CHARS: Final = 128


def _short_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    text = cut_at_control(str(value).strip())
    return utf8_text(text[:_REQUEST_ID_MAX_CHARS].strip())


def fish_request_id(
    body: object,
    headers: Mapping[str, str] | None,
    *,
    header_first: bool = False,
) -> str:
    r"""Return the request id Fish sent, for a log line Fish support can trace.

    Parameters
    ----------
    body : object
        The decoded JSON body. ``transcribe-1-pro`` puts ``request_id`` in it, on
        success and on errors. Anything but a JSON object has none.
    headers : Mapping of str to str, or None
        Response headers. Fish also sends the id as ``x-request-id``. The name is
        looked up as written, so pass ``httpx.Headers`` (case-blind) or a lowercase dict.
    header_first : bool, optional
        Prefer the header when both carry an id. Default False, which prefers the body.

    Returns
    -------
    str
        The id, stripped, cut at its first control character and to 128 characters, or
        ``""`` when neither place has a string or a number.

    Examples
    --------
    >>> fish_request_id({"request_id": "req-1"}, {"x-request-id": "req-2"})
    'req-1'
    >>> fish_request_id(None, {"x-request-id": " req-2\nX-Evil: 1"})
    'req-2'
    """
    data = _as_dict(body)
    from_body = _short_id(data.get("request_id")) if data is not None else ""
    from_header = _short_id(headers.get("x-request-id")) if headers else ""
    first, second = (from_header, from_body) if header_first else (from_body, from_header)
    return first or second


def parse_fish_error(
    status: int,
    raw: Any,
    headers: Mapping[str, str] | None = None,
    *,
    header_first: bool = False,
) -> FishErrorBody:
    """Normalize a Fish error body into a status, a message and Fish's ids.

    Parameters
    ----------
    status : int
        HTTP status of the response. It wins when the body carries a status
        outside 400-599.
    raw : Any
        The body: bytes, text, JSON, a decoded object or a validation list.
    headers : Mapping of str to str, or None, optional
        Response headers, read for ``x-request-id`` as ``fish_request_id`` reads them.
    header_first : bool, optional
        Prefer the header's request id when the body has one too. Default False.

    Returns
    -------
    FishErrorBody
        The status and a readable message, with the body's ``code`` and the request id
        when Fish sent them. Validation lists become their messages joined, a
        plain-text body is used as is, and an empty one becomes ``"HTTP <status>"``.

    Examples
    --------
    >>> parse_fish_error(402, {"message": "Insufficient credits", "status": 402}).message
    'Insufficient credits'
    >>> parse_fish_error(500, b"").message
    'HTTP 500'
    >>> body = {"message": "too long", "code": "audio_too_long", "request_id": "req-1"}
    >>> found = parse_fish_error(400, body)
    >>> found.code, found.request_id
    ('audio_too_long', 'req-1')
    """
    detail, data = _error_detail(status, raw)
    return FishErrorBody(
        detail.status,
        detail.message,
        code=_short_id(data.get("code")) if data is not None else "",
        request_id=fish_request_id(data, headers, header_first=header_first),
    )


def _error_detail(status: int, raw: Any) -> tuple[FishErrorBody, dict[str, Any] | None]:
    # The decoded object is returned too, so the ids are read from the same parse.
    data = _as_dict(raw)
    if data is not None:
        msg = _message_of(data)
        text = str(msg).strip() if msg is not None else ""
        code = parse_number(data.get("status", status), status, int)
        # 200 or 0 in the body must not turn an HTTP 500 into a success.
        if code < 400 or code > 599:
            code = status
        return FishErrorBody.of(code, _fallback_message(text, status)), data

    if isinstance(raw, list):
        summary = _validation_msgs(cast(list[Any], raw))
        if summary:
            return FishErrorBody.of(status, summary), None

    stripped = _decoded(raw).strip()
    if stripped:
        try:
            loaded = json.loads(stripped)
        # Deep nesting such as 20k "[" raises RecursionError. ValueError covers a
        # decode error and an integer past Python's 4300-digit limit.
        except (ValueError, RecursionError):
            return FishErrorBody.of(status, stripped), None
        if isinstance(loaded, list):
            summary = _validation_msgs(cast(list[Any], loaded))
            if summary:
                return FishErrorBody.of(status, summary), None
        elif _as_dict(loaded) is not None:
            return _error_detail(status, loaded)
    return FishErrorBody.of(status, _fallback_message(stripped, status)), None
