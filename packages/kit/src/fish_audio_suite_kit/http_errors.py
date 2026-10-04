"""Fish HTTP error shape and retry policy. No network."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Self, cast

from fish_audio_suite_kit._charsets import utf8_text
from fish_audio_suite_kit.env import parse_number
from fish_audio_suite_kit.payloads import AsrBody

__all__ = [
    "FISH_ASR_PATH",
    "FISH_BACKOFF_CAP_S",
    "FISH_NON_JSON_MESSAGE",
    "FISH_NON_OBJECT_MESSAGE",
    "FISH_RETRY_AFTER_CAP_S",
    "FISH_RETRY_ATTEMPTS",
    "FISH_TIMEOUT_MESSAGE",
    "FISH_TIMEOUT_STATUS",
    "FISH_TTS_PATH",
    "FISH_UNREACHABLE_MESSAGE",
    "FISH_UNREACHABLE_STATUS",
    "FishAudioSuiteError",
    "FishAuthError",
    "FishErrorBody",
    "FishHttpError",
    "FishRateLimitError",
    "FishTimeoutError",
    "FishUpstreamError",
    "bearer",
    "describe_request_error",
    "describe_transport_error",
    "fish_attempt_exhausted",
    "fish_backoff_s",
    "fish_error_body",
    "fish_sleep_before_retry",
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


class FishAudioSuiteError(Exception):
    """Base class for the errors this toolkit raises on purpose."""


class FishHttpError(FishAudioSuiteError):
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
            that catches by type sees the same class whichever way the error was built.
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
    def from_status(
        cls,
        status: int,
        message: str,
        *,
        retry_after: float | None = None,
    ) -> FishHttpError:
        """Build the error class that matches an HTTP status.

        Parameters
        ----------
        status : int
            HTTP status.
        message : str
            Human-readable message.
        retry_after : float or None, optional
            Seconds from a ``Retry-After`` header.

        Returns
        -------
        FishHttpError
            ``FishAuthError`` for 401, 402 and 403, ``FishRateLimitError`` for 429,
            ``FishTimeoutError`` for 504, ``FishUpstreamError`` for any other
            5xx, and a plain ``FishHttpError`` for the rest.

        Examples
        --------
        >>> FishHttpError.from_status(429, "slow down", retry_after=3).retry_after
        3.0
        >>> type(FishHttpError.from_status(503, "down")).__name__
        'FishUpstreamError'
        >>> FishHttpError.from_status(404, "missing").retryable
        False
        """
        return FishHttpError(int(status), message, retry_after=retry_after)

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
        Index into ``FISH_RETRY_ATTEMPTS`` (5), so attempt 4 is the last.

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


def describe_transport_error(exc: BaseException | None, *, timed_out: bool) -> tuple[int, str]:
    """Map a transport failure to a status and message.

    Parameters
    ----------
    exc : BaseException or None
        The caught error. It is accepted so callers keep one call shape, and it is
        never read: its text can name hosts, addresses or errno values, so it
        must not reach a client. A caller that wants the detail logs ``exc``.
    timed_out : bool
        True when the caller already classified ``exc`` as its timeout type.

    Returns
    -------
    tuple of int and str
        ``(504, "Fish request timed out")`` on timeout, otherwise the pair
        ``(502, "Fish upstream unreachable")``.
    """
    del exc
    if timed_out:
        return FISH_TIMEOUT_STATUS, FISH_TIMEOUT_MESSAGE
    return FISH_UNREACHABLE_STATUS, FISH_UNREACHABLE_MESSAGE


def describe_request_error(
    exc: BaseException, timeout_type: type[BaseException]
) -> tuple[int, str]:
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
        See ``describe_transport_error``.
    """
    return describe_transport_error(exc, timed_out=isinstance(exc, timeout_type))


@dataclass(frozen=True, slots=True)
class FishErrorBody:
    """A Fish ``{message, status}`` error, with the text already made safe.

    Attributes
    ----------
    status : int
        HTTP status of the error.
    message : str
        Error text after a UTF-8 round trip.
    """

    status: int
    message: str

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
            The normalized body.
        """
        return cls(int(status), utf8_text(str(message)))

    def as_dict(self) -> dict[str, str | int]:
        """Return the Fish wire shape ``{"message": ..., "status": ...}``."""
        return {"message": self.message, "status": self.status}

    def __getitem__(self, key: Literal["status", "message"]) -> str | int:
        """Read a field by name, for code that still treats the body as a dict.

        Parameters
        ----------
        key : str
            ``"status"`` or ``"message"``.

        Returns
        -------
        str or int
            The field value.

        Raises
        ------
        KeyError
            For any other key.
        """
        if key == "status":
            return self.status
        if key == "message":
            return self.message
        raise KeyError(key)


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
    return FishErrorBody.of(status, message).as_dict()


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


def parse_fish_error(status: int, raw: Any) -> FishErrorBody:
    """Normalize a Fish error body into a status and a message.

    Parameters
    ----------
    status : int
        HTTP status of the response. It wins when the body carries a status
        outside 400-599.
    raw : Any
        The body: bytes, text, JSON, a decoded object or a validation list.

    Returns
    -------
    FishErrorBody
        The status and a readable message. Validation lists become their
        messages joined, a plain-text body is used as is, and an empty one
        becomes ``"HTTP <status>"``.

    Examples
    --------
    >>> parse_fish_error(402, {"message": "Insufficient credits", "status": 402}).message
    'Insufficient credits'
    >>> parse_fish_error(500, b"").message
    'HTTP 500'
    """
    data = _as_dict(raw)
    if data is not None:
        msg = _message_of(data)
        text = str(msg).strip() if msg is not None else ""
        code = parse_number(data.get("status", status), status, int)
        # 200 or 0 in the body must not turn an HTTP 500 into a success.
        if code < 400 or code > 599:
            code = status
        return FishErrorBody.of(code, _fallback_message(text, status))

    if isinstance(raw, list):
        summary = _validation_msgs(cast(list[Any], raw))
        if summary:
            return FishErrorBody.of(status, summary)

    stripped = _decoded(raw).strip()
    if stripped:
        try:
            loaded = json.loads(stripped)
        # Deep nesting such as 20k "[" raises RecursionError, not a decode error.
        except (json.JSONDecodeError, RecursionError):
            return FishErrorBody.of(status, stripped)
        if isinstance(loaded, list):
            summary = _validation_msgs(cast(list[Any], loaded))
            if summary:
                return FishErrorBody.of(status, summary)
        elif _as_dict(loaded) is not None:
            return parse_fish_error(status, loaded)
    return FishErrorBody.of(status, _fallback_message(stripped, status))
