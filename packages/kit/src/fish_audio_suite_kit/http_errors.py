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
        self.status = int(status)
        self.message = utf8_text(str(message))
        super().__init__(f"HTTP {self.status}: {self.message}")


def should_retry_fish_status(status: int) -> bool:
    """Retry 429 and 5xx only. Other 4xx need a different request."""
    return status == 429 or status >= 500


def fish_attempt_exhausted(attempt: int) -> bool:
    return attempt + 1 >= FISH_RETRY_ATTEMPTS


def fish_backoff_seconds(attempt: int) -> float:
    """Docs: `2 ** attempt` for attempts 0..4."""
    return float(2 ** max(0, attempt))


async def fish_retry_pause(attempt: int) -> bool:
    """True when this attempt is the last. Otherwise sleep before the next try."""
    if fish_attempt_exhausted(attempt):
        return True
    await asyncio.sleep(fish_backoff_seconds(attempt))
    return False


def bearer(key: str) -> str:
    return f"Bearer {key}"


def fish_unreachable() -> tuple[int, str]:
    return FISH_UNREACHABLE_STATUS, FISH_UNREACHABLE_MESSAGE


def fish_non_json() -> tuple[int, str]:
    return FISH_UNREACHABLE_STATUS, "Fish returned a non-JSON body"


def fish_non_object() -> tuple[int, str]:
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
    if timed_out:
        return FISH_TIMEOUT_STATUS, FISH_TIMEOUT_MESSAGE
    text = str(exc).strip() if exc is not None else ""
    if text:
        return FISH_UNREACHABLE_STATUS, text
    return fish_unreachable()


def fish_request_error(exc: BaseException, timeout_type: type[BaseException]) -> tuple[int, str]:
    return fish_transport_error(exc, timed_out=isinstance(exc, timeout_type))


def fish_error_body(status: int, message: str) -> dict[str, str | int]:
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


def _message_of(data: dict[str, Any]) -> Any:
    msg: Any = None
    for key in ("message", "detail"):
        if data.get(key) is not None:
            msg = data[key]
            break
    if msg is None:
        msg = _nested_message(data.get("error"))
    if isinstance(msg, dict):
        inner = _nested_message(msg)
        return inner if inner is not None else "error"
    return msg


def _decoded(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="replace")
    if raw is None:
        return ""
    return str(raw)


def _fallback_message(text: str, status: int) -> str:
    return text or f"HTTP {status}"


def parse_fish_error(status: int, raw: Any) -> dict[str, str | int]:
    """Normalize Fish `{message, status}` or a plain-text parse error."""
    data = _as_dict(raw)
    if data is not None:
        msg = _message_of(data)
        text = str(msg).strip() if msg is not None else ""
        code = number_or(data.get("status", status), status, int)
        return fish_error_body(code, _fallback_message(text, status))

    stripped = _decoded(raw).strip()
    if stripped:
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return fish_error_body(status, stripped)
        if _as_dict(loaded) is not None:
            return parse_fish_error(status, loaded)
    return fish_error_body(status, _fallback_message(stripped, status))
