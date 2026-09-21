"""Fish HTTP error shape and retry policy. No network."""

from __future__ import annotations

import json
from typing import Any, cast

FISH_RETRY_ATTEMPTS = 5


class FishHttpError(Exception):
    """REST/WS-adjacent Fish failure after retries or a non-retryable status."""

    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        self.message = str(message)
        super().__init__(f"HTTP {self.status}: {self.message}")


def should_retry_fish_status(status: int) -> bool:
    """Retry 429 and 5xx only. Other 4xx need a different request."""
    return status == 429 or status >= 500


def fish_backoff_seconds(attempt: int) -> float:
    """Docs: `2 ** attempt` for attempts 0..4."""
    return float(2 ** max(0, attempt))


def fish_error_body(status: int, message: str) -> dict[str, str | int]:
    return {"message": str(message), "status": int(status)}


def parse_fish_error(status: int, raw: Any) -> dict[str, str | int]:
    """Normalize Fish `{message, status}` or a plain-text parse error."""
    if isinstance(raw, dict):
        data = cast(dict[str, Any], raw)
        msg: Any = data.get("message")
        if msg is None:
            msg = data.get("detail")
        if msg is None:
            err: Any = data.get("error")
            msg = cast(dict[str, Any], err).get("message") if isinstance(err, dict) else err
        if isinstance(msg, dict):
            inner = cast(dict[str, Any], msg).get("message")
            msg = inner if inner is not None else "error"
        parsed: Any = data.get("status", status)
        try:
            parsed_status = int(parsed)
        except (TypeError, ValueError):
            parsed_status = status
        text = str(msg).strip() if msg is not None else ""
        return fish_error_body(parsed_status, text or f"HTTP {status}")

    if isinstance(raw, (bytes, bytearray)):
        text = bytes(raw).decode("utf-8", errors="replace")
    else:
        text = str(raw) if raw is not None else ""
    stripped = text.strip()
    if stripped:
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return fish_error_body(status, stripped)
        if isinstance(loaded, dict):
            return parse_fish_error(status, loaded)
    return fish_error_body(status, stripped or f"HTTP {status}")
