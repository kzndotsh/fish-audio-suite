"""What a log line may show of a payload: sizes for bodies and bytes, nothing for secrets."""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "public_meta",
    "ws_event_view",
]


_SECRET_HEADER: Final = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie"})


# What the user said or will hear. Logged by size only.
_HIDDEN_KEYS: Final = frozenset({"text", "content", "audio", "messages"})
# Nested metadata deeper than this is cut off, so a large body cannot flood the log.
_META_DEPTH: Final = 3


def _is_secret(key: object) -> bool:
    low = str(key).lower().replace("_", "-")
    if low in _SECRET_HEADER:
        return True
    # A name that only contains a secret word, such as x-api-key or api_key, is one too.
    return any(part in low for part in ("authorization", "api-key", "apikey", "cookie"))


def _plain(val: object) -> bool:
    return isinstance(val, (str, int, float, bool)) or val is None


def _byte_len(key: str, val: object) -> tuple[str, int] | None:
    if isinstance(val, (bytes, bytearray)):
        return f"{key}_bytes", len(val)
    return None


def _hidden_size(key: str, val: object) -> tuple[str, int] | None:
    sized = _byte_len(key, val)
    if sized is not None:
        return sized
    if isinstance(val, str):
        return f"{key}_chars", len(val)
    if isinstance(val, list):
        return f"{key}_len", len(val)
    return None


def _note_size(out: dict[str, Any], sized: tuple[str, int] | None) -> bool:
    if sized is None:
        return False
    name, count = sized
    out[name] = count
    return True


def ws_event_view(data: dict[str, Any]) -> dict[str, Any]:
    """Copy a websocket event, replacing byte payloads with their lengths."""
    view: dict[str, Any] = {}
    for key, val in data.items():
        if _plain(val):
            view[key] = val
            continue
        if _note_size(view, _byte_len(key, val)):
            continue
        view[key] = type(val).__name__
    return view


def public_meta(data: dict[str, Any], *, depth: int = 0) -> dict[str, Any]:
    """JSON-ish metadata without utterance bodies or secrets."""
    if depth > _META_DEPTH:
        return {"_truncated": True}
    out: dict[str, Any] = {}
    for key, val in data.items():
        low = str(key).lower()
        if _is_secret(key):
            continue
        if low in _HIDDEN_KEYS:
            _note_size(out, _hidden_size(key, val))
            continue
        if _plain(val):
            out[key] = val
        elif isinstance(val, dict):
            out[key] = public_meta(val, depth=depth + 1)
        else:
            _note_size(out, _hidden_size(key, val))
    return out
