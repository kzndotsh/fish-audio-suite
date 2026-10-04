"""Tap the Fish websocket SDK to log events, and strip secrets and bodies from log metadata."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Final, TypeGuard

from fishaudio.resources import realtime as _fish_rt
from loguru import logger

from fish_audio_suite_voice.debug import trace

__all__ = [
    "header_meta",
    "install_fish_ws_tap",
    "public_meta",
    "ws_event_view",
]


_SECRET_HEADER: Final = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie"})


_HIDDEN_KEYS: Final = frozenset({"text", "content", "audio", "messages"})


_META_DEPTH: Final = 3


_PUBLIC_HEADERS: Final = frozenset(
    {
        "content-type",
        "retry-after",
        "openai-processing-ms",
        "openai-version",
    }
)


def _is_secret(key: object) -> bool:
    low = str(key).lower().replace("_", "-")
    if low in _SECRET_HEADER:
        return True
    # x-api-key still starts with x-, which the header copy would otherwise keep.
    return any(part in low for part in ("authorization", "api-key", "apikey", "cookie"))


_fish_realtime: Any = _fish_rt


class _WsTap:
    on: bool = False
    skipped: bool = False


_WS_TAP: Final = _WsTap()


# The Fish SDK has no public hook for its websocket events, so the debug tap
# wraps these two private functions. Both are looked up by name and checked
# before use, so an SDK change turns the tap off instead of breaking a stream.
_TAPPED_FUNCTIONS: Final = ("_should_stop", "_process_audio_event")


_Tapped = Callable[[dict[str, Any]], Any]


def _tappable(function: object) -> TypeGuard[_Tapped]:
    if not callable(function):
        return False
    try:
        params = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return len(params) == 1


def install_fish_ws_tap() -> None:
    """Log live Fish WS msgpack events (audio as byte length only).

    Notes
    -----
    This wraps two private functions of the Fish SDK. If either is missing or
    no longer takes one argument, the tap is skipped with one debug line and
    the SDK is left untouched. It never raises.
    """
    if _WS_TAP.on or _WS_TAP.skipped:
        return
    stop_name, proc_name = _TAPPED_FUNCTIONS
    orig_stop = getattr(_fish_realtime, stop_name, None)
    orig_proc = getattr(_fish_realtime, proc_name, None)
    if not (_tappable(orig_stop) and _tappable(orig_proc)):
        _WS_TAP.skipped = True
        unusable = [
            name
            for name, function in ((stop_name, orig_stop), (proc_name, orig_proc))
            if not _tappable(function)
        ]
        logger.debug(
            "debug.tap skipped, the Fish SDK changed: {} missing or has a new signature",
            ", ".join(unusable),
        )
        return

    def stop(data: dict[str, Any]) -> bool:
        kind = data.get("event")
        if kind == "audio":
            trace("tts.audio {} bytes", len(data.get("audio") or b""))
        elif kind == "finish":
            logger.debug("tts.finish reason={}", data.get("reason"))
        else:
            logger.debug("fish.ws {}", ws_event_view(data))
        return bool(orig_stop(data))

    def proc(data: dict[str, Any]) -> bytes | None:
        if data.get("event") not in {"audio", "finish"}:
            logger.debug("fish.ws {}", ws_event_view(data))
        result: bytes | None = orig_proc(data)
        return result

    setattr(_fish_realtime, stop_name, stop)
    setattr(_fish_realtime, proc_name, proc)
    _WS_TAP.on = True


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


def header_meta(headers: Any) -> dict[str, str]:
    """Copy response headers, omitting secrets."""
    out: dict[str, str] = {}
    for key, val in headers.items():
        low = str(key).lower()
        if _is_secret(key):
            continue
        if low.startswith("x-") or low in _PUBLIC_HEADERS:
            out[str(key)] = str(val)
    return out
