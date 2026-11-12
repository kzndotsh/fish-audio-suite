"""Tap the Fish websocket SDK to log its events."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Final, TypeGuard

from fishaudio.resources import realtime as _fish_rt
from loguru import logger

from fish_audio_suite_voice.debug import trace
from fish_audio_suite_voice.log_meta import ws_event_view

__all__ = [
    "install_fish_ws_tap",
]


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
