"""Opt-in duplex debug logs. Off unless FISH_VOICE_DEBUG or --debug."""

from __future__ import annotations

import os
import sys
from typing import Any

from fishaudio.resources import realtime as _fish_rt
from loguru import logger

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_SECRET_HEADER = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie"})
_fish_realtime: Any = _fish_rt


class _WsTap:
    on: bool = False


_WS_TAP = _WsTap()


def env_debug() -> bool:
    return os.environ.get("FISH_VOICE_DEBUG", "").strip().lower() in _TRUTHY


def configure_voice_logging(*, debug: bool) -> None:
    """stderr DEBUG when on; otherwise only ERROR so debug() is silent."""
    if debug:
        os.environ["FISH_VOICE_DEBUG"] = "1"
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "ERROR",
        format="{time:HH:mm:ss.SSS} | {level:<5} | {message}",
        colorize=False,
    )
    if debug:
        install_fish_ws_tap()
        logger.debug("debug on (Fish WS tap + listen/barge/llm meta)")


def install_fish_ws_tap() -> None:
    """Log live Fish WS msgpack events (audio as byte length only)."""
    if _WS_TAP.on:
        return
    orig_stop = _fish_realtime._should_stop
    orig_proc = _fish_realtime._process_audio_event

    def stop(data: dict[str, Any]) -> bool:
        logger.debug("fish.ws {}", ws_event_view(data))
        return orig_stop(data)

    def proc(data: dict[str, Any]) -> bytes | None:
        if data.get("event") != "audio":
            logger.debug("fish.ws {}", ws_event_view(data))
        return orig_proc(data)

    _fish_realtime._should_stop = stop
    _fish_realtime._process_audio_event = proc
    _WS_TAP.on = True


logger.remove()
logger.add(
    sys.stderr,
    level="ERROR",
    format="{time:HH:mm:ss.SSS} | {level:<5} | {message}",
    colorize=False,
)


def ws_event_view(data: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for key, val in data.items():
        if key == "audio" and isinstance(val, (bytes, bytearray)):
            view["audio_bytes"] = len(val)
        elif isinstance(val, (str, int, float, bool)) or val is None:
            view[key] = val
        elif isinstance(val, (bytes, bytearray)):
            view[f"{key}_bytes"] = len(val)
        else:
            view[key] = type(val).__name__
    return view


def public_meta(data: dict[str, Any], *, depth: int = 0) -> dict[str, Any]:
    """JSON-ish metadata without utterance bodies or secrets."""
    if depth > 3:
        return {"_truncated": True}
    out: dict[str, Any] = {}
    for key, val in data.items():
        low = key.lower()
        if low in _SECRET_HEADER:
            continue
        if low in {"text", "content", "audio", "messages"}:
            if isinstance(val, str):
                out[f"{key}_chars"] = len(val)
            elif isinstance(val, (bytes, bytearray)):
                out[f"{key}_bytes"] = len(val)
            elif isinstance(val, list):
                out[f"{key}_len"] = len(val)
            continue
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[key] = val
        elif isinstance(val, list):
            out[f"{key}_len"] = len(val)
        elif isinstance(val, dict):
            out[key] = public_meta(val, depth=depth + 1)
        elif isinstance(val, (bytes, bytearray)):
            out[f"{key}_bytes"] = len(val)
    return out


def header_meta(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in headers.items():
        low = str(key).lower()
        if low in _SECRET_HEADER:
            continue
        if low.startswith("x-") or low in {
            "content-type",
            "retry-after",
            "openai-processing-ms",
            "openai-version",
        }:
            out[str(key)] = str(val)
    return out
