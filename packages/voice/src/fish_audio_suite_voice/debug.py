"""Opt-in duplex debug logs. Off unless FISH_VOICE_DEBUG or --debug."""

from __future__ import annotations

import os
import sys
from typing import Any

from fishaudio.resources import realtime as _fish_rt
from loguru import logger

from fish_audio_suite_kit import env_bool

_SECRET_HEADER = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie"})
_HIDDEN_KEYS = frozenset({"text", "content", "audio", "messages"})
_META_DEPTH = 3
_PUBLIC_HEADERS = frozenset(
    {
        "content-type",
        "retry-after",
        "openai-processing-ms",
        "openai-version",
    }
)


def _is_secret(key: object) -> bool:
    return str(key).lower() in _SECRET_HEADER


_fish_realtime: Any = _fish_rt


class _WsTap:
    on: bool = False


_WS_TAP = _WsTap()


class _ReplyLine:
    open: bool = False


_REPLY = _ReplyLine()


def write_reply_token(text: str) -> None:
    """Stream one LLM token. The line stays open until a debug log or the turn ends."""
    sys.stdout.write(text)
    sys.stdout.flush()
    _REPLY.open = True


def end_reply_line() -> None:
    if not _REPLY.open:
        return
    sys.stdout.write("\n")
    sys.stdout.flush()
    _REPLY.open = False


def env_debug() -> bool:
    return env_bool("FISH_VOICE_DEBUG")


def debug(message: str, *args: Any, **fields: Any) -> None:
    if env_debug():
        end_reply_line()
        logger.debug(message, *args, **fields)


def heartbeat_due(idle_frames: int, every: int) -> bool:
    return env_debug() and idle_frames % every == 0


def _write_stderr(message: str) -> None:
    """Write at emit time so a wrapped stderr (pytest, a later redirect) is the one used."""
    sys.stderr.write(message)


def _stderr_logger(level: str) -> None:
    logger.remove()
    logger.add(
        _write_stderr,
        level=level,
        format="{time:HH:mm:ss.SSS} | {level:<5} | {message}",
        colorize=False,
    )


def configure_voice_logging(*, debug: bool) -> None:
    """stderr DEBUG when on; otherwise only ERROR so debug() is silent."""
    if debug:
        os.environ["FISH_VOICE_DEBUG"] = "1"
    _stderr_logger("DEBUG" if debug else "ERROR")
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


_stderr_logger("ERROR")


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
    out: dict[str, str] = {}
    for key, val in headers.items():
        low = str(key).lower()
        if _is_secret(key):
            continue
        if low.startswith("x-") or low in _PUBLIC_HEADERS:
            out[str(key)] = str(val)
    return out
