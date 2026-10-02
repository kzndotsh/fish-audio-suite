"""Opt-in duplex debug logs. Off unless FISH_VOICE_DEBUG or --debug.

Level 1 (``--debug``) logs events. Level 2 (``--trace`` or ``FISH_VOICE_DEBUG=2``)
adds the mic heartbeats, raw websocket audio and HTTP request lines.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
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
    low = str(key).lower().replace("_", "-")
    if low in _SECRET_HEADER:
        return True
    # x-api-key still starts with x-, which the header copy would otherwise keep.
    return any(part in low for part in ("authorization", "api-key", "apikey", "cookie"))


_fish_realtime: Any = _fish_rt


class _WsTap:
    on: bool = False


_WS_TAP = _WsTap()


class _ReplyLine:
    open: bool = False


_REPLY = _ReplyLine()


def console_print(*args: object, **kwargs: Any) -> None:
    """Print a status line. A closed stdout must not drop the spoken reply."""
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        return


def write_reply_token(text: str) -> None:
    """Stream one LLM token. The line stays open until a debug log or the turn ends."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        return
    _REPLY.open = True


def end_reply_line() -> None:
    """Close the open token line with a newline."""
    if not _REPLY.open:
        return
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        pass
    _REPLY.open = False


class _Debug:
    """Debug level set by ``configure_voice_logging``. Avoids writing os.environ."""

    level: int = 0


_DEBUG = _Debug()
_TRACE_WORDS = frozenset({"2", "trace"})


def debug_level() -> int:
    """Return the debug level: 0 off, 1 events, 2 events plus heartbeats and raw traffic.

    Returns
    -------
    int
        The level set by ``configure_voice_logging``, or the one
        ``FISH_VOICE_DEBUG`` asks for when that is higher. ``2`` or ``trace``
        is level 2, any other true value is level 1.
    """
    raw = os.environ.get("FISH_VOICE_DEBUG", "").strip().lower()
    from_env = int(env_bool("FISH_VOICE_DEBUG"))
    if raw in _TRACE_WORDS:
        from_env = 2
    return max(_DEBUG.level, from_env)


def env_debug() -> bool:
    """Return whether debug logging is on.

    Returns
    -------
    bool
        True after ``configure_voice_logging(debug=True)``, or when
        ``FISH_VOICE_DEBUG`` is set to a true value.
    """
    return debug_level() >= 1


def debug(message: str, *args: Any, **fields: Any) -> None:
    """Log at debug when FISH_VOICE_DEBUG is on, after closing the token line."""
    if env_debug():
        end_reply_line()
        logger.debug(message, *args, **fields)


def trace(message: str, *args: Any, **fields: Any) -> None:
    """Log only at level 2. Use for per-frame and per-chunk detail."""
    if debug_level() >= 2:
        end_reply_line()
        logger.debug(message, *args, **fields)


def heartbeat_due(idle_frames: int, every: int) -> bool:
    """Return whether a heartbeat should print on this idle frame (level 2 only)."""
    return debug_level() >= 2 and idle_frames % every == 0


class _Turn:
    t0: float | None = None
    count: int = 0


_TURN = _Turn()
_RULE_WIDTH = 56


def mark_turn() -> None:
    """Start the per-turn clock and print a rule. Later debug lines show seconds since now."""
    _TURN.t0 = time.perf_counter()
    _TURN.count += 1
    if env_debug():
        end_reply_line()
        title = f" turn {_TURN.count} "
        _write_stderr(_dim(title.center(_RULE_WIDTH, "\u2500")) + "\n")


def clear_turn() -> None:
    """Stop the per-turn clock, so listening lines carry no stale offset."""
    _TURN.t0 = None


_TAGGED = re.compile(r"^([a-z]+)\.([a-z_]+)\b ?(.*)$", re.DOTALL)
_BRACKETED = re.compile(r"^\[([A-Za-z-]+)\]\s*(.*)$", re.DOTALL)
_TAG_WIDTH = 7
_TAG_COLORS = {
    "listen": "36",
    "asr": "33",
    "llm": "35",
    "tts": "32",
    "barge": "31",
    "aec": "34",
    "warn": "93",
}


def _color_on() -> bool:
    return sys.stderr.isatty() and "NO_COLOR" not in os.environ


def _dim(text: str) -> str:
    return f"\x1b[2m{text}\x1b[0m" if _color_on() else text


def _paint(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _color_on() else text


def _split_tag(message: str, level: str) -> tuple[str, str]:
    tagged = _TAGGED.match(message)
    if tagged:
        tag, name, rest = tagged.groups()
        return tag, f"{name} {rest}".rstrip()
    bracketed = _BRACKETED.match(message)
    if bracketed:
        return bracketed.group(1).lower(), bracketed.group(2)
    if message.startswith("HTTP Request:"):
        return "http", message.removeprefix("HTTP Request:").strip()
    return ("warn" if level == "WARNING" else "log"), message


def _offset() -> str:
    t0 = _TURN.t0
    if t0 is None:
        return " " * 8
    return f"{time.perf_counter() - t0:+.2f}s".rjust(8)


def _format_record(record: Any) -> str:
    stamp = record["time"].strftime("%H:%M:%S.") + f"{record['time'].microsecond // 1000:03d}"
    tag, body = _split_tag(str(record["message"]), record["level"].name)
    if record["level"].name == "WARNING":
        tag = "warn" if tag == "log" else tag
        body = _paint("93", body)
    tag_text = _paint(_TAG_COLORS.get(tag, "90"), tag.ljust(_TAG_WIDTH))
    record["extra"]["line"] = f"{_dim(stamp)} {_dim(_offset())}  {tag_text}{body}"
    return "{extra[line]}\n{exception}"


def _write_stderr(message: str) -> None:
    """Write at emit time so a wrapped stderr (pytest, a later redirect) is the one used."""
    sys.stderr.write(message)


class _Configured:
    """True after the CLI installs the stderr sink. warn() must not configure loguru itself."""

    on: bool = False


_CONFIGURED = _Configured()


def warn(message: str) -> None:
    """Stderr diagnostic. Closes an open reply line first. Does not configure logging."""
    end_reply_line()
    if _CONFIGURED.on:
        logger.warning(message)
        return
    _write_stderr(f"{message}\n")


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


_INTERCEPTED = ("httpx", "httpcore", "websockets", "asyncio")
# httpcore DEBUG prints raw header bytes (b'...'), including Set-Cookie.
# httpx INFO is the one-line "HTTP Request: METHOD url status" record.
_HTTPX_DEBUG_LEVEL = logging.INFO


def _intercept_libraries(*, level: int) -> None:
    handler = _InterceptHandler()
    for name in _INTERCEPTED:
        lib = logging.getLogger(name)
        lib.handlers.clear()
        lib.addHandler(handler)
        lib.propagate = False
        if name == "httpx" and level >= 2:
            lib.setLevel(_HTTPX_DEBUG_LEVEL)
        else:
            lib.setLevel(logging.WARNING)


def _stderr_logger(level: str) -> None:
    logger.remove()
    logger.add(
        _write_stderr,
        level=level,
        format=_format_record,
        colorize=False,
    )


def configure_voice_logging(*, debug: bool | int) -> None:
    """Idempotent stderr sink. DEBUG when on; otherwise WARNING. Call from the CLI only.

    Parameters
    ----------
    debug : bool or int
        ``False`` or 0 logs warnings only. ``True`` or 1 adds events. 2 adds the
        heartbeats, raw websocket audio and HTTP request lines.
    """
    level = int(debug)
    _DEBUG.level = level
    _stderr_logger("DEBUG" if level >= 1 else "WARNING")
    _CONFIGURED.on = True
    _intercept_libraries(level=level)
    if level >= 1:
        install_fish_ws_tap()
        shown = "events, heartbeats and raw traffic" if level >= 2 else "events (--trace adds more)"
        logger.debug("debug.on {}", shown)


def install_fish_ws_tap() -> None:
    """Log live Fish WS msgpack events (audio as byte length only)."""
    if _WS_TAP.on:
        return
    orig_stop = _fish_realtime._should_stop
    orig_proc = _fish_realtime._process_audio_event

    def stop(data: dict[str, Any]) -> bool:
        kind = data.get("event")
        if kind == "audio":
            trace("tts.audio {} bytes", len(data.get("audio") or b""))
        elif kind == "finish":
            logger.debug("tts.finish reason={}", data.get("reason"))
        else:
            logger.debug("fish.ws {}", ws_event_view(data))
        return orig_stop(data)

    def proc(data: dict[str, Any]) -> bytes | None:
        if data.get("event") not in {"audio", "finish"}:
            logger.debug("fish.ws {}", ws_event_view(data))
        return orig_proc(data)

    _fish_realtime._should_stop = stop
    _fish_realtime._process_audio_event = proc
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
