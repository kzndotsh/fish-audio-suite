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
from enum import IntEnum
from typing import Any, Final, override

from loguru import logger

from fish_audio_suite_kit import env_bool
from fish_audio_suite_voice.console import console_print, end_reply_line

__all__ = [
    "DebugLevel",
    "clear_turn",
    "configure_voice_logging",
    "conversation",
    "debug",
    "debug_enabled",
    "debug_level",
    "heartbeat_due",
    "mark_turn",
    "short_model",
    "trace",
    "warn",
    "with_detail",
]


def with_detail(message: str, exc: BaseException) -> str:
    """Add an error's own text to a fixed message, for the local terminal.

    Parameters
    ----------
    message : str
        The fixed message, for example the kit's ``Fish upstream unreachable``.
    exc : BaseException
        The error it stands for.

    Returns
    -------
    str
        ``message: text`` when the error has text of its own, otherwise
        ``message``. Kit keeps client-facing messages free of exception text;
        voice is a local tool, so the user sees the detail here.
    """
    detail = str(exc).strip()
    return f"{message}: {detail}" if detail and detail != message else message


def short_model(model: object) -> str:
    """Return a model id without its vendor prefix, for display only.

    Parameters
    ----------
    model : object
        An id such as ``vendor/name-7b``. ``None`` or blank gives ``""``.

    Returns
    -------
    str
        The part after the last ``/``. Any ``:suffix`` stays. Use the full id
        wherever two ids are compared or sent to the API.
    """
    return str(model or "").strip().rsplit("/", 1)[-1]


class DebugLevel(IntEnum):
    """How much the voice CLI logs."""

    OFF = 0
    EVENTS = 1
    TRACE = 2


class _Debug:
    """Debug level set by ``configure_voice_logging``. Avoids writing os.environ.

    ``frozen`` is the effective level once logging is configured. Until then the
    environment is read live, so library callers that never configure logging
    still honor ``FISH_VOICE_DEBUG``.
    """

    level: DebugLevel = DebugLevel.OFF
    frozen: DebugLevel | None = None


_DEBUG: Final = _Debug()
_TRACE_WORDS: Final = frozenset({"2", "trace"})


def debug_level() -> DebugLevel:
    """Return how much to log: off, events, or events plus heartbeats and raw traffic.

    Returns
    -------
    DebugLevel
        The level ``configure_voice_logging`` fixed, which is the higher of its
        argument and ``FISH_VOICE_DEBUG`` at that moment. Before logging is
        configured the environment is read on each call. ``2`` or ``trace`` is
        ``TRACE``, any other true value is ``EVENTS``.

    Notes
    -----
    Mic frames call this up to every 30 ms on two threads, so after
    ``configure_voice_logging`` it never touches ``os.environ``.
    """
    if _DEBUG.frozen is not None:
        return _DEBUG.frozen
    return max(_DEBUG.level, _env_level())


def _env_level() -> DebugLevel:
    if os.environ.get("FISH_VOICE_DEBUG", "").strip().lower() in _TRACE_WORDS:
        return DebugLevel.TRACE
    return DebugLevel.EVENTS if env_bool("FISH_VOICE_DEBUG") else DebugLevel.OFF


def debug_enabled() -> bool:
    """Return whether debug logging is on.

    Returns
    -------
    bool
        True after ``configure_voice_logging(debug=True)``, or when
        ``FISH_VOICE_DEBUG`` is set to a true value.
    """
    return debug_level() >= DebugLevel.EVENTS


def debug(message: str, *args: Any, **fields: Any) -> None:
    """Log at debug when FISH_VOICE_DEBUG is on, after closing the token line."""
    if debug_enabled():
        end_reply_line()
        logger.debug(message, *args, **fields)


def trace(message: str, *args: Any, **fields: Any) -> None:
    """Log only at level 2. Use for per-frame and per-chunk detail."""
    if debug_level() >= DebugLevel.TRACE:
        end_reply_line()
        logger.debug(message, *args, **fields)


def heartbeat_due(idle_frames: int, every: int) -> bool:
    """Return whether a heartbeat should print on this idle frame (level 2 only)."""
    return debug_level() >= DebugLevel.TRACE and idle_frames % every == 0


class _Turn:
    t0: float | None = None
    count: int = 0


_TURN: Final = _Turn()
_RULE_WIDTH: Final = 56


def mark_turn() -> None:
    """Start the per-turn clock and print a rule. Later debug lines show seconds since now."""
    _TURN.t0 = time.perf_counter()
    _TURN.count += 1
    if debug_enabled():
        end_reply_line()
        title = f" turn {_TURN.count} "
        _write_stderr(_dim(title.center(_RULE_WIDTH, "\u2500")) + "\n")


def clear_turn() -> None:
    """Stop the per-turn clock, so listening lines carry no stale offset."""
    _TURN.t0 = None


_TAGGED: Final = re.compile(r"^([a-z]+)\.([a-z_]+)\b ?(.*)$", re.DOTALL)
_BRACKETED: Final = re.compile(r"^\[([A-Za-z-]+)\]\s*(.*)$", re.DOTALL)
_TAG_WIDTH: Final = 7
_TAG_COLORS: Final = {
    "you": "96",
    "listen": "36",
    "asr": "33",
    "llm": "35",
    "turn": "37",
    "tts": "32",
    "barge": "31",
    "aec": "34",
    "warn": "93",
}


def _color_on(stream: Any = None) -> bool:
    target = stream if stream is not None else sys.stderr
    return bool(target.isatty()) and "NO_COLOR" not in os.environ


def _dim(text: str, stream: Any = None) -> str:
    return f"\x1b[2m{text}\x1b[0m" if _color_on(stream) else text


def _paint(code: str, text: str, stream: Any = None) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _color_on(stream) else text


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


def _compose(stamp: str, tag: str, label: str, body: str, stream: Any = None) -> str:
    tag_text = _paint(_TAG_COLORS.get(tag, "90"), label.ljust(_TAG_WIDTH), stream)
    return f"{_dim(stamp, stream)} {_dim(_offset(), stream)}  {tag_text}{body}"


def _format_record(record: Any) -> str:
    stamp = record["time"].strftime("%H:%M:%S.") + f"{record['time'].microsecond // 1000:03d}"
    tag, body = _split_tag(str(record["message"]), record["level"].name)
    if record["level"].name == "WARNING":
        tag = "warn" if tag == "log" else tag
        body = _paint("93", body)
    record["extra"]["line"] = _compose(stamp, tag, tag, body)
    return "{extra[line]}\n{exception}"


def _now_stamp() -> str:
    now = time.time()
    millis = int((now % 1) * 1000)
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{millis:03d}"


_ROLE_COLORS: Final = {"you": "96", "llm": "95"}


def _stdout_tty() -> bool:
    return bool(sys.stdout.isatty())


def conversation(role: str, text: str) -> None:
    """Print one line of the conversation, ``you`` or ``llm``.

    Parameters
    ----------
    role : str
        ``you`` or ``llm``.
    text : str
        What was said.

    Notes
    -----
    The conversation always goes to stdout, so piping it keeps the transcript.
    On a terminal with debug on, the line carries the same time columns and
    role tag as the log lines on stderr. Written and flushed in order, it stays
    in sequence with them. Piped, or with debug off, it is a plain
    ``role \u25b8 text`` line.
    """
    end_reply_line()
    label = f"{role} \u25b8"
    if not (debug_enabled() and _stdout_tty()):
        console_print(f"{label} {text}", flush=True)
        return
    color = _ROLE_COLORS.get(role, "97")
    body = _paint(f"1;{color}", text, sys.stdout)
    console_print(_compose(_now_stamp(), role, label, body, sys.stdout), flush=True)


def _write_stderr(message: str) -> None:
    """Write at emit time so a wrapped stderr (pytest, a later redirect) is the one used."""
    sys.stderr.write(message)


class _Configured:
    """True after the CLI installs the stderr sink. warn() must not configure loguru itself."""

    on: bool = False


_CONFIGURED: Final = _Configured()


def warn(message: str) -> None:
    """Stderr diagnostic. Closes an open reply line first. Does not configure logging."""
    end_reply_line()
    if _CONFIGURED.on:
        logger.warning(message)
        return
    _write_stderr(f"{message}\n")


class _InterceptHandler(logging.Handler):
    @override
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


_INTERCEPTED: Final = ("httpx", "httpcore", "websockets", "asyncio")
# httpcore DEBUG prints raw header bytes (b'...'), including Set-Cookie.
# httpx INFO is the one-line "HTTP Request: METHOD url status" record.
_HTTPX_DEBUG_LEVEL: Final = logging.INFO


def _intercept_libraries(*, level: DebugLevel) -> None:
    handler = _InterceptHandler()
    for name in _INTERCEPTED:
        lib = logging.getLogger(name)
        lib.handlers.clear()
        lib.addHandler(handler)
        lib.propagate = False
        if name == "httpx" and level >= DebugLevel.TRACE:
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


def configure_voice_logging(*, debug: bool | int | DebugLevel) -> None:
    """Idempotent stderr sink. DEBUG when on; otherwise WARNING. Call from the CLI only.

    Parameters
    ----------
    debug : bool or int or DebugLevel
        ``False``, 0 or ``OFF`` logs warnings only. ``True``, 1 or ``EVENTS``
        adds events. 2 or ``TRACE`` adds the heartbeats, raw websocket audio and
        HTTP request lines. Larger numbers count as ``TRACE``.
    """
    level = max(DebugLevel(min(max(int(debug), 0), DebugLevel.TRACE)), _env_level())
    _DEBUG.level = level
    _DEBUG.frozen = level
    _stderr_logger("DEBUG" if level >= DebugLevel.EVENTS else "WARNING")
    _CONFIGURED.on = True
    _intercept_libraries(level=level)
    if level >= DebugLevel.EVENTS:
        shown = (
            "events, heartbeats and raw traffic"
            if level >= DebugLevel.TRACE
            else "events (--trace adds more)"
        )
        logger.debug("debug.on {}", shown)
