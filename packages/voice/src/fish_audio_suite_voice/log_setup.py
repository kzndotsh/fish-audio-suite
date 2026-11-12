"""Where log lines go: the terminal, the run's log file, or a display's log pane.

Called once by the CLI at start-up (``configure_voice_logging``), never at import. The
session's own steps log through ``debug``; this module only decides the sinks.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, override

from loguru import logger

from fish_audio_suite_voice.debug import DebugLevel, debug_enabled, debug_level, fix_debug_level
from fish_audio_suite_voice.events import EVENTS, EventBus, Heard, LogLine, ReplyEnd
from fish_audio_suite_voice.log_file import (
    log_file_on,
    log_file_path,
    log_file_sink,
    log_to_file,
    open_log_file,
)
from fish_audio_suite_voice.log_format import clock_stamp, compose, log_tag, paint

__all__ = [
    "configure_voice_logging",
    "forward_logs",
    "mirror_conversation",
]


def _format_record(record: Any) -> str:
    tag, body = log_tag(str(record["message"]), record["level"].name)
    if record["level"].name == "WARNING":
        body = paint("93", body)
    record["extra"]["line"] = compose(clock_stamp(record["time"]), tag, tag, body)
    return "{extra[line]}\n{exception}"


def _displayed(record: Any) -> bool:
    """Say whether a log record belongs on the screen: not file-only, and not debug when off."""
    if record["extra"].get("file_only"):
        return False
    return record["level"].no >= logger.level("WARNING").no or debug_enabled()


def _write_stderr(message: str) -> None:
    """Write at emit time so a wrapped stderr (pytest, a later redirect) is the one used."""
    sys.stderr.write(message)


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
        hidden = record.levelno < logging.WARNING and debug_level() < DebugLevel.TRACE
        scoped = logger.bind(file_only=True) if hidden else logger
        scoped.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


_INTERCEPTED: Final = ("httpx", "httpcore", "websockets", "asyncio")
# httpcore DEBUG prints raw header bytes (b'...'), including Set-Cookie.
# httpx INFO is the one-line "HTTP Request: METHOD url status" record.
_HTTPX_DEBUG_LEVEL: Final = logging.INFO


def _intercept_libraries(*, level: DebugLevel) -> None:
    if log_file_on():
        level = DebugLevel.TRACE  # the file gets the request lines too
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


def configure_voice_logging(
    *, debug: bool | int | DebugLevel, to_stderr: bool = True, log_dir: str = ""
) -> Path | None:
    """Idempotent stderr sink. DEBUG when on; otherwise WARNING. Call from the CLI only.

    Parameters
    ----------
    debug : bool or int or DebugLevel
        ``False``, 0 or ``OFF`` logs warnings only. ``True``, 1 or ``EVENTS``
        adds events. 2 or ``TRACE`` adds the heartbeats, raw websocket audio and
        HTTP request lines. Larger numbers count as ``TRACE``.
    to_stderr : bool, optional
        False installs no stderr sink, for a display that owns the terminal. Log
        lines then go only to sinks added afterwards, such as ``forward_logs``.
    log_dir : str, optional
        A folder to write a complete log to, in a new file for this run. It gets every line at
        the most detailed level, whatever ``debug`` shows on the screen. Empty keeps an
        earlier file, or none. Calling again without it keeps the same file.

    Returns
    -------
    Path or None
        The log file in use, or None.
    """
    if log_dir:
        open_log_file(log_dir)
    level = fix_debug_level(debug)
    logger.remove()
    if to_stderr:
        logger.add(
            _write_stderr,
            level="DEBUG" if level >= DebugLevel.EVENTS else "WARNING",
            format=_format_record,
            colorize=False,
            filter=_displayed,
        )
    if log_file_on():
        logger.add(log_file_sink, level="DEBUG", format="{message}")
    _intercept_libraries(level=level)
    if level >= DebugLevel.EVENTS:
        shown = (
            "events, heartbeats and raw traffic"
            if level >= DebugLevel.TRACE
            else "events (--trace adds more)"
        )
        logger.debug("debug.on {}", shown)
    if log_dir and log_file_on():
        log_to_file("debug", f"log file {log_file_path()}")
    return log_file_path()


def forward_logs(bus: EventBus = EVENTS, *, level: str = "DEBUG") -> Callable[[], None]:
    """Send each log line the screen would show to ``bus`` as a ``LogLine``.

    Parameters
    ----------
    bus : EventBus, optional
        Where to send them. The session's ``EVENTS`` by default.
    level : str, optional
        The lowest log level to forward.

    Returns
    -------
    Callable
        Call it to stop forwarding. Calling it twice is harmless.

    Notes
    -----
    Use it with ``configure_voice_logging(..., to_stderr=False)`` when a display
    owns the terminal, so log lines reach it instead of corrupting the screen.
    Call it after ``configure_voice_logging``, which removes every other sink.
    """

    def sink(message: Any) -> None:
        record = message.record
        name = record["level"].name
        tag, text = log_tag(str(record["message"]), name)
        bus.emit(LogLine(name, tag, text))

    handler = logger.add(sink, level=level, format="{message}", filter=_displayed)
    stopped = threading.Event()

    def stop() -> None:
        if not stopped.is_set():
            stopped.set()
            logger.remove(handler)

    return stop


def mirror_conversation() -> Callable[[], None]:
    """Write what was heard and what the model answered to the log file.

    Returns
    -------
    Callable
        Call it to stop.
    """

    def write(event: object) -> None:
        if isinstance(event, Heard):
            log_to_file("you", f"▸ {event.text}")
        elif isinstance(event, ReplyEnd):
            log_to_file("llm", f"▸ {event.text}")

    return EVENTS.subscribe(write)
