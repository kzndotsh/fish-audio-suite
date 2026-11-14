"""Opt-in duplex debug logs. Off unless FISH_VOICE_DEBUG or --debug.

Level 1 (``--debug``) logs events. Level 2 (``--trace`` or ``FISH_VOICE_DEBUG=2``)
adds the mic heartbeats, raw websocket audio and HTTP request lines. This is the module
every step logs through: the debug level, ``debug``, ``trace``, ``warn`` and the
per-turn clock. How lines look is ``logutil.format``, the run's file is ``logutil.file``, and
where loguru sends them is set up by ``logutil.setup``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from enum import IntEnum
from typing import Any, Final

from loguru import logger

from fish_audio_suite_kit import env_bool
from fish_audio_suite_voice.logutil.file import log_file_on, log_record
from fish_audio_suite_voice.logutil.format import dim, start_turn_clock, stop_turn_clock

__all__ = [
    "DebugLevel",
    "clear_turn",
    "close_terminal_line",
    "debug",
    "debug_enabled",
    "debug_level",
    "fix_debug_level",
    "heartbeat_due",
    "mark_turn",
    "on_terminal_line",
    "reset_debug_level",
    "trace",
    "warn",
]


class DebugLevel(IntEnum):
    """How much the voice CLI logs."""

    OFF = 0
    EVENTS = 1
    TRACE = 2


class _Debug:
    """Debug level set by ``fix_debug_level``. Avoids writing os.environ.

    ``frozen`` is the effective level once logging is configured. Until then the
    environment is read live, so library callers that never configure logging
    still honor ``FISH_VOICE_DEBUG``. ``configured`` is True once loguru has a
    terminal or log sink, so ``warn`` may go through it.
    """

    level: DebugLevel = DebugLevel.OFF
    frozen: DebugLevel | None = None
    configured: bool = False


def _nothing() -> None:
    return None


class _Terminal:
    """What runs before a line is written to the terminal: a display closing its open line."""

    def __init__(self) -> None:
        self.before_line: Callable[[], None] = _nothing


_DEBUG: Final = _Debug()
_TERMINAL: Final = _Terminal()
_TRACE_WORDS: Final = frozenset({"2", "trace"})
_RULE_WIDTH: Final = 56


def debug_level() -> DebugLevel:
    """Return how much to log: off, events, or events plus heartbeats and raw traffic.

    Returns
    -------
    DebugLevel
        The level ``fix_debug_level`` fixed, which is the higher of its argument and
        ``FISH_VOICE_DEBUG`` at that moment. Before logging is configured the
        environment is read on each call. ``2`` or ``trace`` is ``TRACE``, any other
        true value is ``EVENTS``.

    Notes
    -----
    Mic frames call this up to every 30 ms on two threads, so after logging is
    configured it never touches ``os.environ``.
    """
    if _DEBUG.frozen is not None:
        return _DEBUG.frozen
    return max(_DEBUG.level, _env_level())


def _env_level() -> DebugLevel:
    if os.environ.get("FISH_VOICE_DEBUG", "").strip().lower() in _TRACE_WORDS:
        return DebugLevel.TRACE
    return DebugLevel.EVENTS if env_bool("FISH_VOICE_DEBUG") else DebugLevel.OFF


def fix_debug_level(debug: bool | int | DebugLevel) -> DebugLevel:
    """Fix the debug level for the rest of the run, once logging has its sinks.

    Parameters
    ----------
    debug : bool or int or DebugLevel
        ``False``, 0 or ``OFF`` logs warnings only. ``True``, 1 or ``EVENTS``
        adds events. 2 or ``TRACE`` adds the heartbeats, raw websocket audio and
        HTTP request lines. Larger numbers count as ``TRACE``.

    Returns
    -------
    DebugLevel
        The higher of ``debug`` and ``FISH_VOICE_DEBUG``, which ``debug_level`` returns
        from now on without reading the environment again.
    """
    level = max(DebugLevel(min(max(int(debug), 0), DebugLevel.TRACE)), _env_level())
    _DEBUG.level = level
    _DEBUG.frozen = level
    _DEBUG.configured = True
    return level


def reset_debug_level() -> None:
    """Undo ``fix_debug_level``: read ``FISH_VOICE_DEBUG`` live again, as before logging was set up.

    ``warn`` writes straight to stderr again until logging is configured once more. For
    tests and for a host that sets logging up a second time; it removes no loguru sinks.
    """
    _DEBUG.level = DebugLevel.OFF
    _DEBUG.frozen = None
    _DEBUG.configured = False


def debug_enabled() -> bool:
    """Return whether debug logging is on.

    Returns
    -------
    bool
        True after ``configure_voice_logging(debug=True)``, or when
        ``FISH_VOICE_DEBUG`` is set to a true value.
    """
    return debug_level() >= DebugLevel.EVENTS


def on_terminal_line(before: Callable[[], None]) -> Callable[[], None]:
    """Run ``before`` ahead of each line logging writes to the terminal.

    Parameters
    ----------
    before : Callable
        Called first, on the logging thread. The plain display closes the reply line
        it left open, so the log line does not land in the middle of it.

    Returns
    -------
    Callable
        Call it to stop. It does nothing once another callback has taken over.
    """
    _TERMINAL.before_line = before

    def stop() -> None:
        if _TERMINAL.before_line is before:
            _TERMINAL.before_line = _nothing

    return stop


def close_terminal_line() -> None:
    """Run the ``on_terminal_line`` callback: a line is about to be written to the terminal.

    The stderr log sink calls it for every line it writes, including those a library logs
    through the standard ``logging`` module.
    """
    _TERMINAL.before_line()


def debug(message: str, *args: Any, **fields: Any) -> None:
    """Log at debug when FISH_VOICE_DEBUG is on, after closing the token line.

    With a log file (``FISH_VOICE_LOG_DIR``) a line that is not shown is still written there.
    """
    if debug_enabled():
        _TERMINAL.before_line()
        logger.debug(message, *args, **fields)
    elif log_file_on():
        log_record(message, args, fields)


def trace(message: str, *args: Any, **fields: Any) -> None:
    """Log only at level 2. Use for per-frame and per-chunk detail.

    With a log file a line below that level is still written there.
    """
    if debug_level() >= DebugLevel.TRACE:
        _TERMINAL.before_line()
        logger.debug(message, *args, **fields)
    elif log_file_on():
        log_record(message, args, fields)


def heartbeat_due(idle_frames: int, every: int) -> bool:
    """Return whether a heartbeat should print on this idle frame (level 2, or a log file)."""
    return (debug_level() >= DebugLevel.TRACE or log_file_on()) and idle_frames % every == 0


def mark_turn() -> None:
    """Start the per-turn clock and print a rule. Later debug lines show seconds since now."""
    count = start_turn_clock()
    if debug_enabled():
        _TERMINAL.before_line()
        title = f" turn {count} "
        sys.stderr.write(dim(title.center(_RULE_WIDTH, "─")) + "\n")


def clear_turn() -> None:
    """Stop the per-turn clock, so listening lines carry no stale offset."""
    stop_turn_clock()


def warn(message: str) -> None:
    """Stderr diagnostic. Closes an open reply line first. Does not configure logging."""
    _TERMINAL.before_line()
    if _DEBUG.configured:
        logger.warning(message)
        return
    sys.stderr.write(f"{message}\n")
