"""The run's log file (``FISH_VOICE_LOG_DIR`` or ``--log-dir``): every line, whatever the screen shows."""

from __future__ import annotations

import atexit
import contextlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final, TextIO

from fish_audio_suite_voice.log_format import clock_stamp, compose, log_tag

__all__ = [
    "close_log_file",
    "log_file_on",
    "log_file_sink",
    "log_record",
    "log_to_file",
    "open_log_file",
]


class _FileLog:
    """The open log file, or none. Writes are serialised and never raise."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self._handle: TextIO | None = None
        self._lock = threading.Lock()

    @property
    def on(self) -> bool:
        return self._handle is not None

    def start(self, folder: str) -> Path | None:
        """Open a new log file in ``folder``, once. Returns its path, or None when it cannot."""
        if self._handle is not None:
            return self.path
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = Path(folder).expanduser() / f"fish-voice-{stamp}.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Private: it holds what was said and what the model answered.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._handle = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            sys.stderr.write(f"fish-voice: cannot write the log file {path}: {exc}\n")
            return None
        self.path = path
        atexit.register(self.close)
        return path

    def write(self, tag: str, body: str, stamp: str | None = None) -> None:
        stamp = stamp or clock_stamp()
        with self._lock, contextlib.suppress(OSError, ValueError):
            if self._handle is not None:
                # Laid out like a terminal line; a file is not a terminal, so it is plain.
                line = compose(stamp, tag, tag, body, self._handle).rstrip()
                self._handle.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            handle, self._handle = self._handle, None
            self.path = None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


_FILE: Final = _FileLog()


def open_log_file(folder: str) -> Path | None:
    """Start writing this run's log file in ``folder``, once.

    Parameters
    ----------
    folder : str
        Where the file goes. ``~`` is expanded and the folder is made.

    Returns
    -------
    Path or None
        The file, which is readable only by its owner. The same file on a second call.
        None, after a line on stderr, when it cannot be written.
    """
    return _FILE.start(folder)


def close_log_file() -> None:
    """Close the run's log file and forget it, so the next ``open_log_file`` starts a new one.

    Nothing happens when no file is open. The file is also closed when the process exits.
    """
    _FILE.close()


def log_file_on() -> bool:
    """Return whether a log file is being written."""
    return _FILE.on


def log_file_path() -> Path | None:
    """Return the log file in use, or None when logging to a file is off."""
    return _FILE.path if _FILE.on else None


def log_to_file(tag: str, text: str) -> None:
    """Write one line to the log file, if there is one. For the conversation itself.

    Parameters
    ----------
    tag : str
        The tag column, such as ``you``.
    text : str
        The rest of the line.
    """
    if _FILE.on:
        _FILE.write(tag, text)


def log_record(message: str, args: tuple[Any, ...], fields: dict[str, Any]) -> None:
    """Write a debug or trace line that the screen is not showing.

    Parameters
    ----------
    message : str
        A loguru-style message with ``{}`` fields.
    args : tuple
        Its positional fields.
    fields : dict
        Its named fields.
    """
    try:
        text = message.format(*args, **fields) if args or fields else message
    except (IndexError, KeyError, ValueError):
        text = message
    tag, body = log_tag(text, "DEBUG")
    _FILE.write(tag, body)


def log_file_sink(message: Any) -> None:
    """Write one loguru record to the log file, as a loguru sink.

    Parameters
    ----------
    message : loguru.Message
        What loguru hands a sink.
    """
    record = message.record
    tag, body = log_tag(str(record["message"]), record["level"].name)
    _FILE.write(tag, body, clock_stamp(record["time"]))
