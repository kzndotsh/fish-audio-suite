"""The plain terminal display: print a session from its events.

The conversation goes to stdout, so piping it keeps the transcript. The reply streams onto
one line, which stays open until the reply ends or a log line has to be written; the display
owns that line, and logging asks it to close the line through ``logutil.debug.on_terminal_line``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import TracebackType
from typing import Any, Final, Self

from fish_audio_suite_voice.logutil.debug import debug, debug_enabled, on_terminal_line, trace
from fish_audio_suite_voice.logutil.format import clock_stamp, compose, paint
from fish_audio_suite_voice.session.events import (
    Bye,
    Event,
    EventBus,
    ExitCode,
    Heard,
    Listening,
    Notice,
    ReplyEnd,
    ReplyToken,
    TurnEnded,
)
from fish_audio_suite_voice.session.latency import LatencySnapshot, session_summary, turn_summary

__all__ = [
    "ConsoleSink",
    "console_print",
]

_ROLE_COLORS: Final = {"you": "96", "llm": "95"}


def console_print(*args: object, **kwargs: Any) -> None:
    """Print a status line. A closed stdout must not drop the spoken reply.

    Parameters
    ----------
    *args : object
        What ``print`` takes.
    **kwargs : Any
        What ``print`` takes, such as ``flush=True``.
    """
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        return


def _stdout_tty() -> bool:
    return bool(sys.stdout.isatty())


class ConsoleSink:
    """The plain terminal display: one subscriber that prints what the events say.

    Parameters
    ----------
    bus : EventBus
        The bus to print from.

    Notes
    -----
    A display that draws its own screen leaves this out. With debug on, the
    reply is held and printed whole on one line after the log lines around it,
    and the turn summary goes to the log instead of stdout. When the session
    ends, one line gives the typical (p50) and slow (p90) wait across its turns.
    Use it as a context manager, or call ``close`` to stop printing.
    """

    def __init__(self, bus: EventBus) -> None:
        # The one record of a reply line left open on stdout: it is closed before
        # anything else is written to the terminal.
        self._line_open = False
        # Whether this reply's "llm ▸" label was printed. A warning closes the line
        # mid-reply, and the reply goes on, unlabelled, on the next.
        self._reply_started = False
        self._turns: list[LatencySnapshot] = []
        self._unsubscribe: Callable[[], None] = bus.subscribe(self._on_event)
        self._stop_closing = on_terminal_line(self.end_reply_line)

    def __enter__(self) -> Self:
        """Return the display, which is already printing."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop printing."""
        self.close()

    def close(self) -> None:
        """Stop printing."""
        self._unsubscribe()
        self._stop_closing()

    def end_reply_line(self) -> None:
        """Close the reply line with a newline, if one is open."""
        if not self._line_open:
            return
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except BrokenPipeError:
            pass
        self._line_open = False

    def _write_reply(self, text: str) -> None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except BrokenPipeError:
            return
        self._line_open = True

    def _conversation(self, role: str, text: str) -> None:
        # On a terminal with debug on, the line carries the same time columns and
        # role tag as the log lines on stderr. Written and flushed in order, it stays
        # in sequence with them. Piped, or with debug off, it is a plain line.
        self.end_reply_line()
        label = f"{role} ▸"
        if not (debug_enabled() and _stdout_tty()):
            console_print(f"{label} {text}", flush=True)
            return
        body = paint(f"1;{_ROLE_COLORS.get(role, '97')}", text, sys.stdout)
        console_print(compose(clock_stamp(), role, label, body, sys.stdout), flush=True)

    def _status(self, line: str) -> None:
        # A status line goes below an open reply line, never onto its end.
        self.end_reply_line()
        console_print(line, flush=True)

    def _print_session(self) -> None:
        summary = session_summary(self._turns)
        if not summary:
            return
        if debug_enabled():
            debug("session.summary {}", summary)
        else:
            console_print(f"session {summary}", flush=True)

    def _on_event(self, event: Event) -> None:
        match event:
            case Listening():
                if not debug_enabled():
                    self._status("listening…")
            case Heard(text=text):
                self._conversation("you", text)
            case ReplyToken(text=text):
                if not debug_enabled():
                    if not self._reply_started:
                        self._write_reply("llm ▸ ")
                        self._reply_started = True
                    self._write_reply(text)
            case ReplyEnd(text=text):
                self.end_reply_line()
                self._reply_started = False
                if text and debug_enabled():
                    self._conversation("llm", text)
            case Notice(text=text):
                self._status(f"  {text}")
            case TurnEnded(snapshot=snapshot):
                self._turns.append(snapshot)
                summary = turn_summary(snapshot)
                if debug_enabled():
                    debug("turn.summary {}", summary)
                    trace("turn.timing {}", snapshot.log_line())
                else:
                    self._status(f"  ↳ {summary}" if summary else "")
            case Bye(code=code):
                # A session that stopped mid-reply never ended the reply.
                self.end_reply_line()
                self._print_session()
                if code == ExitCode.OK:
                    console_print("\nbye")
            case _:
                return
