"""Where the next turn comes from: the mic, or text typed into a display."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Protocol

from fish_audio_suite_voice.barge import StopFlag
from fish_audio_suite_voice.debug import mark_turn
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.events import EVENTS, Heard
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "LiveInput",
    "TurnSource",
]

_POLL_S = 0.1


class TurnSource(Protocol):
    """Gives the duplex loop its next turn, however it was made."""

    async def next_turn(self, ctx: DuplexContext, last_user: str) -> HeardLine:
        """Wait for the next turn.

        Parameters
        ----------
        ctx : DuplexContext
            The session.
        last_user : str
            The previous line, used to drop an echo of it.

        Returns
        -------
        HeardLine
            What the loop should do next.
        """
        ...


class _Either:
    def __init__(self, first: StopFlag, second: StopFlag) -> None:
        self._first = first
        self._second = second

    def is_set(self) -> bool:
        return self._first.is_set() or self._second.is_set()


class LiveInput:
    """The mic, plus typed lines and a mute switch, for a display to drive.

    Notes
    -----
    ``submit``, ``mute`` and ``unmute`` are safe to call from any thread. A
    typed line stops the mic at once and is answered like a spoken one. While
    muted, the mic stays closed between turns and the session waits for a typed
    line, ``unmute`` or quit. Mute does not change the barge-in watch during a
    reply. A line typed while a reply is playing stops that reply, like speaking
    over it, unless ``submit`` is told not to.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._typed: deque[str] = deque()
        self._muted = False
        # Set to stop a recording in progress: a typed line, or a change of mute.
        self._interrupt = threading.Event()
        # Seen on the first turn, so a typed line can stop the reply in flight.
        self._session: DuplexSession | None = None
        # Set whenever anything changes, so a muted wait notices at once.
        self._changed = threading.Event()

    @property
    def muted(self) -> bool:
        """Whether the mic is held closed between turns."""
        return self._muted

    def submit(self, text: str, *, interrupt: bool = True) -> None:
        """Queue a line as if the user had said it.

        Parameters
        ----------
        text : str
            The line. Blank text is ignored.
        interrupt : bool, optional
            Stop the reply that is playing, like a barge-in. False lets it finish and
            takes the line on the next turn.
        """
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._typed.append(text)
        self._interrupt.set()
        self._changed.set()
        session = self._session
        if interrupt and session is not None:
            session.turn.fire()

    def mute(self) -> None:
        """Close the mic between turns, and stop a recording in progress."""
        self._muted = True
        self._interrupt.set()
        self._changed.set()

    def unmute(self) -> None:
        """Open the mic again."""
        self._muted = False
        self._changed.set()

    def toggle_mute(self) -> bool:
        """Flip the mute switch, for a single key binding.

        Returns
        -------
        bool
            True when the mic is now muted.
        """
        with self._lock:
            muted = not self._muted
            self._muted = muted
        self._changed.set()
        if muted:
            self._interrupt.set()
        return muted

    def _take_typed(self) -> str | None:
        with self._lock:
            return self._typed.popleft() if self._typed else None

    def _has_typed(self) -> bool:
        with self._lock:
            return bool(self._typed)

    async def next_turn(self, ctx: DuplexContext, last_user: str) -> HeardLine:
        """Wait for a typed line or an utterance, whichever comes first.

        Parameters
        ----------
        ctx : DuplexContext
            The session.
        last_user : str
            The previous line, used to drop an echo of it.

        Returns
        -------
        HeardLine
            The typed line or the heard one, ``bye`` on quit.
        """
        self._session = ctx.session
        quit_requested = ctx.session.quit_requested
        while True:
            typed = self._take_typed()
            if typed is not None:
                return self._typed_line(ctx, typed)
            if quit_requested.is_set():
                return HeardLine("bye")
            if self._muted:
                self._changed.clear()
                if not self._muted or self._has_typed():
                    continue
                await asyncio.to_thread(self._changed.wait, _POLL_S)
                continue
            self._interrupt.clear()
            if self._has_typed() or self._muted:
                continue
            heard = await hear_line(ctx, last_user, stop=_Either(quit_requested, self._interrupt))
            if self._interrupt.is_set() and not quit_requested.is_set():
                # A typed line or a mute cut the recording short, so what the
                # mic caught is dropped.
                continue
            return heard

    def _typed_line(self, ctx: DuplexContext, text: str) -> HeardLine:
        ctx.barge_prefix = b""
        mark_turn()
        EVENTS.emit(Heard(text, 0.0))
        return HeardLine("line", text=text, started=time.perf_counter())
