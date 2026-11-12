"""Where the next turn comes from: the mic, or text typed into a display."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Protocol

from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.debug import mark_turn
from fish_audio_suite_voice.duplex_state import TurnInput
from fish_audio_suite_voice.events import EVENTS, Heard
from fish_audio_suite_voice.heard import HeardLine
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "Hear",
    "LiveInput",
    "TurnSource",
]

type Hear = Callable[[StopFlag], Awaitable[HeardLine]]
"""Listen for one turn with the mic, stopping early when the flag is set."""


class TurnSource(Protocol):
    """Gives the duplex loop its next turn, however it was made."""

    @property
    def muted(self) -> bool:
        """Whether the user has muted the mic, with its device closed."""
        ...

    async def next_turn(self, turn: TurnInput, hear: Hear) -> HeardLine:
        """Wait for the next turn.

        Parameters
        ----------
        turn : TurnInput
            The session's quit flag and reply handles, its mic, what the last barge-in
            carried over, and the mute switch.
        hear : Hear
            Listens with the mic, as the loop would without a source.

        Returns
        -------
        HeardLine
            What the loop should do next.
        """
        ...


class LiveInput:
    """The mic, plus typed lines and a mute switch, for a display to drive.

    Notes
    -----
    ``submit``, ``mute``, ``unmute`` and ``toggle_mute`` are safe to call from any
    thread. A typed line stops the mic at once and is answered like a spoken one.
    Muting closes the mic device at once, and the session then waits for a typed
    line, ``unmute`` or quit. There is no barge-in watch while muted. A reply that
    starts muted is not watched, and muting during a reply ends the watch without
    interrupting the reply. Unmuting does not restart a watch that ended, so it
    applies from the next reply. Typing a line or pressing Escape still
    interrupts. A line typed while a reply is playing stops that reply, like
    speaking over it, unless ``submit`` is told not to.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._typed: deque[str] = deque()
        self._muted = False
        # Set to stop a recording in progress: a typed line, or a change of mute.
        self._interrupt = threading.Event()
        # Seen on the first turn, so a typed line can stop the reply in flight and
        # muting can close the device at once.
        self._session: DuplexSession | None = None
        self._mic_release: Callable[[], None] | None = None
        # Set on the session's loop whenever anything changes, quit included, so a muted
        # wait notices at once.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._changed: asyncio.Event | None = None

    @property
    def muted(self) -> bool:
        """Whether the mic is muted, with its device closed."""
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
        self._note_change()
        if interrupt:
            self.stop_reply()

    def mute(self) -> None:
        """Close the mic device now, and stop a recording in progress."""
        self._set_muted(True)

    def unmute(self) -> None:
        """Let the mic open again on its next read."""
        self._set_muted(False)

    def toggle_mute(self) -> bool:
        """Flip the mute switch, for a single key binding.

        Returns
        -------
        bool
            True when the mic is now muted.
        """
        with self._lock:
            muted = not self._muted
        self._set_muted(muted)
        return muted

    def _set_muted(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted
        if muted:
            self._interrupt.set()
            # The session's input stays open between readers, so muting closes it
            # here rather than waiting for the reply in flight to end.
            release = self._mic_release
            if release is not None:
                release()
        self._note_change()

    def stop_reply(self) -> None:
        """Stop the reply that is being written or played, like speaking over it.

        Notes
        -----
        Does nothing before the first turn or between turns.
        """
        session = self._session
        if session is not None:
            session.turn.fire()

    def _note_change(self) -> None:
        # Any thread: the wait is woken on the session's own loop.
        loop, changed = self._loop, self._changed
        if loop is not None and changed is not None:
            with contextlib.suppress(RuntimeError):  # the session's loop has closed
                loop.call_soon_threadsafe(changed.set)

    def _take_typed(self) -> str | None:
        with self._lock:
            return self._typed.popleft() if self._typed else None

    def _has_typed(self) -> bool:
        with self._lock:
            return bool(self._typed)

    def _join(self, turn: TurnInput) -> asyncio.Event:
        """Meet the session on its first turn: its loop, its mic and its quit."""
        loop = asyncio.get_running_loop()
        if self._changed is None or self._loop is not loop:
            self._loop = loop
            self._changed = asyncio.Event()
        if self._session is not turn.session:
            turn.session.on_quit(self._note_change)
        self._session = turn.session
        self._mic_release = turn.mic.release
        return self._changed

    async def next_turn(self, turn: TurnInput, hear: Hear) -> HeardLine:
        """Wait for a typed line or an utterance, whichever comes first.

        Parameters
        ----------
        turn : TurnInput
            The session's mic side.
        hear : Hear
            Listens with the mic.

        Returns
        -------
        HeardLine
            The typed line or the heard one, ``bye`` on quit.
        """
        changed = self._join(turn)
        quit_requested = turn.session.quit_requested
        while True:
            typed = self._take_typed()
            if typed is not None:
                return self._typed_line(turn, typed)
            if quit_requested.is_set():
                return HeardLine("bye")
            if self._muted:
                turn.mic.release()  # muted before the first turn: nothing had it yet
                changed.clear()
                if not self._muted or self._has_typed() or quit_requested.is_set():
                    continue
                await changed.wait()
                continue
            self._interrupt.clear()
            if self._has_typed() or self._muted:
                continue
            heard = await hear(self._interrupt)
            if self._interrupt.is_set() and not quit_requested.is_set():
                # A typed line or a mute cut the recording short, so what the
                # mic caught is dropped.
                continue
            return heard

    def _typed_line(self, turn: TurnInput, text: str) -> HeardLine:
        turn.carry.take_clip()  # the typed line replaces what the mic caught
        mark_turn()
        EVENTS.emit(Heard(text, 0.0))
        return HeardLine("line", text=text, started=time.perf_counter())
