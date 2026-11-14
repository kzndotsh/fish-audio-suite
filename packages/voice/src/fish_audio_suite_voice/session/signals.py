"""Per-session duplex cancel state. No module-level flags.

Ctrl+C has to stop the TTS turn (``session.turn``) as well as the mic
(``quit_requested``). Cancelling only the mic leaves Fish speaking until the
reply ends. ``TurnSignals.fire`` is safe to call from another thread while the
turn's loop is running; see its notes.
"""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from types import FrameType

__all__ = [
    "DuplexSession",
    "TurnSignals",
]


class TurnSignals:
    """Cancel handles for the reply in flight.

    Notes
    -----
    ``fire`` may run on a signal-handler thread. The asyncio event is set
    through ``call_soon_threadsafe`` on the loop that bound it, because
    ``asyncio.Event.set`` is not thread-safe. After ``end``, every turn bound
    later starts cancelled, so a quit that lands before a turn binds its handles
    still stops it.
    """

    def __init__(self) -> None:
        self.cancel: threading.Event | None = None
        self.llm_cancel: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Re-entrant: ``request_quit`` may run in a signal handler on a thread that holds it.
        self._lock = threading.RLock()
        self._ended = False

    def bind(
        self,
        cancel: threading.Event,
        llm_cancel: asyncio.Event,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Remember the handles for this turn, and set them at once after ``end``.

        Parameters
        ----------
        cancel : threading.Event
            Stops the TTS thread and the barge watcher.
        llm_cancel : asyncio.Event
            Stops the LLM stream.
        loop : asyncio.AbstractEventLoop or None, optional
            Loop that owns ``llm_cancel``. Defaults to the running loop, or
            None when called outside one.
        """
        if loop is None:
            with suppress(RuntimeError):
                loop = asyncio.get_running_loop()
        with self._lock:
            self.cancel = cancel
            self.llm_cancel = llm_cancel
            self._loop = loop
            ended = self._ended
        if ended:
            self.fire()

    def fire(self) -> None:
        """Cancel the bound turn from any thread."""
        with self._lock:
            cancel, event, loop = self.cancel, self.llm_cancel, self._loop
        if cancel is not None:
            cancel.set()
        if event is None:
            return
        if loop is not None and not loop.is_closed():
            # Also right for a stopped loop: the call is queued and runs when
            # it resumes, instead of touching the event from this thread.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(event.set)
                return
        # No loop, or a closed one with nothing left to wake. The flag still
        # sets; waking a waiter on a closed loop raises, so ignore that.
        with suppress(RuntimeError):
            event.set()

    def end(self) -> None:
        """Cancel the bound turn and every turn bound after it, for quit."""
        with self._lock:
            self._ended = True
        self.fire()

    def clear(self) -> None:
        """Forget the bound handles."""
        with self._lock:
            self.cancel = None
            self.llm_cancel = None
            self._loop = None


@dataclass
class DuplexSession:
    """Cancel flags for one duplex run.

    Attributes
    ----------
    quit_requested : threading.Event
        Set on quit. Sticky for the session.
    turn : TurnSignals
        Handles for the reply in flight.
    """

    quit_requested: threading.Event = field(default_factory=threading.Event)
    turn: TurnSignals = field(default_factory=TurnSignals)
    _on_quit: list[Callable[[], None]] = field(default_factory=list, init=False, repr=False)

    def request_quit(self) -> None:
        """Stop listening and cancel the LLM and TTS in flight. Safe from a signal handler."""
        self.quit_requested.set()
        self.turn.end()
        for wake in tuple(self._on_quit):
            wake()

    def on_quit(self, wake: Callable[[], None]) -> None:
        """Call ``wake`` when quit is requested, so a wait that is not on the quit flag ends.

        Parameters
        ----------
        wake : Callable
            Called on the thread that requests quit, which can be a signal handler, so
            it may only call thread-safe things.
        """
        self._on_quit.append(wake)

    def install_sigint(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        before: Callable[[], None] | None = None,
    ) -> None:
        """Route SIGINT to ``request_quit``. A second SIGINT uses the default handler.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The duplex loop.
        before : Callable or None, optional
            Runs first, for example to close the open reply line.

        Notes
        -----
        POSIX uses ``loop.add_signal_handler``. Platforms without it fall back
        to ``signal.signal`` and hop onto the loop with ``call_soon_threadsafe``.
        """

        def fire() -> None:
            if before is not None:
                before()
            self.request_quit()
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        def from_signal(signum: int, frame: FrameType | None) -> None:
            del signum, frame
            loop.call_soon_threadsafe(fire)

        try:
            loop.add_signal_handler(signal.SIGINT, fire)
        except (NotImplementedError, RuntimeError, ValueError):
            signal.signal(signal.SIGINT, from_signal)
