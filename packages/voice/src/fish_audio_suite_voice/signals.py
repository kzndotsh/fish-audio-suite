"""Per-session duplex cancel state. No module-level flags."""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from types import FrameType

from fish_audio_suite_voice.aec import EchoCanceller

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
    ``asyncio.Event.set`` is not thread-safe.
    """

    def __init__(self) -> None:
        self.cancel: threading.Event | None = None
        self.llm_cancel: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(
        self,
        cancel: threading.Event,
        llm_cancel: asyncio.Event,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Remember the handles for this turn.

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
        self.cancel = cancel
        self.llm_cancel = llm_cancel
        if loop is None:
            with suppress(RuntimeError):
                loop = asyncio.get_running_loop()
        self._loop = loop

    def fire(self) -> None:
        """Cancel the bound turn from any thread."""
        if self.cancel is not None:
            self.cancel.set()
        event = self.llm_cancel
        if event is None:
            return
        loop = self._loop
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

    def clear(self) -> None:
        """Forget the bound handles."""
        self.cancel = None
        self.llm_cancel = None
        self._loop = None


@dataclass
class DuplexSession:
    """Cancel flags and echo state for one duplex run.

    Attributes
    ----------
    stop : threading.Event
        Set on quit. Sticky for the session.
    turn : TurnSignals
        Handles for the reply in flight.
    aec : EchoCanceller
        Far-end tap and processor shared by the sink and the barge gate.
    """

    aec: EchoCanceller = field(default_factory=EchoCanceller)
    stop: threading.Event = field(default_factory=threading.Event)
    turn: TurnSignals = field(default_factory=TurnSignals)

    def request_quit(self) -> None:
        """Stop mic listen and cancel in-flight LLM and TTS. Safe from a signal handler."""
        self.stop.set()
        self.turn.fire()

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
