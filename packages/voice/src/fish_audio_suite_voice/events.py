"""What happens in a voice session, as events a display can follow.

The duplex loop reports each step here as well as printing it, so another
display, such as a full-screen terminal app, can follow a session without
reading its console output.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.console import console_print
from fish_audio_suite_voice.debug import warn

__all__ = [
    "EVENTS",
    "Bye",
    "Event",
    "EventBus",
    "Heard",
    "Listening",
    "Notice",
    "ReplyEnd",
    "ReplyToken",
    "TurnEnded",
    "notice",
]


@dataclass(frozen=True, slots=True)
class Listening:
    """The mic is open and waiting for the user."""


@dataclass(frozen=True, slots=True)
class Heard:
    """Fish ASR returned a line that will be answered.

    Attributes
    ----------
    text : str
        The transcript.
    asr_ms : float
        How long the ASR request took.
    """

    text: str
    asr_ms: float


@dataclass(frozen=True, slots=True)
class ReplyToken:
    """One piece of the model's reply, as it arrives.

    Attributes
    ----------
    text : str
        The piece. Joined in order, the pieces make the reply.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ReplyEnd:
    """The model finished, or was cut off.

    Attributes
    ----------
    text : str
        The whole reply as written, which may be empty.
    """

    text: str


@dataclass(frozen=True, slots=True)
class Notice:
    """A short status line about the session, such as a retry or a silent reply.

    Attributes
    ----------
    text : str
        The message, without padding.
    """

    text: str


@dataclass(frozen=True, slots=True)
class TurnEnded:
    """A turn is over, after playback finished or was cut off.

    Attributes
    ----------
    snapshot : LatencySnapshot
        The timings of the turn.
    """

    snapshot: LatencySnapshot


@dataclass(frozen=True, slots=True)
class Bye:
    """The session is ending."""


type Event = Listening | Heard | ReplyToken | ReplyEnd | Notice | TurnEnded | Bye


class EventBus:
    """Hands each event to every subscriber, in the order they subscribed.

    Notes
    -----
    A subscriber runs on the thread that emitted the event, which can be the
    event loop or a worker thread. It has to return quickly and has to hand
    work to its own loop or thread if it needs to. A subscriber that raises is
    reported and does not stop the session or the other subscribers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Start sending events to ``callback``.

        Parameters
        ----------
        callback : Callable
            Called with each event.

        Returns
        -------
        Callable
            Call it to stop. Calling it twice is harmless.
        """
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def emit(self, event: Event) -> None:
        """Send ``event`` to every subscriber.

        Parameters
        ----------
        event : Event
            What happened.
        """
        with self._lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:  # noqa: BLE001 - a display must never break the session
                warn(f"[events] a subscriber failed on {type(event).__name__}: {e!r}")


EVENTS = EventBus()


def notice(text: str) -> None:
    """Print a status line and report it as a ``Notice``.

    Parameters
    ----------
    text : str
        The line as printed, including any leading padding or brackets.
    """
    console_print(text, flush=True)
    EVENTS.emit(Notice(text.strip()))
