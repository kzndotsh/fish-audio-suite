"""What happens in a voice session, as events a display can follow.

The duplex loop reports each step here as well as printing it, so another
display, such as a full-screen terminal app, can follow a session without
reading its console output.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Final, Literal

from loguru import logger

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.debug import log_tag, warn

__all__ = [
    "DEFAULT_KEYS",
    "EVENTS",
    "BargedIn",
    "Bye",
    "Event",
    "EventBus",
    "EventQueue",
    "Heard",
    "Listening",
    "LogLine",
    "MicLevel",
    "Notice",
    "OutputLevel",
    "ReplyEnd",
    "ReplyToken",
    "SessionAction",
    "SessionState",
    "Speaking",
    "StateChanged",
    "StateTracker",
    "TurnEnded",
    "available_actions",
    "forward_logs",
    "next_state",
    "notice",
]

# Events that may be dropped when a consumer falls behind. Everything else is
# a fact about the conversation and is never dropped.
_DROPPABLE: Final = ("MicLevel", "OutputLevel", "LogLine")


def _now() -> float:
    return monotonic()


@dataclass(frozen=True, slots=True)
class Listening:
    """The mic is open and waiting for the user.

    Attributes
    ----------
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class Heard:
    """A line will be answered: Fish ASR returned it, or the user typed it.

    Attributes
    ----------
    text : str
        The transcript, or the typed text.
    asr_ms : float
        How long the ASR request took. 0 for typed text.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    text: str
    asr_ms: float
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class ReplyToken:
    """One piece of the model's reply, as it arrives.

    Attributes
    ----------
    text : str
        The piece. Joined in order, the pieces make the reply.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    text: str
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class ReplyEnd:
    """The model finished, or was cut off.

    Attributes
    ----------
    text : str
        The whole reply as written, which may be empty.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    text: str
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class Speaking:
    """The first audio of a reply reached the speaker.

    Attributes
    ----------
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class BargedIn:
    """The user spoke over the reply and the gate stopped it.

    Attributes
    ----------
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class MicLevel:
    """How loud the mic is, a few times a second while it is open.

    Attributes
    ----------
    rms : float
        RMS of the latest frame, after echo cancellation.
    need : float
        The level a frame has to reach to count as speech.
    source : {"listen", "barge"}
        Whether the mic is waiting for the user or watching for an interruption.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    rms: float
    need: float
    source: Literal["listen", "barge"]
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class OutputLevel:
    """How loud the reply being played is, a few times a second while it plays.

    Attributes
    ----------
    rms : float
        RMS of the latest slice of audio sent to the speaker.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.

    Notes
    -----
    It is taken as each slice goes to the device, so it leads what is heard by the
    device's buffer, usually a few tens of milliseconds.
    """

    rms: float
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class LogLine:
    """One log line, for a display that cannot share the terminal with stderr.

    Attributes
    ----------
    level : str
        ``DEBUG``, ``INFO``, ``WARNING`` or ``ERROR``.
    tag : str
        The part of the message before its first dot or space, such as ``tts``.
    text : str
        The message.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    level: str
    tag: str
    text: str
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class Notice:
    """A short status line about the session, such as a retry or a silent reply.

    Attributes
    ----------
    text : str
        The message, without padding.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    text: str
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class TurnEnded:
    """A turn is over, after playback finished or was cut off.

    Attributes
    ----------
    snapshot : LatencySnapshot
        The timings of the turn.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    snapshot: LatencySnapshot
    at: float = field(default_factory=_now, compare=False, kw_only=True)


@dataclass(frozen=True, slots=True)
class Bye:
    """The session is over, sent once however it ended.

    Attributes
    ----------
    code : int
        The exit code: 0 for a normal quit, 2 for a fatal error such as a bad key or
        missing PortAudio, 1 when the session raised an unexpected exception.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    code: int = 0
    at: float = field(default_factory=_now, compare=False, kw_only=True)


class SessionState(StrEnum):
    """What the session is doing, as a person would say it."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class SessionAction(StrEnum):
    """Something a person can ask a session to do."""

    SEND = "send"
    INTERRUPT = "interrupt"
    MUTE = "mute"
    UNMUTE = "unmute"
    QUIT = "quit"


# Keys a terminal always passes on: Enter, Escape, function keys and Ctrl with a letter
# that is not also another key. Not Ctrl+C (copy in a UI, interrupt in a shell), not
# Ctrl+I or Ctrl+M (a terminal sends them as Tab and Enter), and nothing with Cmd,
# Option or the Windows key, which usually never arrive. Mute and unmute share a key.
DEFAULT_KEYS: Final[Mapping[SessionAction, str]] = {
    SessionAction.SEND: "enter",
    SessionAction.INTERRUPT: "escape",
    SessionAction.MUTE: "f2",
    SessionAction.UNMUTE: "f2",
    SessionAction.QUIT: "ctrl+q",
}


@dataclass(frozen=True, slots=True)
class StateChanged:
    """The session moved to a new state. Made by a ``StateTracker``.

    Attributes
    ----------
    state : SessionState
        The new state.
    at : float
        ``time.monotonic()`` when the event was made. Not part of equality.
    """

    state: SessionState
    at: float = field(default_factory=_now, compare=False, kw_only=True)


type Event = (
    Listening
    | Heard
    | ReplyToken
    | ReplyEnd
    | Speaking
    | BargedIn
    | MicLevel
    | OutputLevel
    | LogLine
    | Notice
    | TurnEnded
    | Bye
    | StateChanged
)


def next_state(state: SessionState, event: Event) -> SessionState:
    """Return the state after ``event``.

    Parameters
    ----------
    state : SessionState
        The state before.
    event : Event
        What just happened.

    Returns
    -------
    SessionState
        The new state. Events that do not change it return ``state``.
    """
    match event:
        case Listening():
            return SessionState.LISTENING
        case Heard():
            return SessionState.THINKING
        case Speaking():
            return SessionState.SPEAKING
        case BargedIn():
            return SessionState.LISTENING
        case TurnEnded() | Bye():
            return SessionState.IDLE
        case _:
            return state


def available_actions(state: SessionState, *, muted: bool) -> frozenset[SessionAction]:
    """Return the actions that make sense now, so a display shows only those keys.

    Parameters
    ----------
    state : SessionState
        What the session is doing.
    muted : bool
        Whether the mic is held closed.

    Returns
    -------
    frozenset of SessionAction
        Sending a line and quitting always apply. Interrupting only applies while the
        model is thinking or the reply is playing, and muting offers the opposite of
        the current setting.
    """
    actions = {SessionAction.SEND, SessionAction.QUIT}
    if state in (SessionState.THINKING, SessionState.SPEAKING):
        actions.add(SessionAction.INTERRUPT)
    actions.add(SessionAction.UNMUTE if muted else SessionAction.MUTE)
    return frozenset(actions)


class _Subscription:
    """One call to ``subscribe``, so the same callback can subscribe twice."""

    __slots__ = ("callback",)

    def __init__(self, callback: Callable[[Event], None]) -> None:
        self.callback = callback


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
        self._subscribers: list[_Subscription] = []
        # A failure is reported through the log, which a subscriber may itself
        # be listening to. This stops that from going round in circles.
        self._reporting = threading.local()

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
        subscription = _Subscription(callback)
        with self._lock:
            self._subscribers.append(subscription)

        def unsubscribe() -> None:
            with self._lock:
                if subscription in self._subscribers:
                    self._subscribers.remove(subscription)

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
        for subscription in subscribers:
            try:
                subscription.callback(event)
            except Exception as e:  # noqa: BLE001 - a display must never break the session
                if getattr(self._reporting, "on", False):
                    continue
                self._reporting.on = True
                try:
                    warn(f"[events] a subscriber failed on {type(event).__name__}: {e!r}")
                finally:
                    self._reporting.on = False


class StateTracker:
    """Follows the events of a bus and announces each change of ``SessionState``.

    Parameters
    ----------
    bus : EventBus
        The bus to follow and to announce on.

    Attributes
    ----------
    state : SessionState
        The current state.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self.state: SessionState = SessionState.IDLE
        self._unsubscribe = bus.subscribe(self._on_event)

    def _on_event(self, event: Event) -> None:
        if isinstance(event, StateChanged):
            return
        with self._lock:
            new = next_state(self.state, event)
            if new is self.state:
                return
            self.state = new
        self._bus.emit(StateChanged(new))

    def close(self) -> None:
        """Stop following the bus."""
        self._unsubscribe()


class EventQueue:
    """Collects the events of a bus for a consumer that runs at its own pace.

    Parameters
    ----------
    bus : EventBus
        The bus to collect from.
    max_droppable : int, optional
        How many ``MicLevel``, ``OutputLevel`` and ``LogLine`` events to keep while the consumer
        is behind. The oldest go first. Every other event is always kept.
    wake : Callable or None, optional
        Called after each event is added, from the thread that emitted it, so it
        may only call thread-safe things. Use it to wake an event loop, for example
        ``loop.call_soon_threadsafe``, or to post a message to a UI.

    Attributes
    ----------
    dropped : int
        How many events were dropped so far.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        max_droppable: int = 256,
        wake: Callable[[], None] | None = None,
    ) -> None:
        self._items: deque[Event] = deque()
        self._droppable = 0
        self._max_droppable = max(1, max_droppable)
        self._wake = wake
        self._cond = threading.Condition()
        self._closed = False
        self.dropped: int = 0
        self._unsubscribe = bus.subscribe(self._put)

    def _put(self, event: Event) -> None:
        with self._cond:
            if self._closed:
                return
            self._items.append(event)
            if type(event).__name__ in _DROPPABLE:
                self._droppable += 1
                if self._droppable > self._max_droppable:
                    self._drop_oldest_droppable()
            self._cond.notify()
        if self._wake is not None:
            self._wake()

    def _drop_oldest_droppable(self) -> None:
        for index, item in enumerate(self._items):
            if type(item).__name__ in _DROPPABLE:
                del self._items[index]
                self._droppable -= 1
                self.dropped += 1
                return

    def _took(self, event: Event) -> None:
        if type(event).__name__ in _DROPPABLE:
            self._droppable -= 1

    def get(self, timeout: float | None = None) -> Event | None:
        """Take the oldest event, waiting for one if there is none.

        Parameters
        ----------
        timeout : float or None, optional
            Seconds to wait. None waits until an event arrives or the queue is closed.

        Returns
        -------
        Event or None
            The event, or None when the wait timed out or the queue was closed.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._items or self._closed, timeout)
            if not self._items:
                return None
            event = self._items.popleft()
            self._took(event)
            return event

    def drain(self) -> list[Event]:
        """Take every event waiting, without blocking.

        Returns
        -------
        list of Event
            The events, oldest first. Empty when there are none.
        """
        with self._cond:
            events = list(self._items)
            self._items.clear()
            self._droppable = 0
            return events

    def close(self) -> None:
        """Stop collecting and wake anyone waiting in ``get``."""
        self._unsubscribe()
        with self._cond:
            self._closed = True
            self._cond.notify_all()


EVENTS = EventBus()


def notice(text: str) -> None:
    """Report a status line as a ``Notice``.

    Parameters
    ----------
    text : str
        The line, such as ``[llm 429, retrying in 4s]``. The console display
        indents it by two spaces.
    """
    EVENTS.emit(Notice(text.strip()))


def forward_logs(bus: EventBus | None = None, *, level: str = "DEBUG") -> Callable[[], None]:
    """Send each log line to ``bus`` as a ``LogLine``.

    Parameters
    ----------
    bus : EventBus or None, optional
        Where to send them. The session's ``EVENTS`` when omitted.
    level : str, optional
        The lowest log level to forward.

    Returns
    -------
    Callable
        Call it to stop forwarding.

    Notes
    -----
    Use it with ``configure_voice_logging(..., to_stderr=False)`` when a display
    owns the terminal, so log lines reach it instead of corrupting the screen.
    Call it after ``configure_voice_logging``, which removes every other sink.
    """
    target = bus if bus is not None else EVENTS

    def sink(message: Any) -> None:
        record = message.record
        name = record["level"].name
        tag, text = log_tag(str(record["message"]), name)
        target.emit(LogLine(name, tag, text))

    handler = logger.add(sink, level=level, format="{message}")

    def stop() -> None:
        try:
            logger.remove(handler)
        except ValueError:
            return

    return stop
