"""The full-screen terminal app for ``fish-voice``, built on Textual.

It follows a session through its events, shows it as one ``SessionView``, and drives it
with typed lines, mute and interrupt. It owns no audio or network code: the session runs
as a worker and reports through ``events``, so everything here happens on the app's own
thread.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.types import CSSPathType
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Log, Static
from textual.worker import Worker, WorkerState

from fish_audio_suite_kit import split_cues
from fish_audio_suite_voice.events import (
    DEFAULT_KEYS,
    EVENTS,
    BargedIn,
    Bye,
    Event,
    EventBus,
    EventQueue,
    Heard,
    LogLine,
    Notice,
    ReplyEnd,
    ReplyToken,
    SessionAction,
    SessionState,
    available_actions,
)
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.reply import turn_summary
from fish_audio_suite_voice.session_view import SessionView, reduce_view, wave_column
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "VoiceApp",
]

_WAVE_HALF = 2  # rows above, and below, the middle line
_WAVE_HISTORY = 240  # bars kept, so a wide terminal is filled
_WAVE_DECAY = 0.82  # how much of the last bar a quieter one keeps, so bars fall smoothly
_QUIT_GRACE_S = 3.0
_NO_TURNS = "no turns yet"


def _cue_content(text: str) -> Content:
    """Return ``text`` as content with each ``[cue]`` styled and nothing read as markup."""
    parts: list[tuple[str, str] | str] = [
        (piece, "italic $text-accent") if is_cue else piece for piece, is_cue in split_cues(text)
    ]
    return Content.assemble(*parts)


class Waveform(Widget):
    """The mic level as a mirrored, scrolling waveform: newest on the right, bars grow from the middle.

    A bar is bright while the mic hears speech (above the threshold) and dim otherwise, and
    a dropped level falls over a few bars instead of snapping.
    """

    DEFAULT_CSS: ClassVar[str] = """
    Waveform {
        height: 4;
        background: $surface;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 - Textual's own name
        super().__init__(id=id)
        self._bars: deque[tuple[float, bool]] = deque(maxlen=_WAVE_HISTORY)

    def push(self, level: float, threshold: float) -> None:
        """Add the newest bar.

        Parameters
        ----------
        level : float
            The mic level, from 0 to 1.
        threshold : float
            Where speech starts to count, from 0 to 1.
        """
        previous = self._bars[-1][0] if self._bars else 0.0
        self._bars.append((max(level, previous * _WAVE_DECAY), level >= threshold > 0.0))
        self.refresh()

    def render(self) -> RenderResult:
        """Draw the newest bars that fit, the oldest cut off on the left."""
        width = self.size.width
        bars = list(self._bars)[-width:] if width else []
        bars = [(0.0, False)] * (width - len(bars)) + bars
        columns = [(wave_column(level, _WAVE_HALF), heard) for level, heard in bars]
        lines: list[Content] = []
        for row in range(_WAVE_HALF * 2):
            cells: list[tuple[str, str] | str] = []
            for column, heard in columns:
                char, reverse = column[row]
                color = "$text-success" if heard else "$text-muted"
                cells.append((char, f"$surface on {color}" if reverse else color))
            lines.append(Content.assemble(*cells))
        return Content("\n").join(lines)


class Conversation(VerticalScroll):
    """The conversation: what you said, the reply as it streams in, and short notes.

    The app tells it what happened and it keeps the rest to itself, including which
    widget is the reply being written, so the app holds no widget state for it.
    """

    BORDER_TITLE: ClassVar[str] = "conversation"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reply: Static | None = None

    def add_user(self, text: str) -> None:
        """Add a line the user said or typed. The reply that follows starts fresh."""
        self._reply = None
        self._add("you", Content.assemble(("you ▸ ", "bold $text-primary"), text))

    def add_note(self, text: str) -> None:
        """Add a dim status line, such as a retry or an interruption."""
        self._add("note", Content.assemble((text, "dim")))

    def show_reply(self, text: str) -> None:
        """Show the reply so far, adding its line on the first call and updating it after."""
        if self._reply is None and not text:
            return
        content = Content.assemble(("llm ▸ ", "bold $text-accent"), _cue_content(text))
        if self._reply is None:
            self._reply = self._add("llm", content)
        else:
            self._reply.update(content)

    def _add(self, role: str, content: Content) -> Static:
        message = Static(content, classes=f"message {role}")
        self.mount(message)
        return message


class VoiceApp(App[int]):
    """Conversation, status, mic meter, timings and log of one ``fish-voice`` session.

    Parameters
    ----------
    runner : Callable
        Runs the session and returns its exit code. It is started as a worker once the
        screen is up and should report through ``bus``.
    live : LiveInput
        The turn source the session uses. Typed lines, mute and interrupt go to it.
    session : DuplexSession
        The running session, so quit can stop it cleanly.
    bus : EventBus, optional
        Where the session reports. The session-wide ``EVENTS`` by default.
    info : str, optional
        What to show next to the title, such as the model and voice.
    """

    TITLE: str | None = "fish-voice"
    CSS_PATH: ClassVar[CSSPathType | None] = "tui.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(DEFAULT_KEYS[SessionAction.INTERRUPT], "interrupt", "Stop", priority=True),
        Binding(DEFAULT_KEYS[SessionAction.MUTE], "toggle_mute", "Mute", priority=True),
        Binding(DEFAULT_KEYS[SessionAction.QUIT], "quit_session", "Quit", priority=True),
    ]

    class Wake(Message):
        """Events are waiting in the queue."""

    def __init__(
        self,
        runner: Callable[[], Awaitable[int]],
        *,
        live: LiveInput,
        session: DuplexSession,
        bus: EventBus = EVENTS,
        info: str = "",
    ) -> None:
        super().__init__()
        self._runner = runner
        self._live = live
        self._session = session
        self._info = info
        self._view = SessionView()
        self._exit_code = 0
        self._wake_pending = threading.Event()
        # Subscribed now, so nothing the session says before the screen is up is lost.
        self._queue = EventQueue(bus, wake=self._wake)
        self._ended = False

    @property
    def view(self) -> SessionView:
        """The latest snapshot of the session."""
        return self._view

    def compose(self) -> ComposeResult:
        """Lay out the conversation, the side panels, the input line and the footer."""
        yield Header()
        with Horizontal(id="body"):
            yield Conversation(id="conversation")
            with Vertical(id="side"):
                with VerticalGroup(id="status-box") as status_box:
                    status_box.border_title = "state"
                    yield Static("idle", id="status")
                with VerticalGroup(id="mic-box") as mic_box:
                    mic_box.border_title = "mic"
                    wave = Waveform(id="meter")
                    wave.tooltip = (
                        "The mic level, newest on the right. Bright means it hears speech."
                    )
                    yield wave
                with VerticalGroup(id="latency-box") as latency_box:
                    latency_box.border_title = "last turn"
                    yield Static(_NO_TURNS, id="latency")
                log = Log(id="log", max_lines=1000)
                log.border_title = "log"
                yield log
        yield Input(placeholder="type a line and press Enter", id="line")
        yield Footer()

    def on_mount(self) -> None:
        """Show the first view, start the session as a worker and focus the input line."""
        self.sub_title = self._info
        self._render_view()
        self.run_worker(self._run(), exclusive=True, exit_on_error=False)
        self.query_one(Input).focus()

    async def _run(self) -> int:
        return await self._runner()

    def _wake(self) -> None:
        # Called from whichever thread emitted the event. One message at a time is
        # enough: the handler drains everything that arrived.
        if not self._wake_pending.is_set():
            self._wake_pending.set()
            self.post_message(self.Wake())

    def on_voice_app_wake(self) -> None:
        """Take every waiting event from the queue and show it."""
        self._wake_pending.clear()
        self._apply(self._queue.drain())

    def _apply(self, events: list[Event]) -> None:
        for event in events:
            self._view = reduce_view(self._view, event)
            self._side_effects(event)
        if events:
            self._render_view()
            self.query_one(Conversation).scroll_end(animate=False)

    def _side_effects(self, event: Event) -> None:
        conversation = self.query_one(Conversation)
        match event:
            case Heard(text=text):
                conversation.add_user(text)
            case ReplyToken() | ReplyEnd():
                # The view has already folded the tokens into the reply so far.
                conversation.show_reply(self._view.reply)
            case BargedIn():
                conversation.add_note("(interrupted)")
            case Notice(text=text):
                conversation.add_note(text)
            case LogLine(level=level, tag=tag, text=text):
                self.query_one("#log", Log).write_line(f"{level[:1]} {tag:<6} {text}")
            case Bye(code=code):
                self._on_bye(code)
            case _:
                return

    def _on_bye(self, code: int) -> None:
        self._ended = True
        self._exit_code = code
        if code == 0:
            self.exit(0)

    def _render_view(self) -> None:
        view = self._view
        status = self.query_one("#status", Static)
        label = view.state.value
        if self._live.muted:
            label += " · muted"
        if self._ended:
            label = f"ended (code {self._exit_code})"
        status.update(Content(label))
        status.set_classes("ended" if self._ended else f"state-{view.state.value}")
        if not self._ended:
            self.query_one(Waveform).push(
                0.0 if self._live.muted else view.mic_fraction, view.mic_need_fraction
            )
        latency = _NO_TURNS
        if view.last_turn is not None:
            latency = turn_summary(view.last_turn).strip().removeprefix("↳ ") or _NO_TURNS
        self.query_one("#latency", Static).update(Content(latency))
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Show Stop only while there is something to stop."""
        del parameters
        if action == "interrupt":
            actions = available_actions(self._view.state, muted=self._live.muted)
            return True if SessionAction.INTERRUPT in actions else None
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Send the typed line to the session and clear the input."""
        self._live.submit(event.value)
        event.input.value = ""

    def action_interrupt(self) -> None:
        """Stop the reply that is playing."""
        self._live.stop_reply()

    def action_toggle_mute(self) -> None:
        """Mute or unmute the mic."""
        self._live.toggle_mute()
        self._render_view()

    def action_quit_session(self) -> None:
        """Ask the session to end, then leave when it has, or after a short grace period."""
        if self._ended:
            self.exit(self._exit_code)
            return
        self._session.request_quit()
        # The session answers with Bye and the app exits. If it does not, leave anyway.
        self.set_timer(_QUIT_GRACE_S, lambda: self.exit(self._exit_code))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Show a session that stopped with an error, or leave when it ended normally."""
        if event.state is WorkerState.ERROR:
            self._ended = True
            self._exit_code = 1
            self.query_one(Conversation).add_note(f"the session stopped: {event.worker.error!r}")
            self._render_view()
        elif event.state is WorkerState.SUCCESS and not self._ended:
            self.exit(int(event.worker.result or 0))

    def on_unmount(self) -> None:
        """Stop collecting events."""
        self._queue.close()

    @property
    def state(self) -> SessionState:
        """What the session is doing now."""
        return self._view.state
