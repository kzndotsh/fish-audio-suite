"""The full-screen terminal app for ``fish-voice``, built on Textual.

It follows a session through its events, shows it as one ``SessionView``, and drives it
with typed lines, mute and interrupt. It owns no audio or network code: the session runs
as a worker and reports through ``events``, so everything here happens on the app's own
thread.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.types import CSSPathType
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
from fish_audio_suite_voice.session_view import SessionView, meter_bar, reduce_view
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "VoiceApp",
]

_METER_WIDTH = 24
_QUIT_GRACE_S = 3.0
_NO_TURNS = "no turns yet"


def _cue_content(text: str) -> Content:
    """Return ``text`` as content with each ``[cue]`` styled and nothing read as markup."""
    parts: list[tuple[str, str] | str] = [
        (piece, "italic $text-accent") if is_cue else piece for piece, is_cue in split_cues(text)
    ]
    return Content.assemble(*parts)


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
        self._reply_widget: Static | None = None
        self._reply_text = ""
        self._ended = False

    @property
    def view(self) -> SessionView:
        """The latest snapshot of the session."""
        return self._view

    def compose(self) -> ComposeResult:
        """Lay out the conversation, the side panels, the input line and the footer."""
        yield Header()
        with Horizontal(id="body"):
            conversation = VerticalScroll(id="conversation")
            conversation.border_title = "conversation"
            yield conversation
            with Vertical(id="side"):
                with VerticalGroup(id="status-box") as status_box:
                    status_box.border_title = "state"
                    yield Static("idle", id="status")
                with VerticalGroup(id="mic-box") as mic_box:
                    mic_box.border_title = "mic"
                    yield Static("", id="meter")
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
            self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _side_effects(self, event: Event) -> None:
        match event:
            case Heard(text=text):
                self._reply_widget = None
                self._reply_text = ""
                self._add_message("you", Content.assemble(("you ▸ ", "bold $text-primary"), text))
            case ReplyToken(text=text):
                self._reply_text += text
                self._show_reply()
            case ReplyEnd(text=text):
                if text or self._reply_widget is not None:
                    self._reply_text = text
                    self._show_reply()
            case BargedIn():
                self._add_note("(interrupted)")
            case Notice(text=text):
                self._add_note(text)
            case LogLine(level=level, tag=tag, text=text):
                self.query_one("#log", Log).write_line(f"{level[:1]} {tag:<6} {text}")
            case Bye(code=code):
                self._on_bye(code)
            case _:
                return

    def _add_message(self, role: str, content: Content) -> Static:
        message = Static(content, classes=f"message {role}")
        self.query_one("#conversation", VerticalScroll).mount(message)
        return message

    def _add_note(self, text: str) -> None:
        self._add_message("note", Content.assemble((text, "dim")))

    def _show_reply(self) -> None:
        content = Content.assemble(("llm ▸ ", "bold $text-accent"), _cue_content(self._reply_text))
        if self._reply_widget is None:
            self._reply_widget = self._add_message("llm", content)
        else:
            self._reply_widget.update(content)

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
        self.query_one("#meter", Static).update(
            Content(meter_bar(view.mic_fraction, view.mic_need_fraction, _METER_WIDTH))
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
            self._add_note(f"the session stopped: {event.worker.error!r}")
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
