"""The full-screen terminal app for ``fish-voice``, built on Textual.

It follows a session through its events, shows it as one ``SessionView``, and drives it
with typed lines, mute, interrupt and changes to the chat. It owns no audio or network code:
the session runs as a worker and reports through ``session.events``, so everything here
happens on the app's own thread.

The chat screen holds only what a person needs while talking: the chat's title, the
conversation, one state line, the composer and the keys. Everything technical is on the
Diagnostics screen (F12, ``cli.diagnostics``). The conversation pane is ``cli.history_pane``
and the dialogs are ``cli.chat_screens``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult, RenderResult, SystemCommand
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.content import Content
from textual.events import Resize
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.types import CSSPathType
from textual.widget import Widget
from textual.widgets import Footer, Input, Static
from textual.worker import Worker, WorkerState

from fish_audio_suite_voice.cli.chat_screens import ChatPicker, KeysScreen, printable
from fish_audio_suite_voice.cli.diagnostics import LOG_LINES, DiagnosticsScreen
from fish_audio_suite_voice.cli.history_pane import Conversation
from fish_audio_suite_voice.cli.session_view import (
    Ballistics,
    SessionView,
    SpeechScale,
    bar_dots,
    reduce_view,
)
from fish_audio_suite_voice.session.chat_store import chat_title
from fish_audio_suite_voice.session.commands import ChatCommand, NewChat
from fish_audio_suite_voice.session.events import (
    DEFAULT_KEYS,
    EVENTS,
    BargedIn,
    Bye,
    ChatShown,
    Event,
    EventBus,
    EventQueue,
    ExitCode,
    Heard,
    Interim,
    LogLine,
    MicLevel,
    Notice,
    OutputLevel,
    ReplyEnd,
    ReplyToken,
    SessionAction,
    SessionState,
    available_actions,
)
from fish_audio_suite_voice.session.inputs import LiveInput
from fish_audio_suite_voice.session.signals import DuplexSession

__all__ = [
    "Composer",
    "VoiceApp",
    "Waveform",
]

_WAVE_HISTORY = 480  # bars kept (two to a character), so a wide strip is filled
# One bar: the mic's own rate, a level every three 30 ms frames. Every other source keeps to it,
# so the scroll is even.
_BAR_S = 0.09
_AUDIO_GAP_S = 0.25  # no audio level for this long inside a reply: a pause, so fill it in
_BLINK_S = 0.45  # the listening cursor's half cycle
_SPEAKER_NEW_REPLY_S = 1.5  # a pause this long means the next audio is a new reply


def _speaker_scale() -> SpeechScale:
    # TTS is steady and loud, and its pauses are digital silence, so it starts higher than a mic.
    return SpeechScale(start_speech_db=-20.0, start_noise_db=-70.0)


_LOUD_STEPS = (0.34, 0.67)  # where a bar passes from dim to medium to bright
_QUIT_GRACE_S = 3.0
_MIN_SIZE = (50, 14)  # under this nothing fits, so the app says so and keeps quit working
_APP_NAME = "fish-voice"
_CHATS_KEY = "ctrl+o"  # a control key, so it works while typing; Ctrl+O reaches the app in raw mode
_DIAGNOSTICS_KEY = "f12"


class Waveform(Widget):
    """The mic or the speaker as one line of level bars in dots, scrolling: newest on the right.

    Both are drawn the same way, so they look alike: each level is scaled to that source's own
    recent peak and eased down a little, with no caps. Colour says what a bar is (green for
    speech the mic hears, grey for quiet, the accent colour for the reply being played) and its
    height and brightness say how loud it is. While listening a cursor blinks at the right
    edge; while speaking the strip keeps scrolling through the reply's pauses.
    """

    DEFAULT_CSS: ClassVar[str] = """
    Waveform {
        height: 1;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 - Textual's own name
        super().__init__(id=id)
        self._bars: deque[tuple[float, str]] = deque(maxlen=_WAVE_HISTORY)
        self._reset_meters()
        self._last_push: float | None = None
        self._last_speaker: float | None = None
        self._mood = "still"
        self._timer: Timer | None = None
        self._cursor_on = True
        self._last_bar = 0.0
        self._audio_seen = False
        self._filling = False

    def _reset_meters(self) -> None:
        self._mic_scale = SpeechScale()
        self._speaker_scale = _speaker_scale()
        self._mic_ease = Ballistics()
        self._speaker_ease = Ballistics()

    @property
    def mood(self) -> str:
        """What the strip is doing besides drawing levels: listening, speaking or still."""
        return self._mood

    def _add(self, level: float, kind: str) -> None:
        self._bars.append((level, kind))
        self._last_bar = time.monotonic()
        self.refresh()

    def _elapsed(self, now: float) -> float:
        elapsed = 0.0 if self._last_push is None else min(0.5, max(0.0, now - self._last_push))
        self._last_push = now
        return elapsed

    def push(self, rms: float, need: float, *, now: float | None = None) -> None:
        """Add the newest bar of mic level.

        Parameters
        ----------
        rms : float
            The mic's RMS level, on the int16 scale.
        need : float
            The RMS that counts as speech. A bar at or above it is drawn as heard.
        now : float, optional
            When the level was taken, in ``time.monotonic`` seconds. Now by default.
        """
        now = time.monotonic() if now is None else now
        elapsed = self._elapsed(now)
        level = self._mic_ease.update(self._mic_scale.update(rms, now), elapsed)[0]
        self._add(level, "heard" if rms >= need > 0.0 else "quiet")

    def push_speaker(self, rms: float, *, now: float | None = None) -> None:
        """Add the newest bar of the reply being played.

        Parameters
        ----------
        rms : float
            RMS of the latest slice of audio sent to the speaker.
        now : float, optional
            When it was taken, in ``time.monotonic`` seconds. Now by default.

        Notes
        -----
        A reply that starts after a pause starts the scale afresh.
        """
        now = time.monotonic() if now is None else now
        gap = None if self._last_speaker is None else now - self._last_speaker
        if gap is None or gap > _SPEAKER_NEW_REPLY_S:
            self._speaker_scale = _speaker_scale()
            self._speaker_ease = Ballistics()
        self._last_speaker = now
        self._audio_seen = True
        self._filling = False
        elapsed = 0.0 if gap is None else min(gap, _SPEAKER_NEW_REPLY_S)
        level = self._speaker_ease.update(self._speaker_scale.update(rms, now), elapsed)[0]
        self._add(level, "speaker")

    def set_mood(self, mood: str) -> None:
        """Choose what the strip does besides drawing levels.

        Parameters
        ----------
        mood : str
            ``"listening"`` (a blinking cursor at the right edge), ``"speaking"`` (the
            reply's bars, with a flat line scrolling until the first audio level arrives and
            through its pauses) or ``"still"``. Every source adds bars at the same rate, so
            the scroll keeps one speed through a turn. Textual's reduced-motion setting turns
            the blinking and the filling off.
        """
        if mood == self._mood:
            return
        self._mood = mood
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        # The scales keep what they have learnt about the room and the voice across turns.
        self._mic_ease = Ballistics()
        self._speaker_ease = Ballistics()
        self._last_push = None
        self._cursor_on = True
        moving = self.app.animation_level != "none"
        if mood == "listening" and moving:
            self._timer = self.set_interval(_BLINK_S, self._blink)
        elif mood == "speaking" and moving:
            self._audio_seen = False
            self._timer = self.set_interval(_BAR_S / 4, self._fill)
        self.refresh()

    def _fill(self) -> None:
        # The reply's first audio takes a moment to arrive, and a reply pauses between
        # sentences. A flat bar at the usual rate in those gaps keeps the strip scrolling at
        # the same speed instead of stopping and then catching up. Once audio levels are
        # coming it waits longer, so it never cuts in between two real ones.
        patient = self._audio_seen and not self._filling
        if time.monotonic() - self._last_bar > (_AUDIO_GAP_S if patient else _BAR_S * 1.05):
            self._add(0.0, "speaker")
            self._filling = True  # in a gap now, so carry on at the usual rate until audio is back

    def _blink(self) -> None:
        self._cursor_on = not self._cursor_on
        self.refresh()

    def render(self) -> RenderResult:
        """Draw the newest bars that fit, the oldest cut off on the left."""
        width = self.size.width
        want = width * 2  # two bars to a character
        # The last character is always kept for the cursor, so the bars do not shift sideways
        # when the strip starts or stops listening.
        room = max(0, want - 2)
        bars = list(self._bars)[-room:] if room else []
        bars = [(0.0, "quiet")] * (room - len(bars)) + bars
        blinking = self._mood == "listening" and self._cursor_on
        bars += [(0.67, "cursor") if blinking else (0.0, "quiet")] * (want - len(bars))
        dots = bar_dots([level for level, _ in bars])
        tones = [_bar_tone(bars[2 * i], bars[2 * i + 1]) for i in range(width)]
        return Content.assemble(*zip(dots, tones, strict=True))


# What a bar is, as a colour, and how strong it is at each of three loudness steps. The steps
# are close together because a Braille dot is small: a faded one is hard to see, so the height
# of the bar says how loud it is and the strength only adds to it. test_tui checks the
# contrast against every built-in theme.
_BAR_COLOURS = {
    "cursor": ("$text-accent", (100, 100, 100)),
    "speaker": ("$text-accent", (85, 93, 100)),
    "heard": ("$text-success", (85, 93, 100)),
    "quiet": ("$foreground", (58, 68, 78)),
}


def _bar_tone(left: tuple[float, str], right: tuple[float, str]) -> str:
    """Return the style of the character that holds two bars, from what and how loud they are."""
    kinds = {left[1], right[1]}
    kind = next(k for k in ("cursor", "heard", "speaker", "quiet") if k in kinds)
    colour, steps = _BAR_COLOURS[kind]
    level = max(left[0], right[0])
    step = 0 if level < _LOUD_STEPS[0] else 1 if level < _LOUD_STEPS[1] else 2
    return f"{colour} {steps[step]}%"


class Composer(Input):
    """The line you type into: Enter sends, Up and Down bring back what you typed before.

    The lines are kept for this session only, newest first. A line you were typing when you
    went back through them is kept, and comes back when you go past the newest one.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "submit", "Send"),
        Binding("up", "recall(-1)", "Previous", key_display="↑"),
        Binding("down", "recall(1)", "Next", show=False),
        Binding("tab", "messages", "Messages"),
    ]

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 - Textual's own name
        super().__init__(placeholder="Type a message, or just talk", id=id)
        self._sent: list[str] = []
        self._at: int | None = None  # where in ``_sent`` the line came from, None for a new one
        self._draft = ""

    async def action_submit(self) -> None:
        """Remember the line, then send it."""
        text = self.value
        if text.strip() and (not self._sent or self._sent[-1] != text):
            self._sent.append(text)
        self._at, self._draft = None, ""
        await super().action_submit()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Offer Previous only when there is a line to bring back, and Next only while looking.

        Parameters
        ----------
        action : str
            The action's name.
        parameters : tuple
            Its parameters.

        Returns
        -------
        bool or None
            False hides the key.
        """
        if action == "recall":
            if parameters == (-1,):
                return bool(self._sent) and self._at != 0
            return self._at is not None
        return super().check_action(action, parameters)

    def action_recall(self, step: int) -> None:
        """Bring back the line typed before (-1) or after (1) the one shown.

        Parameters
        ----------
        step : int
            -1 for an older line, 1 for a newer one.
        """
        if self._at is None:
            if step > 0 or not self._sent:
                return
            self._draft, self._at = self.value, len(self._sent)
        at = min(len(self._sent), max(0, self._at + step))
        self._at = None if at == len(self._sent) else at
        self.value = self._draft if self._at is None else self._sent[at]
        self.cursor_position = len(self.value)
        self.refresh_bindings()

    def action_messages(self) -> None:
        """Move to the messages."""
        self.screen.focus_next(Conversation)


# The state line: a glyph and a word for every state, never colour alone, and a plain hint.
_LISTENING_HINT = "just talk, or type below"


class VoiceApp(App[int]):
    """The chat of one ``fish-voice`` session, with its diagnostics one key away.

    Parameters
    ----------
    runner : Callable
        Runs the session and returns its exit code. It is started as a worker once the
        screen is up and should report through ``bus``.
    live : LiveInput
        The turn source the session uses. Typed lines, mute, interrupt and changes to the
        chat go to it.
    session : DuplexSession
        The running session, so quit can stop it cleanly.
    bus : EventBus, optional
        Where the session reports. The session-wide ``EVENTS`` by default.
    info : str, optional
        What the session runs on, such as the model and voice. Diagnostics shows it.
    chat_dir : Path or None, optional
        The folder of saved chats the chat list shows. None lists none, so only a new chat
        can be started.
    shape : Callable or None, optional
        Cleans a reply the way it is cleaned before it is spoken (``CueStyle.shape``).
        Diagnostics shows the last reply that way, as Fish read it. None shows the model's
        text as written.
    editor : str, optional
        The command Ctrl+E hands a line being edited to, with any arguments
        (``config.settings.editor_from_env``). ``vi`` by default.
    """

    TITLE: str | None = _APP_NAME
    CSS_PATH: ClassVar[CSSPathType | None] = "tui.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(DEFAULT_KEYS[SessionAction.INTERRUPT], "stop", "Stop", priority=True),
        Binding(DEFAULT_KEYS[SessionAction.MUTE], "mute", "Mic off", priority=True),
        Binding(DEFAULT_KEYS[SessionAction.UNMUTE], "unmute", "Mic on", priority=True),
        Binding(_CHATS_KEY, "chats", "Chats", priority=True),
        Binding(_DIAGNOSTICS_KEY, "diagnostics", "Diagnostics", show=False, priority=True),
        Binding(
            DEFAULT_KEYS[SessionAction.QUIT], "quit_session", "Quit", show=False, priority=True
        ),
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
        chat_dir: Path | None = None,
        shape: Callable[[str], str] | None = None,
        editor: str = "vi",
    ) -> None:
        super().__init__()
        self._editor = editor
        self._runner = runner
        self._live = live
        self._session = session
        self._info = info
        self._chat_dir = chat_dir
        self._shape = shape
        self._chat_file: Path | None = None
        self._chat_title = ""
        # While a line is open in the editor the mic is off; this is whether it was before.
        self._editing = False
        self._muted_before_edit = False
        self._view = SessionView()
        self._log_lines: deque[str] = deque(maxlen=LOG_LINES)
        self._wake_pending = threading.Event()
        # Subscribed now, so nothing the session says before the screen is up is lost.
        self._queue = EventQueue(bus, wake=self._wake)

    @property
    def view(self) -> SessionView:
        """The latest snapshot of the session."""
        return self._view

    def compose(self) -> ComposeResult:
        """Lay out the title, the conversation, the state line, the composer and the keys."""
        with Horizontal(id="header"):
            yield Static(_APP_NAME, id="title")
            yield Static("? keys", id="keys-hint")
        yield Conversation(
            id="conversation", send=self._live.send, hold_mic=self._hold_mic, editor=self._editor
        )
        yield Static(
            f"Terminal too small. Make it at least {_MIN_SIZE[0]}x{_MIN_SIZE[1]}, or Ctrl+Q to quit.",
            id="too-small",
        )
        with Horizontal(id="state-line"):
            yield Static(id="state")
            yield Waveform(id="meter")
            yield Static(id="hint")
        yield Composer(id="line")
        yield Footer()

    def on_resize(self, event: Resize) -> None:
        """Say so when the terminal is too small for the conversation."""
        width, height = event.size
        self._main.set_class(width < _MIN_SIZE[0] or height < _MIN_SIZE[1], "tiny")

    def on_mount(self) -> None:
        """Show the first view, start the session as a worker and focus the composer."""
        self._render_view()
        self.run_worker(self._run(), exclusive=True, exit_on_error=False)
        self.query_one(Composer).focus()

    async def _run(self) -> int:
        return await self._runner()

    @property
    def _main(self) -> Screen[object]:
        # The screen under any dialog: a query from the app looks only at the top screen.
        return self.screen_stack[0]

    @property
    def _diagnostics(self) -> DiagnosticsScreen | None:
        return next((s for s in self.screen_stack if isinstance(s, DiagnosticsScreen)), None)

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
        stack = self.screen_stack
        conversation = next(iter(stack[0].query(Conversation)), None) if stack else None
        if not self.is_running or conversation is None or not conversation.is_attached:
            return  # the screen is going: a wake that arrived while the app was closing
        for event in events:
            self._view = reduce_view(self._view, event)
            self._side_effects(event, conversation)
        if events:
            self._render_view()
            conversation.follow()

    def _side_effects(self, event: Event, conversation: Conversation) -> None:
        speaking = self._view.state is SessionState.SPEAKING
        if isinstance(event, MicLevel) and not speaking:
            self._main.query_one(Waveform).push(0.0 if self._live.muted else event.rms, event.need)
        elif isinstance(event, OutputLevel) and speaking:
            self._main.query_one(Waveform).push_speaker(event.rms)
        match event:
            case MicLevel(rms=rms, need=need, source="listen") if rms >= need > 0.0:
                conversation.show_hearing()
            case Interim(text=text):
                conversation.show_words(text)
            case Heard(text=text):
                conversation.add_user(text)
            case ReplyToken() | ReplyEnd():
                # The view has already folded the tokens into the reply so far.
                conversation.show_reply(self._view.reply)
            case BargedIn():
                conversation.cut_reply()
            case Notice(text=text):
                conversation.add_note(text)
            case ChatShown(file=file, turns=turns):
                self._chat_file = file
                self._chat_title = chat_title(turns)
                conversation.show_record(event)
            case LogLine(level=level, tag=tag, text=text):
                line = printable(f"{level[:1]} {tag:<6} {text}")
                self._log_lines.append(line)
                if (diagnostics := self._diagnostics) is not None:
                    diagnostics.add_log(line)
            case Bye(code=code):
                self._on_bye(code)
            case _:
                return

    def _on_bye(self, code: ExitCode) -> None:
        if code == ExitCode.OK:
            self.exit(ExitCode.OK)

    @property
    def _ended(self) -> bool:
        return self._view.exit_code is not None

    def _exit_code(self) -> ExitCode:
        return ExitCode.OK if self._view.exit_code is None else self._view.exit_code

    @property
    def _title_text(self) -> str:
        return printable(self._chat_title) or _APP_NAME

    def _state_line(self) -> tuple[str, str, str, str]:
        """Return the glyph, the word, the hint and the style class for the state now."""
        state, muted = self._view.state, self._live.muted
        if self._ended:
            return "✕", "Session ended", f"exit code {self._exit_code()} · Ctrl+Q to quit", "ended"
        if state is SessionState.SPEAKING:
            hint = "press Esc to stop" if muted else "talk to interrupt, or press Esc"
            return "▶", "Speaking", hint, "speaking"
        if state is SessionState.THINKING:
            return "◌", "Thinking…", "", "thinking"
        if self._editing:
            return "○", "Mic off while you edit", "it comes back when you're done", "muted"
        if muted:
            return "○", "Mic off", "F2 turns it back on", "muted"
        if state is SessionState.LISTENING:
            return "●", "Listening", _LISTENING_HINT, "listening"
        return "○", "Ready", _LISTENING_HINT, "idle"

    def _render_view(self) -> None:
        main = self._main
        main.query_one("#title", Static).update(Content(self._title_text))
        glyph, word, hint, tone = self._state_line()
        state = main.query_one("#state", Static)
        state.update(Content.assemble(f"{glyph} ", (word, "bold")))
        state.set_classes(tone)
        main.query_one("#hint", Static).update(Content(hint))
        meter = main.query_one(Waveform)
        mood = tone if tone in ("listening", "speaking") else "still"
        meter.set_mood(mood)
        meter.display = mood != "still"
        if (diagnostics := self._diagnostics) is not None:
            diagnostics.show(self._view, self._sent_to_fish)
        self.refresh_bindings()

    @property
    def _sent_to_fish(self) -> str:
        reply = self._view.reply
        return self._shape(reply) if self._shape and reply else reply

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Show each app key only while it does something.

        Parameters
        ----------
        action : str
            The action's name.
        parameters : tuple
            Its parameters.

        Returns
        -------
        bool or None
            False hides the key and passes it on, so Escape closes a dialog or goes back to
            typing when there is nothing to stop.
        """
        del parameters
        actions = available_actions(self._view.state, muted=self._live.muted)
        match action:
            case "stop":
                # Over a dialog or Diagnostics, Escape belongs to that screen.
                on_top = self.screen is self._main
                return on_top and not self._ended and SessionAction.INTERRUPT in actions
            case "mute":
                return not self._editing and SessionAction.MUTE in actions
            case "unmute":
                return not self._editing and SessionAction.UNMUTE in actions
            case "chats":
                return not isinstance(self.screen, ModalScreen)
            case _:
                return True

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        """List the app's own commands in the command palette, before Textual's.

        Parameters
        ----------
        screen : Screen
            The screen the palette was opened from.

        Yields
        ------
        SystemCommand
            Try again (while the newest reply can be tried again), New chat, Chats…, Mic on
            or off, Stop speaking (while there is a reply to stop), Diagnostics, Keys and
            Quit, then Textual's own but its Keys and Quit: the app has its own keys list,
            and its Quit would skip ending the session.
        """
        conversation = self._main.query_one(Conversation)
        if conversation.can_try_again:
            yield SystemCommand(
                "Try again", "A new reply to your last message", conversation.try_again
            )
        yield SystemCommand("New chat", "Start an empty chat", self.action_new_chat)
        yield SystemCommand("Chats…", "Open a saved chat, or start a new one", self.action_chats)
        if self._editing:
            pass  # the mic stays off until the editor closes
        elif self._live.muted:
            yield SystemCommand("Mic on", "Listen again", self.action_unmute)
        else:
            yield SystemCommand("Mic off", "Stop listening until you turn it on", self.action_mute)
        if self.check_action("stop", ()):
            yield SystemCommand("Stop speaking", "Stop the reply", self.action_stop)
        yield SystemCommand(
            "Diagnostics", "Timings, the log, and what was sent to Fish", self.action_diagnostics
        )
        yield SystemCommand("Keys", "Every key, in plain words", self.action_keys)
        yield SystemCommand("Quit", "End the session and leave", self.action_quit_session)
        for command in super().get_system_commands(screen):
            if command.title not in ("Keys", "Quit"):
                yield command

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Send the typed line to the session and clear the composer."""
        self._live.submit(event.value)
        event.input.value = ""

    def action_stop(self) -> None:
        """Stop the reply that is playing."""
        self._live.stop_reply()

    def action_new_chat(self) -> None:
        """Start an empty chat."""
        self._live.send(NewChat())

    def action_chats(self) -> None:
        """Open the list of saved chats, to go on with one or start a new one."""

        def chosen(command: ChatCommand | None) -> None:
            if command is not None:
                self._live.send(command)

        self.push_screen(ChatPicker(self._chat_dir, self._chat_file), chosen)

    def action_keys(self) -> None:
        """List every key in plain words."""
        if not isinstance(self.screen, KeysScreen):
            self.push_screen(KeysScreen())

    def action_diagnostics(self) -> None:
        """Open the Diagnostics screen, or go back to the chat from it."""
        if isinstance(self.screen, DiagnosticsScreen):
            self.screen.dismiss(None)
            return
        if self._diagnostics is not None:
            return  # under a dialog: let the dialog close first
        self.push_screen(
            DiagnosticsScreen(
                title=self._title_text,
                info=self._info,
                log=self._log_lines,
                view=self._view,
                sent=self._sent_to_fish,
            )
        )

    def _hold_mic(self, held: bool) -> None:
        """Turn the mic off while a line is edited, and back to what it was after."""
        if held == self._editing:
            return
        self._editing = held
        if held:
            self._muted_before_edit = self._live.muted
            self._live.mute()
        elif not self._muted_before_edit:
            self._live.unmute()
        self._render_view()

    def action_mute(self) -> None:
        """Turn the mic off."""
        self._live.mute()
        self._render_view()

    def action_unmute(self) -> None:
        """Turn the mic back on."""
        self._live.unmute()
        self._render_view()

    def action_quit_session(self) -> None:
        """Ask the session to end, then leave when it has, or after a short grace period."""
        if self._ended:
            self.exit(self._exit_code())
            return
        self._session.request_quit()
        # The session answers with Bye and the app exits. If it does not, leave anyway.
        self.set_timer(_QUIT_GRACE_S, lambda: self.exit(self._exit_code()))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Show a session that stopped with an error, or leave when it ended normally."""
        if event.state is WorkerState.ERROR:
            self._view = replace(self._view, exit_code=ExitCode.CRASH)
            self._main.query_one(Conversation).add_note(
                f"the session stopped: {event.worker.error!r}"
            )
            self._render_view()
        elif event.state is WorkerState.SUCCESS and not self._ended:
            self.exit(int(event.worker.result or ExitCode.OK))

    def on_unmount(self) -> None:
        """Stop collecting events."""
        self._queue.close()

    @property
    def state(self) -> SessionState:
        """What the session is doing now."""
        return self._view.state
