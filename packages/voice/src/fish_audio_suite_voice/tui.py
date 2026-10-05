"""The full-screen terminal app for ``fish-voice``, built on Textual.

It follows a session through its events, shows it as one ``SessionView``, and drives it
with typed lines, mute and interrupt. It owns no audio or network code: the session runs
as a worker and reports through ``events``, so everything here happens on the app's own
thread.
"""

from __future__ import annotations

import math
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalGroup, VerticalScroll
from textual.content import Content
from textual.events import Resize
from textual.message import Message
from textual.timer import Timer
from textual.types import CSSPathType
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Log, Static
from textual.worker import Worker, WorkerState

from fish_audio_suite_kit import LatencySnapshot, split_cues
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
    MicLevel,
    Notice,
    OutputLevel,
    ReplyEnd,
    ReplyToken,
    SessionAction,
    SessionState,
    available_actions,
)
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.reply import turn_summary
from fish_audio_suite_voice.session_view import (
    Ballistics,
    SessionView,
    SpeechScale,
    processing_level,
    reduce_view,
    split_cells,
    turn_timings,
    wave_dots,
)
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "VoiceApp",
]

_WAVE_ROWS = 4  # lines tall; each line is four dot rows
_WAVE_HISTORY = 480  # bars kept (two to a character), so a wide terminal is filled
_WAVE_STILL_BARS = 60  # bars of the frozen thinking wave when animations are off
_THINK_TICK_S = 0.05  # one new bar of the thinking wave
_BLINK_S = 0.45  # the listening cursor's half cycle
_SPEAKER_NEW_REPLY_S = 1.5  # a pause this long means the next audio is a new reply


def _speaker_scale() -> SpeechScale:
    # TTS is steady and loud, and its pauses are digital silence, so it starts higher than a mic.
    return SpeechScale(start_speech_db=-20.0, start_noise_db=-70.0)


_LOUD_STEPS = (0.34, 0.67)  # where a bar passes from dim to medium to bright
_QUIT_GRACE_S = 3.0
_NARROW_BELOW = 72  # columns: under this the side column keeps only the state and the mic
_SHORT_BELOW = 22  # rows: under this the timings panel goes too
_MIN_SIZE = (50, 14)  # under this nothing fits, so the app says so and keeps quit working
_HEARING_HOLD_S = 3.0  # how long "you" keeps pulsing after the last loud moment, with no transcript
_HEARING_TICK_S = 0.1
_PULSE_CELLS = 5  # characters wide, two bars each
_NO_TURNS = "no turns yet"


def _printable(text: str) -> str:
    """Return ``text`` without control characters, which would reach the terminal as escapes.

    Replies, transcripts and log lines are untrusted: an ESC in one could clear the screen
    or retitle the window. A newline and a tab are kept.
    """
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def _cue_content(text: str) -> Content:
    """Return ``text`` as content with each ``[cue]`` styled and nothing read as markup."""
    parts: list[tuple[str, str] | str] = [
        (piece, "italic $text-accent") if is_cue else piece
        for piece, is_cue in split_cues(_printable(text))
    ]
    return Content.assemble(*parts)


class Waveform(Widget):
    """The mic or the speaker as a mirrored, scrolling waveform of dots: newest on the right.

    Both are drawn the same way, so they look alike: each level is scaled to that source's own
    recent peak and eased down a little, with no caps. Colour says what a bar is (green for
    speech the mic hears, grey for quiet, amber for the model thinking, the accent colour for
    the reply being played) and its brightness says how loud it is. What the strip does when
    nothing is said depends on the mood: while listening a cursor blinks at the right edge,
    while thinking a calm wave scrolls by on its own, and otherwise it holds still.
    """

    DEFAULT_CSS: ClassVar[str] = """
    Waveform {
        height: 4;
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
        self._since = 0.0
        self._cursor_on = True

    def _reset_meters(self) -> None:
        self._mic_scale = SpeechScale()
        self._speaker_scale = _speaker_scale()
        self._mic_ease = Ballistics()
        self._speaker_ease = Ballistics()

    @property
    def mood(self) -> str:
        """What the strip is doing besides drawing the mic: listening, thinking or still."""
        return self._mood

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
        self._bars.append((level, "heard" if rms >= need > 0.0 else "quiet"))
        self.refresh()

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
        elapsed = 0.0 if gap is None else min(gap, _SPEAKER_NEW_REPLY_S)
        level = self._speaker_ease.update(self._speaker_scale.update(rms, now), elapsed)[0]
        self._bars.append((level, "speaker"))
        self.refresh()

    def set_mood(self, mood: str) -> None:
        """Choose what the strip does besides drawing the mic.

        Parameters
        ----------
        mood : str
            ``"listening"`` (a blinking cursor at the right edge), ``"thinking"`` (a calm
            wave scrolls by) or ``"still"``. Textual's reduced-motion setting turns the
            blinking and the scrolling into a fixed picture.
        """
        if mood == self._mood:
            return
        if self._mood == "thinking":
            self._bars.clear()  # the made-up wave is not mic history, so do not leave it behind
        self._mood = mood
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._reset_meters()
        self._last_push = None
        self._cursor_on = True
        moving = self.app.animation_level != "none"
        if mood == "thinking":
            self._since = time.monotonic()
            if moving:
                self._timer = self.set_interval(_THINK_TICK_S, self._think)
            else:
                for step in range(_WAVE_STILL_BARS):
                    self._bars.append((processing_level(step * _THINK_TICK_S), "thinking"))
        elif mood == "listening" and moving:
            self._timer = self.set_interval(_BLINK_S, self._blink)
        self.refresh()

    def _think(self) -> None:
        self._bars.append((processing_level(time.monotonic() - self._since), "thinking"))
        self.refresh()

    def _blink(self) -> None:
        self._cursor_on = not self._cursor_on
        self.refresh()

    def render(self) -> RenderResult:
        """Draw the newest bars that fit, the oldest cut off on the left."""
        width = self.size.width
        want = width * 2  # two bars to a character
        cursor = self._mood == "listening"
        room = max(0, want - 2) if cursor else want  # the last character is kept for the cursor
        bars = list(self._bars)[-room:] if room else []
        bars = [(0.0, "quiet")] * (room - len(bars)) + bars
        if cursor:
            bars += [(0.3, "cursor") if self._cursor_on else (0.0, "quiet")] * (want - len(bars))
        rows = wave_dots([level for level, _ in bars], _WAVE_ROWS)
        tones = [_bar_tone(bars[2 * i], bars[2 * i + 1]) for i in range(width)]
        lines = [Content.assemble(*zip(row, tones, strict=True)) for row in rows]
        return Content("\n").join(lines)


# What a bar is, as a colour, and how bright it is at each of three loudness steps.
_BAR_COLOURS = {
    "cursor": ("$text-accent", (100, 100, 100)),
    "speaker": ("$text-accent", (60, 80, 100)),
    "heard": ("$text-success", (60, 80, 100)),
    "thinking": ("$text-warning", (50, 70, 90)),
    "quiet": ("$foreground", (20, 30, 42)),
}


def _bar_tone(left: tuple[float, str], right: tuple[float, str]) -> str:
    """Return the style of the character that holds two bars, from what and how loud they are."""
    kinds = {left[1], right[1]}
    kind = next(k for k in ("cursor", "heard", "speaker", "thinking", "quiet") if k in kinds)
    colour, steps = _BAR_COLOURS[kind]
    level = max(left[0], right[0])
    step = 0 if level < _LOUD_STEPS[0] else 1 if level < _LOUD_STEPS[1] else 2
    return f"{colour} {steps[step]}%"


_STAGE_TONES = {"asr": "$text-secondary", "llm": "$text-primary", "tts": "$text-accent"}
# Each stage has its own pattern as well as its own colour, so the bar reads without colour:
# dots in the top half, in every row, and in the bottom half.
_STAGE_DOTS = {"asr": "\u281b", "llm": "\u28ff", "tts": "\u28e4"}


class Timings(Widget):
    """The last turn: the wait before you heard audio, and where it went.

    Three lines: the wait beside the whole turn's length, one bar split into recognising
    (asr), thinking (llm) and speaking (tts), and the number behind each part. The parts
    differ by pattern as well as colour, with a gap between them, and come in the same order
    as the legend, so the bar reads without colour. The numbers are always shown.
    """

    DEFAULT_CSS: ClassVar[str] = """
    Timings {
        height: 3;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 - Textual's own name
        super().__init__(id=id)
        self._snapshot: LatencySnapshot | None = None

    def show(self, snapshot: LatencySnapshot | None) -> None:
        """Show a finished turn, or the empty state for ``None``."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> RenderResult:
        """Draw the wait, the split bar and the legend, fitted to the width."""
        snapshot = self._snapshot
        if snapshot is None:
            return Content.assemble((_NO_TURNS, "dim"))
        timings = turn_timings(snapshot)
        if timings is None:
            return Content(turn_summary(snapshot).strip().removeprefix("↳ ") or _NO_TURNS)
        width = max(1, self.size.width)
        wait = f"{timings.wait_ms / 1000:.2f}s"
        total = f"total {timings.total_ms / 1000:.2f}s" if timings.total_ms is not None else ""
        gap = " " * max(1, width - len(wait) - len(" wait") - len(total))
        head = Content.assemble((wait, "bold"), " wait", gap, (total, "dim"))
        gaps = max(0, len(timings.stages) - 1)  # a blank cell between parts marks each edge
        cells = split_cells([ms for _, ms in timings.stages], max(1, width - gaps))
        parts: list[tuple[str, str] | str] = []
        for position, ((name, _), count) in enumerate(zip(timings.stages, cells, strict=True)):
            if position and width > gaps:
                parts.append(" ")
            parts.append((_STAGE_DOTS[name] * count, _STAGE_TONES[name]))
        bar = Content.assemble(*parts)
        decimals = 2 if len(_legend(timings.stages, 2)) <= width else 1
        legend = Content.assemble(*_legend_parts(timings.stages, decimals))
        return Content("\n").join([head, bar, legend])


def _legend(stages: tuple[tuple[str, float], ...], decimals: int) -> str:
    return " ".join(f"{name} {ms / 1000:.{decimals}f}" for name, ms in stages)


def _legend_parts(
    stages: tuple[tuple[str, float], ...], decimals: int
) -> list[tuple[str, str] | str]:
    parts: list[tuple[str, str] | str] = []
    for position, (name, ms) in enumerate(stages):
        if position:
            parts.append(" ")
        parts.append((name, _STAGE_TONES[name]))
        parts.append(f" {ms / 1000:.{decimals}f}")
    return parts


class Conversation(VerticalScroll):
    """The conversation: what you said, the reply as it streams in, and short notes.

    The app tells it what happened and it keeps the rest to itself, including which
    widget is the reply being written, so the app holds no widget state for it.
    """

    BORDER_TITLE: ClassVar[str] = "conversation"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reply: Static | None = None
        self._pending: Static | None = None
        self._pulse: Timer | None = None
        self._expiry: Timer | None = None
        self._phase = 0

    def show_hearing(self) -> None:
        """Show that the user is speaking: a pulsing "you" line, until the words arrive.

        Call it on every loud moment. The line is added on the first call, and goes away
        by itself if no words follow, as after a cough.
        """
        if self._pending is None:
            self._phase = 0
            self._pending = self._add("you pending", self._pulse_content())
            if self.app.animation_level != "none":  # Textual's reduced-motion setting
                self._pulse = self.set_interval(_HEARING_TICK_S, self._advance_pulse)
            self.scroll_end(animate=False)
        if self._expiry is not None:
            self._expiry.stop()
        self._expiry = self.set_timer(_HEARING_HOLD_S, self._drop_pending)

    def _pulse_content(self) -> Content:
        levels = [(math.sin(self._phase * 0.7 + i * 0.6) + 1) / 2 for i in range(_PULSE_CELLS * 2)]
        (dots,) = wave_dots(levels, 1)
        return Content.assemble(("you ▸ ", "bold $text-primary"), (dots, "$text-success"))

    def _advance_pulse(self) -> None:
        if self._pending is not None:
            self._phase += 1
            self._pending.update(self._pulse_content())

    def _stop_pending(self) -> Static | None:
        pending, self._pending = self._pending, None
        for timer in (self._pulse, self._expiry):
            if timer is not None:
                timer.stop()
        self._pulse = self._expiry = None
        return pending

    def _drop_pending(self) -> None:
        pending = self._stop_pending()
        if pending is not None:
            pending.remove()

    def add_user(self, text: str) -> None:
        """Add a line the user said or typed. The reply that follows starts fresh.

        If the "you" line was pulsing, it becomes this line where it stands.
        """
        self._reply = None
        content = Content.assemble(("you ▸ ", "bold $text-primary"), _printable(text))
        pending = self._stop_pending()
        if pending is None:
            self._add("you", content)
        else:
            pending.remove_class("pending")
            pending.update(content)

    def add_note(self, text: str) -> None:
        """Add a dim status line, such as a retry or an interruption."""
        self._add("note", Content.assemble((_printable(text), "dim")))

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
                    yield Timings(id="latency")
                log = Log(id="log", max_lines=1000)
                log.border_title = "log"
                yield log
        yield Static(
            f"Terminal too small. Make it at least {_MIN_SIZE[0]}x{_MIN_SIZE[1]}, or Ctrl+Q to quit.",
            id="too-small",
        )
        yield Input(placeholder="type a line and press Enter", id="line")
        yield Footer()

    def on_resize(self, event: Resize) -> None:
        """Drop the secondary panels as the terminal shrinks, and say so when nothing fits."""
        width, height = event.size
        screen = self.screen
        screen.set_class(width < _NARROW_BELOW, "narrow")
        screen.set_class(height < _SHORT_BELOW, "short")
        screen.set_class(width < _MIN_SIZE[0] or height < _MIN_SIZE[1], "tiny")

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
        conversation = next(iter(self.query(Conversation)), None)
        if conversation is None or not conversation.is_attached:
            return  # the screen is already gone: a wake that arrived while the app was closing
        for event in events:
            self._view = reduce_view(self._view, event)
            self._side_effects(event)
        if events:
            self._render_view()
            self.query_one(Conversation).scroll_end(animate=False)

    def _side_effects(self, event: Event) -> None:
        conversation = self.query_one(Conversation)
        speaking = self._view.state is SessionState.SPEAKING
        if isinstance(event, MicLevel) and not speaking:
            self.query_one(Waveform).push(0.0 if self._live.muted else event.rms, event.need)
        elif isinstance(event, OutputLevel) and speaking:
            self.query_one(Waveform).push_speaker(event.rms)
        match event:
            case MicLevel(rms=rms, need=need, source="listen") if rms >= need > 0.0:
                conversation.show_hearing()
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
                self.query_one("#log", Log).write_line(_printable(f"{level[:1]} {tag:<6} {text}"))
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
        self.query_one(Waveform).set_mood(self._wave_mood(view))
        speaking = view.state is SessionState.SPEAKING and not self._ended
        self.query_one("#mic-box").border_title = "speaker" if speaking else "mic"
        self.query_one(Timings).show(view.last_turn)
        self.refresh_bindings()

    def _wave_mood(self, view: SessionView) -> str:
        if self._ended:
            return "still"
        if view.state is SessionState.THINKING:
            return "thinking"
        if view.state is SessionState.LISTENING and not self._live.muted:
            return "listening"
        return "still"

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
