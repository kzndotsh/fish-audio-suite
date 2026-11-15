"""The full-screen app, driven headless: events in, widgets and exit codes out."""

# The user's lines start with a single angle quote mark, which is the design, not a typo.
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.geometry import Region
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Log, Static, TabbedContent

from fish_audio_suite_voice.cli import history_pane
from fish_audio_suite_voice.cli import tui as tui_module
from fish_audio_suite_voice.cli.chat_screens import KeysScreen
from fish_audio_suite_voice.cli.diagnostics import DiagnosticsScreen, Timings
from fish_audio_suite_voice.cli.history_pane import Conversation
from fish_audio_suite_voice.cli.tui import Composer, VoiceApp, Waveform
from fish_audio_suite_voice.session.commands import NewChat
from fish_audio_suite_voice.session.events import (
    BargedIn,
    Bye,
    EventBus,
    ExitCode,
    Heard,
    Interim,
    Listening,
    LogLine,
    MicLevel,
    Notice,
    OutputLevel,
    ReplyEnd,
    ReplyToken,
    SessionState,
    Speaking,
    TurnEnded,
)
from fish_audio_suite_voice.session.inputs import LiveInput
from fish_audio_suite_voice.session.latency import LatencySnapshot
from fish_audio_suite_voice.session.signals import DuplexSession


class _SpyInput(LiveInput):
    """A ``LiveInput`` that remembers what the app asked of it."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.stops = 0

    def submit(self, text: str, *, interrupt: bool = True) -> None:
        self.lines.append(text)
        super().submit(text, interrupt=interrupt)

    def stop_reply(self) -> None:
        self.stops += 1
        super().stop_reply()


class _Harness:
    def __init__(
        self,
        runner: Callable[[_Harness], Awaitable[int]] | None = None,
        shape: Callable[[str], str] | None = None,
    ) -> None:
        self.bus = EventBus()
        self.session = DuplexSession()
        self.live = _SpyInput()
        self.runner = runner or _Harness._until_quit
        self.app = VoiceApp(
            self._run,
            live=self.live,
            session=self.session,
            bus=self.bus,
            info="model / voice",
            models="chat-model · tts-model",
            shape=shape,
        )

    async def _run(self) -> int:
        return await self.runner(self)

    async def _until_quit(self) -> int:
        """Like ``duplex_turns``: runs until asked to quit, then says bye."""
        await asyncio.to_thread(self.session.quit_requested.wait, 15)
        self.bus.emit(Bye(ExitCode.OK))
        return 0

    @property
    def main(self) -> Screen[object]:
        return self.app.screen_stack[0]

    def messages(self) -> list[str]:
        conversation = self.main.query_one("#conversation")
        return [str(child.content) for child in conversation.children if isinstance(child, Static)]

    def text(self, selector: str) -> str:
        return str(self.main.query_one(selector, Static).content)

    def state_line(self) -> str:
        parts = [self.text("#state"), self.text("#hint")]
        return "  ".join(part for part in parts if part)

    def footer_actions(self) -> set[str]:
        """The actions the footer shows a key for now."""
        return {active.binding.action for active in self.app.screen.active_bindings.values()}


def _dots(char: str) -> int:
    """How many dots a Braille character raises."""
    return (ord(char) - 0x2800).bit_count()


_BASELINE = 2  # a quiet character of the meter: one dot along the bottom for each of its bars


def _mic_rms(height: float) -> float:
    """An RMS the mic strip draws at ``height`` (0 to 1) while its scale is still unlearnt.

    A fresh scale puts the top of the bar at -33 dB and the bottom 24 dB below it.
    """
    if height <= 0:
        return 0.0
    return 32768.0 * 10 ** ((-57.0 + 24.0 * height) / 20)


def _drive(harness: _Harness, scenario: Callable[[Pilot[int]], Awaitable[None]]) -> int | None:
    async def main() -> None:
        async with harness.app.run_test(size=(100, 30)) as pilot:
            await scenario(pilot)

    asyncio.run(asyncio.wait_for(main(), 20))
    return harness.app.return_value


def test_a_turn_is_shown_as_it_happens_without_its_cues() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        for event in (
            Listening(),
            Heard("tell me something fun", 120.0),
            ReplyToken("[calm] Once "),
            ReplyToken("upon a time. [war"),  # half a cue, still arriving
        ):
            harness.bus.emit(event)
        await pilot.pause(0.1)
        assert harness.messages() == ["› tell me something fun", "Once upon a time."]
        assert harness.state_line() == "◌ Thinking…"
        harness.bus.emit(ReplyToken("m, teasing] The end."))
        harness.bus.emit(ReplyEnd("[calm] Once upon a time. [warm, teasing] The end."))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        assert harness.messages()[-1] == "Once upon a time. The end."
        assert harness.state_line() == "▶ Speaking  talk to interrupt, or press Esc"
        snapshot = LatencySnapshot(asr_ms=120.0, first_audio_ms=900.0)
        harness.bus.emit(TurnEnded(snapshot))
        harness.bus.emit(Listening())
        await pilot.pause(0.1)
        assert harness.state_line() == "● Listening"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0
    assert harness.app.view.last_turn == LatencySnapshot(asr_ms=120.0, first_audio_ms=900.0)


def test_square_brackets_are_never_read_as_markup() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard("[bold]hi[/bold] $name", 1.0))
        harness.bus.emit(ReplyEnd("[soft encouragement] Say hi to $name."))
        await pilot.pause(0.1)
        assert harness.messages() == ["› [bold]hi[/bold] $name", "Say hi to $name."]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_a_reply_the_user_talks_over_ends_in_an_ellipsis() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(ReplyToken("Well, let me think"))
        harness.bus.emit(Speaking())
        harness.bus.emit(BargedIn())
        await pilot.pause(0.1)
        assert harness.messages() == ["› hi", "Well, let me think…"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_notices_are_shown_in_the_conversation_and_log_lines_are_not() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(LogLine("DEBUG", "tts", "start voice=abc"))
        harness.bus.emit(Notice("[llm 429, retrying in 4s]"))
        await pilot.pause(0.1)
        assert harness.messages() == ["[llm 429, retrying in 4s]"]
        assert not harness.main.query(Log)  # the log lives in Diagnostics only
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_events_from_another_thread_reach_the_screen_without_flooding_it() -> None:
    harness = _Harness()
    wakes: list[int] = []
    original = harness.app.post_message

    def counting(message: Any) -> bool:
        wakes.append(1)
        return original(message)

    harness.app.post_message = counting

    async def scenario(pilot: Pilot[int]) -> None:
        def audio_thread() -> None:
            for n in range(500):
                harness.bus.emit(MicLevel(float(n), 200.0, "listen"))
            harness.bus.emit(Notice("done"))

        thread = threading.Thread(target=audio_thread)
        thread.start()
        await asyncio.to_thread(thread.join)
        await pilot.pause(0.2)
        assert "done" in harness.messages()  # the last event, after all 500 levels
        # 501 events, but the app is woken far fewer times than that.
        assert len(wakes) < 100
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_a_typed_line_goes_to_the_session_and_the_box_is_cleared() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        line = harness.app.query_one(Composer)
        assert line.placeholder == "Type a message, or just talk"
        await pilot.press(*"hello", "enter")
        await pilot.pause(0.1)
        assert harness.live.lines == ["hello"]
        assert line.value == ""
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


async def _press_up(pilot: Pilot[int]) -> None:
    await pilot.press("up")  # an arrow is a key click too, though it changes nothing


async def _paste(pilot: Pilot[int]) -> None:
    pilot.app.query_one(Composer).insert_text_at_cursor("pasted")  # a change with no key
    await pilot.pause(0.05)


@pytest.mark.parametrize("act", [_press_up, _paste], ids=["arrow", "paste"])
def test_the_composer_tells_the_mic_the_user_is_typing(
    act: Callable[[Pilot[int]], Awaitable[None]],
) -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        assert not harness.live.typing
        await act(pilot)
        assert harness.live.typing
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_up_and_down_bring_back_typed_lines_and_keep_the_draft() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        line = harness.app.query_one(Composer)
        await pilot.press("up")
        assert line.value == ""  # nothing typed yet: nothing to bring back
        assert "recall(-1)" not in harness.footer_actions()  # so the footer offers nothing
        for text in ("first", "second"):
            await pilot.press(*text, "enter")
        assert "recall(-1)" in harness.footer_actions()
        await pilot.press(*"half")
        await pilot.press("up")
        assert line.value == "second"  # newest first
        await pilot.press("up")
        assert line.value == "first"
        await pilot.press("up")
        assert line.value == "first"  # the oldest stays
        await pilot.press("down")
        assert line.value == "second"
        await pilot.press("down")
        assert line.value == "half"  # past the newest: the draft is back
        assert line.cursor_position == len("half")
        await pilot.press("down")
        assert line.value == "half"
        await pilot.press("up", "enter")
        assert harness.live.lines == ["first", "second", "second"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_mute_key_turns_the_mic_off_and_the_state_line_says_so() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Listening())
        await pilot.pause(0.05)
        assert "mute" in harness.footer_actions()
        await pilot.press("f2")
        await pilot.pause(0.05)
        assert harness.live.muted
        assert harness.state_line() == "○ Mic off"
        assert harness.footer_actions() >= {"unmute"}
        assert "mute" not in harness.footer_actions()
        await pilot.press("f2")
        await pilot.pause(0.05)
        assert not harness.live.muted
        assert harness.state_line() == "● Listening"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_stop_is_shown_and_works_only_while_there_is_a_reply_to_stop() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        assert "stop" not in harness.footer_actions()
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert harness.live.stops == 0  # idle: nothing to stop
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        assert harness.app.state is SessionState.SPEAKING
        assert "stop" in harness.footer_actions()
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert harness.live.stops == 1
        harness.bus.emit(TurnEnded(LatencySnapshot()))
        await pilot.pause(0.05)
        assert "stop" not in harness.footer_actions()
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_tab_moves_only_between_the_composer_and_the_messages() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        composer = harness.app.query_one(Composer)
        conversation = harness.app.query_one(Conversation)
        assert composer.has_focus
        seen: list[Widget | None] = []
        for _ in range(4):
            await pilot.press("tab")
            seen.append(harness.app.focused)
        assert seen == [conversation, composer, conversation, composer]
        focusable = [w for w in harness.app.screen.walk_children(Widget) if w.focusable]
        assert focusable == [conversation, composer]  # nothing read-only takes focus
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_escape_stops_the_reply_first_then_closes_then_goes_back_to_typing() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        composer = harness.app.query_one(Composer)
        await pilot.press("tab")
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        await pilot.press("escape")  # a reply is playing: it stops, and focus stays
        assert harness.live.stops == 1
        assert not composer.has_focus
        harness.bus.emit(TurnEnded(LatencySnapshot()))
        await pilot.pause(0.05)
        await pilot.press("f12")
        assert isinstance(harness.app.screen, DiagnosticsScreen)
        await pilot.press("escape")  # a screen is open: it closes
        assert not isinstance(harness.app.screen, DiagnosticsScreen)
        assert not composer.has_focus
        await pilot.press("escape")  # nothing else: back to typing
        assert composer.has_focus
        assert harness.live.stops == 1
        await pilot.press("escape")  # already typing: nothing happens
        assert composer.has_focus
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_escape_over_an_open_screen_closes_it_even_while_a_reply_plays() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        await pilot.press("f12")
        assert isinstance(harness.app.screen, DiagnosticsScreen)
        await pilot.press("escape")  # the open screen has the key, not the reply
        assert not isinstance(harness.app.screen, DiagnosticsScreen)
        assert harness.live.stops == 0
        await pilot.press("escape")  # back on the chat: now it stops the reply
        assert harness.live.stops == 1
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_state_line_has_a_glyph_and_a_word_for_every_state() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        seen: list[str] = []
        for event in (Listening(), Heard("hi", 1.0), Speaking()):
            harness.bus.emit(event)
            await pilot.pause(0.05)
            seen.append(harness.text("#state"))
        await pilot.press("f2")  # muted while the reply plays: no barge-in, so say Esc
        await pilot.pause(0.05)
        assert harness.state_line() == "▶ Speaking  press Esc to stop"
        assert seen == ["● Listening", "◌ Thinking…", "▶ Speaking"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_quit_asks_the_session_to_end_and_the_app_leaves_when_it_has() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("ctrl+q")
        await pilot.pause(0.5)

    assert _drive(harness, scenario) == 0
    assert harness.session.quit_requested.is_set()


def test_a_session_that_says_goodbye_on_its_own_closes_the_app() -> None:
    async def says_bye(harness: _Harness) -> int:
        await asyncio.sleep(0.05)
        harness.bus.emit(Bye(ExitCode.OK))
        return 0

    harness = _Harness(says_bye)

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.pause(0.5)

    assert _drive(harness, scenario) == 0
    assert not harness.session.quit_requested.is_set()


def test_a_fatal_end_stays_on_screen_until_the_user_quits_and_keeps_its_code() -> None:
    async def fails(harness: _Harness) -> int:
        await asyncio.sleep(0.05)
        harness.bus.emit(Notice("[tts] 401 unauthorized"))
        harness.bus.emit(Bye(ExitCode.FATAL))
        return 2

    harness = _Harness(fails)

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.pause(0.3)
        assert harness.app.is_running  # still open so the reason can be read
        assert harness.state_line() == "✕ Session ended  exit code 2 · Ctrl+Q to quit"
        assert "[tts] 401 unauthorized" in harness.messages()
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 2


def test_a_session_that_raises_is_reported_and_leaves_code_1() -> None:
    async def crashes(harness: _Harness) -> int:
        del harness
        raise RuntimeError("boom")

    harness = _Harness(crashes)

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.pause(0.3)
        assert any("the session stopped" in m and "boom" in m for m in harness.messages())
        assert harness.state_line().startswith("✕ Session ended  exit code 1")
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 1


def test_events_sent_before_the_screen_is_up_are_not_lost() -> None:
    harness = _Harness()
    harness.bus.emit(Heard("early", 1.0))

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.pause(0.2)
        assert harness.messages() == ["› early"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_header_names_the_chat_or_else_the_app() -> None:
    from fish_audio_suite_voice.session.chat_store import chat_title
    from fish_audio_suite_voice.session.conversation import Conversation as Record
    from fish_audio_suite_voice.session.events import ChatShown

    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        assert harness.text("#title") == "fish-voice"
        assert harness.text("#keys-hint") == "? keys"
        # The chat model and the TTS model stay in view, beside the keys hint.
        assert harness.text("#models") == "chat-model · tts-model"
        record = Record("be brief")
        record.add_user_line("What should the cat be called?")
        title = chat_title(record)
        harness.bus.emit(ChatShown(record.id, tuple(record.turns), frozenset(), title=title))
        await pilot.pause(0.1)
        assert harness.text("#title") == "What should the cat be called?"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_app_stops_collecting_events_when_it_closes() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("ctrl+q")
        await pilot.pause(0.4)

    _drive(harness, scenario)
    harness.bus.emit(Heard("after", 1.0))  # nobody is listening, and nothing fails
    assert harness.app.view.state is not SessionState.THINKING


@pytest.mark.parametrize("size", [(80, 24), (60, 20), (120, 40)])
def test_the_layout_stacks_the_parts_full_width_and_nothing_overlaps(
    size: tuple[int, int],
) -> None:
    harness = _Harness()

    async def main() -> None:
        async with harness.app.run_test(size=size) as pilot:
            await pilot.pause(0.1)
            screen = harness.app.screen.region
            order = ["#header", "#conversation", "#state-line", "#line", "Footer"]
            regions = [harness.app.query_one(selector).region for selector in order]
            for name, region in zip(order, regions, strict=True):
                assert region.width == screen.width, f"{name} is not full width"
                assert region.height > 0, f"{name} has no height"
                assert screen.contains_region(region), f"{name} is off the screen"
            for above, below in itertools.pairwise(regions):
                assert above.bottom <= below.y
            assert regions[0].height == 1
            assert regions[2].height == 1
            await pilot.press("ctrl+q")

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_stylesheet_ships_inside_the_package_next_to_the_app() -> None:
    sheet = Path(tui_module.__file__).with_name("tui.tcss")
    assert sheet.is_file()
    assert VoiceApp.CSS_PATH == "tui.tcss"
    # Hatchling includes every file under the package directory in the wheel.
    config = (Path(tui_module.__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["src/fish_audio_suite_voice"]' in config


def test_the_app_draws_under_every_builtin_theme() -> None:
    """Colors come from theme variables, so a theme that lacks one would break the screen."""
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard("hello", 1.0))
        harness.bus.emit(ReplyEnd("[calm] hi there"))
        harness.bus.emit(Speaking())
        await pilot.pause(0.05)
        themes = sorted(harness.app.available_themes)
        assert len(themes) >= 5
        for name in themes:
            harness.app.theme = name
            await pilot.pause(0.02)
            assert harness.app.theme == name
            assert harness.messages() == ["› hello", "hi there"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_too_small_a_terminal_says_so_and_quit_still_works() -> None:
    harness = _Harness()

    async def main() -> None:
        async with harness.app.run_test(size=(40, 12)) as pilot:
            await pilot.pause(0.1)
            assert harness.app.query_one("#too-small").region.height > 0
            for selector in ("#conversation", "#state-line"):
                widget = harness.app.query_one(selector)
                assert not widget.display or widget.region.height == 0, selector
            assert harness.app.query_one("#line").region.height > 0  # typing still works
            await pilot.press("ctrl+q")

    asyncio.run(asyncio.wait_for(main(), 20))
    assert harness.app.return_value == 0


class _ConversationApp(App[None]):
    """Hosts a ``Conversation`` alone, to test it without a session."""

    def compose(self) -> ComposeResult:
        yield Conversation(send=lambda _command: None, hold_mic=lambda _held: None)


def _conversation_lines(conversation: Conversation) -> list[str]:
    return [str(child.content) for child in conversation.children if isinstance(child, Static)]


def test_the_conversation_widget_adds_its_reply_once_then_updates_it() -> None:
    async def main() -> None:
        app = _ConversationApp()
        async with app.run_test() as pilot:
            conversation = app.query_one(Conversation)
            conversation.show_reply("")  # nothing to show yet: no empty line
            conversation.add_user("hi")
            conversation.show_reply("[calm]")  # only a cue so far: still nothing to show
            conversation.show_reply("[calm] Hel")
            conversation.show_reply("[calm] Hello")
            await pilot.pause(0.05)
            assert _conversation_lines(conversation) == ["› hi", "Hello"]
            conversation.add_user("again")
            conversation.show_reply("Sure")
            await pilot.pause(0.05)
            assert _conversation_lines(conversation) == ["› hi", "Hello", "› again", "Sure"]

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_conversation_widget_keeps_notes_in_order() -> None:
    async def main() -> None:
        app = _ConversationApp()
        async with app.run_test() as pilot:
            conversation = app.query_one(Conversation)
            conversation.add_user("a")
            conversation.add_note("[llm 429, retrying in 4s]")
            conversation.add_note("[a second note]")
            await pilot.pause(0.05)
            assert _conversation_lines(conversation) == [
                "› a",
                "[llm 429, retrying in 4s]",
                "[a second note]",
            ]

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_conversation_shows_no_you_line_until_words_are_recognised() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        # Loud speech moves the meter on the state line only: one level indicator, not two.
        for rms in (1000.0, 1200.0, 900.0):
            harness.bus.emit(MicLevel(rms, 200.0, "listen"))
        await pilot.pause(0.2)
        assert harness.messages() == []
        harness.bus.emit(Interim("hello"))
        await pilot.pause(0.1)
        assert harness.messages() == ["› hello"]  # the words so far, and no dots
        assert harness.main.query_one("#conversation .you").has_class("pending")  # dim
        harness.bus.emit(Heard("hello there", 120.0))
        await pilot.pause(0.1)
        assert harness.messages() == ["› hello there"]  # the same line, now final
        assert not harness.main.query_one("#conversation .you").has_class("pending")
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


async def _settled(pilot: Pilot[int], check: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    """Poll ``check`` while the app runs, so a slow runner gets time to redraw."""
    deadline = time.monotonic() + timeout_s
    while not check() and time.monotonic() < deadline:
        await pilot.pause(0.02)
    return check()


def test_words_that_never_become_a_turn_go_away_and_come_back_for_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(history_pane, "_HEARING_HOLD_S", 0.2)
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Interim("mm"))
        assert await _settled(pilot, lambda: len(harness.messages()) == 1)
        # The turn was dropped as noise: no line followed, so the words go.
        assert await _settled(pilot, lambda: harness.messages() == [])
        harness.bus.emit(Interim("hello"))  # and the line comes back for the next words
        assert await _settled(pilot, lambda: harness.messages() == ["› hello"])
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_a_typed_line_and_barge_in_levels_add_no_pending_line() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(MicLevel(5000.0, 200.0, "barge"))  # the mic hearing the speakers
        await pilot.pause(0.1)
        assert harness.messages() == []
        harness.bus.emit(Heard("typed", 0.0))
        await pilot.pause(0.1)
        assert harness.messages() == ["› typed"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_control_characters_in_untrusted_text_never_reach_the_screen() -> None:
    harness = _Harness()
    evil = "a\x1b]0;pwned\x07b\x1b[2Jc\x9bd\x08e\tf"

    def clean(text: str) -> bool:
        return not any(
            (ord(ch) < 32 and ch not in "\n\t") or 0x7F <= ord(ch) <= 0x9F for ch in text
        )

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard(evil, 1.0))
        harness.bus.emit(ReplyToken(evil))
        harness.bus.emit(Notice(evil))
        harness.bus.emit(LogLine("INFO", "llm", evil))
        await pilot.pause(0.1)
        shown = harness.messages()
        assert len(shown) == 3
        assert all(clean(text) for text in shown)
        assert "› a]0;pwnedb[2Jcde\tf" in shown  # the printable remainder is kept
        await pilot.press("f12")
        await pilot.pause(0.1)
        (logged,) = harness.app.screen.query_one(Log).lines
        assert clean(logged)
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_an_event_that_arrives_as_the_app_closes_is_ignored() -> None:
    harness = _Harness()

    def held(message: Any) -> bool:
        del message
        return True

    async def main() -> None:
        async with harness.app.run_test() as pilot:
            await pilot.pause(0.05)
            # The wake for this event is still on its way when the app exits.
            harness.app.post_message = held
            harness.bus.emit(Heard("too late", 1.0))
            harness.app.exit(0)
        harness.session.request_quit()
        harness.app.on_voice_app_wake()  # it lands once the screen is gone, and changes nothing

    asyncio.run(asyncio.wait_for(main(), 20))
    assert harness.app.return_value == 0
    assert harness.app.view.state is not SessionState.THINKING


def test_the_composer_has_a_rounded_border_in_the_accent_colour_only_when_focused() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        line = harness.app.query_one("#line")
        assert line.has_focus
        top = line.render_lines(Region(0, 0, line.region.width, line.region.height))[0].text
        assert top.startswith("╭")
        assert top.endswith("╮")  # no half-block edges of the default "tall" border
        assert line.region.height == 3
        accent = Color.parse(harness.app.get_css_variables()["accent"])
        assert line.styles.border_top[1] == accent
        await pilot.press("tab")
        assert line.styles.border_top[1] != accent
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_composer_and_footer_share_the_screens_background() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        screen = Color.parse(harness.app.get_css_variables()["background"]).rgb
        for selector in ("#line", "Footer"):
            assert harness.app.query_one(selector).styles.background.rgb == screen, selector
        assert harness.app.query_one(Waveform).styles.background.a == 0  # draws on the screen
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_focus_does_not_tint_past_the_border_of_the_conversation_or_composer() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        for selector in ("#conversation", "#line"):
            widget = harness.app.query_one(selector)
            widget.focus()
            await pilot.pause(0.05)
            assert widget.has_focus, selector
            assert widget.styles.background_tint.a == 0, f"{selector} is tinted when focused"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_command_palette_lists_the_apps_own_commands_first() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        def titles() -> list[str]:
            return [command.title for command in harness.app.get_system_commands(harness.main)]

        assert titles()[:6] == ["New chat", "Chats…", "Mic off", "Diagnostics", "Keys", "Quit"]
        assert len(titles()) > 6  # Textual's own come after
        # Its Keys and Quit are left out: one keys list, and Quit that ends the session first.
        assert titles().count("Keys") == 1
        assert titles().count("Quit") == 1
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await pilot.press("f2")
        await pilot.pause(0.1)
        assert titles()[:4] == ["New chat", "Chats…", "Mic on", "Stop speaking"]
        new_chat = next(c for c in harness.app.get_system_commands(harness.main))
        new_chat.callback()
        assert harness.live.take_commands() == [NewChat()]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_keys_list_opens_from_the_messages_and_closes_with_escape() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("question_mark")  # typing: it is just a character
        assert harness.app.query_one(Composer).value == "?"
        await pilot.press("tab", "question_mark")
        assert isinstance(harness.app.screen, KeysScreen)
        text = str(harness.app.screen.query_one(Static).content)
        assert "Hide from the AI, or show again" in text
        await pilot.press("escape")
        assert not isinstance(harness.app.screen, KeysScreen)
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def _diagnostics(harness: _Harness) -> DiagnosticsScreen:
    screen = harness.app.screen
    assert isinstance(screen, DiagnosticsScreen)
    return screen


def test_diagnostics_shows_the_log_including_lines_that_arrived_while_it_was_closed() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(LogLine("DEBUG", "tts", "start voice=abc"))
        await pilot.pause(0.1)
        await pilot.press("f12")
        await pilot.pause(0.1)
        screen = _diagnostics(harness)
        screen.query_one(TabbedContent).active = "log-tab"
        log = screen.query_one(Log)
        assert list(log.lines) == ["D tts    start voice=abc"]
        harness.bus.emit(LogLine("INFO", "llm", "while open"))
        await pilot.pause(0.1)
        assert list(log.lines)[-1] == "I llm    while open"
        await pilot.press("f12")  # F12 goes back as well
        assert not isinstance(harness.app.screen, DiagnosticsScreen)
        harness.bus.emit(LogLine("INFO", "llm", "while closed again"))
        await pilot.pause(0.1)
        await pilot.press("f12")
        await pilot.pause(0.1)
        lines = list(_diagnostics(harness).query_one(Log).lines)
        assert lines == [
            "D tts    start voice=abc",
            "I llm    while open",
            "I llm    while closed again",
        ]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_log_keeps_only_the_newest_lines_while_diagnostics_is_closed() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        for number in range(1200):
            harness.bus.emit(LogLine("INFO", "llm", f"line {number}"))
            if number % 200 == 0:
                await pilot.pause(0.01)  # the queue keeps 256 between wakes
        await pilot.pause(0.2)
        await pilot.press("f12")
        await pilot.pause(0.1)
        lines = list(_diagnostics(harness).query_one(Log).lines)
        assert len(lines) == 1000
        assert lines[-1].endswith("line 1199")
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_diagnostics_shows_timings_the_reply_as_fish_read_it_and_the_session() -> None:
    from fish_audio_suite_voice.session.conversation import Conversation as Record
    from fish_audio_suite_voice.session.events import ChatShown

    harness = _Harness(shape=lambda text: text.replace("happy", "[happy]"))

    async def scenario(pilot: Pilot[int]) -> None:
        record = Record("be brief")
        for line in ("one", "two"):
            record.add_user_line(line)
            record.add_reply("ok.", interrupted=False)
        harness.bus.emit(ChatShown(record.id, tuple(record.turns), frozenset({record.turns[1].id})))
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(ReplyEnd("happy Hello there."))
        for wait in (900.0, 1500.0):
            harness.bus.emit(
                TurnEnded(
                    LatencySnapshot(asr_ms=200.0, llm_first_token_ms=400.0, first_audio_ms=wait)
                )
            )
        await pilot.pause(0.1)
        await pilot.press("f12")
        await pilot.pause(0.1)
        screen = _diagnostics(harness)
        assert "1.50s wait" in "\n".join(
            strip.text for strip in screen.query_one(Timings).render_lines(Region(0, 0, 60, 3))
        )
        recent = str(screen.query_one("#recent", Static).content).splitlines()
        assert recent[0].split()[:2] == ["turn", "heard"]
        assert recent[1].split()[0] == "2"  # newest first
        assert recent[2].split()[0] == "1"
        assert str(screen.query_one("#sent", Static).content) == "[happy] Hello there."
        session = str(screen.query_one("#session", Static).content)
        assert "model / voice" in session
        assert "the model sees 1 of 2 turns" in session
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def _timings_widget_text(snapshot: LatencySnapshot | None, width: int) -> list[str]:
    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield Timings()

    lines: list[str] = []

    async def main() -> None:
        host = Host()
        async with host.run_test(size=(width, 10)) as pilot:
            panel = host.query_one(Timings)
            panel.show(snapshot)
            await pilot.pause()
            lines.extend(
                strip.text for strip in panel.render_lines(Region(0, 0, panel.size.width, 3))
            )

    asyncio.run(main())
    return lines


def test_the_timings_panel_shows_the_wait_a_split_bar_and_a_legend() -> None:
    snapshot = LatencySnapshot(
        asr_ms=270.0, llm_first_token_ms=670.0, first_audio_ms=1530.0, voice_to_voice_ms=6470.0
    )
    head, bar, legend = _timings_widget_text(snapshot, 26)
    assert head.startswith("1.53s wait")
    assert head.rstrip().endswith("total 6.47s")
    assert len(head) == 26  # the total sits at the right edge
    assert len(bar) == 26
    # One pattern per stage, in legend order, with a blank cell between parts: it reads
    # without colour.
    parts = bar.split()
    assert [part[0] for part in parts] == ["⠛", "⣿", "⣤"]
    assert all(len(set(part)) == 1 for part in parts)
    assert bar.count(" ") == 2
    assert legend.split() == ["asr", "0.27", "llm", "0.67", "tts", "0.59"]


def test_the_timings_panel_copes_with_no_turn_no_asr_a_narrow_width_and_no_wait() -> None:
    assert _timings_widget_text(None, 26)[0].strip() == "no turns yet"
    typed = _timings_widget_text(
        LatencySnapshot(llm_first_token_ms=400.0, first_audio_ms=900.0), 26
    )
    assert typed[2].split() == ["llm", "0.40", "tts", "0.50"]
    wide_values = LatencySnapshot(
        asr_ms=12000.0,
        llm_first_token_ms=13000.0,
        first_audio_ms=40000.0,
        voice_to_voice_ms=90000.0,
    )
    legend = _timings_widget_text(wide_values, 26)[2]
    assert len(legend.rstrip()) <= 26  # drops to one decimal rather than overflow
    fallback = _timings_widget_text(LatencySnapshot(asr_ms=120.0), 40)
    assert "asr 0.12s" in fallback[0]  # no wait to split, so the one-line summary


def test_the_timings_bar_keeps_its_parts_distinct_even_when_one_is_tiny_or_the_panel_is_small() -> (
    None
):
    tiny_asr = LatencySnapshot(asr_ms=5.0, llm_first_token_ms=900.0, first_audio_ms=1500.0)
    bar = _timings_widget_text(tiny_asr, 26)[1]
    assert len(bar) == 26
    assert [part[0] for part in bar.split()] == ["⠛", "⣿", "⣤"]  # the short one still shows
    narrow = _timings_widget_text(tiny_asr, 4)[1]
    assert len(narrow) == 4  # never wider than the panel


class _WaveHost(App[None]):
    def compose(self) -> ComposeResult:
        yield Waveform()


def _wave_line(wave: Waveform) -> str:
    return wave.render_lines(Region(0, 0, wave.size.width, 1))[0].text


def _wave_colours(wave: Waveform) -> list[tuple[int, int, int] | None]:
    strip = wave.render_lines(Region(0, 0, wave.size.width, 1))[0]
    colours: list[tuple[int, int, int] | None] = []
    for segment in strip:
        colour = segment.style.color if segment.style is not None else None
        triplet = tuple(colour.get_truecolor()) if colour is not None else None
        colours.extend([triplet] * len(segment.text))  # type: ignore[list-item]
    return colours


def test_the_meter_scrolls_newest_right_and_decays_in_dots() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            width = wave.size.width
            assert width > 0
            assert wave.size.height == 1
            for level in (
                0.0,
                1.0,
                0.0,
            ):  # quiet, loud, then silence that falls instead of snapping
                wave.push(_mic_rms(level), 1.0)
            await pilot.pause()
            line = _wave_line(wave)
            assert len(line) == width
            assert all(0x2800 <= ord(ch) <= 0x28FF for ch in line)  # all dots
            assert _dots(line[0]) == _BASELINE  # nothing but the thin line before any speech
            assert (
                _dots(line[-2]) > _BASELINE
            )  # the newest characters hold the loud bar and its fall
            assert _dots(line[-2]) > _dots(line[-4])  # (the last one is kept for the cursor)

    asyncio.run(main())


def test_the_meter_draws_dots_in_the_foreground_only_with_no_cell_background() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            for level in (0.0, 0.8, 0.05, 0.9):
                wave.push(_mic_rms(level), 1.0)
            await pilot.pause()
            screen = Color.parse(host.get_css_variables()["background"]).rgb
            strips = wave.render_lines(Region(0, 0, wave.size.width, 1))
            backgrounds = {
                tuple(segment.style.bgcolor.get_truecolor())
                for strip in strips
                for segment in strip
                if segment.style is not None and segment.style.bgcolor is not None
            }
            assert backgrounds <= {screen}  # dots sit on the screen, nothing is filled behind them

    asyncio.run(main())


def test_the_meter_shows_only_while_listening_or_speaking() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        wave = harness.app.query_one(Waveform)
        moods: list[tuple[str, bool]] = []
        for event in (Listening(), Heard("hi", 1.0), Speaking(), Listening()):
            harness.bus.emit(event)
            await pilot.pause(0.1)
            moods.append((wave.mood, wave.display))
        assert moods == [
            ("listening", True),
            ("still", False),  # thinking: the word says it, there is no level to show
            ("speaking", True),
            ("listening", True),
        ]
        await pilot.press("f2")  # muted: nothing is listening, so no meter
        await pilot.pause(0.1)
        assert (wave.mood, wave.display) == ("still", False)
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_while_listening_a_cursor_blinks_at_the_right_edge() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        wave = harness.app.query_one(Waveform)
        harness.bus.emit(Listening())
        await pilot.pause(0.1)
        seen: set[str] = set()
        for _ in range(8):
            seen.add(_wave_line(wave)[-1])  # the cursor's character
            await pilot.pause(0.2)
        assert len(seen) == 2  # on and off
        assert all(_dots(ch) >= _BASELINE for ch in seen)  # even "off" keeps the thin line
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_with_animations_off_the_cursor_holds_still() -> None:
    harness = _Harness()
    harness.app.animation_level = "none"  # what TEXTUAL_ANIMATIONS=none sets

    async def scenario(pilot: Pilot[int]) -> None:
        wave = harness.app.query_one(Waveform)
        harness.bus.emit(Listening())
        await pilot.pause(0.1)
        cursor = _wave_line(wave)
        await pilot.pause(0.7)
        assert _wave_line(wave) == cursor  # no blinking
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def _strip_after(feed: Callable[[Waveform], None]) -> str:
    """Draw a fresh strip after ``feed`` has pushed bars to it, and return its line."""
    lines: list[str] = []

    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            feed(wave)
            await pilot.pause()
            lines.append(_wave_line(wave))

    asyncio.run(main())
    return lines[0]


def test_the_mic_and_the_speaker_draw_alike_once_each_has_learnt_its_source() -> None:
    # The same ups and downs, in dB below each source's loud level, repeated for ten seconds.
    shape = (0.0, -3.0, -6.0, -12.0, -22.0, -9.0, -3.0, 0.0, -18.0, -6.0, -40.0, -40.0)

    def feed(
        wave: Waveform, push: Callable[[Waveform, float, float], None], loud_db: float
    ) -> None:
        for step in range(130):
            rms = 32768.0 * 10 ** ((loud_db + shape[step % len(shape)]) / 20)
            push(wave, rms, step * 0.09)

    def mic(wave: Waveform, rms: float, now: float) -> None:
        wave.push(rms, 1.0, now=now)

    def speaker(wave: Waveform, rms: float, now: float) -> None:
        wave.push_speaker(rms, now=now)

    def ink(line: str) -> int:
        return sum(_dots(ch) for ch in line[-12:])  # the newest twelve cells

    quiet_mic = _strip_after(lambda wave: feed(wave, mic, -32.0))
    hot_speaker = _strip_after(lambda wave: feed(wave, speaker, -8.0))  # 24 dB louder
    assert ink(quiet_mic) == pytest.approx(ink(hot_speaker), rel=0.2)


def test_a_quiet_room_is_not_stretched_into_a_waveform_by_the_mic_scale() -> None:
    def room(wave: Waveform) -> None:
        for step in range(60):  # three and a half seconds of room noise at -50 dB
            wave.push(32768.0 * 10 ** (-50.0 / 20), 1500.0, now=step * 0.06)

    line = _strip_after(room)
    # After the first second, while the scale measures the room, there is only the thin line.
    assert max(_dots(ch) for ch in line[-20:]) <= _BASELINE + 1

    def speech(wave: Waveform) -> None:
        room(wave)
        wave.push(32768.0 * 10 ** (-30.0 / 20), 1500.0, now=4.0)  # then someone speaks

    assert max(_dots(ch) for ch in _strip_after(speech)) > _BASELINE + 2


def test_brightness_follows_loudness_and_colour_follows_what_the_bar_is() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            for level in (0.2, 0.2, 0.55, 0.55, 0.95, 0.95):  # soft, medium, loud, all heard
                wave.push(_mic_rms(level), 1.0, now=0.0)
            await pilot.pause()
            colours = _wave_colours(wave)
            soft, medium, loud = colours[-4], colours[-3], colours[-2]
            assert None not in (soft, medium, loud)
            assert sum(soft) < sum(medium) < sum(loud)  # type: ignore[arg-type]  # brighter as it gets louder

    asyncio.run(main())


def test_while_the_model_speaks_the_meter_shows_the_speaker_and_hands_back_to_the_mic() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        wave = harness.app.query_one(Waveform)
        harness.bus.emit(Listening())
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        before = wave._bars.copy()
        # The mic hearing the speakers (the barge-in watch) is not drawn while the reply plays.
        harness.bus.emit(MicLevel(30000.0, 200.0, "barge"))
        await pilot.pause(0.05)
        assert not any(kind == "heard" for _, kind in list(wave._bars)[len(before) :])
        for _ in range(6):
            harness.bus.emit(OutputLevel(8000.0))
        await pilot.pause(0.1)
        assert max(_dots(ch) for ch in _wave_line(wave)) >= 6  # the reply's level is drawn
        harness.bus.emit(BargedIn())  # the user talks over it: back to the mic
        await pilot.pause(0.1)
        assert wave.mood == "listening"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_speaker_bars_are_the_accent_colour_and_the_mics_are_not() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            for _ in range(2):
                wave.push_speaker(8000.0, now=0.0)
            for _ in range(2):
                wave.push(_mic_rms(0.9), 1.0, now=0.0)  # a loud mic bar, heard
            await pilot.pause()
            colours = _wave_colours(wave)
            speaker, mic = colours[-3], colours[-2]
            assert speaker is not None
            assert mic is not None
            assert speaker != mic

    asyncio.run(main())


def _cell_dots(wave: Waveform, cell: int) -> int:
    return _dots(_wave_line(wave)[cell])


def test_speaker_bars_follow_the_voices_own_ups_and_downs() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            clock = 0.0
            # Two bars to a character: loud, loud | pause, pause | loud, loud | stressed, soft.
            for rms in (9000.0, 9000.0, 60.0, 60.0, 9000.0, 9000.0, 9000.0, 1800.0):
                wave.push_speaker(rms, now=clock)
                clock += 0.25
            await pilot.pause()
            loud, pause, loud_again, stressed = (_cell_dots(wave, i) for i in (-5, -4, -3, -2))
            assert pause < loud  # a pause really is lower
            assert loud_again == loud  # the same loudness, the same height
            assert stressed < loud  # and a softer syllable sits between

    asyncio.run(main())


def test_a_reply_after_a_pause_is_scaled_afresh_not_against_the_last_reply() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            wave.push_speaker(30000.0, now=0.0)  # a very loud first reply
            wave.push_speaker(30000.0, now=0.06)
            wave.push_speaker(1500.0, now=5.0)  # much later: the next reply, much quieter
            wave.push_speaker(1500.0, now=5.06)
            await pilot.pause()
            same_reply = _cell_dots(wave, -2)
            wave.push_speaker(30000.0, now=6.0)
            wave.push_speaker(30000.0, now=6.06)
            wave.push_speaker(1500.0, now=6.1)  # soon after a loud one: the same reply
            wave.push_speaker(1500.0, now=6.16)
            await pilot.pause()
            assert _cell_dots(wave, -2) < same_reply  # judged against the loud moment just before

    asyncio.run(main())


def test_one_loud_transient_does_not_shrink_the_speech_that_follows_it() -> None:
    def speech(wave: Waveform, *, transient: bool) -> None:
        clock = 0.0
        for rms in ([15000.0] if transient else [150.0]) + [150.0] * 2:  # a bump, or nothing
            wave.push(rms, 200.0, now=clock)
            clock += 0.06
        for rms in (3000.0, 4500.0, 4000.0, 3500.0, 4800.0, 4200.0):  # then ordinary speech
            wave.push(rms, 200.0, now=clock)
            clock += 0.06

    bumped = _strip_after(lambda wave: speech(wave, transient=True))
    clean = _strip_after(lambda wave: speech(wave, transient=False))

    def height(line: str) -> int:
        return sum(_dots(ch) for ch in line[-4:])  # the speech's own cells

    assert height(bumped) >= 0.85 * height(clean)  # a bump costs it almost nothing


def test_the_bars_do_not_shift_sideways_when_the_mood_changes() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(30, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            for step in range(60):
                wave.push(_mic_rms(0.8 if step % 6 < 3 else 0.1), 1.0, now=step * 0.09)
            await pilot.pause()

            def without_cursor() -> str:
                return _wave_line(wave)[:-1]  # the last character is the cursor

            still = without_cursor()
            wave.set_mood("listening")
            await pilot.pause()
            assert without_cursor() == still  # starting to listen moves nothing
            wave.set_mood("still")
            await pilot.pause()
            assert without_cursor() == still

    asyncio.run(main())


def test_the_scale_keeps_what_it_learnt_about_the_room_across_turns() -> None:
    async def main() -> None:
        host = _WaveHost()
        async with host.run_test(size=(30, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            room = 32768.0 * 10 ** (-45.0 / 20)  # a noisy room, well above the usual -60 dB guess
            speech = 32768.0 * 10 ** (-25.0 / 20)
            for step in range(130):  # twelve seconds of someone talking in it
                wave.push(speech if step % 8 < 4 else room, 1e9, now=step * 0.09)
            for mood in ("speaking", "still", "listening"):  # a turn: speaking, then listening
                wave.set_mood(mood)
                await pilot.pause(0.05)
            wave.push(room, 1e9, now=40.0)  # the first blocks of the next turn, much later
            wave.push(room, 1e9, now=40.09)
            await pilot.pause()
            assert _cell_dots(wave, -2) == _BASELINE  # a thin line: it still knows the room

    asyncio.run(main())


def _luminance(colour: Color) -> float:
    def channel(value: int) -> float:
        unit = value / 255
        return unit / 12.92 if unit <= 0.03928 else ((unit + 0.055) / 1.055) ** 2.4

    red, green, blue = colour.rgb
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _contrast(first: Color, second: Color) -> float:
    lighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


class _FaintBars(App[None]):
    """One strip per kind of bar, each drawn at its faintest: a level of zero."""

    def compose(self) -> ComposeResult:
        for kind in ("quiet", "heard", "speaker", "cursor"):
            yield Waveform(id=kind)

    def on_mount(self) -> None:
        self.animation_level = "none"  # the cursor holds still
        for _ in range(8):
            self.query_one("#quiet", Waveform).push(0.0, 1.0)
            self.query_one("#heard", Waveform).push(1.0, 1.0)  # at the speech level: heard
            self.query_one("#speaker", Waveform).push_speaker(0.0)
        self.query_one("#cursor", Waveform).set_mood("listening")


def test_every_bar_colour_stays_visible_against_the_background_in_every_builtin_theme() -> None:
    # The newest cells hold the bars just added; the last one is the cursor's.
    cells = {
        "quiet": slice(-4, -1),
        "heard": slice(-4, -1),
        "speaker": slice(-4, -1),
        "cursor": slice(-1, None),
    }

    async def main() -> None:
        app = _FaintBars()
        async with app.run_test(size=(40, 30)) as pilot:
            await pilot.pause()
            for theme in sorted(app.available_themes):
                if theme.startswith("ansi"):
                    continue  # these use the terminal's own colours, which cannot be measured here
                app.theme = theme
                await pilot.pause(0.02)
                background = Color.parse(app.get_css_variables()["background"])
                for kind, where in cells.items():
                    wave = app.query_one(f"#{kind}", Waveform)
                    drawn = {c for c in _wave_colours(wave)[where] if c is not None}
                    assert drawn, f"{kind} draws nothing in {theme}"
                    weakest = min(_contrast(Color(*rgb), background) for rgb in drawn)
                    # Braille dots are small, so they need more than text does. A faint "quiet"
                    # line is meant to be quiet, but it must still be there.
                    minimum = 2.2 if kind == "quiet" else 2.6
                    assert weakest >= minimum, f"{kind} in {theme}: contrast {weakest:.1f}"

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_strip_scrolls_at_one_speed_through_a_whole_turn() -> None:
    """Each source used to add bars at its own rate (10, 19, none, 15 a second), which felt like
    the strip speeding up, stopping and catching up at every handover."""
    harness = _Harness()
    rates: dict[str, float] = {}

    async def scenario(pilot: Pilot[int]) -> None:
        wave = harness.app.query_one(Waveform)

        async def phase(name: str, seconds: float, emit: Callable[[], None] | None = None) -> None:
            before, started = len(wave._bars), time.monotonic()
            while time.monotonic() - started < seconds:
                if emit is not None:
                    emit()
                await pilot.pause(0.09)
            rates[name] = (len(wave._bars) - before) / (time.monotonic() - started)

        harness.bus.emit(Listening())
        await phase("listening", 1.3, lambda: harness.bus.emit(MicLevel(500.0, 200.0, "listen")))
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await phase("before audio", 1.3)  # no audio levels yet: the strip must not stop
        await phase("speaking", 1.3, lambda: harness.bus.emit(OutputLevel(8000.0)))
        await phase("a pause", 1.3)
        harness.bus.emit(BargedIn())
        await phase(
            "listening again", 1.3, lambda: harness.bus.emit(MicLevel(500.0, 200.0, "listen"))
        )
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0
    # About eleven bars a second, the mic's own rate, whatever is feeding the strip.
    for name, rate in rates.items():
        assert 7.0 <= rate <= 13.0, f"{name}: {rate:.1f} bars a second"


def test_the_words_appear_on_the_you_line_as_they_are_recognised_then_become_the_transcript() -> (
    None
):
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Interim("hel"))
        await pilot.pause(0.1)
        (line,) = harness.messages()
        assert line.startswith("› ")
        assert line.endswith(" hel")
        harness.bus.emit(Interim("hello there"))
        harness.bus.emit(Interim("hello there friend\x1b[2J"))  # untrusted text, like any other
        await pilot.pause(0.1)
        (line,) = harness.messages()  # the same line, replaced, not a new one each time
        assert line.endswith(" hello there friend[2J")
        assert "\x1b" not in line
        harness.bus.emit(Heard("hello there friend", 120.0))
        await pilot.pause(0.1)
        assert harness.messages() == ["› hello there friend"]  # the same line, now final
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0
