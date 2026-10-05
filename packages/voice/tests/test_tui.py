"""The full-screen app, driven headless: events in, widgets and exit codes out."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.geometry import Region
from textual.pilot import Pilot
from textual.widgets import Input, Log, Static

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice import tui as tui_module
from fish_audio_suite_voice.events import (
    BargedIn,
    Bye,
    EventBus,
    Heard,
    Listening,
    LogLine,
    MicLevel,
    Notice,
    ReplyEnd,
    ReplyToken,
    SessionState,
    Speaking,
    TurnEnded,
)
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.tui import Conversation, VoiceApp, Waveform


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
    def __init__(self, runner: Callable[[_Harness], Awaitable[int]] | None = None) -> None:
        self.bus = EventBus()
        self.session = DuplexSession()
        self.live = _SpyInput()
        self.runner = runner or _Harness._until_quit
        self.app = VoiceApp(
            self._run, live=self.live, session=self.session, bus=self.bus, info="model / voice"
        )

    async def _run(self) -> int:
        return await self.runner(self)

    async def _until_quit(self) -> int:
        """Like ``duplex_turns``: runs until asked to quit, then says bye."""
        await asyncio.to_thread(self.session.quit_requested.wait, 15)
        self.bus.emit(Bye(0))
        return 0

    def messages(self) -> list[str]:
        conversation = self.app.query_one("#conversation")
        return [str(child.content) for child in conversation.children if isinstance(child, Static)]

    def text(self, selector: str) -> str:
        return str(self.app.query_one(selector, Static).content)

    def log_lines(self) -> list[str]:
        return list(self.app.query_one("#log", Log).lines)


def _drive(harness: _Harness, scenario: Callable[[Pilot[int]], Awaitable[None]]) -> int | None:
    async def main() -> None:
        async with harness.app.run_test() as pilot:
            await scenario(pilot)

    asyncio.run(asyncio.wait_for(main(), 20))
    return harness.app.return_value


def test_a_turn_is_shown_as_it_happens() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        for event in (
            Listening(),
            Heard("tell me something fun", 120.0),
            ReplyToken("[calm] Once "),
            ReplyToken("upon a time."),
        ):
            harness.bus.emit(event)
        await pilot.pause(0.1)
        assert harness.messages() == [
            "you ▸ tell me something fun",
            "llm ▸ [calm] Once upon a time.",
        ]
        assert harness.text("#status") == "thinking"
        harness.bus.emit(ReplyEnd("[calm] Once upon a time."))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        assert harness.text("#status") == "speaking"
        snapshot = LatencySnapshot(asr_ms=120.0, first_audio_ms=900.0)
        harness.bus.emit(TurnEnded(snapshot))
        await pilot.pause(0.1)
        assert harness.text("#status") == "idle"
        assert "first audio" in harness.text("#latency")
        assert harness.messages().count("llm ▸ [calm] Once upon a time.") == 1
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0
    assert harness.app.view.turns == 1


def test_square_brackets_in_a_reply_are_shown_as_they_are_not_read_as_markup() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard("[bold]hi[/bold] $name", 1.0))
        harness.bus.emit(ReplyEnd("[soft encouragement] a[0] [/] [[x]] done"))
        await pilot.pause(0.1)
        assert harness.messages() == [
            "you ▸ [bold]hi[/bold] $name",
            "llm ▸ [soft encouragement] a[0] [/] [[x]] done",
        ]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_mic_meter_and_the_log_follow_the_events() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(MicLevel(1000.0, 200.0, "listen"))
        harness.bus.emit(LogLine("DEBUG", "tts", "start voice=abc"))
        harness.bus.emit(Notice("[llm 429, retrying in 4s]"))
        harness.bus.emit(BargedIn())
        await pilot.pause(0.1)
        wave = harness.app.query_one(Waveform)
        assert wave.size.height == 4
        lines = [strip.text for strip in wave.render_lines(Region(0, 0, wave.size.width, 4))]
        assert any("█" in line or "▇" in line for line in lines)  # the loud bar is drawn
        assert harness.log_lines() == ["D tts    start voice=abc"]
        assert "[llm 429, retrying in 4s]" in harness.messages()
        assert "(interrupted)" in harness.messages()
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
        assert "done" in harness.messages()
        assert harness.app.view.mic_rms == 499.0
        # 501 events, but the app is woken far fewer times than that.
        assert len(wakes) < 100
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_a_typed_line_goes_to_the_session_and_the_box_is_cleared() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press(*"hello", "enter")
        await pilot.pause(0.1)
        assert harness.live.lines == ["hello"]
        assert harness.app.query_one(Input).value == ""
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_mute_key_toggles_the_mic_and_says_so() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("f2")
        await pilot.pause(0.05)
        assert harness.live.muted
        assert harness.text("#status") == "idle · muted"
        await pilot.press("f2")
        await pilot.pause(0.05)
        assert not harness.live.muted
        assert harness.text("#status") == "idle"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_stop_only_works_while_the_model_is_thinking_or_the_reply_is_playing() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert harness.live.stops == 0  # idle: nothing to stop
        harness.bus.emit(Heard("hi", 1.0))
        harness.bus.emit(Speaking())
        await pilot.pause(0.1)
        assert harness.app.state is SessionState.SPEAKING
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert harness.live.stops == 1
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
        harness.bus.emit(Bye(0))
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
        harness.bus.emit(Bye(2))
        return 2

    harness = _Harness(fails)

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.pause(0.3)
        assert harness.app.is_running  # still open so the reason can be read
        assert harness.text("#status") == "ended (code 2)"
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
        assert harness.text("#status") == "ended (code 1)"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 1


def test_events_sent_before_the_screen_is_up_are_not_lost() -> None:
    harness = _Harness()
    harness.bus.emit(Heard("early", 1.0))

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.pause(0.2)
        assert harness.messages() == ["you ▸ early"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_title_shows_what_the_session_uses() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        assert harness.app.sub_title == "model / voice"
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_app_stops_collecting_events_when_it_closes() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("ctrl+q")
        await pilot.pause(0.4)

    _drive(harness, scenario)
    harness.bus.emit(Heard("after", 1.0))  # nobody is listening, and nothing fails
    assert harness.app.view.heard != "after"


@pytest.mark.parametrize("key", ["f2", "escape", "ctrl+q"])
def test_the_default_keys_are_the_ones_bound(key: str) -> None:
    bound = {binding.key for binding in VoiceApp.BINDINGS if isinstance(binding, Binding)}
    assert key in bound


@pytest.mark.parametrize("size", [(80, 24), (60, 20), (120, 40)])
def test_the_layout_fits_the_screen_and_nothing_overlaps(size: tuple[int, int]) -> None:
    harness = _Harness()

    async def main() -> None:
        async with harness.app.run_test(size=size) as pilot:
            await pilot.pause(0.1)
            screen = harness.app.screen.region
            conversation = harness.app.query_one("#conversation").region
            side = harness.app.query_one("#side").region
            line = harness.app.query_one("#line").region
            footer = harness.app.query_one("Footer").region
            header = harness.app.query_one("Header").region
            for name, region in {
                "conversation": conversation,
                "side": side,
                "line": line,
                "footer": footer,
                "header": header,
            }.items():
                assert region.width > 0, f"{name} has no width"
                assert region.height > 0, f"{name} has no height"
                assert screen.contains_region(region), f"{name} is off the screen"
            # The input line sits directly above the footer, below the panes, and
            # the panes do not run into each other.
            assert line.bottom <= footer.y
            assert conversation.bottom <= line.y
            assert side.bottom <= line.y
            assert header.bottom <= conversation.y
            assert conversation.right <= side.x
            await pilot.press("ctrl+q")

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_stylesheet_ships_inside_the_package_next_to_the_app() -> None:
    sheet = Path(tui_module.__file__).with_name("tui.tcss")
    assert sheet.is_file()
    assert VoiceApp.CSS_PATH == "tui.tcss"
    # Hatchling includes every file under the package directory in the wheel.
    config = (Path(tui_module.__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
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
            assert harness.messages() == ["you ▸ hello", "llm ▸ [calm] hi there"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


class _ConversationApp(App[None]):
    """Hosts a ``Conversation`` alone, to test it without a session."""

    def compose(self) -> ComposeResult:
        yield Conversation()


def _conversation_lines(conversation: Conversation) -> list[str]:
    return [str(child.content) for child in conversation.children if isinstance(child, Static)]


def test_the_conversation_widget_adds_its_reply_once_then_updates_it() -> None:
    async def main() -> None:
        app = _ConversationApp()
        async with app.run_test() as pilot:
            conversation = app.query_one(Conversation)
            conversation.show_reply("")  # nothing to show yet: no empty line
            conversation.add_user("hi")
            conversation.show_reply("[calm] Hel")
            conversation.show_reply("[calm] Hello")
            await pilot.pause(0.05)
            assert _conversation_lines(conversation) == ["you ▸ hi", "llm ▸ [calm] Hello"]
            conversation.add_user("again")
            conversation.show_reply("Sure")
            await pilot.pause(0.05)
            assert _conversation_lines(conversation) == [
                "you ▸ hi",
                "llm ▸ [calm] Hello",
                "you ▸ again",
                "llm ▸ Sure",
            ]

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_conversation_widget_keeps_notes_in_order_and_titles_itself() -> None:
    async def main() -> None:
        app = _ConversationApp()
        async with app.run_test() as pilot:
            conversation = app.query_one(Conversation)
            conversation.add_user("a")
            conversation.add_note("(interrupted)")
            conversation.add_note("[llm 429, retrying in 4s]")
            await pilot.pause(0.05)
            assert _conversation_lines(conversation) == [
                "you ▸ a",
                "(interrupted)",
                "[llm 429, retrying in 4s]",
            ]
            assert conversation.border_title == "conversation"

    asyncio.run(asyncio.wait_for(main(), 20))


def test_the_waveform_scrolls_newest_right_dims_below_the_threshold_and_decays() -> None:
    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield Waveform()

    async def main() -> None:
        host = Host()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            await pilot.pause()
            width = wave.size.width
            assert width > 0
            wave.push(0.0, 0.3)  # quiet: a thin line only, so no full cells
            wave.push(1.0, 0.3)  # loud and above the threshold
            wave.push(0.0, 0.3)  # silence again, but the bar falls instead of snapping
            await pilot.pause()
            lines = [s.text for s in wave.render_lines(Region(0, 0, width, 4))]
            assert all(len(line) == width for line in lines)
            top = lines[0]
            assert top[-4] == " "  # nothing before the first push
            assert top[-3] == " "  # the quiet bar reaches only the middle rows
            assert top[-2] == "█"  # the loud bar, newest but one
            assert top[-1] not in (" ", "█")  # the silence after it is still falling
            # Mirrored: the bottom row is the same bar hanging down, so its loud cell is a filled one.
            assert lines[3][-2] == " "

    asyncio.run(main())


def test_a_pulsing_you_line_appears_when_speech_starts_and_becomes_the_transcript() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(MicLevel(100.0, 200.0, "listen"))  # quiet: nothing yet
        await pilot.pause(0.1)
        assert harness.messages() == []
        harness.bus.emit(MicLevel(1000.0, 200.0, "listen"))
        harness.bus.emit(MicLevel(1200.0, 200.0, "listen"))  # more speech: still one line
        await pilot.pause(0.35)  # long enough to animate
        (pending,) = harness.messages()
        assert pending.startswith("you ▸ ")
        assert any(bar in pending for bar in "▁▂▃▄▅▆▇█")
        frames = {pending}
        await pilot.pause(0.35)
        frames.add(harness.messages()[0])
        assert len(frames) == 2  # it moves
        harness.bus.emit(Heard("hello there", 120.0))
        await pilot.pause(0.1)
        assert harness.messages() == ["you ▸ hello there"]  # the same line, now with words
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_pulsing_line_goes_away_when_no_words_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui_module, "_HEARING_HOLD_S", 0.2)
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(MicLevel(1000.0, 200.0, "listen"))
        await pilot.pause(0.1)
        assert len(harness.messages()) == 1
        await pilot.pause(0.4)  # a cough: nothing was heard
        assert harness.messages() == []
        harness.bus.emit(MicLevel(1000.0, 200.0, "listen"))  # and it comes back for real speech
        await pilot.pause(0.1)
        assert len(harness.messages()) == 1
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_a_typed_line_and_barge_in_levels_do_not_start_the_pulse() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(MicLevel(5000.0, 200.0, "barge"))  # the mic hearing the speakers
        await pilot.pause(0.1)
        assert harness.messages() == []
        harness.bus.emit(Heard("typed", 0.0))
        await pilot.pause(0.1)
        assert harness.messages() == ["you ▸ typed"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_control_characters_in_untrusted_text_never_reach_the_screen() -> None:
    harness = _Harness()
    evil = "a\x1b]0;pwned\x07b\x1b[2Jc\x9bd\x08e\tf"

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard(evil, 1.0))
        harness.bus.emit(ReplyToken(evil))
        harness.bus.emit(Notice(evil))
        harness.bus.emit(LogLine("INFO", "llm", evil))
        await pilot.pause(0.1)
        shown = [*harness.messages(), *harness.log_lines()]
        assert len(shown) == 4
        for text in shown:
            assert not any(
                (ord(ch) < 32 and ch not in "\n\t") or 0x7F <= ord(ch) <= 0x9F for ch in text
            )
        assert "you ▸ a]0;pwnedb[2Jcde\tf" in shown  # the printable remainder is kept
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


@pytest.mark.parametrize(
    ("size", "shown", "hidden"),
    [
        ((120, 40), ["#log", "#latency-box", "#mic-box", "#status-box"], ["#too-small"]),
        ((80, 24), ["#log", "#latency-box", "#mic-box", "#status-box"], ["#too-small"]),
        ((60, 24), ["#mic-box", "#status-box"], ["#log", "#latency-box", "#too-small"]),
        ((100, 18), ["#log", "#mic-box", "#status-box"], ["#latency-box", "#too-small"]),
        ((40, 12), ["#too-small", "#line"], ["#body", "#conversation", "#mic-box"]),
    ],
)
def test_the_layout_gives_up_secondary_panels_as_the_terminal_shrinks(
    size: tuple[int, int], shown: list[str], hidden: list[str]
) -> None:
    harness = _Harness()

    async def main() -> None:
        async with harness.app.run_test(size=size) as pilot:
            await pilot.pause(0.1)
            for selector in shown:
                assert harness.app.query_one(selector).display, f"{selector} should show"
                assert harness.app.query_one(selector).region.height > 0
            for selector in hidden:
                widget = harness.app.query_one(selector)
                assert not widget.display or widget.region.height == 0, f"{selector} should hide"
            width = harness.app.query_one("#conversation").region.width
            assert width == 0 or width >= 36  # either hidden with its body, or room to read
            await pilot.press("ctrl+q")  # quit still works at every size

    asyncio.run(asyncio.wait_for(main(), 20))
    assert harness.app.return_value == 0


def test_an_event_that_arrives_as_the_app_closes_is_ignored() -> None:
    harness = _Harness()

    async def main() -> None:
        async with harness.app.run_test() as pilot:
            await pilot.pause(0.05)
            harness.app.exit(0)
        harness.app._apply([Heard("too late", 1.0)])  # the screen is gone by now

    asyncio.run(asyncio.wait_for(main(), 20))
