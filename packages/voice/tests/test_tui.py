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
from textual.color import Color
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
from fish_audio_suite_voice.tui import Conversation, Timings, VoiceApp, Waveform


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


def _dots(char: str) -> int:
    """How many dots a Braille character raises."""
    return (ord(char) - 0x2800).bit_count()


def _timings_text(harness: _Harness) -> str:
    panel = harness.app.query_one(Timings)
    return "\n".join(strip.text for strip in panel.render_lines(Region(0, 0, panel.size.width, 3)))


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
        assert "0.90s wait" in _timings_text(harness)
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
        assert max(_dots(ch) for line in lines for ch in line) >= 4  # the loud bar is drawn
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


def test_the_waveform_scrolls_newest_right_and_decays_in_dots() -> None:
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
            for level in (
                0.0,
                1.0,
                0.0,
            ):  # quiet, loud, then silence that falls instead of snapping
                wave.push(level, 0.3)
            await pilot.pause()
            lines = [s.text for s in wave.render_lines(Region(0, 0, width, 4))]
            assert all(len(line) == width for line in lines)
            assert all(0x2800 <= ord(ch) <= 0x28FF for line in lines for ch in line)  # all dots
            top = lines[0]
            assert top[0] == "\u2800"  # nothing drawn that high before any speech
            assert _dots(top[-1]) > 0  # the newest characters hold the loud bar and its fall
            assert _dots(top[-1]) > _dots(top[-3])
            # Mirrored: each line has as many dots as the one at the same distance below.
            for row in range(2):
                for above, below in zip(lines[row], lines[3 - row], strict=True):
                    assert _dots(above) == _dots(below)

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
        assert any(0x2800 < ord(ch) <= 0x28FF for ch in pending)  # dots, not blocks
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


def test_the_pulse_holds_still_when_animations_are_turned_off() -> None:
    harness = _Harness()
    harness.app.animation_level = "none"  # what TEXTUAL_ANIMATIONS=none sets

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(MicLevel(1000.0, 200.0, "listen"))
        await pilot.pause(0.1)
        (first,) = harness.messages()
        await pilot.pause(0.4)
        assert harness.messages() == [first]  # still there, and not moving
        assert first.startswith("you ▸ ")
        harness.bus.emit(Heard("hi", 1.0))
        await pilot.pause(0.1)
        assert harness.messages() == ["you ▸ hi"]
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_waveform_draws_dots_in_the_foreground_only_with_no_cell_background() -> None:
    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield Waveform()

    async def main() -> None:
        host = Host()
        async with host.run_test(size=(40, 10)) as pilot:
            wave = host.query_one(Waveform)
            for level in (0.0, 0.8, 0.05, 0.9):
                wave.push(level, 0.3)
            await pilot.pause()
            screen = Color.parse(host.get_css_variables()["background"]).rgb
            strips = wave.render_lines(Region(0, 0, wave.size.width, 4))
            backgrounds = {
                tuple(segment.style.bgcolor.get_truecolor())
                for strip in strips
                for segment in strip
                if segment.style is not None and segment.style.bgcolor is not None
            }
            assert backgrounds <= {screen}  # dots sit on the screen, nothing is filled behind them

    asyncio.run(main())


def test_the_input_line_has_the_same_rounded_border_as_the_panels() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        line = harness.app.query_one("#line")
        assert line.has_focus
        top = line.render_lines(Region(0, 0, line.region.width, line.region.height))[0].text
        assert top.startswith("╭")
        assert top.endswith("╮")  # no half-block edges of the default "tall" border
        assert line.region.height == 3
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_the_log_shows_scrollbars_only_when_there_is_something_to_scroll() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        log = harness.app.query_one("#log", Log)
        harness.bus.emit(LogLine("INFO", "llm", "one short line"))
        await pilot.pause(0.1)
        assert not log.show_vertical_scrollbar
        assert not log.show_horizontal_scrollbar
        for number in range(200):
            harness.bus.emit(LogLine("INFO", "llm", f"line {number}"))
        await pilot.pause(0.2)
        assert log.show_vertical_scrollbar
        assert log.scrollbar_size_vertical == 1
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_bordered_widgets_share_the_screens_background_so_none_shows_past_its_border() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        screen = Color.parse(harness.app.get_css_variables()["background"]).rgb
        for selector in ("#log", "#line", "Footer"):
            assert harness.app.query_one(selector).styles.background.rgb == screen, selector
        assert harness.app.query_one(Waveform).styles.background.a == 0  # draws on the screen
        await pilot.press("ctrl+q")

    assert _drive(harness, scenario) == 0


def test_focus_does_not_tint_past_the_border_of_the_log_conversation_or_input() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        for selector in ("#log", "#conversation", "#line"):
            widget = harness.app.query_one(selector)
            widget.focus()
            await pilot.pause(0.05)
            assert widget.has_focus, selector
            assert widget.styles.background_tint.a == 0, f"{selector} is tinted when focused"
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
