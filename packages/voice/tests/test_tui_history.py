"""The full-screen app's conversation pane and chat list, driven headless.

The app never changes the chat: each key sends a ``ChatCommand`` to the session, and the
pane redraws from the ``ChatShown`` the session answers with.
"""

# The user's lines start with a single angle quote mark, which is the design, not a typo.
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import pytest
from session_fakes import FakeBackend, hello_tokens, quick_config, speak_with_fakes
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Input, OptionList, Static, TextArea
from voice_fakes import ScriptedMic, install_gate

from fish_audio_suite_kit import CueStyle
from fish_audio_suite_voice.cli.chat_screens import ChatPicker, EditReplyScreen
from fish_audio_suite_voice.cli.diagnostics import DiagnosticsScreen
from fish_audio_suite_voice.cli.history_pane import Conversation, TurnRow
from fish_audio_suite_voice.cli.tui import VoiceApp
from fish_audio_suite_voice.session.chat_store import ChatFile, load_chat
from fish_audio_suite_voice.session.commands import (
    ChatCommand,
    DeleteTurn,
    EditMessage,
    EditReply,
    NewChat,
    OpenChat,
    SetHidden,
    TryAgain,
    Undo,
)
from fish_audio_suite_voice.session.conversation import Conversation as Record
from fish_audio_suite_voice.session.conversation import HistoryView
from fish_audio_suite_voice.session.duplex import duplex_turns
from fish_audio_suite_voice.session.events import (
    Bye,
    ChatShown,
    EventBus,
    ExitCode,
    Heard,
    Listening,
    Notice,
    ReplyEnd,
    ReplyToken,
)
from fish_audio_suite_voice.session.inputs import LiveInput
from fish_audio_suite_voice.session.signals import DuplexSession
from fish_audio_suite_voice.tts.speaker import FishSpeaker


def _text(widget: Widget) -> str:
    """What a row of the pane shows: each of its visible lines."""
    if isinstance(widget, TurnRow):
        return "\n".join(str(line.content) for line in widget.query(Static) if line.display)
    assert isinstance(widget, Static)
    return str(widget.content)


class _SpyInput(LiveInput):
    """A ``LiveInput`` that remembers each change the app sent and each stop."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[ChatCommand] = []
        self.stops = 0

    def send(self, command: ChatCommand) -> None:
        self.sent.append(command)
        super().send(command)

    def stop_reply(self) -> None:
        self.stops += 1
        super().stop_reply()


class _Harness:
    def __init__(
        self, chat_dir: Path | None = None, shape: Callable[[str], str] | None = None
    ) -> None:
        self.bus = EventBus()
        self.session = DuplexSession()
        self.live = _SpyInput()
        self.toasts: list[str] = []
        self.copied: list[str] = []
        self.app = VoiceApp(
            self._until_quit,
            live=self.live,
            session=self.session,
            bus=self.bus,
            chat_dir=chat_dir,
            shape=shape,
        )
        self.app.notify = lambda message, **_kwargs: self.toasts.append(str(message))

        def copy(text: str) -> None:
            self.copied.append(text)

        self.app.copy_to_clipboard = copy

    async def _until_quit(self) -> int:
        await asyncio.to_thread(self.session.quit_requested.wait, 15)
        self.bus.emit(Bye(ExitCode.OK))
        return 0

    @property
    def pane(self) -> Conversation:
        return self.app.screen_stack[0].query_one(Conversation)

    def rows(self) -> list[TurnRow]:
        return [child for child in self.pane.children if isinstance(child, TurnRow)]

    def lines(self) -> list[str]:
        return [_text(child) for child in self.pane.children]


def _drive(harness: _Harness, scenario: Callable[[Pilot[int]], Awaitable[None]]) -> None:
    async def main() -> None:
        async with harness.app.run_test(size=(120, 40)) as pilot:
            try:
                await scenario(pilot)
            finally:
                harness.session.request_quit()

    asyncio.run(asyncio.wait_for(main(), 20))


def _record() -> Record:
    """Three turns: the first with a cue, the second hidden, the third edited."""
    record = Record("be brief")
    for user, reply in (("one", "[calm] first."), ("two", "second."), ("three", "third.")):
        record.add_user_line(user)
        record.add_reply(reply, interrupted=False)
    record.set_hidden(record.turns[1].id, hidden=True)
    record.edit_reply(record.turns[2].id, "a better third.")
    return record


def _shown(record: Record, *, window: int = 1, file: Path | None = None) -> ChatShown:
    sent = frozenset(turn.id for turn in HistoryView(window).sent(record))
    return ChatShown(record.id, tuple(record.turns), sent, file)


def test_the_pane_shows_every_turn_as_it_was_heard_and_marks_hidden_and_edited_ones() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(_record()))
        await pilot.pause(0.1)
        # The first turn is older than the model is sent, which only Diagnostics says.
        assert harness.lines() == [
            "› one\nfirst.",
            "› two\nsecond.\nhidden from AI",
            "› three\na better third.\nedited",
        ]
        assert [row.has_class("hidden") for row in harness.rows()] == [False, True, False]

    _drive(harness, scenario)


def test_keys_on_the_focused_pane_send_changes_for_the_selected_turn() -> None:
    harness = _Harness()
    record = _record()
    one, two, three = (turn.id for turn in record.turns)

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("tab")
        assert harness.app.focused is harness.pane
        assert harness.pane.selected is not None
        assert harness.pane.selected.id == three  # the newest turn is selected first
        await pilot.press("k", "h")  # up to the hidden turn, and show it again
        await pilot.press("up", "h")  # the first turn, and hide it
        await pilot.press("j", "d", "z")
        assert harness.live.sent == [
            SetHidden(two, hidden=False),
            SetHidden(one, hidden=True),
            DeleteTurn(two),
            Undo(),
        ]
        assert [row.has_class("selected") for row in harness.rows()] == [False, False, True]
        assert harness.toasts == [
            "Message shown to AI again · z to undo",
            "Message hidden from AI · z to undo",
            "Message deleted · z to undo",
        ]

    _drive(harness, scenario)


def test_the_selected_turn_shows_its_keys_and_y_copies_the_reply_without_cues() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(_record()))
        await pilot.pause(0.1)
        await pilot.press("tab", "g")  # the first turn
        await pilot.pause(0.05)
        first = harness.rows()[0]
        assert _text(first).splitlines()[-1] == (
            "e Edit reply   E Edit my message   y Copy   h Hide   d Delete"
        )
        assert "e Edit reply" not in _text(harness.rows()[1])  # only on the selected turn
        await pilot.press("y")
        assert harness.copied == ["first."]
        assert harness.toasts == ["Copied"]
        await pilot.press("j")
        assert _text(harness.rows()[1]).splitlines()[-1].endswith("h Show   d Delete")
        await pilot.press("escape")  # back to typing: no keys under any turn
        await pilot.pause(0.05)
        assert all("Delete" not in _text(row) for row in harness.rows())

    _drive(harness, scenario)


def test_r_tries_the_newest_reply_again_and_is_offered_only_when_it_can() -> None:
    harness = _Harness()
    record = _record()
    three = record.turns[2].id

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("tab")
        assert "r Try again" in _text(harness.rows()[2])
        commands = harness.app.get_system_commands(harness.app.screen)
        assert next(iter(commands)).title == "Try again"
        await pilot.press("k", "r")  # an older turn: nothing to try again
        assert "r Try again" not in _text(harness.rows()[1])
        assert harness.live.sent == []
        await pilot.press("j", "r")
        assert harness.live.sent == [TryAgain(three)]
        assert harness.toasts == ["Trying again · z to undo"]
        # The session drops the reply and answers the same line: it is drawn once.
        record.try_again(three)
        harness.bus.emit(_shown(record))
        harness.bus.emit(Heard("three", 0.0))
        harness.bus.emit(ReplyToken("A third try."))
        await pilot.pause(0.1)
        assert [line.splitlines()[0] for line in harness.lines()[-2:]] == [
            "› three",
            "A third try.",
        ]
        await pilot.press("r")  # a reply is in progress
        assert harness.live.sent == [TryAgain(three)]
        assert "Try again" not in [
            c.title for c in harness.app.get_system_commands(harness.app.screen)
        ]

    _drive(harness, scenario)


def test_g_goes_to_the_first_turn_and_end_follows_the_newest_again() -> None:
    harness = _Harness()
    record = _record()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("tab", "home")
        assert harness.pane.selected is not None
        assert harness.pane.selected.id == record.turns[0].id
        record.add_user_line("four")
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        assert harness.pane.selected.id == record.turns[0].id  # reading back: it stays put
        await pilot.press("G")
        assert harness.pane.selected.id == record.turns[-1].id
        record.add_user_line("five")
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        assert harness.pane.selected.id == record.turns[-1].id  # following: the new turn

    _drive(harness, scenario)


def test_coming_back_to_the_pane_selects_the_newest_turn_again() -> None:
    harness = _Harness()
    record = _record()
    newest = record.turns[-1].id

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        harness.pane.focus()
        await pilot.press("k", "k")
        assert harness.pane.selected is not None
        assert harness.pane.selected.id != newest
        harness.app.query_one("#line").focus()
        harness.pane.focus()
        await pilot.pause(0.1)
        assert harness.pane.selected is not None
        assert harness.pane.selected.id == newest

    _drive(harness, scenario)


def test_the_edit_keys_type_into_the_input_line_while_it_has_focus() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(_record()))
        await pilot.pause(0.1)
        await pilot.press("e", "d", "h", "z", "k", "y", "g")
        assert harness.app.query_one(Input).value == "edhzkyg"
        assert harness.live.sent == []

    _drive(harness, scenario)


def test_a_reply_is_rewritten_in_a_dialog_and_saved_or_left_as_it_was() -> None:
    harness = _Harness()
    record = _record()
    three = record.turns[2].id

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("tab", "e")
        assert isinstance(harness.app.screen, EditReplyScreen)
        editor = harness.app.screen.query_one(TextArea)
        assert editor.text == "a better third."
        await pilot.press("escape")  # closes the dialog, and stops no reply
        assert not isinstance(harness.app.screen, EditReplyScreen)
        assert (harness.live.sent, harness.live.stops) == ([], 0)

        await pilot.press("e")
        editor = harness.app.screen.query_one(TextArea)
        editor.load_text("   ")
        await pilot.press("ctrl+s")  # blank: refused, and the dialog stays
        assert isinstance(harness.app.screen, EditReplyScreen)
        editor.load_text("Calmer")
        editor.move_cursor(editor.document.end)
        await pilot.press("full_stop", "ctrl+s")
        assert not isinstance(harness.app.screen, EditReplyScreen)
        assert harness.live.sent == [EditReply(three, "Calmer.")]
        assert harness.toasts[-1] == "Reply edited · z to undo"

    _drive(harness, scenario)


def test_shift_e_rewrites_the_selected_line_and_sends_it_again() -> None:
    harness = _Harness()
    record = _record()
    two = record.turns[1].id

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("tab", "k")
        assert "E Edit my message" in _text(harness.rows()[1])
        await pilot.press("E")
        editor = harness.app.screen.query_one(TextArea)
        assert editor.text == "two"
        editor.move_cursor(editor.document.end)
        await pilot.press("exclamation_mark", "ctrl+s")
        assert harness.live.sent == [EditMessage(two, "two!")]
        assert harness.toasts[-1] == "Message edited and sent again · z to undo"
        await pilot.press("E")  # the line is being answered: no more edits until it is
        assert isinstance(harness.app.screen.focused, Conversation)

    _drive(harness, scenario)


def test_the_session_s_answer_redraws_the_pane_and_events_still_arrive_under_a_dialog() -> None:
    harness = _Harness()
    record = _record()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("tab", "e")
        harness.bus.emit(Listening())
        harness.bus.emit(Notice("[a note]"))
        record.delete_turn(record.turns[0].id)
        harness.bus.emit(_shown(record))
        await pilot.pause(0.1)
        await pilot.press("escape")
        assert "[a note]" in harness.lines()
        assert [row.turn.user for row in harness.rows()] == ["two", "three"]
        state = harness.app.screen_stack[0].query_one("#state", Static)
        assert str(state.content) == "● Listening"
        await pilot.press("f12")
        assert isinstance(harness.app.screen, DiagnosticsScreen)
        session = str(harness.app.screen.query_one("#session", Static).content)
        assert "the model sees 1 of 2 turns" in session

    _drive(harness, scenario)


def test_a_recorded_turn_takes_the_place_of_the_turn_in_progress() -> None:
    harness = _Harness()
    record = _record()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record, window=10))
        harness.bus.emit(Heard("four", 0.0))
        harness.bus.emit(ReplyToken("[calm] fourth."))
        harness.bus.emit(ReplyEnd("[calm] fourth."))
        harness.bus.emit(Notice("[after the reply]"))
        await pilot.pause(0.1)
        assert harness.lines()[-3:] == ["› four", "fourth.", "[after the reply]"]
        record.add_user_line("four")
        record.add_reply("[calm] fourth.", interrupted=True)
        harness.bus.emit(_shown(record, window=10))
        await pilot.pause(0.1)
        # Cut off: the reply ends in a dim ellipsis.
        assert harness.lines()[-2:] == ["› four\nfourth.…", "[after the reply]"]

    _drive(harness, scenario)


def test_a_line_in_progress_starts_in_the_same_column_as_a_recorded_turn() -> None:
    harness = _Harness()
    record = _record()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(record, window=10))
        harness.bus.emit(Heard("four", 0.0))
        await pilot.pause(0.1)
        recorded = harness.rows()[-1]
        live = [child for child in harness.pane.children if isinstance(child, Static)][-1]
        assert str(live.content) == "› four"
        # Otherwise the line jumps sideways once its turn is recorded.
        assert live.content_region.x == recorded.content_region.x

    _drive(harness, scenario)


def test_a_turn_in_progress_is_laid_out_as_its_recorded_turn_will_be() -> None:
    harness = _Harness(shape=CueStyle.for_model("s2.1-pro", official=True, lead=False).shape)

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(Heard("hello", 0.0))
        harness.bus.emit(ReplyToken("[happy] Hi"))
        harness.bus.emit(ReplyToken(" there."))
        await pilot.pause(0.1)
        you, reply = [child for child in harness.pane.children if isinstance(child, Static)][-2:]
        assert str(reply.content) == "Hi there."
        # No blank line between the user's line and the reply, as in a recorded turn, and
        # the reply is indented under the user's line.
        assert reply.region.y == you.region.y + you.region.height
        assert reply.content_region.x == you.content_region.x + 2

    _drive(harness, scenario)


def test_another_chat_replaces_everything_in_the_pane() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(_record()))
        harness.bus.emit(Notice("[a note about the old chat]"))
        other = Record("be brief")
        other.add_user_line("hello again")
        harness.bus.emit(_shown(other))
        await pilot.pause(0.1)
        assert harness.lines() == ["› hello again"]

    _drive(harness, scenario)


def _save(folder: Path, user: str, updated: str) -> Path:
    record = Record("be brief")
    record.add_user_line(user)
    record.add_reply("ok.", interrupted=False)
    file = ChatFile.new(folder, record)
    file.save(record)
    data = json.loads(file.path.read_text(encoding="utf-8"))
    data["updated"] = updated
    file.path.write_text(json.dumps(data), encoding="utf-8")
    return file.path


def test_the_chat_list_opens_a_saved_chat_or_starts_a_new_one(tmp_path: Path) -> None:
    older = _save(tmp_path, "an older chat", "2026-01-01T10:00:00+00:00")
    newer = _save(tmp_path, "a newer chat", "2026-02-01T10:00:00+00:00")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    harness = _Harness(chat_dir=tmp_path)

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(Record("be brief"), file=older))
        await pilot.press("ctrl+o")  # from the input line: it works while typing
        assert isinstance(harness.app.screen, ChatPicker)
        await pilot.pause(0.2)
        options = harness.app.screen.query_one(OptionList)
        prompts = [str(options.get_option_at_index(i).prompt) for i in range(options.option_count)]
        assert prompts[0] == "+ new chat"
        assert prompts[1].startswith("a newer chat\nstarted ")
        assert " · updated 2026-0" in prompts[1]
        assert prompts[1].endswith(" · 1 turn")
        assert prompts[2].startswith("an older chat  open now")
        assert len(prompts) == 3  # the broken file is left out

        await pilot.press("down", "enter")
        await pilot.press("ctrl+o")
        await pilot.pause(0.2)
        await pilot.press("down", "down", "enter")  # the open chat: nothing to change
        await pilot.press("ctrl+o")
        await pilot.pause(0.2)
        await pilot.press("enter")
        await pilot.press("ctrl+o")
        await pilot.pause(0.2)
        await pilot.press("escape")
        assert not isinstance(harness.app.screen, ChatPicker)
        assert harness.live.sent == [OpenChat(newer), NewChat()]
        assert harness.live.stops == 0

    _drive(harness, scenario)


def test_the_chat_list_drops_control_characters_from_a_title(tmp_path: Path) -> None:
    # A title comes from a file, which could hold a terminal escape that clears the screen.
    _save(tmp_path, "evil\x1b[2Jtitle", "2026-01-01T10:00:00+00:00")
    harness = _Harness(chat_dir=tmp_path)

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("ctrl+o")
        await pilot.pause(0.2)
        options = harness.app.screen.query_one(OptionList)
        title = str(options.get_option_at_index(1).prompt).split("\n")[0]
        assert title == "evil[2Jtitle"
        await pilot.press("escape")

    _drive(harness, scenario)


def test_with_no_chat_folder_the_list_offers_only_a_new_chat() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        await pilot.press("ctrl+o")
        await pilot.pause(0.1)
        assert harness.app.screen.query_one(OptionList).option_count == 1
        await pilot.press("enter")
        assert harness.live.sent == [NewChat()]

    _drive(harness, scenario)


async def _until(pilot: Pilot[int], ready: Callable[[], bool]) -> None:
    for _ in range(200):
        if ready():
            return
        await pilot.pause(0.02)
    raise AssertionError("the app did not get there in time")


def test_a_resumed_chat_is_edited_in_the_app_and_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _save(tmp_path, "an earlier line", "2026-01-01T10:00:00+00:00")
    config = replace(quick_config(), chat_dir=tmp_path, prompt="be brief")
    monkeypatch.setattr(
        "fish_audio_suite_voice.session.duplex.SessionMic", lambda *_a, **_k: ScriptedMic()
    )
    install_gate(monkeypatch)
    tts = FishSpeaker(api_key="k", voice_id="voice")
    speak_with_fakes(monkeypatch, tts)
    session = DuplexSession()
    live = LiveInput()
    live.mute()  # typed lines only: the scripted mic is never read

    async def runner() -> int:
        return await duplex_turns(
            config,
            tts,
            None,
            FakeBackend(hello_tokens),
            session,
            source=live,
            resumed=load_chat(path),
        )

    app = VoiceApp(runner, live=live, session=session, chat_dir=tmp_path)

    def rows() -> list[TurnRow]:
        pane = app.screen_stack[0].query_one(Conversation)
        return [child for child in pane.children if isinstance(child, TurnRow)]

    async def main() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await _until(pilot, lambda: len(rows()) == 1)
            await pilot.press(*"hi there", "enter")
            await _until(pilot, lambda: len(rows()) == 2)
            await pilot.press("tab", "e")
            editor = app.screen.query_one(TextArea)
            assert editor.text == "Hello there friend."
            editor.load_text("Hi. How was your day?")
            await pilot.press("ctrl+s")
            await _until(pilot, lambda: "edited" in _text(rows()[-1]))
            await pilot.press("ctrl+q")

    asyncio.run(asyncio.wait_for(main(), 20))
    turns = load_chat(path).conversation.turns
    assert [turn.user for turn in turns] == ["an earlier line", "hi there"]
    reply = turns[1].reply
    assert reply is not None
    assert (reply.text, reply.edit) == ("Hello there friend.", "Hi. How was your day?")
    assert app.return_value == ExitCode.OK
