"""The full-screen app's conversation pane and chat list, driven headless.

The app never changes the chat: each key sends a ``ChatCommand`` to the session, and the
pane redraws from the ``ChatShown`` the session answers with.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import pytest
from session_fakes import FakeBackend, hello_tokens, quick_config, speak_with_fakes
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Static, TextArea
from voice_fakes import ScriptedMic, install_gate

from fish_audio_suite_voice.cli.chat_screens import ChatPicker, EditReplyScreen
from fish_audio_suite_voice.cli.history_pane import Conversation, TurnRow
from fish_audio_suite_voice.cli.tui import VoiceApp
from fish_audio_suite_voice.config.settings import SystemPrompt
from fish_audio_suite_voice.session.chat_store import ChatFile, load_chat
from fish_audio_suite_voice.session.commands import (
    ChatCommand,
    DeleteTurn,
    EditReply,
    NewChat,
    OpenChat,
    SetHidden,
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
    def __init__(self, chat_dir: Path | None = None) -> None:
        self.bus = EventBus()
        self.session = DuplexSession()
        self.live = _SpyInput()
        self.app = VoiceApp(
            self._until_quit,
            live=self.live,
            session=self.session,
            bus=self.bus,
            chat_dir=chat_dir,
        )

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
        return [str(child.content) for child in self.pane.children if isinstance(child, Static)]


def _drive(harness: _Harness, scenario: Callable[[Pilot[int]], Awaitable[None]]) -> None:
    async def main() -> None:
        async with harness.app.run_test(size=(120, 40)) as pilot:
            try:
                await scenario(pilot)
            finally:
                harness.session.request_quit()

    asyncio.run(asyncio.wait_for(main(), 20))


def _record() -> Record:
    """Three turns: the first one older than a one-turn window, the second hidden, the third edited."""
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


def test_the_pane_shows_every_turn_and_what_the_model_is_sent() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(_record()))
        await pilot.pause(0.1)
        assert harness.lines() == [
            "you ▸ one\nllm ▸ [calm] first.\nolder than the model is sent",
            "you ▸ two\nllm ▸ second.\nhidden from the model",
            "you ▸ three\nllm ▸ a better third.\nedited",
        ]
        older, hidden, sent = harness.rows()
        marks = {"older", "hidden"}
        assert [set(row.classes) & marks for row in (older, hidden, sent)] == [
            {"older"},
            {"hidden"},
            set(),
        ]
        assert harness.pane.border_subtitle == "model sees 1 of 3 turns"

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
        await pilot.press("j", "d", "u")
        assert harness.live.sent == [
            SetHidden(two, hidden=False),
            SetHidden(one, hidden=True),
            DeleteTurn(two),
            Undo(),
        ]
        assert [row.has_class("selected") for row in harness.rows()] == [False, False, True]

    _drive(harness, scenario)


def test_the_edit_keys_type_into_the_input_line_while_it_has_focus() -> None:
    harness = _Harness()

    async def scenario(pilot: Pilot[int]) -> None:
        harness.bus.emit(_shown(_record()))
        await pilot.pause(0.1)
        await pilot.press("e", "d", "h", "u", "k")
        assert harness.app.query_one(Input).value == "edhuk"
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
        assert harness.pane.border_subtitle == "model sees 1 of 2 turns"
        assert str(harness.app.screen_stack[0].query_one("#status", Static).content) == "listening"

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
        assert harness.lines()[-3:] == ["you ▸ four", "llm ▸ [calm] fourth.", "[after the reply]"]
        record.add_user_line("four")
        record.add_reply("fourth.", interrupted=True)
        harness.bus.emit(_shown(record, window=10))
        await pilot.pause(0.1)
        assert harness.lines()[-2:] == ["you ▸ four\nllm ▸ fourth.\ncut off", "[after the reply]"]
        assert harness.pane.border_subtitle == "model sees 3 of 4 turns"

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
        assert harness.lines() == ["you ▸ hello again"]

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
    config = replace(
        quick_config(), chat_dir=tmp_path, prompt=SystemPrompt("be brief", pin_seed=False)
    )
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
            await _until(pilot, lambda: "edited" in str(rows()[-1].content))
            await pilot.press("ctrl+q")

    asyncio.run(asyncio.wait_for(main(), 20))
    turns = load_chat(path).conversation.turns
    assert [turn.user for turn in turns] == ["an earlier line", "hi there"]
    reply = turns[1].reply
    assert reply is not None
    assert (reply.text, reply.edit) == ("Hello there friend.", "Hi. How was your day?")
    assert app.return_value == ExitCode.OK
