"""Changes to the chat a display asks for, as the session loop applies them between turns."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path

import pytest
from session_fakes import (
    FakeBackend,
    ScriptedTurn,
    ScriptedTurns,
    heard_line,
    quick_config,
    speak_with_fakes,
)
from voice_fakes import install_gate

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.asr.outcome import HeardLine
from fish_audio_suite_voice.config.settings import SystemPrompt, VoiceCliConfig
from fish_audio_suite_voice.session.chat_store import ChatFile, list_chats, load_chat
from fish_audio_suite_voice.session.commands import (
    ChatCommand,
    DeleteTurn,
    EditReply,
    NewChat,
    OpenChat,
    SetHidden,
    Undo,
)
from fish_audio_suite_voice.session.conversation import Conversation
from fish_audio_suite_voice.session.duplex import duplex_turns
from fish_audio_suite_voice.session.events import EVENTS, ChatShown, Event, ExitCode, Notice
from fish_audio_suite_voice.session.state import TurnInput
from fish_audio_suite_voice.tts.speaker import FishSpeaker


class _Run:
    """One session run: what each request sent the model, each ChatShown, and each notice."""

    def __init__(self, config: VoiceCliConfig) -> None:
        self.config = config
        self.asked: list[list[ChatMessage]] = []
        self.shown: list[ChatShown] = []
        self.notices: list[str] = []
        self.source = ScriptedTurns()
        self.during_reply: Callable[[], None] | None = None

    def change(self, build: Callable[[ChatShown], list[ChatCommand]]) -> ScriptedTurn:
        """A step that asks for changes, built from the record as last shown."""

        def step(_turn: TurnInput) -> HeardLine:
            self.source.commands.extend(build(self.shown[-1]))
            return HeardLine("again")

        return step

    async def _tokens(self, messages: list[ChatMessage], **_kwargs: object) -> AsyncIterator[str]:
        self.asked.append(copy.deepcopy(messages))
        if self.during_reply is not None:
            self.during_reply()
            self.during_reply = None
        yield f"reply {len(self.asked)}."

    def _keep(self, event: Event) -> None:
        if isinstance(event, ChatShown):
            self.shown.append(event)
        elif isinstance(event, Notice):
            self.notices.append(event.text)

    def run(self, monkeypatch: pytest.MonkeyPatch, *steps: ScriptedTurn) -> None:
        tts = FishSpeaker(api_key="k", voice_id="voice")
        install_gate(monkeypatch)
        speak_with_fakes(monkeypatch, tts)
        self.source = ScriptedTurns(*steps)
        stop = EVENTS.subscribe(self._keep)

        async def session() -> int:
            backend = FakeBackend(self._tokens)
            return await duplex_turns(self.config, tts, None, backend, source=self.source)

        try:
            assert asyncio.run(asyncio.wait_for(session(), 5)) == ExitCode.OK
        finally:
            stop()


def _ids(shown: ChatShown) -> list[str]:
    return [turn.id for turn in shown.turns]


def _dialogue(messages: list[ChatMessage]) -> list[str]:
    return [message["content"] for message in messages if message["role"] != "system"]


@pytest.fixture
def saving(tmp_path: Path) -> VoiceCliConfig:
    return replace(
        quick_config(), chat_dir=tmp_path, prompt=SystemPrompt("be brief", pin_seed=False)
    )


def test_an_edit_reaches_the_next_request_and_the_file(
    monkeypatch: pytest.MonkeyPatch, saving: VoiceCliConfig
) -> None:
    run = _Run(saving)
    run.run(
        monkeypatch,
        heard_line("one"),
        run.change(lambda shown: [EditReply(_ids(shown)[0], "a better reply.")]),
        heard_line("two"),
    )
    assert _dialogue(run.asked[1]) == ["one", "a better reply.", "two"]
    [chat] = list_chats(saving.chat_dir or Path())
    [first, _] = load_chat(chat.path).conversation.turns
    assert first.reply is not None
    assert (first.reply.text, first.reply.edit) == ("reply 1.", "a better reply.")


def test_hiding_and_deleting_leave_turns_out_of_the_next_request(
    monkeypatch: pytest.MonkeyPatch, saving: VoiceCliConfig
) -> None:
    run = _Run(saving)
    run.run(
        monkeypatch,
        heard_line("one"),
        heard_line("two"),
        heard_line("three"),
        run.change(
            lambda shown: [SetHidden(_ids(shown)[0], hidden=True), DeleteTurn(_ids(shown)[2])]
        ),
        heard_line("four"),
    )
    assert _dialogue(run.asked[3]) == ["two", "reply 2.", "four"]
    after = run.shown[-2]  # the record once the changes were applied, before "four"
    assert [turn.user for turn in after.turns] == ["one", "two"]
    assert after.turns[0].hidden
    assert after.sent == {after.turns[1].id}
    saved = load_chat(list_chats(saving.chat_dir or Path())[0].path).conversation
    assert [(turn.user, turn.hidden) for turn in saved.turns] == [
        ("one", True),
        ("two", False),
        ("four", False),
    ]


def test_undo_reverses_the_last_change_and_says_when_there_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Run(quick_config())
    run.run(
        monkeypatch,
        heard_line("one"),
        run.change(lambda shown: [DeleteTurn(_ids(shown)[0]), Undo(), Undo()]),
        heard_line("two"),
    )
    assert _dialogue(run.asked[1]) == ["one", "reply 1.", "two"]
    assert run.notices == ["[nothing to undo]"]


def test_a_change_asked_for_during_a_reply_waits_until_the_reply_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Run(quick_config())

    def hide_the_first() -> None:
        first = run.shown[-1].turns[0].id
        run.source.commands.append(SetHidden(first, hidden=True))

    def second_line(_turn: TurnInput) -> HeardLine:
        run.during_reply = hide_the_first
        return heard_line("two")

    run.run(monkeypatch, heard_line("one"), second_line, heard_line("three"))
    # The reply in flight was asked for, and recorded, with the first turn still in.
    assert _dialogue(run.asked[1]) == ["one", "reply 1.", "two"]
    recorded, changed = run.shown[-3], run.shown[-2]
    assert [(t.user, t.reply.text if t.reply else None, t.hidden) for t in recorded.turns] == [
        ("one", "reply 1.", False),
        ("two", "reply 2.", False),
    ]
    assert changed.turns[0].hidden
    assert _dialogue(run.asked[2]) == ["two", "reply 2.", "three"]


def test_a_stale_or_blank_change_is_refused_with_a_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run(quick_config())
    run.run(
        monkeypatch,
        heard_line("one"),
        run.change(lambda shown: [DeleteTurn("gone"), EditReply(_ids(shown)[0], "  ")]),
        heard_line("two"),
    )
    assert run.notices == [
        "[that turn is no longer in the chat]",
        "[a reply cannot be blank: hide or delete the turn instead]",
    ]
    assert _dialogue(run.asked[1]) == ["one", "reply 1.", "two"]


def _saved_chat(folder: Path, *exchanges: tuple[str, str]) -> Path:
    record = Conversation("an old prompt")
    for user, reply in exchanges:
        record.add_user_line(user)
        record.add_reply(reply, interrupted=False)
    file = ChatFile.new(folder, record)
    file.save(record)
    return file.path


def test_opening_a_saved_chat_goes_on_with_it_in_its_own_file(
    monkeypatch: pytest.MonkeyPatch, saving: VoiceCliConfig, tmp_path: Path
) -> None:
    other = _saved_chat(tmp_path / "elsewhere", ("earlier", "an old reply."))
    run = _Run(saving)
    run.run(
        monkeypatch,
        heard_line("one"),
        run.change(lambda _shown: [OpenChat(other)]),
        heard_line("two"),
    )
    assert run.asked[1][0] == {"role": "system", "content": "be brief"}
    assert _dialogue(run.asked[1]) == ["earlier", "an old reply.", "two"]
    assert [turn.user for turn in load_chat(other).conversation.turns] == ["earlier", "two"]
    assert run.shown[-1].file == other
    assert run.notices[0] == f"[opened {other.name}: 1 turn]"
    # The chat it left is still saved as it was.
    [left] = list_chats(tmp_path)
    assert left.turns == 1


def test_a_chat_that_cannot_be_read_is_reported_and_the_current_chat_goes_on(
    monkeypatch: pytest.MonkeyPatch, saving: VoiceCliConfig, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    run = _Run(saving)
    run.run(
        monkeypatch,
        heard_line("one"),
        run.change(lambda _shown: [OpenChat(broken)]),
        heard_line("two"),
    )
    assert run.notices[0].startswith(f"[cannot open that chat: cannot read the saved chat {broken}")
    assert _dialogue(run.asked[1]) == ["one", "reply 1.", "two"]


def test_a_new_chat_starts_empty_in_a_new_file(
    monkeypatch: pytest.MonkeyPatch, saving: VoiceCliConfig
) -> None:
    run = _Run(saving)
    run.run(
        monkeypatch,
        heard_line("one"),
        run.change(lambda _shown: [NewChat()]),
        heard_line("two"),
    )
    assert _dialogue(run.asked[1]) == ["two"]
    chats = list_chats(saving.chat_dir or Path())
    assert sorted(chat.title for chat in chats) == ["one", "two"]
    assert run.shown[-1].chat_id != run.shown[0].chat_id
    assert run.notices == ["[new chat]"]


def test_the_record_is_shown_at_the_start_and_after_each_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Run(quick_config())
    run.run(monkeypatch, heard_line("one"), heard_line("two"))
    assert [[turn.user for turn in shown.turns] for shown in run.shown] == [
        [],
        ["one"],
        ["one", "two"],
    ]
    assert run.shown[-1].sent == set(_ids(run.shown[-1]))


def test_the_record_says_which_turns_are_older_than_the_model_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _Run(replace(quick_config(), history_turns=2))
    run.run(monkeypatch, *(heard_line(f"line {n}") for n in range(4)))
    last = run.shown[-1]
    assert [turn.user for turn in last.turns if turn.id in last.sent] == ["line 2", "line 3"]
    assert _dialogue(run.asked[-1])[0] == "line 2"
