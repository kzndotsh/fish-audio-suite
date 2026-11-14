"""Saved chats: the file a session writes after each turn, and ``--resume`` reading it back."""

from __future__ import annotations

import asyncio
import json
import re
import stat
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import cast

import pytest
from session_fakes import (
    FakeBackend,
    ScriptedTurns,
    heard_line,
    hello_tokens,
    quick_config,
    speak_with_fakes,
)
from voice_fakes import install_gate

from fish_audio_suite_voice.cli.main import main
from fish_audio_suite_voice.config.settings import (
    DEFAULT_CHAT_DIR,
    SystemPrompt,
    VoiceCliConfig,
    load_config,
)
from fish_audio_suite_voice.session import chat_store
from fish_audio_suite_voice.session.chat_store import (
    ChatFile,
    ChatLoadError,
    SavedChat,
    find_chat,
    list_chats,
    load_chat,
)
from fish_audio_suite_voice.session.conversation import Conversation
from fish_audio_suite_voice.session.duplex import duplex_turns
from fish_audio_suite_voice.session.events import EVENTS, Event, ExitCode, Notice
from fish_audio_suite_voice.tts.speaker import FishSpeaker


def _talk(conversation: Conversation, *exchanges: tuple[str, str]) -> Conversation:
    for user, reply in exchanges:
        conversation.add_user_line(user)
        conversation.add_reply(reply, interrupted=False)
    return conversation


def _saved(folder: Path, record_id: str, *exchanges: tuple[str, str], updated: str = "") -> Path:
    conversation = _talk(Conversation("be brief", id=record_id), *exchanges)
    file = ChatFile.new(folder, conversation)
    file.save(conversation)
    if updated:
        data = json.loads(file.path.read_text(encoding="utf-8"))
        file.path.write_text(json.dumps({**data, "updated": updated}), encoding="utf-8")
    return file.path


def test_a_saved_chat_reads_back_as_it_was_written(tmp_path: Path) -> None:
    conversation = _talk(Conversation("be brief"), ("café?", "[warm] Oui."), ("and you", "Fine."))
    conversation.edit_reply(conversation.turns[0].id, "Yes.")
    conversation.set_hidden(conversation.turns[1].id, hidden=True)
    file = ChatFile.new(tmp_path / "chats", conversation)
    file.save(conversation)
    back = load_chat(file.path)
    assert back.conversation == conversation
    assert back.file == file
    assert re.fullmatch(rf"\d{{8}}-\d{{6}}-{conversation.id[:8]}\.json", file.path.name)
    text = file.path.read_text(encoding="utf-8")
    assert "café?" in text  # readable by hand: not escaped, and indented
    assert '\n  "turns": [' in text


def test_a_failed_write_leaves_the_last_good_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = _talk(Conversation("be brief"), ("one", "First."))
    file = ChatFile.new(tmp_path, conversation)
    file.save(conversation)
    before = file.path.read_bytes()

    def disk_full(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(chat_store.os, "fsync", disk_full)
    _talk(conversation, ("two", "Second."))
    with pytest.raises(OSError, match="No space"):
        file.save(conversation)
    assert file.path.read_bytes() == before
    assert [path.name for path in tmp_path.iterdir()] == [file.path.name]  # no temp file left


def test_the_folder_and_the_file_are_private(tmp_path: Path) -> None:
    path = _saved(tmp_path / "chats", "a1", ("hi", "Hello."))
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_list_is_newest_first_with_a_title_and_a_count(tmp_path: Path) -> None:
    older = _saved(tmp_path, "a1", ("first chat", "Hi."), updated="2026-01-01T10:00:00+00:00")
    long_line = "tell me " + "a very long story " * 10
    newer = _saved(
        tmp_path, "b2", (long_line, "Ok."), ("more", "Sure."), updated="2026-02-01T10:00:00+00:00"
    )
    chats = list_chats(tmp_path)
    assert [chat.path for chat in chats] == [newer, older]
    assert [chat.turns for chat in chats] == [2, 1]
    assert chats[1].title == "first chat"
    assert chats[0].title.startswith("tell me a very long story")
    assert chats[0].title.endswith("…")
    assert len(chats[0].title) == 60


def test_resume_with_no_chat_named_takes_the_most_recently_updated(tmp_path: Path) -> None:
    # Both started in the same second, so the order comes from when each was last saved.
    _saved(tmp_path, "a1", ("hi", "Hello."), updated="2026-02-01T10:00:00+00:00")
    _saved(tmp_path, "b2", ("yo", "Hey."), updated="2026-01-01T10:00:00+00:00")
    assert load_chat(find_chat(tmp_path, "")).conversation.id == "a1"


@pytest.mark.parametrize("newer", ["a1", "b2"])
def test_chats_saved_in_one_second_still_list_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newer: str
) -> None:
    # Switching chats in the full-screen app saves both within a second.
    early, late = (datetime(2026, 1, 1, 10, 0, 0, ms * 1000, UTC) for ms in (100, 900))
    times = iter([early, late, early, early] if newer == "a1" else [early, early, early, late])

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Clock:
            return cast("Clock", next(times))

    monkeypatch.setattr(chat_store, "datetime", Clock)
    _saved(tmp_path, "a1", ("hi", "Hello."))
    _saved(tmp_path, "b2", ("yo", "Hey."))
    assert list_chats(tmp_path)[0].id == newer
    assert load_chat(find_chat(tmp_path, "")).conversation.id == newer


def test_a_lone_surrogate_is_saved_and_read_back(tmp_path: Path) -> None:
    # Half of a split emoji, as a JSON escape decoded alone gives, cannot be UTF-8 encoded.
    conversation = _talk(Conversation("be brief"), ("hi \ud83d", "Hello."))
    file = ChatFile.new(tmp_path, conversation)
    file.save(conversation)
    assert load_chat(file.path).conversation == conversation


def test_resume_takes_an_id_prefix_a_file_name_or_a_path(tmp_path: Path) -> None:
    first = _saved(tmp_path, "abcd1111", ("hi", "Hello."))
    second = _saved(tmp_path, "abcd2222", ("yo", "Hey."))
    assert find_chat(tmp_path, "abcd2") == second
    assert find_chat(tmp_path, "ABCD1") == first
    assert find_chat(tmp_path, first.name) == first
    assert find_chat(None, str(second)) == second


def test_resume_refuses_an_ambiguous_or_unknown_prefix(tmp_path: Path) -> None:
    _saved(tmp_path, "abcd1111", ("hi", "Hello."))
    _saved(tmp_path, "abcd2222", ("yo", "Hey."))
    with pytest.raises(ChatLoadError, match="2 saved chats have an id starting with 'abcd'"):
        find_chat(tmp_path, "abcd")
    with pytest.raises(ChatLoadError, match="no saved chat"):
        find_chat(tmp_path, "ffff")
    with pytest.raises(ChatLoadError, match="no saved chats"):
        find_chat(tmp_path / "empty", "")


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("{not json", "cannot read"),
        ("[" * 100_000, "cannot read"),  # nested past the parser's recursion limit
        ("[]", "no JSON object"),
        (json.dumps({"version": 2, "id": "a"}), "newer fish-voice"),
        (json.dumps({"version": 1, "id": "a", "system": "s", "seed": [], "turns": 3}), "turns is"),
        (
            json.dumps({"version": 1, "id": "a", "system": "s", "seed": [], "turns": []}),
            "started is",
        ),
    ],
)
def test_a_file_that_is_not_a_chat_is_refused_by_name(
    tmp_path: Path, body: str, reason: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ChatLoadError, match=reason) as caught:
        load_chat(path)
    assert str(path) in str(caught.value)
    assert list_chats(tmp_path) == []  # the list leaves it out


def _no_log(**_kwargs: object) -> None:
    return None


def _main_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref: str
) -> tuple[int, list[SavedChat | None]]:
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("FISH_VOICE_CHAT_DIR", str(tmp_path / "chats"))
    monkeypatch.setattr("fish_audio_suite_voice.cli.main.configure_voice_logging", _no_log)
    got: list[SavedChat | None] = []

    async def run(_config: object, *, resumed: SavedChat | None = None) -> int:
        got.append(resumed)
        return ExitCode.OK

    monkeypatch.setattr("fish_audio_suite_voice.cli.main.run_loop", run)
    return main(["--env-file", str(empty), "--resume", *([ref] if ref else [])]), got


def test_the_resume_flag_hands_the_chat_to_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _saved(tmp_path / "chats", "abcd1111", ("hi", "Hello."))
    code, got = _main_resuming(tmp_path, monkeypatch, "")
    assert code == ExitCode.OK
    assert [chat.conversation.id for chat in got if chat is not None] == ["abcd1111"]


def test_a_chat_that_cannot_be_read_stops_the_run_before_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code, got = _main_resuming(tmp_path, monkeypatch, str(bad))
    assert code == ExitCode.FATAL
    assert got == []
    err = capsys.readouterr().err
    assert f"fish-voice --resume: cannot read the saved chat {bad}" in err
    assert "Traceback" not in err


def test_the_chat_folder_follows_the_setting_then_xdg_then_the_home_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_CHAT_DIR", "/srv/chats")
    monkeypatch.setenv("XDG_DATA_HOME", "/data")
    assert load_config().chat_dir == Path("/srv/chats")
    monkeypatch.delenv("FISH_VOICE_CHAT_DIR")
    assert load_config().chat_dir == Path("/data/fish-audio-suite/chats")
    monkeypatch.setenv("XDG_DATA_HOME", "relative/data")  # the XDG spec ignores a relative one
    assert load_config().chat_dir == DEFAULT_CHAT_DIR
    monkeypatch.delenv("XDG_DATA_HOME")
    assert load_config().chat_dir == DEFAULT_CHAT_DIR


def test_chats_are_saved_unless_the_setting_turns_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_SAVE_CHATS", raising=False)
    assert load_config().save_chats
    monkeypatch.setenv("FISH_VOICE_SAVE_CHATS", "0")
    assert not load_config().save_chats


@pytest.fixture
def notices() -> Iterator[list[str]]:
    seen: list[str] = []

    def keep(event: Event) -> None:
        if isinstance(event, Notice):
            seen.append(event.text)

    stop = EVENTS.subscribe(keep)
    yield seen
    stop()


def _session(
    monkeypatch: pytest.MonkeyPatch,
    config: VoiceCliConfig,
    *lines: str,
    resumed: SavedChat | None = None,
) -> list[str]:
    """Run a session that hears ``lines`` and answers each, and return what was spoken."""
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts)
    source = ScriptedTurns(*(heard_line(line) for line in lines))

    async def run() -> int:
        backend = FakeBackend(hello_tokens)
        return await duplex_turns(config, tts, None, backend, source=source, resumed=resumed)

    assert asyncio.run(asyncio.wait_for(run(), 5)) == ExitCode.OK
    return said


def _saving(folder: Path, prompt: str = "be brief") -> VoiceCliConfig:
    return replace(quick_config(), chat_dir=folder, prompt=SystemPrompt(prompt, pin_seed=False))


def test_a_session_saves_each_turn_it_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(monkeypatch, _saving(tmp_path), "hi there", "and then?")
    [chat] = list_chats(tmp_path)
    turns = load_chat(chat.path).conversation.turns
    assert [turn.user for turn in turns] == ["hi there", "and then?"]
    assert [turn.reply.text if turn.reply else None for turn in turns] == [
        "Hello there friend."
    ] * 2


def test_a_session_with_no_turns_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(monkeypatch, _saving(tmp_path / "chats"))
    assert not (tmp_path / "chats").exists()


def test_saving_off_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(monkeypatch, replace(_saving(tmp_path / "chats"), save_chats=False), "hi there")
    assert not (tmp_path / "chats").exists()


def test_a_failed_save_warns_once_and_the_session_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_folder = tmp_path / "taken"
    not_a_folder.write_text("", encoding="utf-8")
    said = _session(monkeypatch, _saving(not_a_folder), "hi there", "and then?", "one more")
    assert len(said) == 3
    assert capsys.readouterr().err.count("cannot save the chat") == 1


def test_a_resumed_session_goes_on_in_the_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notices: list[str]
) -> None:
    _session(monkeypatch, _saving(tmp_path), "hi there")
    [first] = list_chats(tmp_path)
    resumed = load_chat(first.path)
    _session(monkeypatch, _saving(tmp_path), "remember me?", resumed=resumed)
    [chat] = list_chats(tmp_path)
    assert (chat.path, chat.id) == (first.path, first.id)
    users = [turn.user for turn in load_chat(chat.path).conversation.turns]
    assert users == ["hi there", "remember me?"]
    assert notices == [f"[resumed {first.path.name}: 1 turn]"]


def test_a_resumed_chat_uses_the_prompt_set_now_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notices: list[str]
) -> None:
    _session(monkeypatch, _saving(tmp_path, "be brief"), "hi there")
    resumed = load_chat(list_chats(tmp_path)[0].path)
    _session(monkeypatch, _saving(tmp_path, "be kind"), "again", resumed=resumed)
    assert load_chat(resumed.file.path).conversation.system == "be kind"
    assert any("another system prompt" in text for text in notices)
