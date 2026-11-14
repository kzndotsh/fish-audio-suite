"""The conversation record and the messages a request sends the model from it."""

from __future__ import annotations

import itertools
import json

import pytest

from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT, ChatMessage
from fish_audio_suite_voice.config.settings import DEFAULT_HISTORY_TURNS, SystemPrompt
from fish_audio_suite_voice.session.conversation import Conversation, HistoryView

HISTORY_TURNS = DEFAULT_HISTORY_TURNS


def _talk(conversation: Conversation, *pairs: tuple[str, str | None]) -> Conversation:
    for user, reply in pairs:
        conversation.add_user_line(user)
        if reply is not None:
            conversation.add_reply(reply, interrupted=False)
    return conversation


def _chat(turns: int, cap: int = HISTORY_TURNS) -> tuple[Conversation, HistoryView]:
    conversation = Conversation("be brief")
    _talk(conversation, *((f"u{i}", f"a{i}") for i in range(turns)))
    return conversation, HistoryView(cap)


def _contents(messages: list[ChatMessage]) -> list[str]:
    return [m["content"] for m in messages]


def test_the_view_keeps_the_system_prompt_and_drops_the_oldest_turns() -> None:
    conversation, view = _chat(HISTORY_TURNS + 1)
    sent = view.messages(conversation)
    assert sent[0] == {"role": "system", "content": "be brief"}
    kept = _contents(sent[1::2])
    assert kept == [f"u{i}" for i in range(HISTORY_TURNS + 1 - len(kept), HISTORY_TURNS + 1)]
    assert sent[-1]["content"] == f"a{HISTORY_TURNS}"
    assert len(sent) <= 1 + HISTORY_TURNS * 2


def test_the_record_keeps_every_turn_the_view_drops() -> None:
    conversation, view = _chat(HISTORY_TURNS * 2)
    assert len(conversation.turns) == HISTORY_TURNS * 2
    assert len(view.messages(conversation)) <= 1 + HISTORY_TURNS * 2


def test_the_view_drops_whole_turns_so_roles_stay_paired() -> None:
    conversation, view = _chat(HISTORY_TURNS)
    _talk(conversation, ("newest", "answer"))
    sent = view.messages(conversation)
    roles = [m["role"] for m in sent[1:]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)
    first = sent[1]["content"]
    assert sent[2]["content"] == "a" + first.removeprefix("u")
    assert sent[-2:] == [
        {"role": "user", "content": "newest"},
        {"role": "assistant", "content": "answer"},
    ]


def test_the_view_drops_turns_in_blocks_so_its_start_stays_the_same_between_drops() -> None:
    conversation, view = _chat(0)
    floor = HISTORY_TURNS - HISTORY_TURNS // 4
    starts: list[list[ChatMessage]] = []
    for i in range(HISTORY_TURNS * 3):
        conversation.add_user_line(f"u{i}")
        sent = view.messages(conversation)
        turns = len(sent) // 2  # the newest line counts as a turn
        assert turns <= HISTORY_TURNS
        if i >= HISTORY_TURNS:
            assert turns >= floor
        starts.append(sent[:3])
        conversation.add_reply(f"a{i}", interrupted=False)
    # A provider caches the start of the prompt. Dropping one turn at a time changed it
    # on every turn once the cap was reached. Now it changes once per block.
    changes = sum(1 for before, after in itertools.pairwise(starts) if before != after)
    assert changes <= 2 * HISTORY_TURNS // (HISTORY_TURNS // 4)


def test_a_small_cap_still_drops_one_turn_at_a_time() -> None:
    conversation, _ = _chat(4)
    assert _contents(HistoryView(3).messages(conversation))[1:] == [
        "u1",
        "a1",
        "u2",
        "a2",
        "u3",
        "a3",
    ]


def test_the_cap_follows_the_configured_turns() -> None:
    conversation, _ = _chat(5)
    conversation.add_user_line("newest")
    # The oldest user and assistant leave together, so roles stay paired.
    assert _contents(HistoryView(2).messages(conversation)) == ["be brief", "u4", "a4", "newest"]


def test_an_unanswered_user_line_is_merged_with_the_next_one() -> None:
    # A turn the model never answered (a 429, a refusal from the provider) left its
    # user line last. Two user messages in a row reached the model before.
    conversation = Conversation("s")
    conversation.add_user_line("Swear on it.")
    conversation.add_user_line("Swear on it.")
    view = HistoryView(HISTORY_TURNS)
    assert view.messages(conversation) == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Swear on it."},
    ]
    conversation.add_user_line("Are you there?")
    assert view.messages(conversation)[-1] == {
        "role": "user",
        "content": "Swear on it.\nAre you there?",
    }
    assert len(conversation.turns) == 1


def test_a_new_line_after_a_reply_starts_a_turn() -> None:
    conversation = _talk(Conversation("s"), ("hi", "hello"), ("hi", None))
    roles = [m["role"] for m in HistoryView(HISTORY_TURNS).messages(conversation)]
    assert roles == ["system", "user", "assistant", "user"]


def test_a_long_run_of_unanswered_lines_keeps_only_the_latest() -> None:
    conversation = Conversation("s")
    for i in range(10):
        conversation.add_user_line(f"line {i}")
    assert conversation.last_user() == "line 6\nline 7\nline 8\nline 9"


def test_the_record_keeps_what_was_heard_of_a_reply() -> None:
    conversation = Conversation("sys")
    conversation.add_user_line("tell me a story")
    conversation.add_reply("", interrupted=True)  # nothing heard, nothing kept
    conversation.add_reply("tell me a story", interrupted=False)  # an echo is not kept
    assert conversation.last_user() == "tell me a story"
    conversation.add_reply("Once upon", interrupted=True)
    conversation.add_reply("a time.", interrupted=False)  # the rest after a false barge-in
    assert HistoryView(HISTORY_TURNS).messages(conversation)[-1] == {
        "role": "assistant",
        "content": "Once upon a time.",
    }
    reply = conversation.turns[-1].reply
    assert reply is not None
    assert not reply.interrupted


def test_an_interrupted_reply_is_marked() -> None:
    conversation = _talk(Conversation("s"), ("hi", None))
    conversation.add_reply("Hello", interrupted=True)
    reply = conversation.turns[-1].reply
    assert reply is not None
    assert reply.interrupted


def _opening(text: str) -> Conversation:
    """Open a conversation for a prompt written out in full, as the config decides it."""
    prompt = SystemPrompt.written(text)
    return Conversation.opening(prompt.text, seed=prompt.pin_seed)


def test_the_default_prompt_starts_with_a_pinned_multi_cue_exchange() -> None:
    sent = HistoryView(HISTORY_TURNS).messages(_opening(DEFAULT_SYSTEM_PROMPT))
    assert [m["role"] for m in sent] == ["system", "user", "assistant"]
    assert sent[2]["content"].count("[") >= 2


def test_a_custom_prompt_gets_no_seed_exchange() -> None:
    sent = HistoryView(HISTORY_TURNS).messages(_opening("You are a pirate."))
    assert sent == [{"role": "system", "content": "You are a pirate."}]


def test_dropping_turns_never_drops_the_pinned_seed() -> None:
    conversation = _opening(DEFAULT_SYSTEM_PROMPT)
    view = HistoryView(3)
    opening = view.messages(conversation)
    for index in range(30):
        _talk(conversation, (f"question {index}", f"[calm] answer {index}"))
    sent = view.messages(conversation)
    assert sent[: len(opening)] == opening
    assert len(sent) == len(opening) + 3 * 2
    assert sent[-2]["content"] == "question 29"


def test_a_hidden_turn_stays_in_the_record_but_is_not_sent() -> None:
    conversation = _talk(Conversation("s"), ("one", "first"), ("two", "second"), ("three", "third"))
    view = HistoryView(HISTORY_TURNS)
    conversation.set_hidden(conversation.turns[1].id, hidden=True)
    assert _contents(view.messages(conversation)) == ["s", "one", "first", "three", "third"]
    assert len(conversation.turns) == 3
    conversation.set_hidden(conversation.turns[1].id, hidden=False)
    assert "second" in _contents(view.messages(conversation))


def test_hidden_turns_do_not_count_against_the_cap() -> None:
    conversation, _ = _chat(4)
    conversation.set_hidden(conversation.turns[3].id, hidden=True)
    assert _contents(HistoryView(2).messages(conversation)) == ["be brief", "u1", "a1", "u2", "a2"]


def test_a_new_line_after_a_hidden_unanswered_turn_starts_its_own_turn() -> None:
    conversation = _talk(Conversation("s"), ("hello?", None))
    conversation.set_hidden(conversation.turns[0].id, hidden=True)
    conversation.add_user_line("anyone?")
    assert _contents(HistoryView(HISTORY_TURNS).messages(conversation)) == ["s", "anyone?"]


def test_roles_alternate_when_an_unanswered_turn_is_shown_again() -> None:
    conversation = _talk(Conversation("s"), ("hello?", None))
    conversation.set_hidden(conversation.turns[0].id, hidden=True)
    _talk(conversation, ("anyone?", "Yes, I'm here."))
    conversation.undo()
    assert HistoryView(HISTORY_TURNS).messages(conversation)[1:] == [
        {"role": "user", "content": "hello?\nanyone?"},
        {"role": "assistant", "content": "Yes, I'm here."},
    ]


def test_an_edit_replaces_the_reply_the_model_is_sent_and_undo_restores_it() -> None:
    conversation = _talk(Conversation("s"), ("hi", "[calm] Hello. Nice day."))
    view = HistoryView(HISTORY_TURNS)
    turn = conversation.turns[0]
    conversation.edit_reply(turn.id, "Hello.")
    assert view.messages(conversation)[-1]["content"] == "Hello."
    reply = conversation.turns[0].reply
    assert reply is not None
    assert reply.text == "[calm] Hello. Nice day."  # what the model wrote is kept
    assert conversation.undo()
    assert view.messages(conversation)[-1]["content"] == "[calm] Hello. Nice day."


def test_a_blank_edit_or_an_edit_of_no_reply_is_refused() -> None:
    conversation = _talk(Conversation("s"), ("hi", "Hello."), ("there?", None))
    with pytest.raises(ValueError, match="no reply to edit"):
        conversation.edit_reply(conversation.turns[0].id, "  ")
    with pytest.raises(ValueError, match="no reply to edit"):
        conversation.edit_reply(conversation.turns[1].id, "Yes.")
    with pytest.raises(KeyError):
        conversation.edit_reply("missing", "Yes.")


def test_a_deleted_turn_comes_back_in_its_place_on_undo() -> None:
    conversation = _talk(Conversation("s"), ("one", "first"), ("two", "second"), ("three", "third"))
    before = list(conversation.turns)
    conversation.delete_turn(before[1].id)
    assert [turn.user for turn in conversation.turns] == ["one", "three"]
    assert "second" not in _contents(HistoryView(HISTORY_TURNS).messages(conversation))
    assert conversation.undo()
    assert conversation.turns == before


def test_undo_reverses_changes_newest_first_and_then_has_nothing_left() -> None:
    conversation = _talk(Conversation("s"), ("one", "first"), ("two", "second"))
    before = list(conversation.turns)
    first, second = before
    conversation.edit_reply(first.id, "edited")
    conversation.set_hidden(second.id, hidden=True)
    conversation.delete_turn(first.id)
    assert conversation.undo()
    assert conversation.undo()
    assert conversation.turns[1] == second
    assert conversation.undo()
    assert conversation.turns == before
    assert not conversation.undo()


def test_new_turns_keep_coming_in_after_an_edit_is_undone() -> None:
    conversation = _talk(Conversation("s"), ("one", "first"))
    conversation.edit_reply(conversation.turns[0].id, "edited")
    _talk(conversation, ("two", "second"))
    conversation.undo()
    assert _contents(HistoryView(HISTORY_TURNS).messages(conversation)) == [
        "s",
        "one",
        "first",
        "two",
        "second",
    ]


def test_turn_ids_are_unique_and_survive_edits() -> None:
    conversation = _talk(Conversation("s"), ("one", "first"), ("two", "second"))
    ids = [turn.id for turn in conversation.turns]
    assert len(set(ids)) == 2
    conversation.edit_reply(ids[0], "edited")
    conversation.set_hidden(ids[1], hidden=True)
    assert [turn.id for turn in conversation.turns] == ids


def test_the_record_round_trips_through_json() -> None:
    conversation = _talk(_opening(DEFAULT_SYSTEM_PROMPT), ("one", "[calm] first"), ("two", None))
    conversation.add_reply("Half", interrupted=True)
    conversation.edit_reply(conversation.turns[0].id, "first, edited")
    conversation.set_hidden(conversation.turns[1].id, hidden=True)
    text = json.dumps(conversation.to_json())
    back = Conversation.from_json(json.loads(text))
    assert back == conversation
    view = HistoryView(HISTORY_TURNS)
    assert view.messages(back) == view.messages(conversation)


def test_json_names_its_schema_version() -> None:
    data = Conversation("s").to_json()
    assert data["version"] == 1
    with pytest.raises(ValueError, match="version"):
        Conversation.from_json({**data, "version": 2})


@pytest.mark.parametrize(
    "turn",
    [
        {"id": "a", "user": "hi", "reply": None},  # no hidden flag
        {"id": "a", "user": 3, "reply": None, "hidden": False},
        {"id": "a", "user": "hi", "reply": {"text": "x", "interrupted": "no"}, "hidden": False},
        "not a turn",
    ],
)
def test_json_with_a_wrong_field_is_refused(turn: object) -> None:
    data = {**Conversation("s").to_json(), "turns": [turn]}
    with pytest.raises(TypeError):
        Conversation.from_json(data)


def test_json_with_two_turns_of_one_id_is_refused() -> None:
    turn = {"id": "a", "user": "hi", "reply": None, "hidden": False}
    with pytest.raises(ValueError, match="share an id"):
        Conversation.from_json({**Conversation("s").to_json(), "turns": [turn, turn]})


def test_older_replies_lose_their_cues_when_the_setting_says_so() -> None:
    conversation = _talk(
        Conversation("s"),
        ("one", "[calm] First. [happy] Yes."),
        ("two", "[calm] Second."),
        ("three", "[sad] Third."),
    )
    sent = HistoryView(HISTORY_TURNS, cue_turns=1).messages(conversation)
    assert _contents(sent)[2::2] == ["First. Yes.", "Second.", "[sad] Third."]
    assert _contents(HistoryView(HISTORY_TURNS, cue_turns=0).messages(conversation))[2::2] == [
        "First. Yes.",
        "Second.",
        "Third.",
    ]
    # The record keeps them: only what the model is sent changes.
    assert [turn.reply.text for turn in conversation.turns if turn.reply] == [
        "[calm] First. [happy] Yes.",
        "[calm] Second.",
        "[sad] Third.",
    ]


def test_every_reply_keeps_its_cues_by_default_and_the_seed_always_does() -> None:
    conversation = _talk(_opening(DEFAULT_SYSTEM_PROMPT), ("one", "[calm] First."))
    assert _contents(HistoryView(HISTORY_TURNS).messages(conversation))[-1] == "[calm] First."
    sent = HistoryView(HISTORY_TURNS, cue_turns=0).messages(conversation)
    assert sent[2]["content"].count("[") >= 2  # the pinned example is how cues are taught
    assert sent[-1]["content"] == "First."


def test_the_cue_count_is_of_replies_the_model_is_sent() -> None:
    conversation = _talk(Conversation("s"), ("one", "[calm] First."), ("two", "[sad] Second."))
    conversation.set_hidden(conversation.turns[1].id, hidden=True)
    assert _contents(HistoryView(HISTORY_TURNS, cue_turns=1).messages(conversation))[-1] == (
        "[calm] First."
    )


def test_an_edited_reply_is_sent_by_the_same_cue_rule() -> None:
    conversation = _talk(Conversation("s"), ("one", "First."), ("two", "Second."))
    conversation.edit_reply(conversation.turns[0].id, "[happy] First, edited.")
    sent = HistoryView(HISTORY_TURNS, cue_turns=1).messages(conversation)
    assert _contents(sent)[2] == "First, edited."


def test_the_id_is_kept_by_a_copy_json_and_a_reopened_record() -> None:
    conversation = _talk(Conversation("old prompt"), ("one", "First."))
    assert conversation.copy().id == conversation.id
    assert Conversation.from_json(conversation.to_json()).id == conversation.id
    reopened = conversation.reopened(DEFAULT_SYSTEM_PROMPT, seed=True)
    assert (reopened.id, reopened.turns) == (conversation.id, conversation.turns)
    assert reopened.system == DEFAULT_SYSTEM_PROMPT
    assert reopened.seed  # the seed follows the prompt set now
    assert Conversation("s").id != Conversation("s").id


def test_json_with_a_blank_id_is_refused() -> None:
    with pytest.raises(ValueError, match="id is blank"):
        Conversation.from_json({**Conversation("s").to_json(), "id": " "})
