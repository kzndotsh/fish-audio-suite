"""The session's ``History``: its opening, merging unanswered lines, trimming in blocks, and what is kept of a reply."""

from __future__ import annotations

import itertools

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.config.settings import DEFAULT_HISTORY_TURNS, SystemPrompt
from fish_audio_suite_voice.session.history import History, trim_history

HISTORY_TURNS = DEFAULT_HISTORY_TURNS


def test_history_keeps_the_system_prompt_and_drops_the_oldest_turns() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS + 1):
        History(history, HISTORY_TURNS).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    assert history[0] == {"role": "system", "content": "be brief"}
    kept = [m["content"] for m in history[1::2]]
    assert kept == [f"u{i}" for i in range(HISTORY_TURNS + 1 - len(kept), HISTORY_TURNS + 1)]
    assert history[-1]["content"] == f"a{HISTORY_TURNS}"
    assert len(history) <= 1 + HISTORY_TURNS * 2


def test_history_drops_whole_turns_so_roles_stay_paired() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS):
        History(history, HISTORY_TURNS).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    History(history, HISTORY_TURNS).remember_user("newest")
    history.append({"role": "assistant", "content": "answer"})
    roles = [m["role"] for m in history[1:]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)
    first = history[1]["content"]
    assert history[2]["content"] == "a" + first.removeprefix("u")
    assert history[-2:] == [
        {"role": "user", "content": "newest"},
        {"role": "assistant", "content": "answer"},
    ]
    assert len(history) <= 1 + HISTORY_TURNS * 2


def test_the_history_is_trimmed_in_blocks_so_its_start_stays_the_same_between_trims() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    floor = HISTORY_TURNS - HISTORY_TURNS // 4
    starts: list[list[ChatMessage]] = []
    for i in range(HISTORY_TURNS * 3):
        History(history, HISTORY_TURNS).remember_user(f"u{i}")
        turns = (len(history) - 1) // 2  # whole pairs before the newest line
        assert turns <= HISTORY_TURNS
        if i >= HISTORY_TURNS:
            assert turns >= floor - 1
        starts.append([*history[:3]])
        history.append({"role": "assistant", "content": f"a{i}"})
    # A provider caches the start of the prompt. Dropping one turn at a time changed it
    # on every turn once the cap was reached. Now it changes once per block.
    changes = sum(1 for before, after in itertools.pairwise(starts) if before != after)
    assert changes <= 2 * HISTORY_TURNS // (HISTORY_TURNS // 4)


def test_a_small_history_cap_still_drops_one_turn_at_a_time() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(4):
        History(history, 3).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    assert [m["content"] for m in history[1:]] == ["u1", "a1", "u2", "a2", "u3", "a3"]


def test_history_cap_follows_the_configured_turns() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(5):
        History(history, 2).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    History(history, 2).remember_user("newest")
    # The oldest user and assistant leave together, so roles stay paired.
    assert [m["content"] for m in history] == ["be brief", "u4", "a4", "newest"]


def test_an_unanswered_user_line_is_merged_with_the_next_one() -> None:
    # A turn the model never answered (a 429, a refusal from the provider) left its
    # user line last. Two user messages in a row reached the model before.
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    History(history, HISTORY_TURNS).remember_user("Swear on it.")
    History(history, HISTORY_TURNS).remember_user("Swear on it.")
    assert history == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Swear on it."},
    ]
    History(history, HISTORY_TURNS).remember_user("Are you there?")
    assert history[-1] == {"role": "user", "content": "Swear on it.\nAre you there?"}
    assert [m["role"] for m in history] == ["system", "user"]


def test_a_new_line_after_a_reply_is_its_own_message() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    History(history, HISTORY_TURNS).remember_user("hi")
    history.append({"role": "assistant", "content": "hello"})
    History(history, HISTORY_TURNS).remember_user("hi")
    assert [m["role"] for m in history] == ["system", "user", "assistant", "user"]


def test_a_long_run_of_unanswered_lines_keeps_only_the_latest() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    for i in range(10):
        History(history, HISTORY_TURNS).remember_user(f"line {i}")
    assert len(history) == 2
    assert history[-1]["content"] == "line 6\nline 7\nline 8\nline 9"


def test_history_remembers_what_was_heard_of_a_reply() -> None:
    history = History.opening("sys", seed=False, turns=DEFAULT_HISTORY_TURNS)
    history.remember_user("tell me a story")
    history.remember_assistant("")  # nothing heard, nothing kept
    assert history.messages[-1] == {"role": "user", "content": "tell me a story"}
    history.remember_assistant("tell me a story")  # an echo of the line is not kept
    assert history.messages[-1]["role"] == "user"
    history.remember_assistant("Once upon")
    history.remember_assistant("a time.", resumes=True)  # the rest after a false barge-in
    assert history.messages[-1] == {"role": "assistant", "content": "Once upon a time."}
    history.remember_assistant("A new reply.")  # not resuming: a new message
    assert [m["content"] for m in history.messages[-2:]] == ["Once upon a time.", "A new reply."]


def test_a_resumed_reply_with_no_assistant_message_before_it_starts_one() -> None:
    history = History.opening("sys", seed=False, turns=DEFAULT_HISTORY_TURNS)
    history.remember_user("hi")
    history.remember_assistant("rest of it", resumes=True)
    assert history.messages[-1] == {"role": "assistant", "content": "rest of it"}


def _opening(text: str) -> History:
    """Open a session's history for a prompt written out in full, as the config decides it."""
    prompt = SystemPrompt.written(text)
    return History.opening(prompt.text, seed=prompt.pin_seed, turns=DEFAULT_HISTORY_TURNS)


def test_default_prompt_starts_with_a_pinned_multi_cue_exchange() -> None:
    from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT

    opened = _opening(DEFAULT_SYSTEM_PROMPT)
    history, pinned = opened.messages, opened.pinned
    assert pinned == 3
    assert [m["role"] for m in history] == ["system", "user", "assistant"]
    assert history[2]["content"].count("[") >= 2


def test_a_custom_prompt_gets_no_seed_exchange() -> None:
    opened = _opening("You are a pirate.")
    history, pinned = opened.messages, opened.pinned
    assert pinned == 1
    assert history == [{"role": "system", "content": "You are a pirate."}]


def test_trimming_never_drops_the_pinned_seed() -> None:
    from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT

    opened = _opening(DEFAULT_SYSTEM_PROMPT)
    history, pinned = opened.messages, opened.pinned
    for index in range(30):
        History(history, 3, pinned).remember_user(f"question {index}")
        history.append({"role": "assistant", "content": f"[calm] answer {index}"})
    assert history[:pinned] == _opening(DEFAULT_SYSTEM_PROMPT).messages
    assert len(history) == pinned + 3 * 2
    assert history[-2]["content"] == "question 29"


def test_trim_history_still_pairs_user_and_assistant_without_a_seed() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    for index in range(5):
        history.append({"role": "user", "content": f"u{index}"})
        history.append({"role": "assistant", "content": f"a{index}"})
    trim_history(history, 2)
    assert [m["content"] for m in history] == ["s", "u3", "a3", "u4", "a4"]
