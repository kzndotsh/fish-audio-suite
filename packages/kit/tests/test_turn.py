from __future__ import annotations

import json
from pathlib import Path

import pytest
from kit_helpers import assert_linear_time

from fish_audio_suite_kit import judge_turn


@pytest.mark.parametrize(
    "text",
    [
        "I was going to say that, but",
        "So what I think is that we should go to the",
        "I went to the store and",
        "Well, um",
        "Okay,",
        "I was thinking...",
        "I was thinking\u2026",
        "The thing is:",
        "He told me \u2014",
        "It was late, so",
        "The numbers are 4, 8, 15",
        "one, two, three",
        "I want to talk about",
        "She said she would be there because.",
        '"Maybe we could," she said, and',
        "I was tired, and then",
    ],
)
def test_a_thought_that_stops_short_is_not_finished(text: str) -> None:
    assert not judge_turn(text).finished, text


@pytest.mark.parametrize(
    "text",
    [
        "Can you tell me what time it is?",
        "That sounds great.",
        "Wow!",
        "Yes",
        "No thanks",
        "I think so",
        "I hope not",
        "Tell me a joke about cats",
        "What are you looking for?",
        "My number is 42",
        "What is 7 plus 8",
        "I am 25.",
        "And?",
        "\u4f60\u597d\uff0c\u4eca\u5929\u600e\u4e48\u6837\uff1f",
        "",
        "   ",
    ],
)
def test_a_complete_sentence_or_short_answer_is_finished(text: str) -> None:
    assert judge_turn(text).finished, text


def test_the_reason_says_what_decided_it() -> None:
    assert judge_turn("Okay,").reason == "open_punctuation"
    assert judge_turn("and the").reason == "dangling_word"
    assert judge_turn("1, 2, 3").reason == "number_list"
    assert judge_turn("Fine.").reason == "terminal_punctuation"
    assert judge_turn("Sure thing").reason == "short_answer"
    assert judge_turn("Tell me a joke about cats").reason == "complete"
    assert judge_turn("").reason == "empty"


def test_closing_quotes_and_spaces_after_the_last_word_do_not_hide_it() -> None:
    assert not judge_turn('He said "wait,"  ').finished
    assert judge_turn('He said "wait."  ').finished


def test_only_the_end_of_a_long_text_matters() -> None:
    assert not judge_turn("This is a long story. " * 200 + "and then I").finished
    assert judge_turn("and then I went. " * 200).finished


@pytest.mark.perf
@pytest.mark.parametrize(
    "unit", [" ", "a", "a,", "a ", ", ", "1,", "\"'", "x-", "I'm ", "\u2019a", "so,"]
)
def test_judging_hostile_text_is_linear(unit: str) -> None:
    assert_linear_time(judge_turn, unit * 20_000)


_CORPUS = json.loads((Path(__file__).parent / "golden" / "turn_corpus.json").read_text("utf-8"))
_CASES = [
    pytest.param(
        case["text"],
        case["finished"],
        id=f"{case['group']}-{n}",
        marks=[pytest.mark.xfail(strict=True, reason="a known miss")] if case["known_miss"] else [],
    )
    for n, case in enumerate(_CORPUS["cases"])
]


@pytest.mark.parametrize(("text", "finished"), _CASES)
def test_judge_turn_agrees_with_the_pipecat_corpus(text: str, finished: bool) -> None:
    """A known miss is a strict xfail, so improving the judge shows up as a failure to update."""
    assert judge_turn(text).finished is finished


def test_the_corpus_never_makes_a_finished_turn_wait() -> None:
    """A wrong "not finished" only costs a wait, but none of the corpus should need one."""
    finished = [c["text"] for c in _CORPUS["cases"] if c["finished"]]
    assert finished
    assert all(judge_turn(text).finished for text in finished)
