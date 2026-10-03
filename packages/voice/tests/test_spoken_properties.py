"""Properties of the heard-so-far prefix that history relies on."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fish_audio_suite_voice.spoken import _word_prefix

_LATIN = (
    "Hello",
    " there",
    " friend",
    ".",
    "!",
    ",",
    " ",
    "\n",
    " ",
    "3.14",
    "there.Friend",
    "don't",
    "well-known",
)
_CJK = ("你好", "我们", "。", " ", "\n", "今天")

_latin_texts = st.lists(st.sampled_from(_LATIN), min_size=1, max_size=10).map("".join)
_cjk_texts = st.lists(st.sampled_from(_CJK), min_size=1, max_size=10).map("".join)
_texts = st.one_of(_latin_texts, _cjk_texts)

_SETTINGS = settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _glyphs(text: str) -> str:
    return "".join(text.split())


@_SETTINGS
@given(text=_texts, n=st.integers(min_value=0, max_value=80))
def test_the_heard_prefix_is_a_prefix_of_what_was_cut(text: str, n: int) -> None:
    heard = _word_prefix(text, n)
    assert _glyphs(text[:n]).startswith(_glyphs(heard))


@_SETTINGS
@given(text=_texts)
def test_playing_more_never_shrinks_what_counts_as_heard(text: str) -> None:
    sizes = [len(_glyphs(_word_prefix(text, n))) for n in range(len(text) + 2)]
    assert sizes == sorted(sizes)


@_SETTINGS
@given(text=_texts)
def test_playing_everything_hears_everything(text: str) -> None:
    assert _word_prefix(text, len(text)) == text.strip()


def test_a_cut_inside_a_number_does_not_count_the_digit_before_the_point() -> None:
    # "3" before "." is the start of "3.14", not a finished word.
    assert _word_prefix("3.14", 1) == ""
    assert _word_prefix("see 3.14 today", 5) == "see"
    assert _word_prefix("你好3.14", 3) == "你好"
    assert _word_prefix("there.3.14 x", 8) == "there."


@pytest.mark.xfail(strict=True, reason="a cut inside a Latin word after CJK drops finished Latin")
def test_known_gap_cjk_then_latin_keeps_the_finished_latin_word() -> None:
    text = "你好there.Friend"
    # "你好there." was played in full at cut 8, so a cut inside "Friend" keeps it.
    assert _word_prefix(text, 9) == "你好there."
