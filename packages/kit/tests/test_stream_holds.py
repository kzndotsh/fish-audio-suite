from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    tts_hold_at,
)


def test_tts_hold_at_keeps_an_unfinished_span() -> None:
    assert tts_hold_at("<think", line_start=True, sentence_start=True) == 0
    partial = "See https://exa"
    assert tts_hold_at(partial, line_start=True, sentence_start=False) < len(partial)
    fence = "Open the door.\n```"
    assert tts_hold_at(fence, line_start=True, sentence_start=False) <= fence.index("`")


@pytest.mark.parametrize(
    ("text", "held_at"),
    [
        ("Hello <thinking>still going", 6),
        ("Hello <analysis>still going", 6),
        ("Hello <scratch", 6),
        ("Hello <thinking>done</thinking> there", None),
        ("Hello <whisper>quiet", 6),
        ("Hello <script>x</style>", 6),
        ("Hello <script>x</script> there", None),
        ("code ```print(1)", 5),
        ("code ~~~print(1)", 5),
        ("see https://exam", 4),
        ("see <http", 4),
        ("Hello (she smiles", 6),
        ("Hello (she smiles) there", None),
        ("Hello <thi", 6),
    ],
)
def test_unfinished_spans_hold_at_their_opener(text: str, held_at: int | None) -> None:
    result = tts_hold_at(text, line_start=True, sentence_start=False)
    assert result == (held_at if held_at is not None else len(text))


@pytest.mark.parametrize("text", ["a < b and", "x < y", "if a &lt; b then", "1 < 2 and 3 &lt; 4"])
def test_a_spaced_less_than_is_a_comparison_and_is_not_held(text: str) -> None:
    assert tts_hold_at(text, line_start=True, sentence_start=True) == len(text)


@pytest.mark.parametrize("text", ["ok <b", "ok </ p", "ok <", "go &lt;br", "go &lt;/b"])
def test_an_unfinished_tag_is_still_held(text: str) -> None:
    assert tts_hold_at(text, line_start=True, sentence_start=True) < len(text)


@pytest.mark.parametrize("text", ["Title\n### ", "Intro\n- ", "Intro\n* \t", "### "])
def test_a_marker_with_no_words_after_it_yet_is_held_with_its_space(text: str) -> None:
    # Shaped alone, "Title\n### " loses its line, and the mood word after it is
    # then read as mid-sentence.
    held = text.rstrip(" \t").rstrip("#-*")
    assert tts_hold_at(text, line_start=True, sentence_start=True) == len(held)


def test_a_mid_line_star_and_space_is_not_held() -> None:
    assert tts_hold_at("2 * ", line_start=False, sentence_start=True) == len("2 * ")
