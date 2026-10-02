from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    extract_quoted_speech,
    is_tts_junk,
    scrub_tts,
    skip_empty_delta,
)


def test_unclosed_cue_is_junk() -> None:
    assert is_tts_junk("[warm, leftover")


def test_tts_junk_thin_and_narration() -> None:
    assert is_tts_junk("")
    assert not is_tts_junk("hi")
    assert is_tts_junk("a")
    assert is_tts_junk("[clear] a")
    assert not is_tts_junk("[clear] hi")
    assert is_tts_junk("She smiles and leans closer now", drop_narration=True)
    assert is_tts_junk("[clear] She smiles and leans closer now", drop_narration=True)
    assert not is_tts_junk("[clear] Hello there friend")
    assert not is_tts_junk("你好，很开心认识你")
    assert not is_tts_junk("100")
    assert not is_tts_junk("3.14")
    assert is_tts_junk("42")
    assert not is_tts_junk("ok")
    assert not is_tts_junk("Été")
    assert not is_tts_junk("Привет, как дела сегодня")
    assert not is_tts_junk("Да")


def test_guillemets_and_corner_quotes_stay_speech() -> None:
    for sample in (
        "She smiles. «Hello there friend.»",
        "She smiles. 「Hello there friend.」",
        "She smiles. 『Hello there friend.』",
        "She smiles. ‹Hello there friend.›",
        "She smiles. ›Hello there friend.‹",
        "She smiles. 〈Hello there friend.〉",
        "She smiles. 》Hello there friend.《",
    ):
        quoted = extract_quoted_speech(scrub_tts(sample))
        assert "Hello there friend." in quoted
        assert "smiles" not in quoted
        assert not is_tts_junk(scrub_tts(sample), drop_narration=True)
    assert is_tts_junk(scrub_tts("She smiles and looks away today."), drop_narration=True)


def test_a_stage_direction_does_not_silence_the_next_sentence() -> None:
    spoken = "She smiles. It's open today friend."
    assert not is_tts_junk(spoken, drop_narration=True)
    assert "open today friend" in spoken
    assert not is_tts_junk("She smiles.\nOpen the door today friend.", drop_narration=True)
    assert not is_tts_junk("She smiles. 'Open the door today friend.'", drop_narration=True)
    assert is_tts_junk("She smiles and looks away today.", drop_narration=True)
    assert is_tts_junk("The door looks open today friend.", drop_narration=True)
    assert is_tts_junk("He looks tired today friend please.", drop_narration=True)
    assert not is_tts_junk("She said the door is open today friend.", drop_narration=True)


def test_extract_quoted_keeps_quotes() -> None:
    spoken = extract_quoted_speech(scrub_tts('[warm] "Loud and clear." Stage note.'))
    assert "Loud and clear" in spoken
    assert "Stage note" not in spoken


def test_extract_quoted_open_passthrough_and_empty() -> None:
    open_q = extract_quoted_speech('[warm] "hello there friend')
    assert "hello there friend" in open_q
    assert extract_quoted_speech('He said "x" but wait') == ""
    inches = 'Use a 5" pipe for the drain today.'
    assert extract_quoted_speech(inches) == inches
    mixed = 'The board is 2" wide. She said "hold it steady today."'
    assert extract_quoted_speech(mixed) == '"hold it steady today."'
    assert extract_quoted_speech('She said "room 2" and left the building today.') == '"room 2"'
    assert extract_quoted_speech('She said "100" and left the building today.') == '"100"'
    assert extract_quoted_speech('She said "3.14" and left the building today.') == '"3.14"'
    assert extract_quoted_speech('She said "42" and left the building today.') == ""
    assert extract_quoted_speech("no quotes here at all") == "no quotes here at all"
    assert extract_quoted_speech('[warm] "[clear]"') == '[warm] "[clear]"'
    assert extract_quoted_speech('旁白。"你好朋友" 然后离开。') == '"你好朋友"'
    wrapped = 'She smiles.\n"Hello there\nfriend today."'
    assert extract_quoted_speech(wrapped) == '"Hello there\nfriend today."'
    assert "smiles" not in extract_quoted_speech(wrapped)
    assert "你好朋友" in extract_quoted_speech('"你好朋友')


def test_skip_empty_delta() -> None:
    assert skip_empty_delta("  ")
    assert not skip_empty_delta("hi")


@pytest.mark.parametrize(
    "line",
    [
        "The weather looks great today.",
        "A cat moves fast.",
        "He looks fine, I promise.",
        "She smiles when she is happy.",
        "The door shifts in the wind.",
        "no",
        "No.",
        "ok",
        "hi",
        "yes",
    ],
)
def test_ordinary_speech_is_never_junk_by_default(line: str) -> None:
    assert not is_tts_junk(line)


@pytest.mark.parametrize(
    "line",
    ["The weather looks great today.", "A cat moves fast.", "He looks fine, I promise."],
)
def test_narration_dropping_is_opt_in(line: str) -> None:
    assert is_tts_junk(line, drop_narration=True)


@pytest.mark.parametrize("line", ["", "   ", "a", "!!", "[warm]", "[ ]", "42"])
def test_noise_is_still_junk(line: str) -> None:
    assert is_tts_junk(line)


def test_the_letter_floor_is_a_parameter() -> None:
    assert is_tts_junk("no", min_letters=9, short_words=frozenset())
    assert not is_tts_junk("no", min_letters=9)


def test_nested_quotes_of_another_style_stay_inside_the_speech() -> None:
    assert extract_quoted_speech('She nods. 「こんにちは "hi" 友よ」 and leaves.') == (
        '「こんにちは "hi" 友よ」'
    )
    assert extract_quoted_speech("She said “I think ‹so› too” then went.") == ("“I think ‹so› too”")
    assert extract_quoted_speech("Elle dit «bonjour mon ami» puis part.") == "«bonjour mon ami»"


def test_an_empty_quote_pair_is_not_speech_and_hides_the_narration() -> None:
    assert extract_quoted_speech('She smiles. "" Then waits for you.') == ""


def test_an_unclosed_trailing_quote_is_kept_after_narration_or_a_closed_quote() -> None:
    assert extract_quoted_speech('She smiles. "Hello there my fri') == '"Hello there my fri"'
    assert (
        extract_quoted_speech('She nods. "Hello there." She waits. "And I was thinking about')
        == '"Hello there." "And I was thinking about"'
    )
    assert extract_quoted_speech('The pipe is 5" wide. She said "hold on a second') == (
        '"hold on a second"'
    )
