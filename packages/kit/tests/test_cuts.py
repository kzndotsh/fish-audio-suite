from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    ends_sentence,
    normalize_cues,
    scrub_tts,
    split_tts_piece,
)
from fish_audio_suite_kit.cuts import next_tts_cut
from fish_audio_suite_kit.stream_holds import sentence_closer_hold_at


def test_next_tts_cut_skips_abbreviations() -> None:
    buf = "Dr. Smith is here. Next"
    cut = next_tts_cut(buf)
    assert cut > 0
    assert buf[:cut].strip().endswith("here.")
    assert "Dr." in buf[:cut]
    restart = "1. Restart the service now please and thank you extra"
    cut2 = next_tts_cut(restart)
    assert not restart[:cut2].strip().endswith("1.")
    priced = "The total is 3.14. Please pay the invoice today."
    price_cut = next_tts_cut(priced, partial_chars=80)
    assert priced[:price_cut] == "The total is 3.14. "


def test_a_time_or_page_number_ends_the_sentence() -> None:
    clock = normalize_cues(scrub_tts("Meet at 10:30. Excited, hi there friend."), lead=True)
    assert "[excited]" in clock
    assert "Excited" not in clock
    page = normalize_cues("See page 12. Excited, hi there friend.", lead=True)
    assert "[excited]" in page
    assert ends_sentence("Meet at 10:30.")
    assert ends_sentence("See page 12.")
    assert not ends_sentence("1. ")


def test_ends_sentence_skips_abbreviations_and_list_numbers() -> None:
    assert not ends_sentence("Dr.")
    assert ends_sentence("No. ")
    assert not ends_sentence("1. ")
    assert not ends_sentence("A.")
    assert not ends_sentence("Hello. Dr.")
    assert ends_sentence("Hello.")
    assert ends_sentence("3.14.")
    assert ends_sentence("你好。")
    assert not ends_sentence("Hello؟")
    assert ends_sentence("Hello؟ ")
    assert ends_sentence('He said "Done."')
    assert ends_sentence('He said "Done." ')
    assert not ends_sentence('Dr."')
    assert ends_sentence('Hello؟ "')
    assert sentence_closer_hold_at("你好。") == 2
    assert sentence_closer_hold_at("你好。”") == 2
    assert sentence_closer_hold_at("你好。” ") is None
    assert sentence_closer_hold_at("你好。再") is None
    assert sentence_closer_hold_at("Hello. ") is None
    assert sentence_closer_hold_at("Hello!") == 5
    assert sentence_closer_hold_at("Anxious!", lead=True) is None
    assert sentence_closer_hold_at("Anxious!") == 7
    assert sentence_closer_hold_at("Words (aside.)") is None
    assert sentence_closer_hold_at("Done.)") == 4


def test_cjk_period_and_ellipsis_end_the_sentence() -> None:
    buf = "今天天气很好。我们打算下午出去走走顺便买些东西然后回家做饭再休息一会儿才出门见朋友。"
    cut = next_tts_cut(buf, partial_chars=40)
    assert buf[:cut] == "今天天气很好。"
    spaced = "你好。 Excited, goodbye now please."
    spaced_cut = next_tts_cut(spaced, partial_chars=80)
    assert spaced[:spaced_cut] == "你好。 "
    spoken = "Hello… world is ready today please continue."
    ellipsis = next_tts_cut(spoken, partial_chars=80)
    assert spoken[:ellipsis] == "Hello… "


def test_a_long_finished_sentence_still_cuts_early() -> None:
    buf = ("word " * 30).strip() + "."
    assert len(buf) > 40
    cut = next_tts_cut(buf)
    assert 0 < cut < len(buf)
    assert buf[cut - 1] == " "
    short = "Done.Excited, hi there friend."
    assert next_tts_cut(short, partial_chars=80) == short.index("E")


def test_next_tts_cut_forty_chars() -> None:
    buf = "this is a long spoken fragment without any sentence end yet"
    cut = next_tts_cut(buf)
    assert 0 < cut <= 40
    assert buf[cut - 1] == " "
    assert buf[cut:]
    assert next_tts_cut("short") == -1
    assert next_tts_cut("x" * 50) == 40
    glued = "x" * 36 + "[whispering] come closer today friend please"
    cue_cut = next_tts_cut(glued, partial_chars=40)
    assert glued[:cue_cut] == "x" * 36
    assert glued[cue_cut:].startswith("[whispering]")
    noted = "Note [hello. there] and then more words please today friend."
    noted_cut = next_tts_cut(noted, partial_chars=80)
    assert "[hello. there]" in noted[:noted_cut]
    assert noted[:noted_cut].count("[") == noted[:noted_cut].count("]")
    spaced = "x" * 20 + " [hello. there] and then more words please today friend."
    spaced_cut = next_tts_cut(spaced, partial_chars=80)
    assert spaced[spaced_cut:].startswith("[hello. there]")
    assert spaced[:spaced_cut].count("[") == spaced[:spaced_cut].count("]")
    lines = "\n".join(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"])
    line_cut = next_tts_cut(lines, partial_chars=40)
    assert lines[:line_cut].endswith("\n")
    assert lines[line_cut:].startswith("golf")


def test_a_nonbreaking_space_is_a_word_cut() -> None:
    head = "Please open the"
    buf = head + "\u00a0" + ("door" * 20)
    cut = next_tts_cut(buf, partial_chars=40)
    assert buf[:cut] == head + "\u00a0"
    assert buf[cut:].startswith("door")
    ideo = head + "\u3000" + ("door" * 20)
    ideo_cut = next_tts_cut(ideo, partial_chars=40)
    assert ideo[:ideo_cut] == head + "\u3000"
    assert ideo[ideo_cut:].startswith("door")


def test_zero_partial_window_does_not_return_an_empty_piece() -> None:
    buf = "hello there friend"
    assert split_tts_piece(buf, 0, flush_rest=False) is None
    assert split_tts_piece(buf, -5, flush_rest=True) == (buf, "")


def test_split_tts_piece_waits_until_flush() -> None:
    assert split_tts_piece("short", 40, flush_rest=False) is None
    assert split_tts_piece("short", 40, flush_rest=True) == ("short", "")
    split = split_tts_piece("word " * 20, 20, flush_rest=False)
    assert split is not None
    piece, tail = split
    assert len(piece) <= 20
    assert piece.endswith(" ")
    assert tail


def test_no_ends_a_sentence_and_does_not_glue_the_next_one() -> None:
    text = "No. I will not do that."
    assert next_tts_cut(text) == len("No. ")
    assert split_tts_piece(text, 40, flush_rest=False) == ("No. ", "I will not do that.")
    assert ends_sentence("No.")


@pytest.mark.parametrize("title", ["Dr.", "Mr.", "Mrs.", "Prof.", "etc."])
def test_titles_do_not_end_a_sentence(title: str) -> None:
    assert not ends_sentence(title)


def test_no_before_a_number_and_co_before_a_suffix_are_abbreviations() -> None:
    assert next_tts_cut("See No. 5 for details. Next one.", partial_chars=400) == 23
    assert next_tts_cut("Acme Co. Ltd. is here. Okay.", partial_chars=400) == 23
    assert next_tts_cut("Acme Co. LLC is here. Okay.", partial_chars=400) == 22


def test_no_and_co_before_a_word_still_end_the_sentence() -> None:
    assert next_tts_cut("No. I will not do that.", partial_chars=400) == 4
    assert next_tts_cut("He said no. Then left.", partial_chars=400) == 12
    assert next_tts_cut("We met at the co. Then left.", partial_chars=400) == 18


def test_many_short_sentences_are_cut_in_linear_time() -> None:
    import time

    for text in ("a. " * 7_000, "Dr. " * 5_000, "No. 5 " * 3_000):
        started = time.perf_counter()
        next_tts_cut(text)
        ends_sentence(text)
        assert time.perf_counter() - started < 0.5
