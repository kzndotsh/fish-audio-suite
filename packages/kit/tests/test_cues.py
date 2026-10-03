from __future__ import annotations

import pytest
from kit_timing import assert_linear_time

from fish_audio_suite_kit import (
    ensure_lead_cue,
    hold_tts,
    normalize_cues,
    scrub_tts,
    strip_cue_tags,
)
from fish_audio_suite_kit.cues import mood_lead_hold_at


def test_mood_lead_becomes_cue() -> None:
    assert normalize_cues("Excited, hello", lead=True) == "[excited] hello"


def test_official_emotion_lead_only_at_sentence_start() -> None:
    assert normalize_cues("Anxious: wait", lead=True) == "[anxious] wait"
    assert normalize_cues("I am anxious today", lead=True) == "I am anxious today"
    assert normalize_cues("excited, hello", lead=True, continued=True) == "excited, hello"
    assert (
        normalize_cues("excited, hello. Sad, bye", lead=True, continued=True)
        == "excited, hello. [sad] bye"
    )
    assert (
        normalize_cues("你好。Excited, goodbye now please.", lead=True)
        == "你好。[excited] goodbye now please."
    )
    assert (
        normalize_cues("你好。 Excited, goodbye now please.", lead=True)
        == "你好。 [excited] goodbye now please."
    )
    assert (
        normalize_cues("你好！Sad, the news is bad today friend.", lead=True)
        == "你好！[sad] the news is bad today friend."
    )
    assert (
        normalize_cues("Hello؟ Excited, yes today friend.", lead=True)
        == "Hello؟ [excited] yes today friend."
    )
    assert (
        normalize_cues("Hello؟Excited, yes today friend.", lead=True)
        == "Hello؟Excited, yes today friend."
    )
    assert (
        normalize_cues("今天天气很好。我们打算下午出去走走。", lead=True)
        == "今天天气很好。我们打算下午出去走走。"
    )
    assert (
        normalize_cues("Dr. Happy, the patient arrived today.", lead=True)
        == "Dr. Happy, the patient arrived today."
    )
    assert (
        normalize_cues("1. Excited, the first item is ready today.", lead=True)
        == "1. Excited, the first item is ready today."
    )
    assert (
        normalize_cues("The visit ended. Excited, goodbye now please.", lead=True)
        == "The visit ended. [excited] goodbye now please."
    )


def test_tone_lead_becomes_cue() -> None:
    assert normalize_cues("Shouting, hey", lead=True) == "[shouting] hey"


def test_hold_tts_keeps_a_mood_until_the_comma() -> None:
    assert hold_tts("Excited,", line_start=True, sentence_start=True, lead=True) == 0
    finished = "Excited, hello"
    assert hold_tts(finished, line_start=True, sentence_start=True, lead=True) == len(finished)


def test_mood_lead_holds_only_an_unfinished_word() -> None:
    assert mood_lead_hold_at("Exc", sentence_start=True) == 0
    assert mood_lead_hold_at("  Exc", sentence_start=True) == 2
    assert mood_lead_hold_at("Excited,", sentence_start=True) == 0
    assert mood_lead_hold_at("Excited, hi", sentence_start=True) is None
    assert mood_lead_hold_at("Anxious!", sentence_start=True) == 0
    assert mood_lead_hold_at("Anxious !!", sentence_start=True) == 0
    assert mood_lead_hold_at("Anxious! Hello", sentence_start=True) is None
    assert mood_lead_hold_at("Excited*", sentence_start=True) == 0
    assert mood_lead_hold_at("Example", sentence_start=True) is None
    assert mood_lead_hold_at("Happy birthday", sentence_start=True) is None
    assert mood_lead_hold_at("Happy ", sentence_start=True) == 0
    assert mood_lead_hold_at("Happ ", sentence_start=True) is None
    assert mood_lead_hold_at("* E", sentence_start=True) == 2
    assert mood_lead_hold_at("* Excited, hi", sentence_start=True) is None
    assert mood_lead_hold_at("- [x] Ex", sentence_start=True) == 6
    assert mood_lead_hold_at("### Ex", sentence_start=True) == 4
    assert mood_lead_hold_at("*Sad", sentence_start=True) == 1
    assert mood_lead_hold_at("excited", sentence_start=False) is None
    assert mood_lead_hold_at("Hello\nEx", sentence_start=False) == 6


def test_sound_effect_lead_stays_spoken() -> None:
    assert normalize_cues("Pause, wait", lead=True) == "Pause, wait"


def test_unlisted_lead_stays_spoken() -> None:
    assert normalize_cues("Soft, hello", lead=True) == "Soft, hello"


def test_cue_lowercase() -> None:
    assert normalize_cues("[Excited] Hello.", lead=True) == "[excited] Hello."


def test_laugh_aliases() -> None:
    assert "[laughing]" in normalize_cues("[laugh] ha", lead=True)
    assert "[laughing]" in normalize_cues("[laughs] ha", lead=True)


def test_sigh_and_chuckle_aliases() -> None:
    assert "[sighing]" in normalize_cues("[sigh] ok", lead=True)
    assert "[chuckling]" in normalize_cues("[chuckle] ha", lead=True)


def test_stacked_leading_cues_kept() -> None:
    assert (
        normalize_cues("[sad][whispering] I miss you", lead=True) == "[sad] [whispering] I miss you"
    )
    assert "[whisper in small voice]" in normalize_cues(
        "[whisper in small voice] come closer", lead=True
    )
    # A doubled bracket is not a cue stack. The inner "[" used to take a space.
    assert (
        normalize_cues("[[page]] and then more words", lead=True) == "[[page]] and then more words"
    )


def test_whisper_aliases() -> None:
    assert "[whispering]" in normalize_cues("[whispers] psst", lead=True)
    assert "[whispering]" in normalize_cues("[whisper] psst", lead=True)
    assert "[whispering]" in normalize_cues("<whisper>psst</whisper>", lead=True)


def test_inline_chuckle_and_cough() -> None:
    out = normalize_cues("I'll call you back [chuckle] in a minute [cough]", lead=True)
    assert "[chuckling]" in out
    assert "[cough]" in out
    assert "[coughing]" not in out


def test_ensure_lead_cue_only_when_missing() -> None:
    assert ensure_lead_cue("[curious] yeah") == "[curious] yeah"
    assert ensure_lead_cue("I'll call you back [chuckle] in a minute").startswith("I'll")
    assert ensure_lead_cue("yeah i hear you") == "yeah i hear you"
    assert ensure_lead_cue("yeah i hear you", default="calm") == "[calm] yeah i hear you"
    assert ensure_lead_cue("  ") == "  "


def test_pause_alias_vs_moss_duration() -> None:
    assert "[break]" in normalize_cues("[pause] wait", lead=True)
    stripped = scrub_tts("hello [pause 3.2s] there")
    assert "pause" not in stripped.lower()
    assert "hello" in stripped
    assert "there" in stripped


def test_fullwidth_lead_punctuation_is_a_cue() -> None:
    out = normalize_cues(scrub_tts("Excited， hello there friend."), lead=True)
    assert "[excited]" in out
    assert "Excited" not in out
    followed = normalize_cues(
        scrub_tts("Hello there friend。Excited， hi there friend."), lead=True
    )
    assert "[excited]" in followed
    assert "[anxious]" in normalize_cues("Anxious！ hello there friend.", lead=True)


def test_mood_glued_to_a_stop_is_a_cue() -> None:
    out = normalize_cues(scrub_tts("Done.Excited, hi there friend."), lead=True)
    assert "[excited]" in out
    assert "Excited" not in out
    assert "3.14" in scrub_tts("It is 3.14 exactly today friend.")
    assert "[happy]" not in normalize_cues(scrub_tts("Dr. Happy birthday today friend."), lead=True)
    # "No." ends a sentence, so the mood after it is a lead.
    assert "[excited]" in normalize_cues(scrub_tts("No. Excited, hello there friend."), lead=True)
    corner = normalize_cues(
        scrub_tts("Open the door today friend。」 Excited, wait there."), lead=True
    )
    assert "[excited]" in corner
    assert "Excited" not in corner
    guillemet = normalize_cues(
        scrub_tts("She said «open the door today friend.» Excited, wait there."), lead=True
    )
    assert "[excited]" in guillemet
    assert "Excited" not in guillemet
    assert "[excited]" not in normalize_cues(
        scrub_tts("Hello」 there today friend please."), lead=True
    )


def test_s1_parens_become_cues() -> None:
    out = normalize_cues(scrub_tts("(happy) Hello there friend (break)"), lead=True)
    assert "[happy]" in out
    assert "[break]" in out
    assert "(" not in out


def test_cue_before_a_parenthetical_stays_a_cue() -> None:
    out = scrub_tts("[happy](softly) Hello there friend")
    assert "[happy]" in out
    assert "softly" not in out
    assert "Hello there friend" in out


def test_mood_lead_with_no_remainder_is_only_the_cue() -> None:
    assert normalize_cues("", lead=True) == ""
    assert normalize_cues("Excited, ", lead=True) == "[excited]"
    assert ensure_lead_cue("hello", default="  ") == "hello"


@pytest.mark.parametrize(
    "text",
    [
        "Curious, isn't it?",
        "Calm down. Sad, but true.",
        "Proud, that is what I am.",
        "Moved, the sofa is by the door.",
        "Excited, hello",
        "Anxious: wait",
    ],
)
def test_mood_words_are_left_alone_by_default(text: str) -> None:
    assert normalize_cues(text) == text
    assert normalize_cues(text, lead=False) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Curious, isn't it?", "[curious] isn't it?"),
        ("Calm down. Sad, but true.", "Calm down. [sad] but true."),
    ],
)
def test_mood_leads_are_rewritten_only_on_request(text: str, expected: str) -> None:
    assert normalize_cues(text, lead=True) == expected


def test_a_continued_chunk_skips_only_its_first_sentence() -> None:
    assert normalize_cues("excited, hello", lead=True, continued=True) == "excited, hello"
    assert (
        normalize_cues("excited, hello. Sad, bye", lead=True, continued=True)
        == "excited, hello. [sad] bye"
    )
    assert normalize_cues("excited, hello", continued=True) == "excited, hello"


def test_explicit_cues_are_normalized_without_the_lead_switch() -> None:
    assert normalize_cues("[Happy] hi. (sigh) ok") == "[happy] hi. [sighing] ok"


def test_strip_cue_tags_keeps_what_the_speaker_says() -> None:
    assert strip_cue_tags("[happy] Hello  [Whispering] there [clear]") == "Hello there"
    assert strip_cue_tags("a[i][j] is an index") == "a[i][j] is an index"
    assert strip_cue_tags("[laughing]") == ""
    assert strip_cue_tags("") == ""


def test_the_mood_hold_is_off_unless_leads_are_on() -> None:
    assert hold_tts("Excited,", line_start=True, sentence_start=True) == len("Excited,")
    assert hold_tts("Exc", line_start=True, sentence_start=True) == len("Exc")
    assert hold_tts("Exc", line_start=True, sentence_start=True, lead=True) == 0


def test_strip_cue_tags_keeps_a_tag_nested_in_a_larger_bracket() -> None:
    assert strip_cue_tags("[[happy]] hi") == "[[happy]] hi"
    assert strip_cue_tags("[happy] hi [calm] there") == "hi there"
    assert strip_cue_tags("[sad][whispering] hi") == "hi"


@pytest.mark.perf
@pytest.mark.parametrize(
    "text",
    [
        "<whisper>" + "<whisper>a" * 8_000,
        "[" + "[\\" * 8_000,
        "[" * 20_000,
        "[ " * 10_000,
        "<whisper " * 8_000,
    ],
)
def test_cue_rewriting_is_linear_on_hostile_input(text: str) -> None:
    assert_linear_time(normalize_cues, text)
    assert_linear_time(strip_cue_tags, text)


def test_a_cue_or_whisper_longer_than_the_cap_is_left_as_text() -> None:
    long_cue = "[" + "x" * 250 + "]"
    assert normalize_cues(long_cue) == long_cue
    assert normalize_cues("<whisper>hush</whisper>") == "[whispering] hush"
    assert normalize_cues("[happy] hi") == "[happy] hi"
