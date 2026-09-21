from __future__ import annotations

from fish_audio_suite_kit import (
    is_asr_hallucination,
    is_backchannel,
    is_tts_junk,
    next_tts_cut,
    normalize_cues,
    scrub_asr,
    scrub_tts,
)


def test_mood_lead_becomes_cue() -> None:
    assert normalize_cues("Excited, hello") == "[excited] hello"


def test_cue_lowercase() -> None:
    assert normalize_cues("[Excited] Hello.") == "[excited] Hello."


def test_laugh_aliases() -> None:
    assert "[laughing]" in normalize_cues("[laugh] ha")
    assert "[laughing]" in normalize_cues("[laughs] ha")


def test_whisper_aliases() -> None:
    assert "[whispering]" in normalize_cues("[whispers] psst")
    assert "[whispering]" in normalize_cues("[whisper] psst")
    assert "[whispering]" in normalize_cues("<whisper>psst</whisper>")


def test_pause_alias_vs_moss_duration() -> None:
    assert "[break]" in normalize_cues("[pause] wait")
    stripped = scrub_tts("hello [pause 3.2s] there")
    assert "pause" not in stripped.lower()
    assert "hello" in stripped and "there" in stripped


def test_unclosed_cue_is_junk() -> None:
    assert is_tts_junk("[warm, leftover")


def test_cjk_asr_drop() -> None:
    assert is_asr_hallucination("谢谢观看")


def test_aside_and_bold_stripped() -> None:
    out = scrub_tts("Hello (aside) *bold* world.")
    assert "aside" not in out
    assert "*" not in out
    assert "Hello" in out
    assert "world" in out


def test_url_and_heading_strip() -> None:
    out = scrub_tts("# Title\nSee https://example.com/x and [docs](https://x.test) later.")
    assert "http" not in out.lower()
    assert "#" not in out
    assert "Title" in out
    assert "docs" in out


def test_s1_and_nospeech() -> None:
    assert "[S1]" not in scrub_tts("[S1] Hello there friend")
    assert is_asr_hallucination("<|nospeech|>")
    assert is_asr_hallucination("<|HAPPY|>")


def test_speaker_and_timestamp_asr() -> None:
    out = scrub_asr("[0.0 - 1.2] Speaker 1: hello there")
    assert "Speaker" not in out
    assert "0.0" not in out
    assert "hello there" in out


def test_backchannel() -> None:
    assert is_backchannel("yeah")
    assert is_backchannel("uh huh")
    assert is_backchannel("mm hmm")
    assert not is_backchannel("yeah can you repeat that")


def test_next_tts_cut_skips_abbreviations() -> None:
    buf = "Dr. Smith is here. Next"
    cut = next_tts_cut(buf)
    assert cut > 0
    assert buf[:cut].strip().endswith("here.")
    assert "Dr." in buf[:cut]
    restart = "1. Restart the service now please and thank you extra"
    cut2 = next_tts_cut(restart)
    assert not restart[:cut2].strip().endswith("1.")


def test_next_tts_cut_forty_chars() -> None:
    buf = "this is a long spoken fragment without any sentence end yet"
    assert len(buf) >= 40
    cut = next_tts_cut(buf)
    assert cut >= 40 or (cut > 0 and buf[cut - 1] == " ")
    assert next_tts_cut("short") == -1


def test_thanks_for_watching() -> None:
    assert is_asr_hallucination("thanks for watching")
    assert is_asr_hallucination("Thanks for watching.")


def test_dialogue_only_keeps_quotes() -> None:
    spoken = scrub_tts('[warm] "Loud and clear." Stage note.', dialogue_only=True)
    assert "Loud and clear" in spoken
    assert "Stage note" not in spoken
