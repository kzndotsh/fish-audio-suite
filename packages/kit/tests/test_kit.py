from __future__ import annotations

from fish_audio_suite_kit import (
    LatencySnapshot,
    SuiteDefaults,
    extract_quoted_speech,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_tts_junk,
    next_tts_cut,
    normalize_cues,
    scrub_asr,
    scrub_tts,
    skip_empty_delta,
)


def test_mood_lead_becomes_cue() -> None:
    assert normalize_cues("Excited, hello") == "[excited] hello"


def test_cue_lowercase() -> None:
    assert normalize_cues("[Excited] Hello.") == "[excited] Hello."


def test_laugh_aliases() -> None:
    assert "[laughing]" in normalize_cues("[laugh] ha")
    assert "[laughing]" in normalize_cues("[laughs] ha")


def test_sigh_and_chuckle_aliases() -> None:
    assert "[sighing]" in normalize_cues("[sigh] ok")
    assert "[chuckling]" in normalize_cues("[chuckle] ha")


def test_stacked_leading_cues_kept() -> None:
    assert normalize_cues("[sad][whispering] I miss you") == "[sad] [whispering] I miss you"
    assert "[whisper in small voice]" in normalize_cues("[whisper in small voice] come closer")


def test_whisper_aliases() -> None:
    assert "[whispering]" in normalize_cues("[whispers] psst")
    assert "[whispering]" in normalize_cues("[whisper] psst")
    assert "[whispering]" in normalize_cues("<whisper>psst</whisper>")


def test_pause_alias_vs_moss_duration() -> None:
    assert "[break]" in normalize_cues("[pause] wait")
    stripped = scrub_tts("hello [pause 3.2s] there")
    assert "pause" not in stripped.lower()
    assert "hello" in stripped
    assert "there" in stripped


def test_unclosed_cue_is_junk() -> None:
    assert is_tts_junk("[warm, leftover")


def test_tts_junk_thin_and_narration() -> None:
    assert is_tts_junk("")
    assert is_tts_junk("hi")
    assert is_tts_junk("She smiles and leans closer now")
    assert not is_tts_junk("[clear] Hello there friend")
    assert not is_tts_junk("你好，很开心认识你")


def test_cjk_asr_keeps_speech_drops_thanks() -> None:
    assert is_asr_hallucination("谢谢观看")
    assert not is_asr_hallucination("你好，很开心认识你")
    assert not is_asr_hallucination("<|speaker:0|>你好")


def test_asr_english_not_nuked_by_script() -> None:
    assert not is_asr_hallucination("hello there friend")


def test_asr_emoji_and_mixed() -> None:
    assert is_asr_hallucination("🔥🔥🔥")
    assert not is_asr_hallucination("AB你")


def test_aside_and_bold_stripped() -> None:
    out = scrub_tts("Hello (aside) *bold* world.")
    assert "aside" not in out
    assert "*" not in out
    assert "Hello" in out
    assert "world" in out


def test_s1_parens_become_cues() -> None:
    out = normalize_cues(scrub_tts("(happy) Hello there friend (break)"))
    assert "[happy]" in out
    assert "[break]" in out
    assert "(" not in out


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


def test_scrub_tts_keeps_fish_control_tokens() -> None:
    speaker = scrub_tts("<|speaker:0|> [happy] Hello there friend")
    assert "<|speaker:0|>" in speaker
    assert "[happy]" in speaker
    phoneme = scrub_tts("<|phoneme_start|>HH AH0 L OW1<|phoneme_end|>")
    assert "<|phoneme_start|>" in phoneme
    assert "<|phoneme_end|>" in phoneme
    assert "HH AH0 L OW1" in phoneme
    asr = scrub_asr("<|speaker:0|> hello there")
    assert "<|" not in asr
    assert "hello there" in asr


def test_speaker_and_timestamp_asr() -> None:
    out = scrub_asr("[0.0 - 1.2] Speaker 1: hello there")
    assert "Speaker" not in out
    assert "0.0" not in out
    assert "hello there" in out


def test_scrub_asr_keeps_speakers_when_asked() -> None:
    out = scrub_asr("Speaker 1: hello there", strip_speakers=False)
    assert "Speaker 1" in out
    assert "hello there" in out


def test_backchannel() -> None:
    assert is_backchannel("yeah")
    assert is_backchannel("uh huh")
    assert is_backchannel("mm hmm")
    assert not is_backchannel("yeah can you repeat that")


def test_quit_utterance() -> None:
    assert is_quit_utterance("bye")
    assert is_quit_utterance("Goodbye!")
    assert not is_quit_utterance("don't stop now")


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


def test_extract_quoted_keeps_quotes() -> None:
    spoken = extract_quoted_speech(scrub_tts('[warm] "Loud and clear." Stage note.'))
    assert "Loud and clear" in spoken
    assert "Stage note" not in spoken


def test_extract_quoted_open_passthrough_and_empty() -> None:
    open_q = extract_quoted_speech('[warm] "hello there friend')
    assert "hello there friend" in open_q
    assert extract_quoted_speech('He said "x" but wait') == ""
    assert extract_quoted_speech("no quotes here at all") == "no quotes here at all"
    assert extract_quoted_speech('[warm] "[clear]"') == '[warm] "[clear]"'


def test_skip_empty_delta() -> None:
    assert skip_empty_delta("  ")
    assert not skip_empty_delta("hi")


def test_suite_defaults_and_timing() -> None:
    d = SuiteDefaults()
    assert d.tts_model == "s2.1-pro"
    assert d.tts_partial_chars == 40
    line = LatencySnapshot(ttfa=12.4).log_line()
    assert "ttfa=12ms" in line
    assert "srt=-1" in line
