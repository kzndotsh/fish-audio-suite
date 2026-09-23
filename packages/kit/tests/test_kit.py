from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    CaptionCue,
    FishHttpError,
    LatencySnapshot,
    SuiteDefaults,
    canonical_traceparent,
    chunk_length_hi,
    clamp_num,
    ensure_lead_cue,
    ensure_trace_headers,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_off,
    env_text,
    env_token,
    extract_quoted_speech,
    fish_backoff_seconds,
    fish_error_body,
    format_as_srt,
    format_as_vtt,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_tts_junk,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    known_tts_model,
    make_traceparent,
    next_tts_cut,
    normalize_cues,
    parse_asr_body,
    parse_fish_error,
    scrub_asr,
    scrub_tts,
    should_retry_fish_status,
    skip_empty_delta,
    trace_id_of,
    w3c_trace_headers,
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


def test_inline_chuckle_and_cough() -> None:
    out = normalize_cues("I'll call you back [chuckle] in a minute [cough]")
    assert "[chuckling]" in out
    assert "[cough]" in out
    assert "[coughing]" not in out


def test_ensure_lead_cue_only_when_missing() -> None:
    assert ensure_lead_cue("[curious] yeah") == "[curious] yeah"
    assert ensure_lead_cue("I'll call you back [chuckle] in a minute").startswith("I'll")
    assert ensure_lead_cue("yeah i hear you") == "[clear] yeah i hear you"
    assert ensure_lead_cue("  ") == "  "


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
    assert is_asr_hallucination("嗯")
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


def test_fish_error_message_is_utf8() -> None:
    body = fish_error_body(502, "bad \ud800 byte")
    assert "\ud800" not in str(body["message"])
    str(body["message"]).encode("utf-8")
    err = FishHttpError(502, "bad \ud800 byte")
    err.message.encode("utf-8")


def test_scrub_asr_replaces_lone_surrogates() -> None:
    out = scrub_asr("hello \ud800 there")
    out.encode("utf-8")
    assert out.startswith("hello ")
    assert "\ud800" not in out
    assert "there" in out


def test_scrub_asr_keeps_speakers_when_asked() -> None:
    out = scrub_asr("Speaker 1: hello there", strip_speakers=False)
    assert "Speaker 1" in out
    assert "hello there" in out


def test_backchannel() -> None:
    assert is_backchannel("yeah")
    assert is_backchannel("uh huh")
    assert is_backchannel("mm hmm")
    assert is_backchannel("嗯。")
    assert is_backchannel("嗯嗯")
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
    assert is_asr_hallucination("Subtitles by the Amara.org community")
    assert is_asr_hallucination("I hope you enjoyed the video.")
    loop = "If the sentence is cut off, do not make up words. " * 12
    assert is_asr_hallucination(loop)
    assert not is_asr_hallucination("hello there friend")


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


def test_bitrate_snaps_to_documented_values() -> None:
    assert known_mp3_bitrate(64) == 64
    assert known_mp3_bitrate(96) == 128
    assert known_mp3_bitrate(192) == 192
    assert known_opus_bitrate(24000) == 24000
    assert known_opus_bitrate(1) == -1000


def test_clamp_num_keeps_fish_ranges() -> None:
    assert clamp_num(900, 100, chunk_length_hi("https://api.fish.audio"), 200, int) == 300
    assert clamp_num(800, 100, chunk_length_hi("http://127.0.0.1:8080"), 200, int) == 800
    assert clamp_num("nope", 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_num(float("nan"), 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_num(float("inf"), 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_num(float("inf"), 100, 300, 200, int) == 200
    assert clamp_num(9, 0.5, 2.0, 1.05, float) == 2.0


def test_parse_asr_body_text_or_502() -> None:
    data, text = parse_asr_body({"text": "hello"})
    assert data["text"] == "hello"
    assert text == "hello"
    assert parse_asr_body({})[1] == ""
    assert parse_asr_body({"text": None})[1] == ""
    with pytest.raises(FishHttpError) as missing:
        parse_asr_body(["hello"])
    assert missing.value.status == 502
    with pytest.raises(FishHttpError) as bad_text:
        parse_asr_body({"text": ["hello"]})
    assert bad_text.value.message == "Fish returned a non-object body"


def test_known_model_and_latency() -> None:
    assert known_tts_model(" S2.1-PRO ") == "s2.1-pro"
    assert known_tts_model("MyModel") == "MyModel"
    assert known_latency(" Normal ", "balanced") == "normal"
    assert known_latency("turbo", "balanced") == "balanced"


def test_suite_defaults_and_timing() -> None:
    d = SuiteDefaults()
    assert d.tts_model == "s2.1-pro"
    assert d.tts_partial_chars == 40
    assert d.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "mid-sentence" in DEFAULT_SYSTEM_PROMPT
    assert "Leave [cough] as [cough]" in DEFAULT_SYSTEM_PROMPT
    assert "do not tag every sentence" in DEFAULT_SYSTEM_PROMPT
    assert "You write every cue" in DEFAULT_SYSTEM_PROMPT
    assert "every two sentences" in DEFAULT_SYSTEM_PROMPT
    assert "[excited]" in DEFAULT_SYSTEM_PROMPT
    assert "vague one-word tags" in DEFAULT_SYSTEM_PROMPT
    assert "[playful], [whispering]" not in DEFAULT_SYSTEM_PROMPT
    line = LatencySnapshot(ttfa=12.4).log_line()
    assert "ttfa=12ms" in line
    assert "asr=" not in line
    heard = LatencySnapshot(asr_ms=40).log_line()
    assert "asr=40ms" in heard
    assert "trace=" not in line
    traced = LatencySnapshot(ttfa=12.4, trace_id="4bf92f3577b34da6a3ce929d0e0e4736").log_line()
    assert "trace=4bf92f3577b34da6a3ce929d0e0e4736" in traced


_SAMPLE_PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_canonical_traceparent() -> None:
    assert canonical_traceparent(_SAMPLE_PARENT) == _SAMPLE_PARENT
    assert canonical_traceparent(_SAMPLE_PARENT.upper()) == _SAMPLE_PARENT
    assert canonical_traceparent("not-a-trace") is None
    assert canonical_traceparent("00-" + "0" * 32 + "-" + "0" * 16 + "-01") is None


def test_w3c_trace_headers_and_mint() -> None:
    headers = w3c_trace_headers({"Traceparent": _SAMPLE_PARENT, "tracestate": "congo=t61rcWkgMzE"})
    assert headers["traceparent"] == _SAMPLE_PARENT
    assert headers["tracestate"] == "congo=t61rcWkgMzE"
    assert w3c_trace_headers({}) == {}
    forwarded = ensure_trace_headers({"traceparent": _SAMPLE_PARENT})
    assert forwarded["traceparent"] == _SAMPLE_PARENT
    minted_out = ensure_trace_headers({})
    assert canonical_traceparent(minted_out["traceparent"]) == minted_out["traceparent"]
    minted = make_traceparent()
    assert canonical_traceparent(minted) == minted
    child = make_traceparent(trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
    assert trace_id_of(child) == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert child != minted


def test_fish_error_shape_and_retry_policy() -> None:
    assert should_retry_fish_status(429)
    assert should_retry_fish_status(503)
    assert not should_retry_fish_status(400)
    assert not should_retry_fish_status(401)
    assert not should_retry_fish_status(402)
    assert not should_retry_fish_status(404)
    assert fish_backoff_seconds(0) == 1.0
    assert fish_backoff_seconds(2) == 4.0
    assert parse_fish_error(401, {"message": "Invalid Token", "status": 401}) == {
        "message": "Invalid Token",
        "status": 401,
    }
    assert parse_fish_error(400, b"not json")["message"] == "not json"
    assert parse_fish_error(502, {"error": {"message": "upstream"}})["message"] == "upstream"
    assert fish_error_body(402, "Insufficient credits")["status"] == 402


def test_format_as_srt_and_vtt() -> None:
    cues = [
        CaptionCue(0.0, 0.6, "hello"),
        CaptionCue(0.6, 1.5, "there"),
    ]
    assert format_as_srt(cues) == (
        "1\n00:00:00,000 --> 00:00:00,600\nhello\n\n2\n00:00:00,600 --> 00:00:01,500\nthere\n"
    )
    assert format_as_vtt(cues) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.600\nhello\n\n00:00:00.600 --> 00:00:01.500\nthere\n"
    )
    assert format_as_srt([]) == ""
    assert format_as_vtt([]) == "WEBVTT\n"
    assert format_as_srt([CaptionCue(0.0, 2.0, "  hello  ")]) == (
        "1\n00:00:00,000 --> 00:00:02,000\nhello\n"
    )
    assert format_as_srt([CaptionCue(float("nan"), float("inf"), "hello")]) == (
        "1\n00:00:00,000 --> 00:00:00,000\nhello\n"
    )
    assert format_as_srt([CaptionCue(0.0, 1e308, "hello")]) == (
        "1\n00:00:00,000 --> 00:00:00,000\nhello\n"
    )
    assert format_as_srt([CaptionCue(0.0, 1.0, "hello\n\nthere")]) == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\nthere\n"
    )
    assert format_as_srt([CaptionCue(0.0, 1.0, "hello\r\n\r\nthere")]) == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\nthere\n"
    )
    assert "hello\n\nthere" not in format_as_vtt([CaptionCue(0.0, 1.0, "hello\n\nthere")])


def test_env_number_keeps_default_when_blank_or_junk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TEST_NUM", "12")
    assert env_int("FISH_TEST_NUM", 3) == 12
    assert env_float("FISH_TEST_NUM", 0.5) == 12.0
    monkeypatch.setenv("FISH_TEST_NUM", "1.5")
    assert env_float("FISH_TEST_NUM", 0.5) == 1.5
    assert env_int("FISH_TEST_NUM", 3) == 3
    monkeypatch.setenv("FISH_TEST_NUM", "nope")
    assert env_int("FISH_TEST_NUM", 3) == 3
    assert env_float("FISH_TEST_NUM", 0.5) == 0.5
    monkeypatch.setenv("FISH_TEST_NUM", "nan")
    assert env_float("FISH_TEST_NUM", 0.5) == 0.5
    monkeypatch.setenv("FISH_TEST_NUM", "inf")
    assert env_float("FISH_TEST_NUM", 0.5) == 0.5
    monkeypatch.setenv("FISH_TEST_NUM", "  ")
    assert env_int("FISH_TEST_NUM", 3) == 3
    assert env_int("FISH_TEST_MISSING", 3) == 3
    monkeypatch.setenv("FISH_TEST_FLAG", " YES ")
    assert env_bool("FISH_TEST_FLAG") is True
    monkeypatch.setenv("FISH_TEST_FLAG", "0")
    assert env_bool("FISH_TEST_FLAG", default=True) is False
    monkeypatch.setenv("FISH_TEST_FLAG", "   ")
    assert env_bool("FISH_TEST_FLAG", default=True) is True
    assert env_bool("FISH_TEST_FLAG_MISSING") is False
    assert env_off("FISH_TEST_OFF_MISSING") is False
    monkeypatch.setenv("FISH_TEST_OFF", " maybe ")
    assert env_off("FISH_TEST_OFF") is False
    monkeypatch.setenv("FISH_TEST_OFF", " OFF ")
    assert env_off("FISH_TEST_OFF") is True
    assert env_base("FISH_TEST_BASE_MISSING", "https://api.fish.audio/") == "https://api.fish.audio"
    assert env_text("FISH_TEST_TEXT_MISSING", "plain") == "plain"
    monkeypatch.setenv("FISH_TEST_TEXT", "  kept  ")
    assert env_text("FISH_TEST_TEXT") == "kept"
    monkeypatch.setenv("FISH_TEST_TOKEN", "  ")
    assert env_token("FISH_TEST_TOKEN", "normal") == "normal"
    monkeypatch.setenv("FISH_TEST_TOKEN", " low ")
    assert env_token("FISH_TEST_TOKEN", "normal") == "low"
    monkeypatch.setenv("FISH_TEST_BASE", "https://example.test/v1/")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    monkeypatch.setenv("FISH_TEST_BASE", "")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == ""
    monkeypatch.setenv("FISH_TEST_BASE", "  https://example.test/v1/  ")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    assert env_bool("FISH_TEST_FLAG_MISSING", default=True) is True
