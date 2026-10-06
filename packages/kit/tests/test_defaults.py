from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    LatencySnapshot,
    SuiteDefaults,
    chunk_length_hi,
    clamp_number,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_off,
    env_text,
    env_token,
    is_insecure_fish_base,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    normalize_tts_model,
    parse_number,
)


def test_bitrate_snaps_to_documented_values() -> None:
    assert known_mp3_bitrate(64) == 64
    assert known_mp3_bitrate(96) == 128
    assert known_mp3_bitrate(192) == 192
    assert known_opus_bitrate(24000) == 24000
    assert known_opus_bitrate(1) == -1000


def test_clamp_number_keeps_fish_ranges() -> None:
    assert clamp_number(900, 100, chunk_length_hi("https://api.fish.audio"), 200, int) == 300
    assert clamp_number(800, 100, chunk_length_hi("http://127.0.0.1:8080"), 200, int) == 800
    assert clamp_number("nope", 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_number(float("nan"), 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_number(float("inf"), 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_number(float("inf"), 100, 300, 200, int) == 200
    assert clamp_number(9, 0.5, 2.0, 1.05, float) == 2.0
    assert clamp_number(True, 0.0, 1.0, 0.7, float) == 0.7
    assert parse_number("16000.0", 44100, int) == 16000
    assert parse_number("16,000", 44100, int) == 44100
    assert parse_number("16,5", 44100, int) == 44100
    assert parse_number(True, 0.0, float) == 0.0
    assert parse_number(False, 3, int) == 3


def test_known_model_and_latency() -> None:
    assert normalize_tts_model(" S2.1-PRO ") == "s2.1-pro"
    assert normalize_tts_model("MyModel") == "MyModel"
    assert normalize_tts_model("custom\r\nX-Injected: 1") == "s2.1-pro"
    assert normalize_tts_model("MyModel\ud800") == "s2.1-pro"
    assert known_latency(" Normal ", "balanced") == "normal"
    assert known_latency("turbo", "balanced") == "balanced"


def test_suite_defaults_and_timing() -> None:
    d = SuiteDefaults()
    assert d.tts_model == "s2.1-pro"
    assert d.tts_partial_chars == 40
    assert d.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "no markdown" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "never spoken" in DEFAULT_SYSTEM_PROMPT
    assert "excited," in DEFAULT_SYSTEM_PROMPT  # the list of one-word moods
    assert "never a description" in DEFAULT_SYSTEM_PROMPT
    assert "Every sentence starts with its own cue" in DEFAULT_SYSTEM_PROMPT
    assert "change the cue" in DEFAULT_SYSTEM_PROMPT
    assert "do not tag every sentence" not in DEFAULT_SYSTEM_PROMPT
    assert "first_audio=850ms" in LatencySnapshot(first_audio_ms=850).log_line()
    line = LatencySnapshot(tts_first_audio_ms=12.4).log_line()
    assert "tts_first_audio=12ms" in line
    assert "asr=" not in line
    heard = LatencySnapshot(asr_ms=40).log_line()
    assert "asr=40ms" in heard
    assert "trace=" not in line
    traced = LatencySnapshot(
        tts_first_audio_ms=12.4, trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
    ).log_line()
    assert "trace=4bf92f3577b34da6a3ce929d0e0e4736" in traced


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
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://api.fish.audio"
    monkeypatch.setenv("FISH_TEST_BASE", "   ")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio/") == "https://api.fish.audio"
    monkeypatch.setenv("FISH_TEST_BASE", "  https://example.test/v1/  ")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    monkeypatch.setenv("FISH_TEST_BASE", "https://example.test/v1\nbad")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    assert env_bool("FISH_TEST_FLAG_MISSING", default=True) is True


@pytest.mark.parametrize(
    "banned",
    ["decline", "refuse", "adult", "character", "roleplay", "scene"],
)
def test_the_default_prompt_is_neutral(banned: str) -> None:
    assert banned not in DEFAULT_SYSTEM_PROMPT.lower()


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.fish.audio", 300),
        ("https://api.fish.audio/v1", 300),
        ("API.FISH.AUDIO", 300),
        ("api.fish.audio:443", 300),
        ("https://eu.api.fish.audio", 300),
        ("http://127.0.0.1:8080", 1000),
        ("http://localhost:8080/proxy/api.fish.audio", 1000),
        ("https://api.fish.audio.evil.example", 1000),
        ("https://notapi.fish.audio.example", 1000),
        ("", 1000),
    ],
)
def test_chunk_length_hi_reads_the_hostname(base: str, expected: int) -> None:
    assert chunk_length_hi(base) == expected


def test_chunk_length_hi_can_be_forced() -> None:
    assert chunk_length_hi("https://fish.internal", self_hosted=False) == 300
    assert chunk_length_hi("https://api.fish.audio", self_hosted=True) == 1000


def test_latency_snapshot_keeps_its_positional_fields() -> None:
    snapshot = LatencySnapshot(1.0, 2.0, 3.0, 4.0, 5.0, "trace", 6.0)
    assert snapshot == LatencySnapshot(
        asr_ms=1.0,
        llm_first_token_ms=2.0,
        tts_first_text_ms=3.0,
        tts_first_audio_ms=4.0,
        voice_to_voice_ms=5.0,
        trace_id="trace",
        first_audio_ms=6.0,
    )
    assert LatencySnapshot(1.0).first_audio_ms is None


def test_latency_snapshot_log_line_names_every_time() -> None:
    line = LatencySnapshot(1.0, 2.0, 3.0, 4.0, 5.0, "t", 6.0).log_line()
    assert line == (
        "[timing asr=1ms llm_first_token=2ms tts_first_text=3ms tts_first_audio=4ms"
        " first_audio=6ms voice_to_voice=5ms trace=t]"
    )


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("api.fish.audio/v1//edge", 300),
        ("https://api.fish.audio", 300),
        ("//api.fish.audio", 300),
        ("http://127.0.0.1:8080//x", 1000),
        ("evil.com/api.fish.audio//", 1000),
    ],
)
def test_cloud_is_detected_from_the_host_even_with_a_double_slash_in_the_path(
    base: str, expected: int
) -> None:
    assert chunk_length_hi(base) == expected


def test_the_package_root_still_exports_the_helpers_it_used_to() -> None:
    import fish_audio_suite_kit as kit

    for name in (
        "FISH_LATENCIES",
        "canonical_traceparent",
        "fish_error_body",
        "describe_transport_error",
        "mood_lead_hold_at",
        "sentence_closer_hold_at",
    ):
        assert name in kit.__all__
        assert callable(getattr(kit, name)) or getattr(kit, name)


@pytest.mark.parametrize(
    ("base", "insecure"),
    [
        ("http://10.0.0.5:8080", True),
        ("http://fish.internal", True),
        ("HTTP://Fish.Example.com/v1", True),
        ("https://api.fish.audio", False),
        ("https://10.0.0.5:8080", False),
        ("http://127.0.0.1:8080", False),
        ("http://127.5.5.5", False),
        ("http://localhost:8080", False),
        ("http://api.localhost", False),
        ("http://[::1]:8080", False),
        ("http://[2001:db8::1]:8080", True),
        ("api.fish.audio", False),
        ("", False),
        ("http://[::1", False),
    ],
)
def test_is_insecure_fish_base(base: str, insecure: bool) -> None:
    assert is_insecure_fish_base(base) is insecure


@pytest.mark.parametrize(
    "value",
    ["\uff11\uff10", "\u0663", "1_000", "0x10", "1 000", "ten", "nan", "inf", "1e", "--5"],
)
def test_env_numbers_ignore_lookalike_and_malformed_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("FISH_TEST_NUM", value)
    assert env_int("FISH_TEST_NUM", 7) == 7
    assert env_float("FISH_TEST_NUM", 7.5) == 7.5


@pytest.mark.parametrize(
    ("value", "as_int", "as_float"),
    [
        ("10", 10, 10.0),
        (" 10 ", 10, 10.0),
        ("-3", -3, -3.0),
        ("+4", 4, 4.0),
        ("16000.0", 16000, 16000.0),
        ("1e3", 1000, 1000.0),
        ("2.5", 7, 2.5),
        (".5", 7, 0.5),
    ],
)
def test_env_numbers_still_accept_ordinary_decimals(
    monkeypatch: pytest.MonkeyPatch, value: str, as_int: int, as_float: float
) -> None:
    monkeypatch.setenv("FISH_TEST_NUM", value)
    assert env_int("FISH_TEST_NUM", 7) == as_int
    assert env_float("FISH_TEST_NUM", 7.5) == as_float


def test_the_default_asr_model_is_transcribe_1_pro() -> None:
    # Fish recommends pro and bills both models the same.
    assert SuiteDefaults().asr_model == "transcribe-1-pro"
