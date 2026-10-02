from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    LatencySnapshot,
    SuiteDefaults,
    chunk_length_hi,
    clamp_num,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_off,
    env_text,
    env_token,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    known_tts_model,
    number_or,
)


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
    assert clamp_num(True, 0.0, 1.0, 0.7, float) == 0.7
    assert number_or("16000.0", 44100, int) == 16000
    assert number_or("16,000", 44100, int) == 44100
    assert number_or("16,5", 44100, int) == 44100
    assert number_or(True, 0.0, float) == 0.0
    assert number_or(False, 3, int) == 3


def test_known_model_and_latency() -> None:
    assert known_tts_model(" S2.1-PRO ") == "s2.1-pro"
    assert known_tts_model("MyModel") == "MyModel"
    assert known_tts_model("custom\r\nX-Injected: 1") == "s2.1-pro"
    assert known_tts_model("MyModel\ud800") == "s2.1-pro"
    assert known_latency(" Normal ", "balanced") == "normal"
    assert known_latency("turbo", "balanced") == "balanced"


def test_suite_defaults_and_timing() -> None:
    d = SuiteDefaults()
    assert d.tts_model == "s2.1-pro"
    assert d.tts_partial_chars == 40
    assert d.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "no markdown" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "never spoken" in DEFAULT_SYSTEM_PROMPT
    assert "[excited]" in DEFAULT_SYSTEM_PROMPT
    assert "change the cue" in DEFAULT_SYSTEM_PROMPT
    assert "do not tag every sentence" not in DEFAULT_SYSTEM_PROMPT
    assert "first_audio=850ms" in LatencySnapshot(first_audio=850).log_line()
    line = LatencySnapshot(ttfa=12.4).log_line()
    assert "ttfa=12ms" in line
    assert "asr=" not in line
    heard = LatencySnapshot(asr_ms=40).log_line()
    assert "asr=40ms" in heard
    assert "trace=" not in line
    traced = LatencySnapshot(ttfa=12.4, trace_id="4bf92f3577b34da6a3ce929d0e0e4736").log_line()
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
    snapshot = LatencySnapshot(1.0, 2.0, 3.0, 4.0, 5.0, "trace")
    assert snapshot.voice_to_voice == 5.0
    assert snapshot.trace_id == "trace"
    assert snapshot.first_audio is None


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
        "fish_non_object",
        "fish_transport_error",
        "mood_lead_hold_at",
        "sentence_closer_hold_at",
    ):
        assert name in kit.__all__
        assert callable(getattr(kit, name)) or getattr(kit, name)
