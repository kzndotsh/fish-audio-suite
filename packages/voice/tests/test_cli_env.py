from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from fish_audio_suite_voice import signals
from fish_audio_suite_voice.cli import _parse_device, apply_cli_env_files, cfg, run_loop
from fish_audio_suite_voice.duplex import EXIT_FATAL
from fish_audio_suite_voice.signals import request_quit


def test_env_file_fills_missing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        "FISH_VOICE_ID=vid-from-file\nexport FISH_API_KEY='key-from-file'\n", encoding="utf-8"
    )
    missing = tmp_path / "missing.env"
    loaded = apply_cli_env_files([path], required=True)
    assert path in loaded
    assert not missing.exists()
    assert cfg().fish_voice_id == "vid-from-file"
    assert cfg().fish_api_key == "key-from-file"


def test_process_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_ID", "from-shell")
    path = tmp_path / "voice.env"
    path.write_text("FISH_VOICE_ID=from-file\n", encoding="utf-8")
    apply_cli_env_files([path], required=True)
    assert cfg().fish_voice_id == "from-shell"


def test_first_file_wins_later_fills_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_LLM_MODEL", raising=False)
    first = tmp_path / "a.env"
    first.write_text("FISH_VOICE_ID=a\n", encoding="utf-8")
    second = tmp_path / "b.env"
    second.write_text("FISH_VOICE_ID=b\nFISH_LLM_MODEL=openai/gpt-4o-mini\n", encoding="utf-8")
    apply_cli_env_files([first, second], required=True)
    c = cfg()
    assert c.fish_voice_id == "a"
    assert c.llm_model == "openai/gpt-4o-mini"


def test_env_file_bad_utf8_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    bad = tmp_path / "bad.env"
    bad.write_bytes(b"FISH_VOICE_ID=from-bad\xff\n")
    good = tmp_path / "good.env"
    good.write_text("FISH_VOICE_ID=from-good\n", encoding="utf-8")
    loaded = apply_cli_env_files([bad, good], required=True)
    assert loaded == [good]
    assert cfg().fish_voice_id == "from-good"
    assert "not utf-8" in capsys.readouterr().err


def test_llm_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FISH_LLM_BASE",
        "OPENROUTER_BASE_URL",
        "FISH_LLM_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "FISH_LLM_MODEL",
        "OPENROUTER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/fallback")
    c = cfg()
    assert c.llm_base == "https://example.test/v1"
    assert c.llm_key == "openai-key"
    assert c.llm_model == "vendor/fallback"
    monkeypatch.setenv("FISH_LLM_BASE", "")
    monkeypatch.setenv("FISH_LLM_KEY", "  fish-key  ")
    monkeypatch.setenv("FISH_LLM_MODEL", "   ")
    c = cfg()
    assert c.llm_base == ""
    assert c.llm_key == "fish-key"
    assert c.llm_model == "vendor/fallback"
    monkeypatch.setenv("FISH_LLM_BASE", "  ")
    assert cfg().llm_base == ""
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1/")
    assert cfg().llm_base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_MODEL", " kept/model ")
    assert cfg().llm_model == "kept/model"


def test_temperature_and_top_p_stay_in_unit_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TEMPERATURE", "5")
    monkeypatch.setenv("FISH_TOP_P", "-1")
    settings = cfg()
    assert settings.fish_temperature == 1.0
    assert settings.fish_top_p == 0.0


def test_speed_and_chunk_stay_in_fish_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_SPEED", "9")
    monkeypatch.setenv("FISH_CHUNK_LENGTH", "900")
    monkeypatch.setenv("FISH_MIN_CHUNK_LENGTH", "250")
    settings = cfg()
    assert settings.fish_speed == 2.0
    assert settings.fish_chunk == 300
    assert settings.fish_min_chunk == 100
    monkeypatch.setenv("FISH_BASE", "http://127.0.0.1:8080")
    monkeypatch.setenv("FISH_CHUNK_LENGTH", "800")
    assert cfg().fish_chunk == 800


def test_sample_rate_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_SAMPLE_RATE", "0")
    assert cfg().fish_sample_rate == 44100
    monkeypatch.setenv("FISH_SAMPLE_RATE", "-1")
    assert cfg().fish_sample_rate == 44100
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16000")
    assert cfg().fish_sample_rate == 16000


def test_run_loop_rejects_bad_playback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_VOICE_ID", "v")
    monkeypatch.setenv("FISH_LLM_KEY", "k")
    monkeypatch.setenv("FISH_LLM_MODEL", "m")
    c = replace(cfg(), playback="nope")
    assert asyncio.run(run_loop(c)) == EXIT_FATAL
    assert "unknown playback sink" in capsys.readouterr().err


def test_playback_mode_is_lowercase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_PLAYBACK", " MPV ")
    assert cfg().playback == "mpv"


def test_catalog_model_and_latency_are_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TTS_MODEL", "S2.1-PRO")
    monkeypatch.setenv("FISH_LATENCY", " Normal ")
    settings = cfg()
    assert settings.fish_tts_model == "s2.1-pro"
    assert settings.fish_latency == "normal"
    monkeypatch.setenv("FISH_LATENCY", "turbo")
    monkeypatch.setenv("FISH_TTS_MODEL", "MyModel")
    settings = cfg()
    assert settings.fish_tts_model == "MyModel"
    assert settings.fish_latency == "normal"


def test_parse_device_strips_index_and_blank() -> None:
    assert _parse_device(None) is None
    assert _parse_device("") is None
    assert _parse_device("  ") is None
    assert _parse_device(" 3 ") == 3
    assert _parse_device(" hw:0 ") == "hw:0"


def test_request_quit_sets_mic_and_turn_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    stop = threading.Event()
    turn = threading.Event()
    llm = threading.Event()
    monkeypatch.setattr(signals, "STOP_RECORD", stop)
    monkeypatch.setattr(signals.TURN, "cancel", turn)
    monkeypatch.setattr(signals.TURN, "llm_cancel", llm)
    request_quit()
    assert stop.is_set()
    assert turn.is_set()
    assert llm.is_set()
