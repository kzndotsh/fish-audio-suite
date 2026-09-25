from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from fish_audio_suite_voice import signals
from fish_audio_suite_voice.cli import _parse_device, apply_cli_env_files, cfg, main, run_loop
from fish_audio_suite_voice.config import OPENROUTER_API_BASE
from fish_audio_suite_voice.duplex import EXIT_FATAL
from fish_audio_suite_voice.signals import _TurnSignals, request_quit


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


def test_export_tab_and_bom_still_set_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_bytes(b"\xef\xbb\xbfexport\tFISH_VOICE_ID=vid-from-file\n")
    apply_cli_env_files([path], required=True)
    assert cfg().fish_voice_id == "vid-from-file"
    glued = tmp_path / "glued.env"
    glued.write_text("exportFISH_VOICE_ID=vid-glued\n", encoding="utf-8")
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    apply_cli_env_files([glued], required=True)
    assert cfg().fish_voice_id == ""


def test_env_file_drops_an_unquoted_inline_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        'FISH_VOICE_ID=vid-from-file # speaker\nFISH_API_KEY="key # not a comment"\n',
        encoding="utf-8",
    )
    apply_cli_env_files([path], required=True)
    assert cfg().fish_voice_id == "vid-from-file"
    assert cfg().fish_api_key == "key # not a comment"
    commented = tmp_path / "commented.env"
    commented.write_text(
        'FISH_API_KEY="sk-real" # rotated "yesterday"\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    apply_cli_env_files([commented], required=True)
    assert cfg().fish_api_key == "sk-real"
    assert cfg().fish_voice_id == "vid"


def test_quoted_prompt_keeps_the_following_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        'FISH_SYSTEM_PROMPT="You are helpful.\nBe brief."\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    apply_cli_env_files([path], required=True)
    assert cfg().system_prompt == "You are helpful.\nBe brief."
    assert cfg().fish_voice_id == "vid"


def test_single_quoted_backslash_does_not_swallow_the_next_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_text("FISH_SYSTEM_PROMPT='C:\\\\temp\\\\'\nFISH_VOICE_ID=vid\n", encoding="utf-8")
    apply_cli_env_files([path], required=True)
    assert cfg().fish_voice_id == "vid"
    assert cfg().system_prompt == "C:\\\\temp\\\\"


def test_escaped_quote_and_newline_stay_in_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        'FISH_SYSTEM_PROMPT="Say \\"hi\\"\\nthere."\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    apply_cli_env_files([path], required=True)
    assert cfg().system_prompt == 'Say "hi"\nthere.'
    assert cfg().fish_voice_id == "vid"
    monkeypatch.delenv("FISH_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    split = tmp_path / "split.env"
    split.write_text(
        'FISH_SYSTEM_PROMPT="Say \\"hi\nthere."\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    apply_cli_env_files([split], required=True)
    assert cfg().system_prompt == 'Say "hi\nthere.'
    assert cfg().fish_voice_id == "vid"


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
    assert c.llm_base == "https://example.test/v1"
    assert c.llm_key == "fish-key"
    assert c.llm_model == "vendor/fallback"
    monkeypatch.setenv("FISH_LLM_BASE", "  ")
    assert cfg().llm_base == "https://example.test/v1"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "  ")
    assert cfg().llm_base == OPENROUTER_API_BASE
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1/")
    assert cfg().llm_base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1\nbad")
    assert cfg().llm_base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_MODEL", " kept/model ")
    assert cfg().llm_model == "kept/model"


def test_temperature_and_top_p_stay_in_unit_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TEMPERATURE", "5")
    monkeypatch.setenv("FISH_TOP_P", "-1")
    settings = cfg()
    assert settings.temperature == 1.0
    assert settings.top_p == 0.0


def test_speed_and_chunk_stay_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_SPEED", "9")
    monkeypatch.setenv("FISH_CHUNK_LENGTH", "900")
    monkeypatch.setenv("FISH_MIN_CHUNK_LENGTH", "250")
    settings = cfg()
    assert settings.speed == 2.0
    assert settings.chunk_length == 300
    assert settings.min_chunk_length == 100
    monkeypatch.setenv("FISH_BASE", "http://127.0.0.1:8080")
    monkeypatch.setenv("FISH_CHUNK_LENGTH", "800")
    assert cfg().chunk_length == 800


def test_sample_rate_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_SAMPLE_RATE", "0")
    assert cfg().sample_rate == 44100
    monkeypatch.setenv("FISH_SAMPLE_RATE", "-1")
    assert cfg().sample_rate == 44100
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16000")
    assert cfg().sample_rate == 16000
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16000.0")
    assert cfg().sample_rate == 16000
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16,000")
    assert cfg().sample_rate == 16000


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
    assert settings.tts_model == "s2.1-pro"
    assert settings.latency == "normal"
    monkeypatch.setenv("FISH_LATENCY", "turbo")
    monkeypatch.setenv("FISH_TTS_MODEL", "MyModel")
    settings = cfg()
    assert settings.tts_model == "MyModel"
    assert settings.latency == "normal"


def test_parse_device_strips_index_and_blank() -> None:
    assert _parse_device(None) is None
    assert _parse_device("") is None
    assert _parse_device("  ") is None
    assert _parse_device(" 3 ") == 3
    assert _parse_device("1.0") == 1
    assert _parse_device("0.0") == 0
    assert _parse_device("1.5") == "1.5"
    assert _parse_device(" hw:0 ") == "hw:0"
    assert _parse_device("3\nbad") == 3
    assert _parse_device("hw:0\nbad") == "hw:0"


def test_cleared_turn_does_not_cancel_the_old_events() -> None:
    turn = threading.Event()
    llm = asyncio.Event()
    signals_ = _TurnSignals()
    signals_.fire()
    signals_.bind(turn, llm)
    signals_.clear()
    signals_.fire()
    assert not turn.is_set()
    assert not llm.is_set()


def _no_env(*_args: object, **_kwargs: object) -> list[Path]:
    return []


def _no_log(**_kwargs: object) -> None:
    return None


def _fake_cfg() -> object:
    return object()


def _ignore_signal(*_args: object, **_kwargs: object) -> None:
    return None


def _stub_main(monkeypatch: pytest.MonkeyPatch, run: object) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.cli.apply_cli_env_files", _no_env)
    monkeypatch.setattr("fish_audio_suite_voice.cli.configure_voice_logging", _no_log)
    monkeypatch.setattr("fish_audio_suite_voice.cli.cfg", _fake_cfg)
    monkeypatch.setattr("fish_audio_suite_voice.cli.signal.signal", _ignore_signal)
    monkeypatch.setattr("fish_audio_suite_voice.cli.run_loop", run)


def test_main_restarts_a_cancelled_loop_until_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def run(_config: object) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.CancelledError
        return 7

    stop = threading.Event()
    monkeypatch.setattr("fish_audio_suite_voice.cli.STOP_RECORD", stop)
    _stub_main(monkeypatch, run)
    assert main([]) == 7
    assert calls["n"] == 2
    assert not stop.is_set()


def test_main_does_not_restart_after_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def run(_config: object) -> int:
        calls["n"] += 1
        raise asyncio.CancelledError

    stop = threading.Event()
    stop.set()
    monkeypatch.setattr("fish_audio_suite_voice.cli.STOP_RECORD", stop)
    _stub_main(monkeypatch, run)
    assert main([]) == 0
    assert calls["n"] == 1


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
