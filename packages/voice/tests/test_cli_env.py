from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from fish_audio_suite_voice.cli import (
    _parse_device,
    apply_cli_env_files,
    cfg,
    main,
    run_loop,
    smoke_test,
)
from fish_audio_suite_voice.config import OPENROUTER_API_BASE
from fish_audio_suite_voice.duplex import EXIT_FATAL
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.signals import DuplexSession, TurnSignals


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
    assert c.llm.model == "openai/gpt-4o-mini"


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
    assert c.llm.base == "https://example.test/v1"
    assert c.llm.key == "openai-key"
    assert c.llm.model == "vendor/fallback"
    monkeypatch.setenv("FISH_LLM_BASE", "")
    monkeypatch.setenv("FISH_LLM_KEY", "  fish-key  ")
    monkeypatch.setenv("FISH_LLM_MODEL", "   ")
    c = cfg()
    assert c.llm.base == "https://example.test/v1"
    assert c.llm.key == "fish-key"
    assert c.llm.model == "vendor/fallback"
    monkeypatch.setenv("FISH_LLM_BASE", "  ")
    assert cfg().llm.base == "https://example.test/v1"
    monkeypatch.setenv("OPENROUTER_BASE_URL", "  ")
    assert cfg().llm.base == OPENROUTER_API_BASE
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1/")
    assert cfg().llm.base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1\nbad")
    assert cfg().llm.base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_MODEL", " kept/model ")
    assert cfg().llm.model == "kept/model"


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
    assert cfg().sample_rate == 44100


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
    signals_ = TurnSignals()
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


def _stub_main(monkeypatch: pytest.MonkeyPatch, run: object) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.cli.apply_cli_env_files", _no_env)
    monkeypatch.setattr("fish_audio_suite_voice.cli.configure_voice_logging", _no_log)
    monkeypatch.setattr("fish_audio_suite_voice.cli.cfg", _fake_cfg)
    monkeypatch.setattr("fish_audio_suite_voice.cli.warn_if_insecure_base", lambda _c: False)
    monkeypatch.setattr("fish_audio_suite_voice.cli.run_loop", run)


def test_main_exits_2_on_an_unexpected_cancel_without_restarting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = {"n": 0}

    async def run(_config: object) -> int:
        calls["n"] += 1
        raise asyncio.CancelledError

    _stub_main(monkeypatch, run)
    assert main([]) == EXIT_FATAL
    assert calls["n"] == 1
    assert "cancelled unexpectedly" in capsys.readouterr().err


def test_main_says_bye_on_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(_config: object) -> int:
        raise KeyboardInterrupt

    _stub_main(monkeypatch, run)
    assert main([]) == 0


def test_request_quit_sets_mic_and_turn_cancel() -> None:
    session = DuplexSession()
    turn = threading.Event()
    llm = asyncio.Event()
    session.turn.bind(turn, llm)
    session.request_quit()
    assert session.stop.is_set()
    assert turn.is_set()
    assert llm.is_set()


def test_request_quit_from_a_thread_sets_the_llm_event_on_its_loop() -> None:
    async def scenario() -> bool:
        session = DuplexSession()
        llm = asyncio.Event()
        session.turn.bind(threading.Event(), llm)
        worker = threading.Thread(target=session.request_quit)
        worker.start()
        await asyncio.wait_for(llm.wait(), timeout=1)
        worker.join()
        return session.stop.is_set()

    assert asyncio.run(scenario())


def test_sessions_do_not_share_quit_state() -> None:
    first = DuplexSession()
    second = DuplexSession()
    first.request_quit()
    assert first.stop.is_set()
    assert not second.stop.is_set()


def test_llm_keys_never_cross_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FISH_LLM_BASE",
        "OPENROUTER_BASE_URL",
        "FISH_LLM_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "FISH_LLM_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    assert cfg().llm.openrouter
    assert cfg().llm.key == ""
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only")
    assert cfg().llm.key == "router-only"
    monkeypatch.setenv("FISH_LLM_BASE", "http://localhost:11434/v1")
    local = cfg().llm
    assert not local.openrouter
    assert local.backend == "openai"
    assert local.key == "openai-only"


def test_llm_backend_env_picks_the_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_LLM_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("FISH_LLM_BACKEND", "openrouter")
    assert cfg().llm.openrouter
    monkeypatch.setenv("FISH_LLM_BACKEND", "bogus")
    assert cfg().llm.backend == "openai"
    assert "FISH_LLM_BACKEND" in capsys.readouterr().err


def test_llm_nitro_and_continuation_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FISH_LLM_NITRO", "FISH_LLM_CONTINUE"):
        monkeypatch.delenv(name, raising=False)
    assert not cfg().llm.nitro
    assert not cfg().llm.continuation
    monkeypatch.setenv("FISH_LLM_NITRO", "1")
    monkeypatch.setenv("FISH_LLM_CONTINUE", "yes")
    assert cfg().llm.nitro
    assert cfg().llm.continuation


def test_bad_numeric_env_warns_and_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "lots")
    monkeypatch.setenv("FISH_LLM_TEMPERATURE", "9")
    monkeypatch.setenv("FISH_HISTORY_TURNS", "0")
    c = cfg()
    assert c.barge.hit_frames == 10
    assert c.llm.temperature == 0.8
    assert c.history_turns == 20
    err = capsys.readouterr().err
    for name in ("FISH_VOICE_BARGE_FRAMES", "FISH_LLM_TEMPERATURE", "FISH_HISTORY_TURNS"):
        assert name in err


def test_roleplay_features_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_MOOD_LEAD", raising=False)
    monkeypatch.delenv("FISH_DROP_NARRATION", raising=False)
    c = cfg()
    assert not c.mood_lead
    assert not c.drop_narration
    monkeypatch.setenv("FISH_MOOD_LEAD", "1")
    monkeypatch.setenv("FISH_DROP_NARRATION", "true")
    c = cfg()
    assert c.mood_lead
    assert c.drop_narration


def test_asr_model_env_accepts_only_native_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_ASR_MODEL", raising=False)
    assert cfg().asr_model == "transcribe-1"
    monkeypatch.setenv("FISH_ASR_MODEL", " Transcribe-1-Pro ")
    assert cfg().asr_model == "transcribe-1-pro"
    monkeypatch.setenv("FISH_ASR_MODEL", "whisper-1")
    assert cfg().asr_model == "transcribe-1"


class _SmokeTts:
    def speak_isolated(self, text: str, sink: PlaybackSink) -> object:
        sink.start()
        sink.write(b"\x00\x01" * 2000)
        sink.finish()
        return type("R", (), {"error_status": None, "error_message": None, "bytes_played": 4000})()


def test_smoke_writes_to_the_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.cli._fish_tts", lambda *_a: _SmokeTts())
    c = replace(cfg(), fish_api_key="k", fish_voice_id="v")
    out = tmp_path / "hello.wav"
    assert asyncio.run(smoke_test(c, out)) == 0
    assert out.stat().st_size > 1000


def test_smoke_default_path_is_unique_and_not_a_shared_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.cli._fish_tts", lambda *_a: _SmokeTts())
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    c = replace(cfg(), fish_api_key="k", fish_voice_id="v")
    assert asyncio.run(smoke_test(c)) == 0
    assert asyncio.run(smoke_test(c)) == 0
    files = sorted(p.name for p in tmp_path.glob("fish-audio-suite-smoke-*.wav"))
    assert len(files) == 2
    assert "fish-audio-suite-smoke.wav" not in {p.name for p in tmp_path.iterdir()}
    assert "smoke: wrote" in capsys.readouterr().out


def test_repeat_window_env_is_read_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_REPEAT_WINDOW_S", raising=False)
    assert cfg().repeat_window_s == 1.5
    monkeypatch.setenv("FISH_VOICE_REPEAT_WINDOW_S", "0")
    assert cfg().repeat_window_s == 0.0
    monkeypatch.setenv("FISH_VOICE_REPEAT_WINDOW_S", "-3")
    assert cfg().repeat_window_s == 1.5


def test_stream_tts_is_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_STREAM_TTS", raising=False)
    assert cfg().stream_tts is False
    monkeypatch.setenv("FISH_STREAM_TTS", "1")
    assert cfg().stream_tts is True


def test_a_blank_llm_key_does_not_hide_the_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fish_audio_suite_voice.tune import LlmTune

    for name in ("FISH_LLM_BASE", "OPENROUTER_BASE_URL", "FISH_LLM_BACKEND", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FISH_LLM_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert LlmTune.from_env().key == "or-key"
    monkeypatch.setenv("FISH_LLM_KEY", "own-key")
    assert LlmTune.from_env().key == "own-key"


def test_a_malformed_llm_base_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fish_audio_suite_voice.tune import LlmTune, openrouter_host

    monkeypatch.delenv("FISH_LLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("FISH_LLM_BASE", "http://[::1")
    tune = LlmTune.from_env()
    assert tune.base == OPENROUTER_API_BASE
    assert "not a valid URL" in capsys.readouterr().err
    assert openrouter_host("http://[::1") is False


def test_main_checks_the_fish_and_llm_base_urls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(_config: object) -> int:
        raise KeyboardInterrupt

    _stub_main(monkeypatch, run)
    seen: list[object] = []

    def record(config: object) -> bool:
        seen.append(config)
        return False

    monkeypatch.setattr("fish_audio_suite_voice.cli.warn_if_insecure_base", record)
    assert main([]) == 0
    assert len(seen) == 1
