from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from voice_fakes import make_result

from fish_audio_suite_voice.capture.portaudio import parse_device
from fish_audio_suite_voice.cli.main import main, run_loop, smoke_test
from fish_audio_suite_voice.config.dotenv import apply_cli_env_files
from fish_audio_suite_voice.config.settings import load_config
from fish_audio_suite_voice.llm.tune import OPENROUTER_API_BASE
from fish_audio_suite_voice.session.events import ExitCode
from fish_audio_suite_voice.tts.playback import PlaybackSink
from fish_audio_suite_voice.tts.speaker import FishSpeaker
from fish_audio_suite_voice.tts.websocket import TtsResult


def test_env_file_fills_missing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        "FISH_VOICE_ID=vid-from-file\nexport FISH_API_KEY='key-from-file'\n", encoding="utf-8"
    )
    loaded = apply_cli_env_files([path], required=True)
    assert loaded == [path]
    assert load_config().fish_voice_id == "vid-from-file"
    assert load_config().fish_api_key == "key-from-file"


def test_export_tab_and_bom_still_set_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_bytes(b"\xef\xbb\xbfexport\tFISH_VOICE_ID=vid-from-file\n")
    apply_cli_env_files([path], required=True)
    assert load_config().fish_voice_id == "vid-from-file"
    glued = tmp_path / "glued.env"
    glued.write_text("exportFISH_VOICE_ID=vid-glued\n", encoding="utf-8")
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    apply_cli_env_files([glued], required=True)
    assert load_config().fish_voice_id == ""


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
    assert load_config().fish_voice_id == "vid-from-file"
    assert load_config().fish_api_key == "key # not a comment"
    commented = tmp_path / "commented.env"
    commented.write_text(
        'FISH_API_KEY="sk-real" # rotated "yesterday"\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    apply_cli_env_files([commented], required=True)
    assert load_config().fish_api_key == "sk-real"
    assert load_config().fish_voice_id == "vid"


def test_quoted_prompt_keeps_the_following_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_VOICE_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        'FISH_VOICE_SYSTEM_PROMPT="You are helpful.\nBe brief."\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    apply_cli_env_files([path], required=True)
    assert load_config().prompt.text == "You are helpful.\nBe brief."
    assert load_config().fish_voice_id == "vid"


def test_single_quoted_backslash_does_not_swallow_the_next_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_VOICE_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        "FISH_VOICE_SYSTEM_PROMPT='C:\\\\temp\\\\'\nFISH_VOICE_ID=vid\n", encoding="utf-8"
    )
    apply_cli_env_files([path], required=True)
    assert load_config().fish_voice_id == "vid"
    assert load_config().prompt.text == "C:\\\\temp\\\\"


def test_escaped_quote_and_newline_stay_in_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FISH_VOICE_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        'FISH_VOICE_SYSTEM_PROMPT="Say \\"hi\\"\\nthere."\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    apply_cli_env_files([path], required=True)
    assert load_config().prompt.text == 'Say "hi"\nthere.'
    assert load_config().fish_voice_id == "vid"
    monkeypatch.delenv("FISH_VOICE_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    split = tmp_path / "split.env"
    split.write_text(
        'FISH_VOICE_SYSTEM_PROMPT="Say \\"hi\nthere."\nFISH_VOICE_ID=vid\n',
        encoding="utf-8",
    )
    apply_cli_env_files([split], required=True)
    assert load_config().prompt.text == 'Say "hi\nthere.'
    assert load_config().fish_voice_id == "vid"


def test_process_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_ID", "from-shell")
    path = tmp_path / "voice.env"
    path.write_text("FISH_VOICE_ID=from-file\n", encoding="utf-8")
    apply_cli_env_files([path], required=True)
    assert load_config().fish_voice_id == "from-shell"


def test_first_file_wins_later_fills_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_LLM_MODEL", raising=False)
    first = tmp_path / "a.env"
    first.write_text("FISH_VOICE_ID=a\n", encoding="utf-8")
    second = tmp_path / "b.env"
    second.write_text("FISH_VOICE_ID=b\nFISH_LLM_MODEL=openai/gpt-4o-mini\n", encoding="utf-8")
    apply_cli_env_files([first, second], required=True)
    c = load_config()
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
    assert load_config().fish_voice_id == "from-good"
    assert "not utf-8" in capsys.readouterr().err


def test_llm_settings_read_only_the_fish_llm_names(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FISH_LLM_BASE",
        "FISH_LLM_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "FISH_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    # Old OpenRouter names are not read before 1.0: they move nothing.
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/old-name")
    c = load_config()
    assert c.llm.base == OPENROUTER_API_BASE
    assert c.llm.model == ""
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("FISH_LLM_MODEL", "vendor/model")
    c = load_config()
    assert c.llm.base == "https://example.test/v1"
    assert c.llm.api_key == "openai-key"
    assert c.llm.model == "vendor/model"
    monkeypatch.setenv("FISH_LLM_API_KEY", "  fish-key  ")
    monkeypatch.setenv("FISH_LLM_MODEL", "   ")
    c = load_config()
    assert c.llm.api_key == "fish-key"
    assert c.llm.model == ""
    monkeypatch.setenv("FISH_LLM_BASE", "  ")
    assert load_config().llm.base == OPENROUTER_API_BASE
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1/")
    assert load_config().llm.base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_BASE", "https://example.test/v1\nbad")
    assert load_config().llm.base == "https://example.test/v1"
    monkeypatch.setenv("FISH_LLM_MODEL", " kept/model ")
    assert load_config().llm.model == "kept/model"


def test_temperature_and_top_p_stay_in_unit_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TEMPERATURE", "5")
    monkeypatch.setenv("FISH_TOP_P", "-1")
    settings = load_config()
    assert settings.tts.temperature == 1.0
    assert settings.tts.top_p == 0.0


def test_speed_and_chunk_stay_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_SPEED", "9")
    monkeypatch.setenv("FISH_CHUNK_LENGTH", "900")
    monkeypatch.setenv("FISH_MIN_CHUNK_LENGTH", "250")
    settings = load_config()
    assert settings.tts.speed == 2.0
    assert settings.tts.chunk_length == 300
    assert settings.tts.min_chunk_length == 100
    # A self-hosted HTTP API takes up to 1000, but this websocket goes through the SDK,
    # which refuses more than 300 for any base, so 300 is what is sent.
    monkeypatch.setenv("FISH_BASE", "http://127.0.0.1:8080")
    monkeypatch.setenv("FISH_CHUNK_LENGTH", "800")
    assert load_config().tts.chunk_length == 300


def test_sample_rate_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_SAMPLE_RATE", "0")
    assert load_config().tts.sample_rate == 44100
    monkeypatch.setenv("FISH_SAMPLE_RATE", "-1")
    assert load_config().tts.sample_rate == 44100
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16000")
    assert load_config().tts.sample_rate == 16000
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16000.0")
    assert load_config().tts.sample_rate == 16000
    monkeypatch.setenv("FISH_SAMPLE_RATE", "16,000")
    assert load_config().tts.sample_rate == 44100


def test_run_loop_rejects_bad_playback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_VOICE_ID", "v")
    monkeypatch.setenv("FISH_LLM_API_KEY", "k")
    monkeypatch.setenv("FISH_LLM_MODEL", "m")
    c = replace(load_config(), playback="nope")
    assert asyncio.run(run_loop(c)) == ExitCode.FATAL
    assert "unknown playback sink" in capsys.readouterr().err


def test_playback_mode_is_lowercase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_PLAYBACK", " MPV ")
    assert load_config().playback == "mpv"


def test_catalog_model_and_latency_are_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TTS_MODEL", "S2.1-PRO")
    monkeypatch.setenv("FISH_LATENCY", " Normal ")
    settings = load_config()
    assert settings.tts_model == "s2.1-pro"
    assert settings.tts.latency == "normal"
    monkeypatch.setenv("FISH_LATENCY", "turbo")
    monkeypatch.setenv("FISH_TTS_MODEL", "MyModel")
    settings = load_config()
    assert settings.tts_model == "MyModel"
    assert settings.tts.latency == "normal"


def test_parse_device_strips_index_and_blank() -> None:
    assert parse_device(None) is None
    assert parse_device("") is None
    assert parse_device("  ") is None
    assert parse_device(" 3 ") == 3
    assert parse_device("1.0") == 1
    assert parse_device("0.0") == 0
    assert parse_device("1.5") == "1.5"
    assert parse_device(" hw:0 ") == "hw:0"
    assert parse_device("3\nbad") == 3
    assert parse_device("hw:0\nbad") == "hw:0"


def _no_log(**_kwargs: object) -> None:
    return None


def _stub_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: object) -> list[str]:
    """Run ``main`` with an empty env file, no logging set-up and ``run`` as the session.

    Returns the arguments to pass, so ``./.env`` is never read.
    """
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr("fish_audio_suite_voice.cli.main.configure_voice_logging", _no_log)
    monkeypatch.setattr("fish_audio_suite_voice.cli.main.run_loop", run)
    return ["--env-file", str(empty)]


def test_main_exits_2_on_an_unexpected_cancel_without_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = {"n": 0}

    async def run(_config: object, **_kwargs: object) -> int:
        calls["n"] += 1
        raise asyncio.CancelledError

    argv = _stub_main(monkeypatch, tmp_path, run)
    assert main(argv) == ExitCode.FATAL
    assert calls["n"] == 1
    assert "cancelled unexpectedly" in capsys.readouterr().err


def test_main_says_bye_on_ctrl_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def run(_config: object, **_kwargs: object) -> int:
        raise KeyboardInterrupt

    argv = _stub_main(monkeypatch, tmp_path, run)
    assert main(argv) == ExitCode.OK
    assert capsys.readouterr().out.endswith("\nbye\n")


def test_llm_keys_never_cross_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FISH_LLM_BASE",
        "FISH_LLM_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only")
    assert load_config().llm.uses_openrouter_sdk
    assert load_config().llm.api_key == ""
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only")
    assert load_config().llm.api_key == "router-only"
    monkeypatch.setenv("FISH_LLM_BASE", "http://localhost:11434/v1")
    local = load_config().llm
    assert not local.uses_openrouter_sdk
    assert local.backend == "openai"
    assert local.api_key == "openai-only"


def test_the_llm_backend_follows_the_base_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_LLM_BASE", "http://localhost:11434/v1")
    assert load_config().llm.backend == "openai"
    monkeypatch.setenv("FISH_LLM_BASE", "https://openrouter.ai/api/v1")
    assert load_config().llm.backend == "openrouter"


def test_llm_nitro_and_continuation_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FISH_LLM_NITRO", "FISH_LLM_CONTINUE"):
        monkeypatch.delenv(name, raising=False)
    assert not load_config().llm.nitro
    assert not load_config().llm.continuation
    monkeypatch.setenv("FISH_LLM_NITRO", "1")
    monkeypatch.setenv("FISH_LLM_CONTINUE", "yes")
    assert load_config().llm.nitro
    assert load_config().llm.continuation


def test_bad_numeric_env_warns_and_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "lots")
    monkeypatch.setenv("FISH_LLM_TEMPERATURE", "9")
    monkeypatch.setenv("FISH_VOICE_HISTORY_TURNS", "0")
    c = load_config()
    assert c.barge.hit_frames == 10
    assert c.llm.temperature == 0.8
    assert c.history_turns == 20
    err = capsys.readouterr().err
    for name in ("FISH_VOICE_BARGE_FRAMES", "FISH_LLM_TEMPERATURE", "FISH_VOICE_HISTORY_TURNS"):
        assert name in err


def test_roleplay_features_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_TTS_MOOD_LEAD", raising=False)
    monkeypatch.delenv("FISH_TTS_DROP_NARRATION", raising=False)
    c = load_config()
    assert not c.mood_lead
    assert not c.drop_narration
    monkeypatch.setenv("FISH_TTS_MOOD_LEAD", "1")
    monkeypatch.setenv("FISH_TTS_DROP_NARRATION", "true")
    c = load_config()
    assert c.mood_lead
    assert c.drop_narration


def test_asr_model_env_accepts_only_native_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_ASR_MODEL", raising=False)
    assert load_config().asr_model == "transcribe-1-pro"
    monkeypatch.setenv("FISH_ASR_MODEL", " Transcribe-1 ")
    assert load_config().asr_model == "transcribe-1"
    monkeypatch.setenv("FISH_ASR_MODEL", "whisper-1")
    assert load_config().asr_model == "transcribe-1-pro"


def _fish_says_hello(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the Fish websocket: write 4000 bytes of audio to the sink."""

    def speak(_self: FishSpeaker, text: str, sink: PlaybackSink, **_kw: object) -> TtsResult:
        sink.start()
        sink.write(b"\x00\x01" * 2000)
        sink.finish()
        return make_result(text, bytes_played=4000, got_audio=True)

    monkeypatch.setattr(FishSpeaker, "speak", speak)


def test_smoke_writes_to_the_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fish_says_hello(monkeypatch)
    c = replace(load_config(), fish_api_key="k", fish_voice_id="v")
    out = tmp_path / "hello.wav"
    assert asyncio.run(smoke_test(c, out)) == 0
    assert out.stat().st_size > 1000


def test_smoke_default_path_is_unique_and_not_a_shared_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fish_says_hello(monkeypatch)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    c = replace(load_config(), fish_api_key="k", fish_voice_id="v")
    assert asyncio.run(smoke_test(c)) == 0
    assert asyncio.run(smoke_test(c)) == 0
    files = sorted(p.name for p in tmp_path.glob("fish-audio-suite-smoke-*.wav"))
    assert len(files) == 2
    assert "fish-audio-suite-smoke.wav" not in {p.name for p in tmp_path.iterdir()}
    assert "smoke: wrote" in capsys.readouterr().out


def test_repeat_window_env_is_read_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_REPEAT_WINDOW", raising=False)
    assert load_config().repeat_window_s == 1.5
    monkeypatch.setenv("FISH_VOICE_REPEAT_WINDOW", "0")
    assert load_config().repeat_window_s == 0.0
    monkeypatch.setenv("FISH_VOICE_REPEAT_WINDOW", "-3")
    assert load_config().repeat_window_s == 1.5


def test_stream_tts_is_off_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_STREAM_TTS", raising=False)
    assert load_config().stream_tts is False
    monkeypatch.setenv("FISH_VOICE_STREAM_TTS", "1")
    assert load_config().stream_tts is True


def test_a_blank_llm_key_does_not_hide_the_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fish_audio_suite_voice.llm.tune import LlmTune

    for name in ("FISH_LLM_BASE", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FISH_LLM_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert LlmTune.from_env().api_key == "or-key"
    monkeypatch.setenv("FISH_LLM_API_KEY", "own-key")
    assert LlmTune.from_env().api_key == "own-key"


def test_a_malformed_llm_base_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fish_audio_suite_voice.llm.tune import LlmTune, is_openrouter_host

    monkeypatch.setenv("FISH_LLM_BASE", "http://[::1")
    tune = LlmTune.from_env()
    assert tune.base == OPENROUTER_API_BASE
    assert "not a valid URL" in capsys.readouterr().err
    assert is_openrouter_host("http://[::1") is False


def test_main_warns_once_for_each_key_sent_over_plain_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def run(_config: object, **_kwargs: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setenv("FISH_API_KEY", "fish-key")
    monkeypatch.setenv("FISH_BASE", "http://10.0.0.5:8080")
    monkeypatch.setenv("FISH_LLM_API_KEY", "llm-key")
    monkeypatch.setenv("FISH_LLM_BASE", "http://10.0.0.6/v1")
    argv = _stub_main(monkeypatch, tmp_path, run)
    assert main(argv) == ExitCode.OK
    err = capsys.readouterr().err
    assert err.count("10.0.0.5 is plain http") == 1
    assert err.count("10.0.0.6 is plain http") == 1
