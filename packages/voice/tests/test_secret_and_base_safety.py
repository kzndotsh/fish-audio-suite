from __future__ import annotations

import httpx
import pytest

from fish_audio_suite_voice.config import VoiceCliConfig, warn_if_insecure_base
from fish_audio_suite_voice.debug import with_detail
from fish_audio_suite_voice.llm_tune import LlmTune
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.wire import _classify_fish_exc

SECRET = "sk-very-secret-value"


def _config(
    *,
    fish_api_key: str = SECRET,
    fish_base: str = "https://api.fish.audio",
    llm: LlmTune | None = None,
) -> VoiceCliConfig:
    return VoiceCliConfig(
        fish_api_key=fish_api_key,
        fish_base=fish_base,
        fish_voice_id="voice",
        fish_asr_language="",
        tts_model="s2.1-pro",
        latency="balanced",
        speed=1.0,
        temperature=0.7,
        top_p=0.7,
        repetition_penalty=1.2,
        chunk_length=200,
        min_chunk_length=50,
        volume=0.0,
        sample_rate=44100,
        playback="stdout",
        system_prompt="be brief",
        device=None,
        llm=llm
        or LlmTune(backend="openai", base="https://llm.example/v1", api_key=SECRET, model="m"),
    )


def test_no_secret_appears_in_a_config_repr() -> None:
    config = _config()
    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    assert SECRET not in repr(config.llm)
    assert "fish_voice_id" in repr(config)


def test_no_secret_appears_in_the_tts_client_or_turn_spec_repr() -> None:
    tts = FishSpeaker(api_key=SECRET, voice_id="voice")
    assert SECRET not in repr(tts)
    assert SECRET not in str(tts)
    assert "voice_id" in repr(tts)
    assert SECRET not in repr(tts._spec())


def test_with_detail_adds_the_errors_own_text() -> None:
    assert with_detail("Fish upstream unreachable", OSError("refused")) == (
        "Fish upstream unreachable: refused"
    )
    assert with_detail("Fish request timed out", httpx.ReadTimeout("")) == "Fish request timed out"
    assert with_detail("same", RuntimeError("same")) == "same"


def test_a_transport_error_keeps_its_detail_for_the_local_user() -> None:
    retry, status, message = _classify_fish_exc(httpx.ConnectError("connect to 10.0.0.5 refused"))
    assert retry is True
    assert status == 502
    assert message.startswith("Fish upstream unreachable")
    assert "10.0.0.5" in message


def test_plain_http_to_a_remote_host_warns_for_both_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes: list[str] = []
    monkeypatch.setattr("fish_audio_suite_voice.config.warn", notes.append)
    config = _config(
        fish_base="http://10.0.0.5:8080",
        llm=LlmTune(backend="openai", base="http://10.0.0.6/v1", api_key=SECRET, model="m"),
    )
    assert warn_if_insecure_base(config) is True
    assert len(notes) == 2
    assert all(SECRET not in note for note in notes)
    assert "10.0.0.5" in notes[0]
    assert "10.0.0.6" in notes[1]


def test_https_and_loopback_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[str] = []
    monkeypatch.setattr("fish_audio_suite_voice.config.warn", notes.append)
    assert warn_if_insecure_base(_config()) is False
    local = _config(
        fish_base="http://127.0.0.1:8080",
        llm=LlmTune(backend="openai", base="http://localhost:1234/v1", api_key=SECRET, model="m"),
    )
    assert warn_if_insecure_base(local) is False
    assert notes == []


def test_no_key_means_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    notes: list[str] = []
    monkeypatch.setattr("fish_audio_suite_voice.config.warn", notes.append)
    keyless = _config(
        fish_api_key="",
        fish_base="http://10.0.0.5:8080",
        llm=LlmTune(backend="openai", base="http://10.0.0.6/v1", api_key="", model="m"),
    )
    assert warn_if_insecure_base(keyless) is False
    assert notes == []


def test_the_warning_prints_only_the_host_never_credentials_in_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes: list[str] = []
    monkeypatch.setattr("fish_audio_suite_voice.config.warn", notes.append)
    config = _config(
        fish_base="http://alice:hunter2@10.0.0.5:8080/v1?token=abc123",
        llm=LlmTune(
            backend="openai",
            base="http://bob:s3cret@10.0.0.6/v1?key=zzz",
            api_key=SECRET,
            model="m",
        ),
    )
    assert warn_if_insecure_base(config) is True
    text = " ".join(notes)
    assert "10.0.0.5" in text
    assert "10.0.0.6" in text
    for leaked in ("alice", "hunter2", "abc123", "bob", "s3cret", "zzz"):
        assert leaked not in text
