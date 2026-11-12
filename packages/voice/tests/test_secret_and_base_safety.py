from __future__ import annotations

import threading
from dataclasses import replace
from types import MappingProxyType

import httpx
import pytest
from fishaudio.types import TTSConfig

from fish_audio_suite_voice.config import SystemPrompt, VoiceCliConfig, warn_if_insecure_base
from fish_audio_suite_voice.llm_tune import LlmTune
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tts_tune import TtsTune
from fish_audio_suite_voice.tts_turn import turn_failure
from fish_audio_suite_voice.tune import DeepgramTune, SttTune
from fish_audio_suite_voice.turn_stream import RecogniserSettings
from fish_audio_suite_voice.wire import TurnSpec
from fish_audio_suite_voice.wording import with_detail

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
        tts=TtsTune(latency="balanced", temperature=0.7, top_p=0.7, repetition_penalty=1.2),
        playback="stdout",
        prompt=SystemPrompt("be brief", pin_seed=True),
        device=None,
        llm=llm or LlmTune(base="https://llm.example/v1", api_key=SECRET, model="m"),
    )


def test_no_secret_appears_in_a_config_repr() -> None:
    config = replace(_config(), stt=SttTune(deepgram=DeepgramTune(key=SECRET)))
    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    assert SECRET not in repr(config.llm)
    assert "fish_voice_id" in repr(config)
    assert "eot_threshold" in repr(config)
    streaming = RecogniserSettings(config.stt, fish_key=SECRET, fish_base=config.fish_base)
    assert SECRET not in repr(streaming)


def test_no_secret_appears_in_the_tts_client_or_turn_spec_repr() -> None:
    tts = FishSpeaker(api_key=SECRET, voice_id="voice")
    assert SECRET not in repr(tts)
    assert SECRET not in str(tts)
    assert "voice_id" in repr(tts)
    spec = TurnSpec(
        api_key=SECRET,
        base_url="https://api.fish.audio",
        voice_id="voice",
        model="s2.1-pro",
        trace_headers=MappingProxyType({}),
        config=TTSConfig(),
    )
    assert SECRET not in repr(spec)
    assert "voice_id" in repr(spec)


def test_with_detail_adds_the_errors_own_text() -> None:
    assert with_detail("Fish upstream unreachable", OSError("refused")) == (
        "Fish upstream unreachable: refused"
    )
    assert with_detail("Fish request timed out", httpx.ReadTimeout("")) == "Fish request timed out"
    assert with_detail("same", RuntimeError("same")) == "same"


def test_a_transport_error_keeps_its_detail_for_the_local_user() -> None:
    refused = httpx.ConnectError("connect to 10.0.0.5 refused")
    first = turn_failure(refused, sent_text="Hi.", got_audio=False, cancel=threading.Event())
    assert first.retry is True
    last = first.give_up()  # what fish_retries does after the last attempt
    assert last.retry is False
    assert last.error_status == 502
    assert last.error_message is not None
    assert last.error_message.startswith("Fish upstream unreachable")
    assert "10.0.0.5" in last.error_message


def test_plain_http_to_a_remote_host_warns_for_both_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(
        fish_base="http://10.0.0.5:8080",
        llm=LlmTune(base="http://10.0.0.6/v1", api_key=SECRET, model="m"),
    )
    assert warn_if_insecure_base(config) is True
    notes = capsys.readouterr().err.splitlines()
    assert len(notes) == 2
    assert all(SECRET not in note for note in notes)
    assert "10.0.0.5" in notes[0]
    assert "10.0.0.6" in notes[1]


def test_https_and_loopback_do_not_warn(capsys: pytest.CaptureFixture[str]) -> None:
    assert warn_if_insecure_base(_config()) is False
    local = _config(
        fish_base="http://127.0.0.1:8080",
        llm=LlmTune(base="http://localhost:1234/v1", api_key=SECRET, model="m"),
    )
    assert warn_if_insecure_base(local) is False
    assert capsys.readouterr().err == ""


def test_no_key_means_no_warning(capsys: pytest.CaptureFixture[str]) -> None:
    keyless = _config(
        fish_api_key="",
        fish_base="http://10.0.0.5:8080",
        llm=LlmTune(base="http://10.0.0.6/v1", api_key="", model="m"),
    )
    assert warn_if_insecure_base(keyless) is False
    assert capsys.readouterr().err == ""


def test_the_warning_prints_only_the_host_never_credentials_in_the_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(
        fish_base="http://alice:hunter2@10.0.0.5:8080/v1?token=abc123",
        llm=LlmTune(
            base="http://bob:s3cret@10.0.0.6/v1?key=zzz",
            api_key=SECRET,
            model="m",
        ),
    )
    assert warn_if_insecure_base(config) is True
    text = capsys.readouterr().err
    assert "10.0.0.5" in text
    assert "10.0.0.6" in text
    for leaked in ("alice", "hunter2", "abc123", "bob", "s3cret", "zzz"):
        assert leaked not in text
