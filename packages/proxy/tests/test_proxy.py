from __future__ import annotations

from fastapi.testclient import TestClient

from fish_audio_suite_kit import is_asr_hallucination, is_tts_junk
from fish_audio_suite_proxy.server import _resolve_asr_model, app, prepare_tts_text


def test_prepare_tts_normalizes_cues() -> None:
    spoken = prepare_tts_text("Excited, hello there friend", dialogue_only=False)
    assert spoken.startswith("[excited]")
    assert "hello" in spoken


def test_junk_unclosed_cue() -> None:
    spoken = prepare_tts_text("[warm,", dialogue_only=False)
    assert is_tts_junk(spoken)


def test_whisper_alias_on_speech_path() -> None:
    spoken = prepare_tts_text("[whispers] keep this quiet now", dialogue_only=False)
    assert "[whispering]" in spoken


def test_asr_model_whisper_remap() -> None:
    assert _resolve_asr_model("whisper-1", "transcribe-1") == "transcribe-1"


def test_asr_hallucination_without_network() -> None:
    assert is_asr_hallucination("谢谢")
    assert not is_asr_hallucination("hello there friend")


def test_health_without_api_key() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
