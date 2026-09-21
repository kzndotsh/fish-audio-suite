from __future__ import annotations

from fastapi.testclient import TestClient

from fish_audio_suite_kit import is_asr_hallucination, is_tts_junk
from fish_audio_suite_proxy.server import (
    _chunk_length_hi,
    _pick_reference_id,
    _resolve_asr_model,
    app,
    prepare_tts_text,
)


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


def test_prepare_keeps_stacked_cues_and_speaker_token() -> None:
    spoken = prepare_tts_text(
        "<|speaker:0|> [sad][whispering] I miss you so much",
        dialogue_only=False,
    )
    assert "<|speaker:0|>" in spoken
    assert "[sad]" in spoken
    assert "[whispering]" in spoken


def test_asr_model_whisper_remap() -> None:
    assert _resolve_asr_model("whisper-1", "transcribe-1") == "transcribe-1"


def test_asr_hallucination_without_network() -> None:
    assert is_asr_hallucination("谢谢观看")
    assert not is_asr_hallucination("你好")
    assert not is_asr_hallucination("hello there friend")


def test_chunk_length_hi_cloud_vs_self_host() -> None:
    assert _chunk_length_hi("https://api.fish.audio") == 300
    assert _chunk_length_hi("http://127.0.0.1:8080") == 1000


def test_pick_reference_id_list() -> None:
    assert _pick_reference_id({"reference_id": ["a", "b"]}) == ["a", "b"]
    assert _pick_reference_id({"voice": "solo"}) == "solo"


def test_health_without_api_key() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["defaults"]["asr_strip_speakers"] is False
        assert body["defaults"]["tts_dialogue_only"] is False
        models = client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert "s2.1-pro-free" in ids
        assert "drama-3-preview" in ids
