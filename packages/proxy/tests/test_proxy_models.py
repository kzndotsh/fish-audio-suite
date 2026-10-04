from __future__ import annotations

from fish_audio_suite_proxy.models import catalog_ids, resolve_asr_model, resolve_tts_model


def test_asr_model_whisper_remap() -> None:
    assert resolve_asr_model("whisper-1", "transcribe-1") == "transcribe-1"
    assert resolve_asr_model("whisper-1", " Transcribe-1 ") == "transcribe-1"
    assert resolve_asr_model(None, "fish-audio/Transcribe-1-Pro") == "transcribe-1-pro"
    assert resolve_asr_model("gpt-4o-transcribe", "CustomAsr") == "CustomAsr"
    assert resolve_asr_model("gpt-4o-transcribe", "transcribe-1\nbad") == "transcribe-1-pro"
    assert resolve_asr_model("whisper-1", "custom\nid") == "transcribe-1-pro"
    assert resolve_asr_model("whisper-1", "   ") == "transcribe-1-pro"


def test_tts_and_asr_model_aliases() -> None:
    assert resolve_tts_model("fish-audio/s2.1-pro", "s1") == "s2.1-pro"
    assert resolve_tts_model("tts-1", "s1") == "s1"
    assert resolve_tts_model("gpt-4o-mini-tts", "s2.1-pro") == "s2.1-pro"
    assert resolve_tts_model("playai-tts", "s1") == "playai-tts"
    assert resolve_tts_model("voice-a", "s1", {"voice-a": "s2-pro"}) == "s2-pro"
    assert resolve_tts_model("s2.1-pro-free", "s2.1-pro") == "s2.1-pro-free"
    assert resolve_tts_model("drama-3-preview", "s2.1-pro") == "drama-3-preview"
    assert resolve_tts_model("S2.1-PRO", "s1") == "s2.1-pro"
    assert resolve_tts_model("fish-audio/ Drama-3-Preview ", "s1") == "drama-3-preview"
    assert resolve_tts_model("CustomModel", "s1") == "CustomModel"
    assert resolve_tts_model("custom\nid", "s1") == "s1"
    assert resolve_tts_model(["s2.1-pro"], "s1") == "s1"
    assert resolve_tts_model("   ", "s1") == "s1"
    assert resolve_asr_model(["whisper-1"], "transcribe-1") == "transcribe-1"
    assert resolve_asr_model("fish-audio/transcribe-1", "transcribe-1-pro") == "transcribe-1"
    assert resolve_asr_model("gpt-4o-transcribe", "transcribe-1") == "transcribe-1"
    assert resolve_asr_model("fish-audio/transcribe-1-pro", "transcribe-1") == "transcribe-1-pro"


def test_catalog_includes_fish_audio_slugs() -> None:
    ids = catalog_ids()
    assert "s2.1-pro" in ids
    assert "fish-audio/s2.1-pro" in ids
    assert "fish-audio/transcribe-1" in ids
    assert "whisper-1" in ids
    assert "tts-1" in ids
    assert "my-voice" in catalog_ids({"my-voice": "s2-pro"})
    assert "tts-1" not in catalog_ids({"my-voice": "s2-pro"})
