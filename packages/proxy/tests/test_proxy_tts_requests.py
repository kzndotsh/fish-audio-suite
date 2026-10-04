from __future__ import annotations

import base64
from typing import Any

import httpx
import ormsgpack
import pytest
from fastapi.testclient import TestClient
from proxy_helpers import (
    AudioStream,
    capture_upstream,
    content_part,
    json_part,
    not_response,
)

from fish_audio_suite_kit import (
    SuiteDefaults,
    chunk_length_hi,
    is_tts_junk,
)
from fish_audio_suite_proxy.audio import pcm_sample_rate
from fish_audio_suite_proxy.errors import ProxyError
from fish_audio_suite_proxy.request_fields import (
    prepare_tts_text,
    read_flag,
    read_format,
    read_reference_id,
)
from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.settings import load_settings, runtime_defaults
from fish_audio_suite_proxy.speech import (
    SpeechControls,
    _fish_tts_payload,
    pack_tts,
    speech_controls,
)


def test_prepare_tts_normalizes_cues() -> None:
    spoken = prepare_tts_text("Excited, hello there friend", dialogue_only=False, mood_lead=True)
    assert spoken.startswith("[excited]")
    assert "hello" in spoken


def test_a_mood_word_is_spoken_as_written_unless_the_lead_is_on() -> None:
    assert prepare_tts_text("Curious, isn't it?", dialogue_only=False) == "Curious, isn't it?"


def test_dialogue_only_drops_the_stage_note() -> None:
    raw = 'She smiles. "Hello there friend."'
    spoken = prepare_tts_text(raw, dialogue_only=True)
    assert "Hello there friend" in spoken
    assert "smiles" not in spoken
    assert "smiles" in prepare_tts_text(raw, dialogue_only=False)


def test_read_flag_keeps_false_and_uses_the_default_when_absent() -> None:
    assert read_flag({}, "dialogue_only", default=True) is True
    assert read_flag({"dialogue_only": False}, "dialogue_only", default=True) is False
    assert read_flag({"dialogue_only": None}, "dialogue_only", default=True) is True
    assert read_flag({"dialogue_only": "false"}, "dialogue_only", default=True) is False
    assert read_flag({"dialogue_only": "no"}, "dialogue_only", default=True) is False
    assert read_flag({"dialogue_only": "true"}, "dialogue_only", default=False) is True
    assert read_flag({"dialogue_only": "maybe"}, "dialogue_only", default=True) is True


def test_no_format_uses_the_default_and_an_unknown_format_name_is_a_400() -> None:
    assert read_format({}, "opus") == "opus"
    assert read_format({"format": "WAV "}, "mp3") == "wav"
    with pytest.raises(ProxyError, match="unsupported response_format"):
        read_format({"format": "not-a-format"}, "mp3")


@pytest.mark.parametrize("name", ["nope", "aac", "flac"])
def test_an_unsupported_format_is_a_400_not_other_bytes(name: str) -> None:
    with pytest.raises(ProxyError) as caught:
        read_format({"response_format": name}, "mp3")
    assert caught.value.status == 400
    assert name in caught.value.message


def test_junk_pcm_and_wav_are_silence() -> None:
    with TestClient(app) as client:
        pcm = client.post(
            "/v1/audio/speech",
            json={"input": "...", "response_format": "pcm"},
        )
        wav = client.post(
            "/v1/audio/speech",
            json={"input": "...", "response_format": "wav"},
        )
        huge = client.post(
            "/v1/audio/speech",
            json={"input": "...", "response_format": "wav", "sample_rate": 2**32},
        )
        mp3 = client.post("/v1/audio/speech", json={"input": "..."})
        opus = client.post(
            "/v1/audio/speech",
            json={"input": "...", "response_format": "opus"},
        )
    assert pcm.status_code == 200
    assert pcm.headers["content-type"].startswith("audio/pcm")
    assert pcm.content == b"\x00\x00"
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert wav.content.startswith(b"RIFF")
    assert huge.status_code == 200
    assert huge.content.startswith(b"RIFF")
    assert mp3.status_code == 200
    assert mp3.headers["content-type"].startswith("audio/mpeg")
    assert mp3.content.startswith(b"\xff\xfb")
    assert opus.status_code == 200
    assert opus.headers["content-type"].startswith("audio/opus")
    assert opus.content.startswith(b"OggS")
    assert not opus.content.startswith(b"\xff\xfb")


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


def test_mp3_bitrate_env_snaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_MP3_BITRATE", "96")
    assert runtime_defaults().mp3_bitrate == 128
    monkeypatch.setenv("FISH_MP3_BITRATE", "192")
    assert runtime_defaults().mp3_bitrate == 192


def test_format_env_uses_the_request_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TTS_FORMAT", " AAC ")
    assert runtime_defaults().audio_format == "mp3"
    assert load_settings().tts_format == "mp3"
    monkeypatch.setenv("FISH_TTS_FORMAT", "PCM16")
    # Fish only knows "pcm". The client-facing default stays "pcm16" (24 kHz).
    assert runtime_defaults().audio_format == "pcm"
    settings = load_settings()
    assert settings.tts_format == "pcm16"
    controls = speech_controls({}, settings.defaults, default_format=settings.tts_format)
    assert controls.fmt == "pcm16"
    monkeypatch.setenv("FISH_ASR_MODEL", " fish-audio/Transcribe-1 ")
    assert runtime_defaults().asr_model == "transcribe-1"
    monkeypatch.delenv("FISH_ASR_MODEL")
    assert runtime_defaults().asr_model == "transcribe-1-pro"


def test_pcm16_format_and_rate() -> None:
    assert read_format({"response_format": "pcm16"}, "mp3") == "pcm16"
    assert pcm_sample_rate("pcm16", {}, 44100) == 24000
    assert pcm_sample_rate("pcm16", {"sample_rate": 16000}, 44100) == 16000
    assert pcm_sample_rate("pcm16", {"sample_rate": "16000.0"}, 44100) == 16000
    assert pcm_sample_rate("pcm16", {"sample_rate": "16000.5"}, 44100) == 24000
    assert pcm_sample_rate("pcm", {}, 44100) == 44100
    assert pcm_sample_rate("pcm", {"sample_rate": 0}, 44100) == 44100
    assert pcm_sample_rate("pcm16", {"sample_rate": -1}, 44100) == 24000
    assert pcm_sample_rate("opus", {"sample_rate": 0}, 48000) == 48000


def test_chunk_length_hi_cloud_vs_self_host() -> None:
    assert chunk_length_hi("https://api.fish.audio") == 300
    assert chunk_length_hi("http://127.0.0.1:8080") == 1000


def test_payload_snaps_bitrate_and_unit_interval() -> None:
    defaults = SuiteDefaults()
    mp3 = _fish_tts_payload(
        {"mp3_bitrate": 96, "temperature": 5},
        defaults,
        SpeechControls("s2.1-pro", 1.0, "mp3", "normal", 200, 50),
        "hi",
    )
    assert mp3["mp3_bitrate"] == 128
    assert mp3["temperature"] == 1.0
    opus = _fish_tts_payload(
        {"opus_bitrate": 1, "top_p": -3, "early_stop_threshold": 4},
        defaults,
        SpeechControls("s2.1-pro", 1.0, "opus", "normal", 200, 50),
        "hi",
    )
    assert opus["opus_bitrate"] == -1000
    assert opus["top_p"] == 0.0
    assert opus["early_stop_threshold"] == 1.0


def test_omitted_chunk_length_clamps_the_default() -> None:
    cloud = SuiteDefaults(
        chunk_length=900, min_chunk_length=250, fish_base="https://api.fish.audio"
    )
    controls = speech_controls({}, cloud)
    assert controls.chunk_length == 300
    assert controls.min_chunk_length == 100
    local = SuiteDefaults(chunk_length=800, fish_base="http://127.0.0.1:8080")
    assert speech_controls({}, local).chunk_length == 800


def test_read_reference_id_list() -> None:
    assert read_reference_id({"reference_id": ["a", "b"]}) == ["a", "b"]
    assert read_reference_id({"voice": "solo"}) == "solo"
    assert read_reference_id({"reference_id": ["  a  ", None, ""]}) == ["a"]
    assert read_reference_id({"voice": "  solo  "}) == "solo"
    assert read_reference_id({"reference_id": [None]}) is None
    assert read_reference_id({"reference_id": [{"id": "x"}, " a ", 1]}) == ["a"]
    assert read_reference_id({"reference_id": [1]}) is None
    assert read_reference_id({"reference_id": [None, ""], "voice": "kept"}) == "kept"
    assert read_reference_id({"reference_id": "", "voice": "kept"}) == "kept"
    assert read_reference_id({"reference_id": ["a"], "voice": "ignored"}) == ["a"]


def test_speech_input_must_be_a_string() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json={"input": ["hello"]})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "input must be a string"
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_speech_bad_json_is_400() -> None:
    with TestClient(app) as client:
        broken = client.post(
            "/v1/audio/speech",
            content=b"{",
            headers={"content-type": "application/json"},
        )
        assert broken.status_code == 400
        assert broken.json()["error"]["message"] == "invalid JSON body"
        raw = client.post(
            "/v1/audio/speech",
            content=b"\xff",
            headers={"content-type": "application/json"},
        )
        assert raw.status_code == 400
        assert raw.json()["error"]["message"] == "invalid JSON body"
        deep = client.post(
            "/v1/audio/speech",
            content=b"[" * 20_000,
            headers={"content-type": "application/json"},
        )
        assert deep.status_code == 400
        assert deep.json()["error"]["message"] == "invalid JSON body"
        array = client.post("/v1/audio/speech", json=["nope"])
        assert array.status_code == 400
        assert array.json()["error"]["message"] == "JSON body must be an object"


def test_normalize_loudness_false_reaches_fish(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = capture_upstream(monkeypatch, AudioStream())
    with TestClient(app) as client:
        off = client.post(
            "/v1/audio/speech",
            json={"input": "Hello there friend", "normalize_loudness": False},
        )
        assert off.status_code == 200
        assert captured["json"]["prosody"]["normalize_loudness"] is False
        on = client.post("/v1/audio/speech", json={"input": "Hello there friend"})
        assert on.status_code == 200
        assert captured["json"]["prosody"]["normalize_loudness"] is True
        word = client.post(
            "/v1/audio/speech",
            json={"input": "Hello there friend", "normalize_loudness": "false"},
        )
        assert word.status_code == 200
        assert captured["json"]["prosody"]["normalize_loudness"] is False


def test_dialogue_only_string_false_keeps_the_stage_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_upstream(monkeypatch, AudioStream())
    raw = 'She smiles. "Hello there friend."'
    with TestClient(app) as client:
        kept = client.post("/v1/audio/speech", json={"input": raw, "dialogue_only": "false"})
        assert kept.status_code == 200
        assert "smiles" in captured["json"]["text"]
        dropped = client.post("/v1/audio/speech", json={"input": raw, "dialogue_only": "true"})
        assert dropped.status_code == 200
        assert "smiles" not in captured["json"]["text"]
        assert "Hello there friend" in captured["json"]["text"]


def test_guillemet_dialogue_is_spoken(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = capture_upstream(monkeypatch, AudioStream())
    raw = "She smiles. «Hello there friend.»"
    with TestClient(app) as client:
        kept = client.post("/v1/audio/speech", json={"input": raw})
        assert kept.status_code == 200
        assert "Hello there friend" in captured["json"]["text"]
        dropped = client.post("/v1/audio/speech", json={"input": raw, "dialogue_only": True})
        assert dropped.status_code == 200
        assert "smiles" not in captured["json"]["text"]
        assert "Hello there friend" in captured["json"]["text"]


def test_null_chunk_length_means_the_default_and_zero_is_a_real_value() -> None:
    defaults = SuiteDefaults()
    nulls = speech_controls({"chunk_length": None, "min_chunk_length": None}, defaults)
    assert nulls.chunk_length == defaults.chunk_length
    assert nulls.min_chunk_length == defaults.min_chunk_length
    given = speech_controls({"chunk_length": 120, "min_chunk_length": 0}, defaults)
    assert given.chunk_length == 120
    assert given.min_chunk_length == 0


def test_the_old_request_field_spellings_are_ignored() -> None:
    defaults = SuiteDefaults()
    controls = speech_controls(
        {"fish_chunk_length": 180, "fish_min_chunk_length": 40, "fish_latency": "low"}, defaults
    )
    assert controls.chunk_length == defaults.chunk_length
    assert controls.min_chunk_length == defaults.min_chunk_length
    assert controls.latency == defaults.latency
    assert speech_controls({"fish_format": "wav"}, defaults).fmt == defaults.audio_format


def test_quality_guard_reads_the_flag_then_the_features_list() -> None:
    defaults = SuiteDefaults()
    spoken = "Hello there friend"

    def features(body: dict[str, Any]) -> object:
        packed = not_response(pack_tts(body, defaults, {}, speech_controls(body, defaults), spoken))
        return json_part(packed).get("features")

    assert features({"input": spoken, "quality_guard": True}) == ["quality-guard"]
    assert features({"input": spoken, "quality_guard": False}) is None
    assert features({"input": spoken, "quality_guard": None, "features": ["quality-guard"]}) == [
        "quality-guard"
    ]
    assert features({"input": spoken, "fish_quality_guard": True}) is None


def test_surrogate_in_pronunciation_still_encodes() -> None:
    defaults = SuiteDefaults()
    body = {
        "input": "Hello there friend",
        "voice": "voice-\ud800",
        "pronunciation_dictionary": [
            {"items": [{"value": "<|phoneme_start|>\ud800ah<|phoneme_end|>"}]}
        ],
    }
    packed = not_response(
        pack_tts(
            body,
            defaults,
            {},
            speech_controls(body, defaults),
            "Hello there friend",
        )
    )
    request = httpx.Request("POST", "https://api.fish.audio/v1/tts", **packed.request_kw)
    request.read()
    payload = json_part(packed)
    assert "\ud800" not in payload["reference_id"]
    value = payload["pronunciation_dictionary"][0]["items"][0]["value"]
    assert "\ud800" not in value
    assert "ah" in value
    assert "phoneme" not in value


def test_surrogate_in_speech_text_still_encodes() -> None:
    raw = "Hello \ud800 there friend"
    spoken = prepare_tts_text(raw, dialogue_only=False)
    assert "\ud800" not in spoken
    assert "Hello" in spoken
    assert "there friend" in spoken
    defaults = SuiteDefaults()
    body = {"input": raw}
    packed = not_response(pack_tts(body, defaults, {}, speech_controls(body, defaults), spoken))
    request = httpx.Request("POST", "https://api.fish.audio/v1/tts", **packed.request_kw)
    request.read()
    sample = base64.b64encode(b"RIFF").decode()
    clip_body = {
        "input": "Hello there friend",
        "references": [{"audio": sample, "text": "sample \ud800 line"}],
    }
    clip_packed = not_response(
        pack_tts(
            clip_body,
            defaults,
            {},
            speech_controls(clip_body, defaults),
            "Hello there friend",
        )
    )
    unpacked = ormsgpack.unpackb(content_part(clip_packed))
    clip_text = unpacked["references"][0]["text"]
    assert "\ud800" not in clip_text
    assert "sample" in clip_text
    assert "line" in clip_text


def test_speech_json_bom_is_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = capture_upstream(monkeypatch, AudioStream())
    body = b'\xef\xbb\xbf{"input": "Hello there friend"}'
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 200
    assert "Hello there friend" in captured["json"]["text"]
