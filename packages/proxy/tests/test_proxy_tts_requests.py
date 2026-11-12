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
    fresh_app_env,
    json_part,
    post_speech,
)

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_proxy.audio import pcm_sample_rate
from fish_audio_suite_proxy.request_body import read_flag
from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.settings import ProxySettings, load_settings
from fish_audio_suite_proxy.speech import (
    PackedTts,
    SpeechControls,
    read_speech,
)


def _controls(body: dict[str, Any], defaults: SuiteDefaults | None = None) -> SpeechControls:
    settings = ProxySettings(defaults=defaults or SuiteDefaults())
    return read_speech({"input": "Hello there friend", **body}, settings).controls


def _pack(body: dict[str, Any]) -> PackedTts:
    return read_speech(body, ProxySettings()).fish_tts({})


def _spoken(monkeypatch: pytest.MonkeyPatch, text: str) -> str:
    """The text the speech route sends Fish for ``text``."""
    response, captured = post_speech(monkeypatch, {"input": text})
    assert response.status_code == 200
    return captured["json"]["text"]


def test_a_mood_word_leads_only_when_the_mood_lead_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch)
    monkeypatch.delenv("FISH_TTS_MOOD_LEAD", raising=False)
    assert _spoken(monkeypatch, "Curious, isn't it?") == "Curious, isn't it?"
    monkeypatch.setenv("FISH_TTS_MOOD_LEAD", "1")
    spoken = _spoken(monkeypatch, "Excited, hello there friend")
    assert spoken.startswith("[excited]")
    assert "hello" in spoken


def test_read_flag_keeps_false_and_uses_the_default_when_absent() -> None:
    assert read_flag({}, "dialogue_only", default=True) is True
    assert read_flag({"dialogue_only": False}, "dialogue_only", default=True) is False
    assert read_flag({"dialogue_only": None}, "dialogue_only", default=True) is True
    assert read_flag({"dialogue_only": "false"}, "dialogue_only", default=True) is False
    assert read_flag({"dialogue_only": "no"}, "dialogue_only", default=True) is False
    assert read_flag({"dialogue_only": "true"}, "dialogue_only", default=False) is True
    assert read_flag({"dialogue_only": "maybe"}, "dialogue_only", default=True) is True


def test_no_format_uses_the_default_and_format_beats_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_app_env(monkeypatch, FISH_TTS_FORMAT="opus")
    default, captured = post_speech(monkeypatch, {"input": "Hello there friend"})
    assert default.headers["content-type"].startswith("audio/opus")
    assert captured["json"]["format"] == "opus"
    named, captured = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "format": "WAV ", "response_format": "mp3"},
    )
    assert named.headers["content-type"].startswith("audio/wav")
    assert captured["json"]["format"] == "wav"
    pcm16, captured = post_speech(
        monkeypatch, {"input": "Hello there friend", "response_format": "pcm16"}
    )
    assert pcm16.headers["content-type"].startswith("audio/pcm")
    assert (captured["json"]["format"], captured["json"]["sample_rate"]) == ("pcm", 24000)


@pytest.mark.parametrize(
    "body",
    [
        {"response_format": "nope"},
        {"response_format": "aac"},
        {"response_format": "flac"},
        {"format": "flac", "response_format": "mp3"},
    ],
)
def test_an_unsupported_format_is_a_400_not_other_bytes(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, str]
) -> None:
    response, captured = post_speech(monkeypatch, {"input": "Hello there friend", **body})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    # Quoted in docs/TROUBLESHOOTING.md.
    assert error["message"].startswith("unsupported response_format")
    named = body.get("format", body["response_format"])
    assert repr(named) in error["message"]
    assert captured == {}


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
        # Fits the 32-bit rate field but not the byte rate (rate x 2).
        over_byte_rate = client.post(
            "/v1/audio/speech",
            json={"input": "...", "response_format": "wav", "sample_rate": 2**31},
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
    assert over_byte_rate.status_code == 200
    assert over_byte_rate.content.startswith(b"RIFF")
    assert mp3.status_code == 200
    assert mp3.headers["content-type"].startswith("audio/mpeg")
    assert mp3.content.startswith(b"\xff\xfb")
    assert opus.status_code == 200
    assert opus.headers["content-type"].startswith("audio/opus")
    assert opus.content.startswith(b"OggS")
    assert not opus.content.startswith(b"\xff\xfb")


def test_an_unclosed_cue_alone_is_silence_without_calling_fish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FISH_TTS_FORMAT", raising=False)
    response, captured = post_speech(monkeypatch, {"input": "[warm,"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content.startswith(b"\xff\xfb")
    assert captured == {}


def test_whisper_alias_on_speech_path(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "[whispering]" in _spoken(monkeypatch, "[whispers] keep this quiet now")


def test_stacked_cues_and_the_speaker_token_reach_fish(monkeypatch: pytest.MonkeyPatch) -> None:
    spoken = _spoken(monkeypatch, "<|speaker:0|> [sad][whispering] I miss you so much")
    assert "<|speaker:0|>" in spoken
    assert "[sad]" in spoken
    assert "[whispering]" in spoken


def test_mp3_bitrate_env_snaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_MP3_BITRATE", "96")
    assert load_settings().defaults.mp3_bitrate == 128
    monkeypatch.setenv("FISH_MP3_BITRATE", "192")
    assert load_settings().defaults.mp3_bitrate == 192


def test_format_env_uses_the_request_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TTS_FORMAT", " AAC ")
    assert load_settings().defaults.audio_format == "mp3"
    assert load_settings().tts_format == "mp3"
    monkeypatch.setenv("FISH_TTS_FORMAT", "PCM16")
    # Fish only knows "pcm". The client-facing default stays "pcm16" (24 kHz).
    assert load_settings().defaults.audio_format == "pcm"
    settings = load_settings()
    assert settings.tts_format == "pcm16"
    assert read_speech({"input": "Hello there friend"}, settings).controls.fmt == "pcm16"
    monkeypatch.setenv("FISH_ASR_MODEL", " fish-audio/Transcribe-1 ")
    assert load_settings().defaults.asr_model == "transcribe-1"
    monkeypatch.delenv("FISH_ASR_MODEL")
    assert load_settings().defaults.asr_model == "transcribe-1-pro"


def test_pcm16_format_and_rate() -> None:
    assert pcm_sample_rate("pcm16", {}, 44100) == 24000
    assert pcm_sample_rate("pcm16", {"sample_rate": 16000}, 44100) == 16000
    assert pcm_sample_rate("pcm16", {"sample_rate": "16000.0"}, 44100) == 16000
    assert pcm_sample_rate("pcm16", {"sample_rate": "16000.5"}, 44100) == 24000
    assert pcm_sample_rate("pcm", {}, 44100) == 44100
    assert pcm_sample_rate("pcm", {"sample_rate": 0}, 44100) == 44100
    assert pcm_sample_rate("pcm16", {"sample_rate": -1}, 44100) == 24000
    assert pcm_sample_rate("opus", {"sample_rate": 0}, 48000) == 48000


def test_payload_snaps_bitrate_and_unit_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    _, captured = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "format": "mp3", "mp3_bitrate": 96, "temperature": 5},
    )
    assert captured["json"]["mp3_bitrate"] == 128
    assert captured["json"]["temperature"] == 1.0
    _, captured = post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "format": "opus",
            "opus_bitrate": 1,
            "top_p": -3,
            "early_stop_threshold": 4,
        },
    )
    assert captured["json"]["opus_bitrate"] == -1000
    assert captured["json"]["top_p"] == 0.0
    assert captured["json"]["early_stop_threshold"] == 1.0


def test_omitted_chunk_length_clamps_the_default() -> None:
    cloud = SuiteDefaults(
        chunk_length=900, min_chunk_length=250, fish_base="https://api.fish.audio"
    )
    controls = _controls({}, cloud)
    assert controls.chunk_length == 300
    assert controls.min_chunk_length == 100
    local = SuiteDefaults(chunk_length=800, fish_base="http://127.0.0.1:8080")
    assert _controls({}, local).chunk_length == 800


@pytest.mark.parametrize(
    ("voice_fields", "sent"),
    [
        ({"reference_id": ["a", "b"]}, ["a", "b"]),
        ({"voice": "solo"}, "solo"),
        ({"reference_id": ["  a  ", None, ""]}, ["a"]),
        ({"voice": "  solo  "}, "solo"),
        ({"reference_id": [None]}, None),
        ({"reference_id": [{"id": "x"}, " a ", 1]}, ["a"]),
        ({"reference_id": [1]}, None),
        ({"reference_id": [None, ""], "voice": "kept"}, "kept"),
        ({"reference_id": "", "voice": "kept"}, "kept"),
        ({"reference_id": ["a"], "voice": "ignored"}, ["a"]),
    ],
)
def test_the_voice_sent_to_fish(
    monkeypatch: pytest.MonkeyPatch, voice_fields: dict[str, Any], sent: object
) -> None:
    response, captured = post_speech(monkeypatch, {"input": "Hello there friend", **voice_fields})
    assert response.status_code == 200
    assert captured["json"].get("reference_id") == sent


def test_speech_input_must_be_a_string() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json={"input": ["hello"]})
    assert response.status_code == 400
    assert "input" in response.json()["error"]["message"]
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
        # Past Python's 4300-digit limit, int() raises a plain ValueError.
        huge = client.post(
            "/v1/audio/speech",
            content=b'{"input": "hi", "speed": ' + b"1" * 5000 + b"}",
            headers={"content-type": "application/json"},
        )
        assert huge.status_code == 400
        assert huge.json()["error"]["message"] == "invalid JSON body"
        array = client.post("/v1/audio/speech", json=["nope"])
        assert array.status_code == 400
        assert array.json()["error"]["type"] == "invalid_request_error"
        assert "object" in array.json()["error"]["message"]


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
    nulls = _controls({"chunk_length": None, "min_chunk_length": None})
    assert nulls.chunk_length == defaults.chunk_length
    assert nulls.min_chunk_length == defaults.min_chunk_length
    given = _controls({"chunk_length": 120, "min_chunk_length": 0})
    assert given.chunk_length == 120
    assert given.min_chunk_length == 0


def test_the_old_request_field_spellings_are_ignored() -> None:
    defaults = SuiteDefaults()
    controls = _controls(
        {"fish_chunk_length": 180, "fish_min_chunk_length": 40, "fish_latency": "low"}
    )
    assert controls.chunk_length == defaults.chunk_length
    assert controls.min_chunk_length == defaults.min_chunk_length
    assert controls.latency == defaults.latency
    assert _controls({"fish_format": "wav"}).fmt == ProxySettings().tts_format


def test_quality_guard_reads_the_flag_then_the_features_list() -> None:
    spoken = "Hello there friend"

    def features(body: dict[str, Any]) -> object:
        return json_part(_pack(body)).get("features")

    assert features({"input": spoken, "quality_guard": True}) == ["quality-guard"]
    assert features({"input": spoken, "quality_guard": False}) is None
    assert features({"input": spoken, "quality_guard": None, "features": ["quality-guard"]}) == [
        "quality-guard"
    ]
    assert features({"input": spoken, "fish_quality_guard": True}) is None


def test_surrogate_in_pronunciation_still_encodes() -> None:
    body = {
        "input": "Hello there friend",
        "voice": "voice-\ud800",
        "pronunciation_dictionary": [
            {"items": [{"value": "<|phoneme_start|>\ud800ah<|phoneme_end|>"}]}
        ],
    }
    packed = _pack(body)
    request = httpx.Request("POST", "https://api.fish.audio/v1/tts", **packed.request_kw)
    request.read()
    payload = json_part(packed)
    assert "\ud800" not in payload["reference_id"]
    value = payload["pronunciation_dictionary"][0]["items"][0]["value"]
    assert "\ud800" not in value
    assert "ah" in value
    assert "phoneme" not in value


def test_surrogate_in_speech_text_still_encodes() -> None:
    packed = _pack({"input": "Hello \ud800 there friend"})
    spoken = json_part(packed)["text"]
    assert "\ud800" not in spoken
    assert "Hello" in spoken
    assert "there friend" in spoken
    request = httpx.Request("POST", "https://api.fish.audio/v1/tts", **packed.request_kw)
    request.read()
    sample = base64.b64encode(b"RIFF").decode()
    clip_body = {
        "input": "Hello there friend",
        "references": [{"audio": sample, "text": "sample \ud800 line"}],
    }
    clip_packed = _pack(clip_body)
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
