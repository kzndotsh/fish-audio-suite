from __future__ import annotations

import base64
from typing import Any

import httpx
import ormsgpack
import pytest
from fastapi.testclient import TestClient
from proxy_helpers import (
    AudioStream,
    FakeUpstream,
    asr_body,
    capture_upstream,
    content_part,
    json_part,
    not_response,
    run_fish_send,
)
from starlette.datastructures import FormData
from starlette.responses import JSONResponse

from fish_audio_suite_kit import (
    CaptionCue,
    SuiteDefaults,
    chunk_length_hi,
    ensure_trace_headers,
    is_asr_hallucination,
    is_tts_junk,
)
from fish_audio_suite_proxy.errors import ProxyError
from fish_audio_suite_proxy.fields import (
    pcm_sample_rate,
    prepare_tts_text,
    read_flag,
    read_format,
    read_reference_id,
)
from fish_audio_suite_proxy.models import catalog_ids, resolve_asr_model, resolve_tts_model
from fish_audio_suite_proxy.phrases import caption_cues
from fish_audio_suite_proxy.server import _uvicorn_run_kwargs, app
from fish_audio_suite_proxy.settings import load_settings, runtime_defaults
from fish_audio_suite_proxy.speech import (
    SpeechControls,
    _fish_tts_payload,
    pack_tts,
    speech_controls,
)
from fish_audio_suite_proxy.transcribe import _granularity_list, form_strings, transcription_body


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


def test_asr_model_whisper_remap() -> None:
    assert resolve_asr_model("whisper-1", "transcribe-1") == "transcribe-1"
    assert resolve_asr_model("whisper-1", " Transcribe-1 ") == "transcribe-1"
    assert resolve_asr_model(None, "fish-audio/Transcribe-1-Pro") == "transcribe-1-pro"
    assert resolve_asr_model("gpt-4o-transcribe", "CustomAsr") == "CustomAsr"
    assert resolve_asr_model("gpt-4o-transcribe", "transcribe-1\nbad") == "transcribe-1-pro"
    assert resolve_asr_model("whisper-1", "custom\nid") == "transcribe-1-pro"
    assert resolve_asr_model("whisper-1", "   ") == "transcribe-1-pro"


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


def test_catalog_includes_fish_audio_slugs() -> None:
    ids = catalog_ids()
    assert "s2.1-pro" in ids
    assert "fish-audio/s2.1-pro" in ids
    assert "fish-audio/transcribe-1" in ids
    assert "whisper-1" in ids
    assert "tts-1" in ids
    assert "my-voice" in catalog_ids({"my-voice": "s2-pro"})
    assert "tts-1" not in catalog_ids({"my-voice": "s2-pro"})


def test_asr_hallucination_without_network() -> None:
    assert is_asr_hallucination("谢谢观看")
    assert not is_asr_hallucination("你好")
    assert not is_asr_hallucination("hello there friend")


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


def test_empty_transcription_is_fish_error_shape() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"", "audio/wav")},
        )
        assert r.status_code == 400
        assert r.json() == {
            "error": {
                "code": 400,
                "message": "empty audio upload",
                "type": "invalid_request_error",
            }
        }


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


def test_transcription_bad_multipart_is_openai_error() -> None:
    with TestClient(app) as client:
        broken = client.post(
            "/v1/audio/transcriptions",
            content=b"not-a-form",
            headers={"content-type": "multipart/form-data"},
        )
        assert broken.status_code == 400
        body = broken.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == "invalid multipart form body"


def test_transcription_bad_json_is_400() -> None:
    with TestClient(app) as client:
        broken = client.post(
            "/v1/audio/transcriptions",
            content=b"{",
            headers={"content-type": "application/json"},
        )
        assert broken.status_code == 400
        assert broken.json()["error"]["message"] == "invalid JSON body"


def test_fish_client_timeout_and_user_agent() -> None:
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        http = app.state.http
        assert isinstance(http, httpx.AsyncClient)
        assert http.timeout.connect == 10.0
        assert http.timeout.read == 120.0
        assert http.timeout.write == 120.0
        assert http.timeout.pool == 5.0
        assert http.headers["User-Agent"].startswith("fish-audio-suite-proxy/")


def test_health_without_api_key() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["defaults"]["asr_strip_speakers"] is True
        assert body["defaults"]["asr_strip_cues"] is False
        assert body["defaults"]["tts_dialogue_only"] is False
        models = client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert "s2.1-pro-free" in ids
        assert "drama-3-preview" in ids
        assert "fish-audio/s2.1-pro" in ids


def test_upstream_trace_headers_forward_or_mint() -> None:
    sample = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    forwarded = ensure_trace_headers({"traceparent": sample, "tracestate": "vendor=1"})
    assert forwarded["traceparent"] == sample
    assert forwarded["tracestate"] == "vendor=1"
    minted = ensure_trace_headers({})
    assert minted["traceparent"].startswith("00-")


def test_uvicorn_run_kwargs_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_PROXY_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("FISH_PROXY_LIMIT_CONCURRENCY", raising=False)
    monkeypatch.delenv("FISH_PROXY_GRACEFUL_SHUTDOWN", raising=False)
    monkeypatch.delenv("FISH_PROXY_PORT", raising=False)
    kw = _uvicorn_run_kwargs()
    assert kw["workers"] == 1
    assert kw["port"] == 8849
    assert kw["loop"] == "auto"
    assert kw["http"] == "auto"
    assert kw["ws"] == "none"
    assert kw["timeout_graceful_shutdown"] == 120
    assert kw["timeout_keep_alive"] == 5
    assert "limit_concurrency" not in kw


def test_uvicorn_port_out_of_range_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_PROXY_PORT", "70000")
    assert _uvicorn_run_kwargs()["port"] == 8849
    monkeypatch.setenv("FISH_PROXY_PORT", "0")
    assert _uvicorn_run_kwargs()["port"] == 8849
    monkeypatch.setenv("FISH_PROXY_PORT", "9000")
    assert _uvicorn_run_kwargs()["port"] == 9000


def test_uvicorn_timeouts_cannot_be_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_PROXY_KEEP_ALIVE", "-1")
    monkeypatch.setenv("FISH_PROXY_GRACEFUL_SHUTDOWN", "-5")
    kw = _uvicorn_run_kwargs()
    assert kw["timeout_keep_alive"] == 5
    assert kw["timeout_graceful_shutdown"] == 120
    monkeypatch.setenv("FISH_PROXY_KEEP_ALIVE", "0")
    monkeypatch.setenv("FISH_PROXY_GRACEFUL_SHUTDOWN", "0")
    kw = _uvicorn_run_kwargs()
    assert kw["timeout_keep_alive"] == 0
    assert kw["timeout_graceful_shutdown"] == 0


def test_uvicorn_run_kwargs_workers_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_PROXY_WORKERS", "4")
    monkeypatch.setenv("FISH_PROXY_LIMIT_CONCURRENCY", "32")
    monkeypatch.setenv("FISH_PROXY_GRACEFUL_SHUTDOWN", "90")
    kw = _uvicorn_run_kwargs()
    assert kw["workers"] == 4
    assert kw["limit_concurrency"] == 32
    assert kw["timeout_graceful_shutdown"] == 90


def test_fish_send_retries_429_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [
            FakeUpstream(429, b'{"message": "slow down", "status": 429}'),
            FakeUpstream(200),
        ],
    )
    assert isinstance(out, FakeUpstream)
    assert out.status_code == 200
    assert client.sends == 2
    sleeps.assert_awaited_once()


def test_fish_send_does_not_treat_a_redirect_as_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(302, b"<html>moved</html>")],
    )
    assert isinstance(out, JSONResponse)
    assert out.status_code == 302
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_does_not_retry_401(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(401, b'{"message": "Invalid Token", "status": 401}')],
    )
    assert out.status_code == 401
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_retries_a_connect_timeout_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [httpx.ConnectTimeout("timed out"), FakeUpstream(200)],
    )
    assert isinstance(out, FakeUpstream)
    assert out.status_code == 200
    assert client.sends == 2
    sleeps.assert_awaited_once()


def test_form_strings_merges_bracket_alias() -> None:
    both = FormData(
        [
            ("timestamp_granularities", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
    )
    assert form_strings(both, "timestamp_granularities", "timestamp_granularities[]") == [
        "word",
        "segment",
    ]
    bracket = FormData([("timestamp_granularities[]", "segment")])
    assert form_strings(bracket, "timestamp_granularities", "timestamp_granularities[]") == [
        "segment"
    ]
    padded = FormData([("timestamp_granularities", " word ")])
    assert form_strings(padded, "timestamp_granularities") == ["word"]


def test_granularity_list_keeps_strings_only() -> None:
    assert _granularity_list([" word ", 1, None, {"type": "word"}]) == ["word"]


def test_verbose_duration_rejects_a_boolean() -> None:
    body = transcription_body(
        "verbose_json",
        "hello",
        [CaptionCue(0.0, 1.5, "hello")],
        {"duration": True},
        language=None,
        granularities=[],
    )
    assert isinstance(body, dict)
    assert body["duration"] is None
    kept = transcription_body(
        "verbose_json",
        "hello",
        [CaptionCue(0.0, 1.5, "hello")],
        {"duration": 1.5},
        language=None,
        granularities=[],
    )
    assert isinstance(kept, dict)
    assert kept["duration"] == 1.5


def test_verbose_words_accepts_padded_granularity() -> None:
    body = transcription_body(
        "verbose_json",
        "hello",
        [CaptionCue(0.0, 1.0, "hello")],
        {},
        language=None,
        granularities=[" word "],
    )
    assert isinstance(body, dict)
    assert "words" not in body
    with_words = transcription_body(
        "verbose_json",
        "hello",
        [CaptionCue(0.0, 1.0, "hello")],
        asr_body({"words": [{"word": "hello", "start": 0.0, "end": 0.4}, {"word": 7}, "junk"]}),
        language=None,
        granularities=[" word "],
    )
    assert isinstance(with_words, dict)
    assert with_words["words"] == [{"word": "hello", "start": 0.0, "end": 0.4}]


def test_caption_cues_drop_a_watermark_segment() -> None:
    # Fish segments are words. The watermark sentence becomes its own phrase and is
    # dropped; the speech after it stays.
    cues = caption_cues(
        {
            "duration": 2.0,
            "segments": [
                {"text": "Thanks", "start": 0, "end": 0.2},
                {"text": "for", "start": 0.2, "end": 0.3},
                {"text": "watching", "start": 0.3, "end": 0.4},
                {"text": "ok", "start": 0.4, "end": 0.6},
                {"text": "hello", "start": 0.6, "end": 1.0},
                {"text": "there", "start": 1.0, "end": 1.3},
                {"text": "friend", "start": 1.3, "end": 1.5},
            ],
        },
        "Thanks for watching. Ok. Hello there friend.",
        strip_speakers=False,
    )
    assert [cue.text for cue in cues] == ["Ok.", "Hello there friend."]
    only = caption_cues(
        {
            "duration": 1.0,
            "segments": [{"text": "Thanks for watching.", "start": 0, "end": 1}],
        },
        "hello there friend",
        strip_speakers=False,
    )
    assert [cue.text for cue in only] == ["hello there friend"]


def test_caption_cues_skip_non_string_segment_text() -> None:
    cues = caption_cues(
        asr_body(
            {
                "segments": [
                    {"text": ["hello"], "start": 0, "end": 1},
                    {"text": "there", "start": 1, "end": 2},
                ]
            }
        ),
        "there",
        strip_speakers=False,
    )
    assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 2.0, "there")]


def test_caption_cues_keep_zero_when_times_are_junk() -> None:
    cues = caption_cues(
        asr_body({"segments": [{"text": "hello", "start": "nope", "end": None}]}),
        "hello",
        strip_speakers=False,
    )
    assert [(c.start, c.end, c.text) for c in cues] == [(0.0, 0.0, "hello")]
    whole = caption_cues({"duration": float("nan")}, "hello", strip_speakers=False)
    assert [(c.start, c.end, c.text) for c in whole] == [(0.0, 0.0, "hello")]
    negative = caption_cues({"duration": -3}, "hello", strip_speakers=False)
    assert [(c.start, c.end, c.text) for c in negative] == [(0.0, 0.0, "hello")]
    flagged = caption_cues({"duration": True}, "hello", strip_speakers=False)
    assert [(c.start, c.end, c.text) for c in flagged] == [(0.0, 0.0, "hello")]
    flagged_start = caption_cues(
        {"segments": [{"text": "hello", "start": True, "end": False}]},
        "hello",
        strip_speakers=False,
    )
    assert [(c.start, c.end, c.text) for c in flagged_start] == [(0.0, 0.0, "hello")]
    backwards = caption_cues(
        {"segments": [{"text": "hello", "start": 2.0, "end": -1.0}]},
        "hello",
        strip_speakers=False,
    )
    assert [(c.start, c.end, c.text) for c in backwards] == [(2.0, 2.0, "hello")]
