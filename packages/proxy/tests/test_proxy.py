from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock

import httpx
import ormsgpack
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import FormData
from starlette.responses import JSONResponse

from fish_audio_suite_kit import CaptionCue, SuiteDefaults, is_asr_hallucination, is_tts_junk
from fish_audio_suite_proxy.errors import json_from_upstream
from fish_audio_suite_proxy.fields import (
    catalog_ids,
    chunk_length_hi,
    explicit_bool,
    pcm_sample_rate,
    pick_format,
    pick_reference_id,
    prepare_tts_text,
    resolve_asr_model,
    resolve_tts_model,
    runtime_defaults,
    upstream_trace_headers,
)
from fish_audio_suite_proxy.server import _fish_send, _uvicorn_run_kwargs, app
from fish_audio_suite_proxy.speech import (
    _fish_tts_payload,
    _scrub_pronunciation_dictionary,
    _seed,
    _SpeechControls,
    pack_tts,
    speech_controls,
)
from fish_audio_suite_proxy.transcribe import (
    _granularity_list,
    caption_cues,
    form_strings,
    transcription_body,
)


class _FakeUpstream:
    def __init__(self, status_code: int, body: bytes = b"") -> None:
        self.status_code = status_code
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        return None


class _FakeFishClient:
    def __init__(self, outcomes: list[Exception | _FakeUpstream]) -> None:
        self._outcomes = list(outcomes)
        self.sends = 0

    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        headers: Any = None,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "url": url,
            "content": content,
            "data": data,
            "files": files,
            "json": json,
            "headers": headers,
        }

    async def send(self, request: Any, *, stream: bool = False) -> _FakeUpstream:
        del request, stream
        self.sends += 1
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _run_fish_send(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[Exception | _FakeUpstream],
) -> tuple[Any, _FakeFishClient, AsyncMock]:
    sleeps = AsyncMock()
    monkeypatch.setattr("fish_audio_suite_kit.http_errors.asyncio.sleep", sleeps)
    client = _FakeFishClient(outcomes)

    async def _run() -> Any:
        return await _fish_send(
            client,
            stream=False,
            method="POST",
            url="https://api.fish.audio/v1/tts",
        )

    return asyncio.run(_run()), client, sleeps


def test_prepare_tts_normalizes_cues() -> None:
    spoken = prepare_tts_text("Excited, hello there friend", dialogue_only=False)
    assert spoken.startswith("[excited]")
    assert "hello" in spoken


def test_dialogue_only_drops_the_stage_note() -> None:
    raw = 'She smiles. "Hello there friend."'
    spoken = prepare_tts_text(raw, dialogue_only=True)
    assert "Hello there friend" in spoken
    assert "smiles" not in spoken
    assert "smiles" in prepare_tts_text(raw, dialogue_only=False)


def test_explicit_bool_keeps_false_and_uses_the_default_when_absent() -> None:
    assert explicit_bool({}, "dialogue_only", default=True) is True
    assert explicit_bool({"dialogue_only": False}, "dialogue_only", default=True) is False
    assert explicit_bool({"dialogue_only": None}, "dialogue_only", default=True) is True
    assert explicit_bool({"dialogue_only": "false"}, "dialogue_only", default=True) is False
    assert explicit_bool({"dialogue_only": "no"}, "dialogue_only", default=True) is False
    assert explicit_bool({"dialogue_only": "true"}, "dialogue_only", default=False) is True
    assert explicit_bool({"dialogue_only": "maybe"}, "dialogue_only", default=True) is True


def test_unknown_format_with_an_unknown_default_is_mp3() -> None:
    assert pick_format({"format": "nope"}, "not-a-format") == "mp3"


def test_junk_pcm_and_wav_are_silence() -> None:
    with TestClient(app) as client:
        pcm = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "response_format": "pcm"},
        )
        wav = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "response_format": "wav"},
        )
        huge = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "response_format": "wav", "sample_rate": 2**32},
        )
        mp3 = client.post("/v1/audio/speech", json={"input": "hi"})
        opus = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "response_format": "opus"},
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
    assert resolve_asr_model("gpt-4o-transcribe", "transcribe-1\nbad") == "transcribe-1"
    assert resolve_asr_model("whisper-1", "custom\nid") == "transcribe-1"


def test_mp3_bitrate_env_snaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_MP3_BITRATE", "96")
    assert runtime_defaults().mp3_bitrate == 128
    monkeypatch.setenv("FISH_MP3_BITRATE", "192")
    assert runtime_defaults().mp3_bitrate == 192


def test_format_env_uses_the_request_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_FORMAT", " AAC ")
    assert runtime_defaults().audio_format == "mp3"
    monkeypatch.setenv("FISH_FORMAT", "PCM16")
    assert runtime_defaults().audio_format == "pcm16"
    monkeypatch.setenv("FISH_ASR_MODEL", " fish-audio/Transcribe-1 ")
    assert runtime_defaults().asr_model == "transcribe-1"


def test_tts_and_asr_model_aliases() -> None:
    assert resolve_tts_model("fish-audio/s2.1-pro", "s1") == "s2.1-pro"
    assert resolve_tts_model("tts-1", "s1") == "s2.1-pro"
    assert resolve_tts_model("playai-tts", "s1") == "s2.1-pro"
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
    assert pick_format({"response_format": "pcm16"}, "mp3") == "pcm16"
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
        _SpeechControls("s2.1-pro", 1.0, "mp3", "normal", 200, 50),
        "hi",
    )
    assert mp3["mp3_bitrate"] == 128
    assert mp3["temperature"] == 1.0
    opus = _fish_tts_payload(
        {"opus_bitrate": 1, "top_p": -3, "early_stop_threshold": 4},
        defaults,
        _SpeechControls("s2.1-pro", 1.0, "opus", "normal", 200, 50),
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


def test_pick_reference_id_list() -> None:
    assert pick_reference_id({"reference_id": ["a", "b"]}) == ["a", "b"]
    assert pick_reference_id({"voice": "solo"}) == "solo"
    assert pick_reference_id({"reference_id": ["  a  ", None, ""]}) == ["a"]
    assert pick_reference_id({"voice": "  solo  "}) == "solo"
    assert pick_reference_id({"reference_id": [None]}) is None
    assert pick_reference_id({"reference_id": [{"id": "x"}, " a ", 1]}) == ["a"]
    assert pick_reference_id({"reference_id": [1]}) is None
    assert pick_reference_id({"reference_id": [None, ""], "voice": "kept"}) == "kept"
    assert pick_reference_id({"reference_id": "", "voice": "kept"}) == "kept"
    assert pick_reference_id({"reference_id": ["a"], "voice": "ignored"}) == ["a"]


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
    captured = _capture_upstream(monkeypatch, _AudioStream())
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
    captured = _capture_upstream(monkeypatch, _AudioStream())
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
    captured = _capture_upstream(monkeypatch, _AudioStream())
    raw = "She smiles. «Hello there friend.»"
    with TestClient(app) as client:
        kept = client.post("/v1/audio/speech", json={"input": raw})
        assert kept.status_code == 200
        assert "Hello there friend" in captured["json"]["text"]
        dropped = client.post("/v1/audio/speech", json={"input": raw, "dialogue_only": True})
        assert dropped.status_code == 200
        assert "smiles" not in captured["json"]["text"]
        assert "Hello there friend" in captured["json"]["text"]


def test_null_chunk_length_does_not_hide_the_alias() -> None:
    defaults = SuiteDefaults()
    controls = speech_controls(
        {
            "chunk_length": None,
            "fish_chunk_length": 180,
            "min_chunk_length": None,
            "fish_min_chunk_length": 40,
        },
        defaults,
    )
    assert controls.chunk_length == 180
    assert controls.min_chunk_length == 40
    primary = speech_controls(
        {"chunk_length": 120, "fish_chunk_length": 180, "min_chunk_length": 0},
        defaults,
    )
    assert primary.chunk_length == 120
    assert primary.min_chunk_length == 0


def test_null_quality_guard_does_not_hide_the_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_audio_suite_proxy.speech.quality_guard_env", lambda: False)
    defaults = SuiteDefaults()
    spoken = "Hello there friend"

    def features(body: dict[str, Any]) -> object:
        packed = pack_tts(body, defaults, {}, speech_controls(body, defaults), spoken)
        return packed.request_kw["json"].get("features")

    assert features({"input": spoken, "quality_guard": None, "fish_quality_guard": True}) == [
        "quality-guard"
    ]
    assert features({"input": spoken, "quality_guard": False, "fish_quality_guard": True}) is None


def test_surrogate_in_pronunciation_still_encodes() -> None:
    defaults = SuiteDefaults()
    body = {
        "input": "Hello there friend",
        "voice": "voice-\ud800",
        "pronunciation_dictionary": [
            {"items": [{"value": "<|phoneme_start|>\ud800ah<|phoneme_end|>"}]}
        ],
    }
    packed = pack_tts(
        body,
        defaults,
        {},
        speech_controls(body, defaults),
        "Hello there friend",
    )
    request = httpx.Request("POST", "https://api.fish.audio/v1/tts", **packed.request_kw)
    request.read()
    payload = packed.request_kw["json"]
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
    packed = pack_tts(body, defaults, {}, speech_controls(body, defaults), spoken)
    request = httpx.Request("POST", "https://api.fish.audio/v1/tts", **packed.request_kw)
    request.read()
    sample = base64.b64encode(b"RIFF").decode()
    clip_body = {
        "input": "Hello there friend",
        "references": [{"audio": sample, "text": "sample \ud800 line"}],
    }
    clip_packed = pack_tts(
        clip_body,
        defaults,
        {},
        speech_controls(clip_body, defaults),
        "Hello there friend",
    )
    unpacked = ormsgpack.unpackb(clip_packed.request_kw["content"])
    clip_text = unpacked["references"][0]["text"]
    assert "\ud800" not in clip_text
    assert "sample" in clip_text
    assert "line" in clip_text


def test_speech_json_bom_is_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_upstream(monkeypatch, _AudioStream())
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
        assert "boundary" in body["error"]["message"].lower()


def test_transcription_bad_json_is_400() -> None:
    with TestClient(app) as client:
        broken = client.post(
            "/v1/audio/transcriptions",
            content=b"{",
            headers={"content-type": "application/json"},
        )
        assert broken.status_code == 400
        assert broken.json()["error"]["message"] == "invalid JSON body"


def test_speech_without_key_is_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    with TestClient(app) as client:
        r = client.post("/v1/audio/speech", json={"input": "[clear] Hello there friend"})
        assert r.status_code == 401
        body = r.json()
        assert body["error"]["code"] == 401
        assert body["error"]["type"] == "authentication_error"
        assert "message" in body["error"]


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
        assert body["defaults"]["asr_strip_speakers"] is False
        assert body["defaults"]["tts_dialogue_only"] is False
        models = client.get("/v1/models")
        ids = {m["id"] for m in models.json()["data"]}
        assert "s2.1-pro-free" in ids
        assert "drama-3-preview" in ids
        assert "fish-audio/s2.1-pro" in ids


def test_upstream_trace_headers_forward_or_mint() -> None:
    sample = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    forwarded = upstream_trace_headers({"traceparent": sample, "tracestate": "vendor=1"})
    assert forwarded["traceparent"] == sample
    assert forwarded["tracestate"] == "vendor=1"
    minted = upstream_trace_headers({})
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
    out, client, sleeps = _run_fish_send(
        monkeypatch,
        [
            _FakeUpstream(429, b'{"message": "slow down", "status": 429}'),
            _FakeUpstream(200),
        ],
    )
    assert isinstance(out, _FakeUpstream)
    assert out.status_code == 200
    assert client.sends == 2
    sleeps.assert_awaited_once()


def test_fish_send_does_not_treat_a_redirect_as_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, client, sleeps = _run_fish_send(
        monkeypatch,
        [_FakeUpstream(302, b"<html>moved</html>")],
    )
    assert isinstance(out, JSONResponse)
    assert out.status_code == 302
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_does_not_retry_401(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = _run_fish_send(
        monkeypatch,
        [_FakeUpstream(401, b'{"message": "Invalid Token", "status": 401}')],
    )
    assert out.status_code == 401
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_retries_timeout_then_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = _run_fish_send(
        monkeypatch,
        [httpx.TimeoutException("timed out"), _FakeUpstream(200)],
    )
    assert isinstance(out, _FakeUpstream)
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
    assert body["words"] == [{"word": "hello", "start": 0.0, "end": 1.0}]


def test_caption_cues_drop_a_watermark_segment() -> None:
    cues = caption_cues(
        {
            "duration": 1.5,
            "segments": [
                {"text": "Thanks for watching.", "start": 0, "end": 0.4},
                {"text": "ok", "start": 0.4, "end": 0.6},
                {"text": "hello there friend", "start": 0.6, "end": 1.5},
            ],
        },
        "hello there friend",
        strip_speakers=False,
    )
    assert [cue.text for cue in cues] == ["ok", "hello there friend"]
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
        {
            "segments": [
                {"text": ["hello"], "start": 0, "end": 1},
                {"text": "there", "start": 1, "end": 2},
            ]
        },
        "there",
        strip_speakers=False,
    )
    assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 2.0, "there")]


def test_caption_cues_keep_zero_when_times_are_junk() -> None:
    cues = caption_cues(
        {"segments": [{"text": "hello", "start": "nope", "end": None}]},
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


_WAV_UPLOAD = {"file": ("a.wav", b"xx", "audio/wav")}


class _AsrJson:
    def json(self) -> dict[str, Any]:
        return {
            "text": "hello there",
            "duration": 1.5,
            "language": "en",
            "segments": [
                {"text": "hello", "start": 0, "end": 0.6},
                {"text": "there", "start": 0.6, "end": 1.5},
            ],
        }


def test_transcriptions_srt_and_granularities_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_upstream(monkeypatch, _AsrJson())
    with TestClient(app) as client:

        def post(data: dict[str, str]) -> Any:
            return client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD, data=data)

        srt = post({"response_format": "srt"})
        assert srt.status_code == 200
        assert srt.headers["content-type"].startswith("application/x-subrip")
        assert "00:00:00,000 --> 00:00:00,600" in srt.text
        assert "hello" in srt.text
        injected = post({"response_format": "srt\nbad"})
        assert injected.status_code == 200
        assert injected.headers["content-type"].startswith("application/x-subrip")
        assert "00:00:00,000 --> 00:00:00,600" in injected.text
        vtt = post({"response_format": "vtt"})
        assert vtt.text.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:00.600" in vtt.text
        json_body = post({"timestamp_granularities[]": "segment"})
        assert json_body.json() == {"text": "hello there"}
        assert captured["data"]["ignore_timestamps"] == "false"
        verbose = post(
            {
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            }
        )
        payload = verbose.json()
        assert payload["text"] == "hello there"
        assert payload["segments"][0]["text"] == "hello"
        assert payload["words"] == [
            {"word": "hello", "start": 0.0, "end": 0.6},
            {"word": "there", "start": 0.6, "end": 1.5},
        ]
        verbose_seg = post({"response_format": "verbose_json"})
        assert "words" not in verbose_seg.json()


def test_verbose_language_surrogate_still_encodes() -> None:
    body = transcription_body(
        "verbose_json",
        "hello there",
        [CaptionCue(0.0, 1.0, "hello there")],
        {"language": "en\ud800", "duration": 1.0},
        language=None,
        granularities=[],
    )
    encoded = JSONResponse(body).body
    assert b"hello there" in encoded
    assert "\\ud800" not in encoded.decode()


def test_blank_transcript_keeps_segment_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SegmentsOnly:
        def json(self) -> dict[str, Any]:
            return {
                "text": "",
                "duration": 1.5,
                "segments": [
                    {"text": "hello", "start": 0, "end": 0.6},
                    {"text": "there", "start": 0.6, "end": 1.5},
                ],
            }

    _capture_upstream(monkeypatch, _SegmentsOnly())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "hello there"


def test_watermark_transcript_keeps_real_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Watermark:
        def json(self) -> dict[str, Any]:
            return {
                "text": "Thanks for watching.",
                "duration": 1.5,
                "segments": [
                    {"text": "hello", "start": 0, "end": 0.6},
                    {"text": "there", "start": 0.6, "end": 1.5},
                ],
            }

    _capture_upstream(monkeypatch, _Watermark())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "hello there"


def test_watermark_segment_is_left_out_of_the_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Mixed:
        def json(self) -> dict[str, Any]:
            return {
                "text": "Thanks for watching.",
                "duration": 2.0,
                "segments": [
                    {"text": "Thanks for watching.", "start": 0, "end": 0.4},
                    {"text": "ok", "start": 0.4, "end": 0.7},
                    {"text": "hello there friend", "start": 0.7, "end": 2.0},
                ],
            }

    _capture_upstream(monkeypatch, _Mixed())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "ok hello there friend"


def test_real_transcript_drops_a_trailing_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MixedText:
        def json(self) -> dict[str, Any]:
            return {
                "text": "hello there friend. Thanks for watching.",
                "duration": 2.0,
                "segments": [
                    {"text": "hello there friend.", "start": 0, "end": 1.2},
                    {"text": "Thanks for watching.", "start": 1.2, "end": 2.0},
                ],
            }

    _capture_upstream(monkeypatch, _MixedText())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "hello there friend."
    assert "watching" not in response.json()["text"]


def test_srt_drops_a_watermark_when_it_is_the_only_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OnlyWatermarkSegment:
        def json(self) -> dict[str, Any]:
            return {
                "text": "hello there friend. Thanks for watching.",
                "duration": 2.0,
                "segments": [
                    {"text": "Thanks for watching.", "start": 1.2, "end": 2.0},
                ],
            }

    _capture_upstream(monkeypatch, _OnlyWatermarkSegment())
    with TestClient(app) as client:
        srt = client.post(
            "/v1/audio/transcriptions",
            files=_WAV_UPLOAD,
            data={"response_format": "srt"},
        )
        body = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert srt.status_code == 200
    assert "watching" not in srt.text
    assert "hello there friend." in srt.text
    assert body.json()["text"] == "hello there friend."


def test_transcription_non_string_text_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadText:
        def json(self) -> dict[str, Any]:
            return {"text": ["hello"]}

    _capture_upstream(monkeypatch, _BadText())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-object body"


def test_transcription_non_object_upstream_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ListBody:
        def json(self) -> list[str]:
            return ["hello"]

    _capture_upstream(monkeypatch, _ListBody())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-object body"
    assert response.json()["error"]["type"] == "api_error"


def test_verbose_language_nan_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NanLang:
        def json(self) -> dict[str, Any]:
            return {
                "text": "hello there",
                "language_code": float("nan"),
                "language": "en",
            }

    _capture_upstream(monkeypatch, _NanLang())
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files=_WAV_UPLOAD,
            data={"response_format": "verbose_json"},
        )
    assert response.status_code == 200
    assert response.json()["language"] == "en"


def test_verbose_duration_infinity_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Infinite:
        def json(self) -> dict[str, Any]:
            return {"text": "hello there", "duration": float("inf")}

    _capture_upstream(monkeypatch, _Infinite())
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files=_WAV_UPLOAD,
            data={"response_format": "verbose_json"},
        )
    assert response.status_code == 200
    assert response.json()["duration"] is None


def test_transcription_non_json_upstream_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        def json(self) -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "", 0)

    _capture_upstream(monkeypatch, _Broken())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-JSON body"
    assert response.json()["error"]["type"] == "api_error"

    class _NotUtf8:
        def json(self) -> dict[str, Any]:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    _capture_upstream(monkeypatch, _NotUtf8())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=_WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-JSON body"


def test_upstream_errors_use_openai_envelope() -> None:
    resp = json_from_upstream(402, {"message": "no credits", "status": 402})
    assert resp.status_code == 402
    body = json.loads(bytes(resp.body))
    assert body == {
        "error": {
            "code": 402,
            "message": "no credits",
            "type": "provider_error",
            "metadata": {"provider_name": "fish-audio"},
        }
    }
    bad = json_from_upstream(502, {"message": "bad \ud800 byte", "status": 502})
    assert bad.status_code == 502
    encoded = bytes(bad.body)
    encoded.decode("utf-8")
    assert "\ud800" not in encoded.decode("utf-8")


def test_json_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_upstream(monkeypatch, _AsrJson())
    audio = base64.b64encode(b"RIFF").decode()
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": "fish-audio/transcribe-1",
                "input_audio": {"data": audio, "format": "wav"},
            },
        )
    assert r.status_code == 200
    assert r.json()["text"] == "hello there"
    name, data, content_type = captured["files"]["audio"]
    assert name == "utterance.wav"
    assert data == b"RIFF"
    assert content_type == "audio/wav"


class _AudioStream:
    async def aiter_bytes(self, _n: int = 4096):
        yield b"mp3"

    async def aclose(self) -> None:
        return None


def _capture_upstream(monkeypatch: pytest.MonkeyPatch, result: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_send(*_args: object, **kwargs: object) -> Any:
        captured.update(kwargs)
        return result

    monkeypatch.setenv("FISH_API_KEY", "test-key")
    monkeypatch.setattr("fish_audio_suite_proxy.server._fish_send", fake_send)
    return captured


def _post_speech(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
) -> tuple[httpx.Response, dict[str, Any]]:
    captured = _capture_upstream(monkeypatch, _AudioStream())
    with TestClient(app) as client:
        return client.post("/v1/audio/speech", json=body), captured


def test_data_uri_scheme_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": f"DATA:audio/wav;base64,{sample}"},
                },
                {"type": "text", "text": "sample line"},
            ],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_input_references_sent_as_msgpack(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": f"data:audio/wav;base64,{sample}"},
                },
                {"type": "text", "text": "sample line"},
            ],
        },
    )
    assert response.status_code == 200
    assert captured["headers"]["Content-Type"] == "application/msgpack"
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_two_input_audio_parts_both_reach_fish(monkeypatch: pytest.MonkeyPatch) -> None:
    first = base64.b64encode(b"RIFF").decode()
    second = base64.b64encode(b"WAVE").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "input_audio", "input_audio": {"data": first}},
                {"type": "text", "text": "first line"},
                {"type": "input_audio", "input_audio": {"data": second}},
                {"type": "text", "text": "second line"},
            ],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [
        {"audio": b"RIFF", "text": "first line"},
        {"audio": b"WAVE", "text": "second line"},
    ]


def test_broken_input_audio_does_not_wipe_an_earlier_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "input_audio", "input_audio": {"data": sample}},
                {"type": "input_audio", "input_audio": "nope"},
                {"type": "text", "text": "sample line"},
            ],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_empty_later_audio_does_not_wipe_an_earlier_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "input_audio", "input_audio": {"data": sample}},
                {"type": "input_audio", "input_audio": {"data": ""}},
                {"type": "input_audio", "input_audio": {"data": "!!!!"}},
                {"type": "text", "text": "sample line"},
            ],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_missing_text_part_does_not_wipe_an_earlier_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "text", "text": "sample line"},
                {"type": "input_audio", "input_audio": {"data": sample}},
                {"type": "text"},
            ],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_input_references_without_audio_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [{"type": "input_audio", "input_audio": "nope"}],
        },
    )
    assert response.status_code == 400
    assert "input_audio" in response.json()["error"]["message"]
    empty, _captured_empty = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [{"type": "input_audio", "input_audio": {"data": ""}}],
        },
    )
    assert empty.status_code == 400


def test_later_non_string_text_keeps_an_earlier_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "text", "text": 1},
                {"type": "text", "text": "sample line"},
                {"type": "input_audio", "input_audio": {"data": sample}},
                {"type": "text", "text": ["nope"]},
            ],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_reference_text_must_be_a_string(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, _captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": [{"audio": sample, "text": ["nope"]}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "reference text must be a string"
    only_bad, _captured_bad = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "input_audio", "input_audio": {"data": sample}},
                {"type": "text", "text": ["nope"]},
            ],
        },
    )
    assert only_bad.status_code == 400
    assert only_bad.json()["error"]["message"] == "reference text must be a string"


def test_null_reference_does_not_drop_a_real_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": [{"audio": sample, "text": "sample line"}, None],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]
    only_null, _captured = _post_speech(
        monkeypatch,
        {"input": "Hello there friend", "references": [None]},
    )
    assert only_null.status_code == 400


def test_reference_string_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    response, captured = _post_speech(
        monkeypatch,
        {"input": "Hello there friend", "references": "not-a-clip"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "references must be clips"
    assert "content" not in captured


def test_references_sent_as_msgpack(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": [{"audio": sample, "text": "sample line"}],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_urlsafe_reference_audio_is_a_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = bytes(range(64))
    sample = base64.urlsafe_b64encode(raw).decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": [{"audio": f"data:audio/wav;base64,{sample}", "text": "sample line"}],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": raw, "text": "sample line"}]


def test_one_reference_object_is_a_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": {"audio": sample, "text": "sample line"},
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]


def test_seed_keeps_a_padded_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    response, captured = _post_speech(
        monkeypatch,
        {"input": "Hello there friend", "seed": " 42 "},
    )
    assert response.status_code == 200
    assert captured["json"]["seed"] == 42
    _, captured = _post_speech(monkeypatch, {"input": "Hello there friend", "seed": "42.0"})
    assert captured["json"]["seed"] == 42
    _, captured = _post_speech(monkeypatch, {"input": "Hello there friend", "seed": "42.5"})
    assert "seed" not in captured["json"]
    _, captured = _post_speech(monkeypatch, {"input": "Hello there friend", "seed": True})
    assert "seed" not in captured["json"]


def test_seed_infinity_is_omitted() -> None:
    assert _seed(float("inf")) is None
    assert _seed(float("-inf")) is None
    assert _seed(2**64 - 1) == 2**64 - 1
    assert _seed(-(2**63)) == -(2**63)
    assert _seed(2**64) is None
    assert _seed(-(2**63) - 1) is None


def test_huge_token_and_rate_with_a_clip_keep_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "max_new_tokens": 2**64,
            "sample_rate": 2**64,
            "references": [{"audio": sample, "text": "sample line"}],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["max_new_tokens"] == 1024
    assert packed["sample_rate"] == 44100
    assert packed["references"][0]["audio"] == b"RIFF"
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "format": "pcm16",
            "sample_rate": 2**64,
            "references": [{"audio": sample, "text": "sample line"}],
        },
    )
    assert response.status_code == 200
    assert ormsgpack.unpackb(captured["content"])["sample_rate"] == 24_000


def test_huge_seed_with_a_reference_clip_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "seed": 2**64,
            "references": [{"audio": sample, "text": "sample line"}],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert "seed" not in packed
    assert packed["references"][0]["audio"] == b"RIFF"


def test_memory_cache_strips_and_lowercases(monkeypatch: pytest.MonkeyPatch) -> None:
    response, captured = _post_speech(
        monkeypatch,
        {"input": "Hello there friend", "use_memory_cache": " ON "},
    )
    assert response.status_code == 200
    assert captured["json"]["use_memory_cache"] == "on"


def test_pronunciation_huge_int_with_a_clip_is_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, _captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "pronunciation_dictionary": [{"items": [{"value": "ah", "n": 2**64}]}],
            "references": [{"audio": sample, "text": "sample line"}],
        },
    )
    assert response.status_code == 400
    assert "encoded" in response.json()["error"]["message"]


def test_pronunciation_scrub_keeps_non_strings() -> None:
    cleaned = _scrub_pronunciation_dictionary(
        [{"items": [{"value": "<|phoneme_start|>ah<|phoneme_end|>"}, {"value": ["nope"]}]}]
    )
    assert cleaned[0]["items"][0]["value"] == "ah"
    assert cleaned[0]["items"][1]["value"] == ["nope"]


def test_json_asr_format_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_upstream(monkeypatch, _AsrJson())
    audio = base64.b64encode(b"RIFF").decode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            json={"input_audio": {"data": audio, "format": " wav "}},
        )
    assert response.status_code == 200
    name, _data, content_type = captured["files"]["audio"]
    assert name == "utterance.wav"
    assert content_type == "audio/wav"


def test_json_asr_format_cannot_split_the_part_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_upstream(monkeypatch, _AsrJson())
    audio = base64.b64encode(b"RIFF").decode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            json={"input_audio": {"data": audio, "format": "wav\r\nX-Injected: 1"}},
        )
    assert response.status_code == 200
    name, _data, content_type = captured["files"]["audio"]
    assert name == "utterance.wav"
    assert content_type == "audio/wav"


def test_json_asr_language_cannot_inject_a_form_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_upstream(monkeypatch, _AsrJson())
    audio = base64.b64encode(b"RIFF").decode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            json={
                "input_audio": {"data": audio, "format": "wav"},
                "language": 'en\r\nContent-Disposition: form-data; name="hack"',
            },
        )
    assert response.status_code == 200
    assert captured["data"]["language"] == "en"
    assert "hack" not in captured["data"]


def test_json_asr_non_string_format_stays_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_upstream(monkeypatch, _AsrJson())
    audio = base64.b64encode(b"RIFF").decode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            json={"input_audio": {"data": audio, "format": ["wav"]}},
        )
    assert response.status_code == 200
    name, _data, content_type = captured["files"]["audio"]
    assert name == "utterance.wav"
    assert content_type == "audio/wav"


def test_bad_reference_audio_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    response, _captured = _post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [
                {"type": "input_audio", "input_audio": {"data": "!!!!"}},
            ],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
