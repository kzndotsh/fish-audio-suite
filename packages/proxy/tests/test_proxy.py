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

from fish_audio_suite_kit import CaptionCue, SuiteDefaults, is_asr_hallucination, is_tts_junk
from fish_audio_suite_proxy.contract import (
    caption_cues,
    catalog_ids,
    chunk_length_hi,
    form_strings,
    json_from_upstream,
    pcm_sample_rate,
    pick_format,
    pick_reference_id,
    prepare_tts_text,
    resolve_asr_model,
    resolve_tts_model,
    runtime_defaults,
    speech_controls,
    transcription_body,
    upstream_trace_headers,
)
from fish_audio_suite_proxy.server import _fish_send, _uvicorn_run_kwargs, app
from fish_audio_suite_proxy.speech import (
    _fish_tts_payload,
    _scrub_pronunciation_dictionary,
    _seed,
    _SpeechControls,
)
from fish_audio_suite_proxy.transcribe import _granularity_list


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
    flagged = caption_cues({"duration": True}, "hello", strip_speakers=False)
    assert [(c.start, c.end, c.text) for c in flagged] == [(0.0, 1.0, "hello")]


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


def test_seed_keeps_a_padded_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    response, captured = _post_speech(
        monkeypatch,
        {"input": "Hello there friend", "seed": " 42 "},
    )
    assert response.status_code == 200
    assert captured["json"]["seed"] == 42
    _, captured = _post_speech(monkeypatch, {"input": "Hello there friend", "seed": True})
    assert "seed" not in captured["json"]


def test_seed_infinity_is_omitted() -> None:
    assert _seed(float("inf")) is None
    assert _seed(float("-inf")) is None


def test_memory_cache_strips_and_lowercases(monkeypatch: pytest.MonkeyPatch) -> None:
    response, captured = _post_speech(
        monkeypatch,
        {"input": "Hello there friend", "use_memory_cache": " ON "},
    )
    assert response.status_code == 200
    assert captured["json"]["use_memory_cache"] == "on"


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
