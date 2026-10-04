from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from proxy_helpers import (
    WAV_UPLOAD,
    AsrJson,
    FakeUpstream,
    asr_body,
    capture_upstream,
    post_speech,
    run_fish_send,
)
from starlette.datastructures import FormData, UploadFile
from starlette.responses import JSONResponse

from fish_audio_suite_kit import (
    CaptionCue,
    is_asr_hallucination,
)
from fish_audio_suite_proxy.errors import provider_json_from_raw
from fish_audio_suite_proxy.phrases import caption_cues
from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.settings import load_settings
from fish_audio_suite_proxy.transcribe import _granularity_list, form_strings, transcription_body
from fish_audio_suite_proxy.upstream import fish_send


def test_transcriptions_srt_and_granularities_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:

        def post(data: dict[str, str]) -> Any:
            return client.post("/v1/audio/transcriptions", files=WAV_UPLOAD, data=data)

        srt = post({"response_format": "srt"})
        assert srt.status_code == 200
        assert srt.headers["content-type"].startswith("application/x-subrip")
        assert "00:00:00,000 --> 00:00:01,500" in srt.text
        assert "hello there" in srt.text
        injected = post({"response_format": "srt\nbad"})
        assert injected.status_code == 200
        assert injected.headers["content-type"].startswith("application/x-subrip")
        assert "00:00:00,000 --> 00:00:01,500" in injected.text
        vtt = post({"response_format": "vtt"})
        assert vtt.text.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.500" in vtt.text
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
        # Fish segments are words. They are grouped into one phrase segment and
        # returned as OpenAI words.
        assert payload["segments"] == [{"id": 0, "text": "hello there", "start": 0, "end": 1.5}]
        assert payload["words"] == [
            {"word": "hello", "start": 0, "end": 0.6},
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
    encoded = bytes(JSONResponse(body).body)
    assert b"hello there" in encoded
    assert "\\ud800" not in encoded.decode()


def test_blank_transcript_keeps_segment_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SegmentsOnly(AsrJson):
        def json(self) -> dict[str, Any]:
            return {
                "text": "",
                "duration": 1.5,
                "segments": [
                    {"text": "hello", "start": 0, "end": 0.6},
                    {"text": "there", "start": 0.6, "end": 1.5},
                ],
            }

    capture_upstream(monkeypatch, _SegmentsOnly())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "hello there"


def test_watermark_transcript_keeps_real_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Watermark(AsrJson):
        def json(self) -> dict[str, Any]:
            return {
                "text": "Thanks for watching.",
                "duration": 1.5,
                "segments": [
                    {"text": "hello", "start": 0, "end": 0.6},
                    {"text": "there", "start": 0.6, "end": 1.5},
                ],
            }

    capture_upstream(monkeypatch, _Watermark())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "hello there"


def test_watermark_segment_is_left_out_of_the_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Mixed(AsrJson):
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

    capture_upstream(monkeypatch, _Mixed())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "ok hello there friend"


def test_real_transcript_drops_a_trailing_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MixedText(AsrJson):
        def json(self) -> dict[str, Any]:
            return {
                "text": "hello there friend. Thanks for watching.",
                "duration": 2.0,
                "segments": [
                    {"text": "hello there friend.", "start": 0, "end": 1.2},
                    {"text": "Thanks for watching.", "start": 1.2, "end": 2.0},
                ],
            }

    capture_upstream(monkeypatch, _MixedText())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 200
    assert response.json()["text"] == "hello there friend."
    assert "watching" not in response.json()["text"]


def test_srt_drops_a_watermark_when_it_is_the_only_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OnlyWatermarkSegment(AsrJson):
        def json(self) -> dict[str, Any]:
            return {
                "text": "hello there friend. Thanks for watching.",
                "duration": 2.0,
                "segments": [
                    {"text": "Thanks for watching.", "start": 1.2, "end": 2.0},
                ],
            }

    capture_upstream(monkeypatch, _OnlyWatermarkSegment())
    with TestClient(app) as client:
        srt = client.post(
            "/v1/audio/transcriptions",
            files=WAV_UPLOAD,
            data={"response_format": "srt"},
        )
        body = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert srt.status_code == 200
    assert "watching" not in srt.text
    assert "hello there friend." in srt.text
    assert body.json()["text"] == "hello there friend."


def test_transcription_non_string_text_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadText(AsrJson):
        def json(self) -> dict[str, Any]:
            return {"text": ["hello"]}

    capture_upstream(monkeypatch, _BadText())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-object body"


def test_transcription_non_object_upstream_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ListBody(AsrJson):
        def json(self) -> list[str]:
            return ["hello"]

    capture_upstream(monkeypatch, _ListBody())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-object body"
    assert response.json()["error"]["type"] == "api_error"


def test_verbose_language_nan_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NanLang(AsrJson):
        def json(self) -> dict[str, Any]:
            return {
                "text": "hello there",
                "language_code": float("nan"),
                "language": "en",
            }

    capture_upstream(monkeypatch, _NanLang())
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files=WAV_UPLOAD,
            data={"response_format": "verbose_json"},
        )
    assert response.status_code == 200
    assert response.json()["language"] == "en"


def test_verbose_duration_infinity_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Infinite(AsrJson):
        def json(self) -> dict[str, Any]:
            return {"text": "hello there", "duration": float("inf")}

    capture_upstream(monkeypatch, _Infinite())
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            files=WAV_UPLOAD,
            data={"response_format": "verbose_json"},
        )
    assert response.status_code == 200
    assert response.json()["duration"] is None


def test_transcription_non_json_upstream_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken(AsrJson):
        def json(self) -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "", 0)

    capture_upstream(monkeypatch, _Broken())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-JSON body"
    assert response.json()["error"]["type"] == "api_error"

    class _NotUtf8(AsrJson):
        def json(self) -> dict[str, Any]:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    capture_upstream(monkeypatch, _NotUtf8())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-JSON body"


def test_upstream_errors_use_openai_envelope() -> None:
    resp = provider_json_from_raw(402, {"message": "no credits", "status": 402})
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
    bad = provider_json_from_raw(502, {"message": "bad \ud800 byte", "status": 502})
    assert bad.status_code == 502
    encoded = bytes(bad.body)
    encoded.decode("utf-8")
    assert "\ud800" not in encoded.decode("utf-8")


def test_json_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = capture_upstream(monkeypatch, AsrJson())
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


def _spy_on_upload_close(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    closed: list[str | None] = []
    original = UploadFile.close

    async def spy(self: UploadFile) -> None:
        closed.append(self.filename)
        await original(self)

    monkeypatch.setattr(UploadFile, "close", spy)
    return closed


def test_transcription_closes_the_uploaded_file(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = _spy_on_upload_close(monkeypatch)
    capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 200
    assert closed == ["a.wav"]


def test_transcription_closes_an_empty_upload_on_the_early_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _spy_on_upload_close(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions", files={"file": ("empty.wav", b"", "audio/wav")}
        )
    assert response.status_code == 400
    assert closed == ["empty.wav"]


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        ("!!!!", "input_audio is not valid base64"),
        ("", "input_audio is empty"),
        (None, "input_audio is empty"),
    ],
)
def test_bad_json_input_audio_names_input_audio_not_a_reference(
    monkeypatch: pytest.MonkeyPatch, audio: str | None, message: str
) -> None:
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            json={"input_audio": {"data": audio, "format": "wav"}},
        )
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["message"] == message
    assert "reference" not in error["message"]
    assert error["type"] == "invalid_request_error"
    assert captured == {}


def test_speaker_labels_are_stripped_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Labelled(AsrJson):
        def json(self) -> dict[str, Any]:
            return {"text": "<|speaker:0|> Speaker 1: hello there"}

    monkeypatch.delenv("FISH_PROXY_ASR_STRIP_SPEAKERS", raising=False)
    capture_upstream(monkeypatch, _Labelled())
    with TestClient(app) as client:
        stripped = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD).json()
    assert stripped == {"text": "hello there"}
    monkeypatch.setenv("FISH_PROXY_ASR_STRIP_SPEAKERS", "0")
    with TestClient(app) as client:
        kept = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD).json()
    assert kept == {"text": "Speaker 1: hello there"}


def test_transcription_gets_the_asr_timeout_and_speech_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FISH_PROXY_ASR_TIMEOUT", raising=False)
    monkeypatch.delenv("FISH_PROXY_RETRY_DEADLINE", raising=False)
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        assert client.post("/v1/audio/transcriptions", files=WAV_UPLOAD).status_code == 200
    assert captured["read_timeout_s"] == 900.0
    assert captured["policy"].deadline_s == 900.0

    monkeypatch.setenv("FISH_PROXY_ASR_TIMEOUT", "1800")
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        assert client.post("/v1/audio/transcriptions", files=WAV_UPLOAD).status_code == 200
        assert client.get("/health").json()["defaults"]["asr_timeout_s"] == 1800.0
    assert captured["read_timeout_s"] == 1800.0
    assert captured["policy"].deadline_s == 1800.0

    reply, spoken = post_speech(monkeypatch, {"input": "hello there", "voice": "v"})
    assert reply.status_code == 200
    assert spoken.get("read_timeout_s") is None
    assert spoken["policy"].deadline_s == 90.0


def test_a_bad_asr_timeout_keeps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("0", "-5", "soon"):
        monkeypatch.setenv("FISH_PROXY_ASR_TIMEOUT", raw)
        assert load_settings().asr_timeout_s == 900.0


def test_fish_send_sets_the_read_timeout_on_that_request_only() -> None:
    seen: list[dict[str, Any]] = []

    class _Client:
        def __init__(self) -> None:
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0, pool=5.0))

        def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
            return self.http.build_request(method, url, **kwargs)

        async def send(self, request: httpx.Request, *, stream: bool = False) -> httpx.Response:
            del stream
            seen.append(dict(request.extensions["timeout"]))
            return httpx.Response(200, request=request)

    async def run() -> None:
        client = _Client()
        await fish_send(client, stream=False, method="POST", url="https://fish.test/v1/asr")
        await fish_send(
            client,
            stream=False,
            method="POST",
            url="https://fish.test/v1/asr",
            read_timeout_s=900.0,
        )
        await client.http.aclose()

    asyncio.run(run())
    assert seen[0] == {"connect": 10.0, "read": 120.0, "write": 120.0, "pool": 5.0}
    assert seen[1] == {"connect": 10.0, "read": 900.0, "write": 900.0, "pool": 5.0}


@pytest.mark.parametrize(
    ("client_hint", "env_hint", "sent"),
    [
        ("en-US", "", "en"),
        ("EN", "", "en"),
        ("zh_CN", "", "zh"),
        ("English", "", None),
        ("", "", None),
        ("", "ja-JP", "ja"),
        ("English", "de", "de"),
        ("en\r\nx", "", "en"),
    ],
)
def test_the_language_hint_sent_to_fish_is_a_two_letter_code(
    monkeypatch: pytest.MonkeyPatch, client_hint: str, env_hint: str, sent: str | None
) -> None:
    monkeypatch.setenv("FISH_ASR_LANGUAGE", env_hint)
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        reply = client.post(
            "/v1/audio/transcriptions", files=WAV_UPLOAD, data={"language": client_hint}
        )
    assert reply.status_code == 200
    assert captured["data"].get("language") == sent


def _post_pro(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, str], *, model: str = "transcribe-1-pro"
) -> tuple[Any, dict[str, Any]]:
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        reply = client.post(
            "/v1/audio/transcriptions", files=WAV_UPLOAD, data={"model": model, **fields}
        )
    return reply, captured


def test_pro_fields_are_forwarded_to_transcribe_1_pro(monkeypatch: pytest.MonkeyPatch) -> None:
    reply, captured = _post_pro(
        monkeypatch,
        {"diarize": " TRUE ", "min_speakers": "2", "max_speakers": "3", "tag_audio_events": "0"},
    )
    assert reply.status_code == 200
    assert captured["headers"]["model"] == "transcribe-1-pro"
    form = captured["data"]
    assert form["diarize"] == "true"
    assert (form["min_speakers"], form["max_speakers"]) == ("2", "3")
    assert form["tag_audio_events"] == "false"
    assert "num_speakers" not in form


def test_pro_fields_are_not_sent_to_transcribe_1(monkeypatch: pytest.MonkeyPatch) -> None:
    reply, captured = _post_pro(
        monkeypatch, {"diarize": "auto", "num_speakers": "2"}, model="transcribe-1"
    )
    assert reply.status_code == 200
    assert captured["headers"]["model"] == "transcribe-1"
    assert not {"diarize", "num_speakers", "tag_audio_events"} & set(captured["data"])


def test_pro_fields_in_a_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = capture_upstream(monkeypatch, AsrJson())
    audio = base64.b64encode(b"RIFFxxxx").decode()
    with TestClient(app) as client:
        reply = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": "transcribe-1-pro",
                "input_audio": {"data": audio, "format": "wav"},
                "diarize": False,
                "tag_audio_events": True,
            },
        )
        assert reply.status_code == 200
        assert captured["data"]["diarize"] == "false"
        assert captured["data"]["tag_audio_events"] == "true"
        counted = client.post(
            "/v1/audio/transcriptions",
            json={
                "model": "transcribe-1-pro",
                "input_audio": {"data": audio, "format": "wav"},
                "num_speakers": 2,
            },
        )
        assert counted.status_code == 200
        assert captured["data"]["num_speakers"] == "2"
        bad = client.post(
            "/v1/audio/transcriptions",
            json={"input_audio": {"data": audio, "format": "wav"}, "num_speakers": True},
        )
        assert bad.status_code == 400


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"diarize": "maybe"}, "diarize must be auto, true or false"),
        ({"num_speakers": "0"}, "num_speakers must be a whole number of at least 1"),
        ({"min_speakers": "-1"}, "min_speakers must be a whole number of at least 1"),
        ({"max_speakers": "two"}, "max_speakers must be a whole number of at least 1"),
        ({"num_speakers": "2.5"}, "num_speakers must be a whole number of at least 1"),
        (
            {"num_speakers": "2", "min_speakers": "1"},
            "num_speakers cannot be combined with min_speakers or max_speakers",
        ),
        (
            {"num_speakers": "2", "max_speakers": "3"},
            "num_speakers cannot be combined with min_speakers or max_speakers",
        ),
        (
            {"min_speakers": "3", "max_speakers": "2"},
            "min_speakers cannot be more than max_speakers",
        ),
        (
            {"diarize": "false", "num_speakers": "2"},
            "speaker counts cannot be sent with diarize=false",
        ),
        (
            {"diarize": "false", "max_speakers": "2"},
            "speaker counts cannot be sent with diarize=false",
        ),
        ({"tag_audio_events": "yes please"}, "tag_audio_events must be true or false"),
    ],
)
def test_invalid_pro_fields_are_a_400(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, str], message: str
) -> None:
    reply, captured = _post_pro(monkeypatch, fields)
    assert reply.status_code == 400
    error = reply.json()["error"]
    assert error["message"] == message
    assert error["type"] == "invalid_request_error"
    assert not captured


def test_blank_pro_fields_are_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    reply, captured = _post_pro(monkeypatch, {"diarize": " ", "num_speakers": ""})
    assert reply.status_code == 200
    assert not {"diarize", "num_speakers"} & set(captured["data"])


def test_a_pro_error_keeps_fish_code_and_request_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    body = json.dumps(
        {
            "status": 400,
            "message": "audio is longer than 60 minutes",
            "code": "audio_too_long",
            "request_id": "req-123",
        }
    ).encode()
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"):
        out, _client, _sleeps = run_fish_send(
            monkeypatch, [FakeUpstream(400, body, {"x-request-id": "req-header"})]
        )
    assert isinstance(out, JSONResponse)
    assert out.status_code == 400
    error = json.loads(bytes(out.body))["error"]
    # error.code stays the HTTP status, as on every other proxy error.
    assert error["code"] == 400
    assert error["type"] == "provider_error"
    assert error["message"] == "audio is longer than 60 minutes"
    assert error["metadata"] == {
        "provider_name": "fish-audio",
        "provider_code": "audio_too_long",
        "request_id": "req-123",
    }
    assert "code=audio_too_long request_id=req-123" in caplog.text


def test_a_fish_error_takes_the_request_id_from_the_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _client, _sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(401, b'{"message": "bad key"}', {"x-request-id": " rid-9\r\nX: y "})],
    )
    assert isinstance(out, JSONResponse)
    metadata = json.loads(bytes(out.body))["error"]["metadata"]
    assert metadata == {"provider_name": "fish-audio", "request_id": "rid-9"}


def test_a_fish_error_without_ids_has_only_the_provider_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _client, _sleeps = run_fish_send(
        monkeypatch, [FakeUpstream(402, b"not json", {}), FakeUpstream(402, b"", {})]
    )
    assert isinstance(out, JSONResponse)
    assert json.loads(bytes(out.body))["error"]["metadata"] == {"provider_name": "fish-audio"}


def test_a_transcription_log_names_the_fish_request_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _WithId(AsrJson):
        def json(self) -> dict[str, Any]:
            return {"text": "hello there", "request_id": "0b6f4c1e"}

    capture_upstream(monkeypatch, _WithId())
    with caplog.at_level("INFO", logger="fish-audio-suite-proxy"), TestClient(app) as client:
        assert client.post("/v1/audio/transcriptions", files=WAV_UPLOAD).status_code == 200
    assert "request_id=0b6f4c1e" in caplog.text


@pytest.mark.parametrize(
    ("fish", "hint", "language"),
    [
        ({"language": "English", "language_code": "en"}, "en", "english"),
        ({"language": "Chinese"}, None, "chinese"),
        ({"language": "", "language_code": "zh"}, "en", "zh"),
        ({"language_code": "ja"}, None, "ja"),
        ({}, "de", "de"),
        ({"language": 7}, None, None),
    ],
)
def test_verbose_language_is_the_lowercase_name(
    fish: dict[str, Any], hint: str | None, language: str | None
) -> None:
    body = transcription_body(
        "verbose_json",
        "hi",
        [CaptionCue(0.0, 1.0, "hi")],
        asr_body({"text": "hi", **fish}),
        language=hint,
        granularities=[],
    )
    assert isinstance(body, dict)
    assert body["language"] == language


def test_verbose_language_falls_back_to_the_hint_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoLanguage(AsrJson):
        def json(self) -> dict[str, Any]:
            return {"text": "hallo", "duration": 1.0}

    monkeypatch.delenv("FISH_ASR_LANGUAGE", raising=False)
    capture_upstream(monkeypatch, _NoLanguage())
    with TestClient(app) as client:
        reply = client.post(
            "/v1/audio/transcriptions",
            files=WAV_UPLOAD,
            data={"response_format": "verbose_json", "language": "de-DE"},
        )
    assert reply.json()["language"] == "de"


def test_asr_hallucination_without_network() -> None:
    assert is_asr_hallucination("谢谢观看")
    assert not is_asr_hallucination("你好")
    assert not is_asr_hallucination("hello there friend")


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
