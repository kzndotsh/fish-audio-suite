from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from fish_audio_suite_kit import (
    CaptionCue,
)
from fish_audio_suite_proxy.errors import json_from_upstream
from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.transcribe import (
    transcription_body,
)

from .helpers import WAV_UPLOAD, AsrJson, capture_upstream


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
        # Fish sent phrases, not word timings, so no words are invented.
        assert "words" not in payload
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

    capture_upstream(monkeypatch, _SegmentsOnly())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
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

    capture_upstream(monkeypatch, _Watermark())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
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

    capture_upstream(monkeypatch, _Mixed())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
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

    capture_upstream(monkeypatch, _MixedText())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
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
    class _BadText:
        def json(self) -> dict[str, Any]:
            return {"text": ["hello"]}

    capture_upstream(monkeypatch, _BadText())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-object body"


def test_transcription_non_object_upstream_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ListBody:
        def json(self) -> list[str]:
            return ["hello"]

    capture_upstream(monkeypatch, _ListBody())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
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
    class _Infinite:
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
    class _Broken:
        def json(self) -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "", 0)

    capture_upstream(monkeypatch, _Broken())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Fish returned a non-JSON body"
    assert response.json()["error"]["type"] == "api_error"

    class _NotUtf8:
        def json(self) -> dict[str, Any]:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    capture_upstream(monkeypatch, _NotUtf8())
    with TestClient(app) as client:
        response = client.post("/v1/audio/transcriptions", files=WAV_UPLOAD)
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
