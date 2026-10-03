from __future__ import annotations

import base64

import ormsgpack
import pytest
from fastapi.testclient import TestClient
from proxy_helpers import AsrJson, capture_upstream, post_speech

from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.speech import (
    _scrub_pronunciation_dictionary,
    _seed,
)


def test_data_uri_scheme_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, _captured = post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "input_references": [{"type": "input_audio", "input_audio": "nope"}],
        },
    )
    assert response.status_code == 400
    assert "input_audio" in response.json()["error"]["message"]
    empty, _captured_empty = post_speech(
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
    response, captured = post_speech(
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
    response, _captured = post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": [{"audio": sample, "text": ["nope"]}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "reference text must be a string"
    only_bad, _captured_bad = post_speech(
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
    response, captured = post_speech(
        monkeypatch,
        {
            "input": "Hello there friend",
            "references": [{"audio": sample, "text": "sample line"}, None],
        },
    )
    assert response.status_code == 200
    packed = ormsgpack.unpackb(captured["content"])
    assert packed["references"] == [{"audio": b"RIFF", "text": "sample line"}]
    only_null, _captured = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "references": [None]},
    )
    assert only_null.status_code == 400


def test_reference_string_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    response, captured = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "references": "not-a-clip"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "references must be clips"
    assert "content" not in captured


def test_references_sent_as_msgpack(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "seed": " 42 "},
    )
    assert response.status_code == 200
    assert captured["json"]["seed"] == 42
    _, captured = post_speech(monkeypatch, {"input": "Hello there friend", "seed": "42.0"})
    assert captured["json"]["seed"] == 42
    _, captured = post_speech(monkeypatch, {"input": "Hello there friend", "seed": "42.5"})
    assert "seed" not in captured["json"]
    _, captured = post_speech(monkeypatch, {"input": "Hello there friend", "seed": True})
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
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
    response, captured = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "use_memory_cache": " ON "},
    )
    assert response.status_code == 200
    assert captured["json"]["use_memory_cache"] == "on"


def test_pronunciation_huge_int_with_a_clip_is_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = base64.b64encode(b"RIFF").decode()
    response, _captured = post_speech(
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
    captured = capture_upstream(monkeypatch, AsrJson())
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
    captured = capture_upstream(monkeypatch, AsrJson())
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
    captured = capture_upstream(monkeypatch, AsrJson())
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
    captured = capture_upstream(monkeypatch, AsrJson())
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
    response, _captured = post_speech(
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
