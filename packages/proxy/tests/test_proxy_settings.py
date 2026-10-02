from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import ormsgpack
import pytest
from fastapi.testclient import TestClient

from fish_audio_suite_proxy.fields import (
    SILENT_MP3,
)
from fish_audio_suite_proxy.server import _uvicorn_run_kwargs, app
from fish_audio_suite_proxy.settings import load_settings
from fish_audio_suite_proxy.upstream import RetryPolicy, retry_after_s

from .helpers import (
    AsrJson,
    FakeUpstream,
    capture_upstream,
    fresh_app_env,
    post_speech,
    run_fish_send,
)


def test_proxy_listens_on_loopback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_PROXY_HOST", raising=False)
    assert _uvicorn_run_kwargs()["host"] == "127.0.0.1"
    assert load_settings().exposed is False
    monkeypatch.setenv("FISH_PROXY_HOST", "0.0.0.0")
    assert load_settings().exposed is True


def test_client_key_is_required_only_when_keys_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 200
    fresh_app_env(monkeypatch, FISH_PROXY_API_KEYS=" alpha , beta ")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        wrong = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert wrong.json()["error"]["type"] == "authentication_error"
        assert client.get("/v1/models", headers={"Authorization": "Basic alpha"}).status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer beta"}).status_code == 200
        speech = client.post("/v1/audio/speech", json={"input": "hello there friend"})
        assert speech.status_code == 401
        health = client.get("/health").json()["defaults"]
        assert health["auth_required"] is True
        assert "alpha" not in json.dumps(health)


def test_a_missing_server_key_is_a_503_not_the_callers_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_app_env(monkeypatch)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    with TestClient(app) as client:
        r = client.post("/v1/audio/speech", json={"input": "hello there friend"})
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "api_error"
    assert "FISH_API_KEY" in r.json()["error"]["message"]


def test_an_oversized_body_is_a_413(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch, FISH_PROXY_MAX_BODY_BYTES="200")
    big = {"input": "hello there friend " * 40}
    with TestClient(app) as client:
        declared = client.post("/v1/audio/speech", json=big)
        assert declared.status_code == 413

        def chunks() -> Any:
            yield b'{"input": "'
            yield b"x" * 300
            yield b'"}'

        streamed = client.post(
            "/v1/audio/speech",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert streamed.status_code == 413
        small = client.post("/v1/audio/speech", json={"input": "hi"})
        assert small.status_code != 413


def test_input_longer_than_the_cap_is_a_400(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch, FISH_PROXY_MAX_INPUT_CHARS="10")
    with TestClient(app) as client:
        r = client.post("/v1/audio/speech", json={"input": "this is far too long"})
    assert r.status_code == 400
    assert "10" in r.json()["error"]["message"]


def test_an_unsupported_response_format_is_a_400(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch)
    response, _captured = post_speech(
        monkeypatch, {"input": "Hello there friend", "response_format": "flac"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_pronunciation_dictionary_must_be_a_bounded_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch)
    bad, _ = post_speech(
        monkeypatch, {"input": "Hello there friend", "pronunciation_dictionary": "x" * 10}
    )
    assert bad.status_code == 400
    too_many, _ = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "pronunciation_dictionary": [{}] * 600},
    )
    assert too_many.status_code == 400


def test_speech_without_clips_is_json_and_with_clips_is_msgpack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_app_env(monkeypatch)
    plain_response, plain = post_speech(monkeypatch, {"input": "Hello there friend"})
    assert plain_response.status_code == 200
    assert plain["headers"]["Content-Type"] == "application/json"
    assert plain["json"]["text"] == "Hello there friend"
    assert plain.get("content") is None
    sample = base64.b64encode(b"RIFF").decode()
    clip_response, clip = post_speech(
        monkeypatch,
        {"input": "Hello there friend", "references": [{"audio": sample, "text": "line"}]},
    )
    assert clip_response.status_code == 200
    assert clip["headers"]["Content-Type"] == "application/msgpack"
    assert clip.get("json") is None
    assert ormsgpack.unpackb(clip["content"])["references"][0]["audio"] == b"RIFF"


def test_models_have_the_openai_fields_and_list_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch, FISH_TTS_ALIASES="my-voice=s2-pro, broken ,=x")
    with TestClient(app) as client:
        data = client.get("/v1/models").json()["data"]
    assert data
    assert all(m["object"] == "model" and m["owned_by"] == "fish-audio" for m in data)
    assert all(isinstance(m["created"], int) for m in data)
    ids = {m["id"] for m in data}
    assert {"tts-1", "my-voice", "whisper-1", "fish-audio/s2.1-pro"} <= ids


def test_custom_alias_resolves_on_the_speech_route(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch, FISH_TTS_ALIASES="my-voice=s2-pro")
    response, captured = post_speech(
        monkeypatch, {"input": "Hello there friend", "model": "my-voice"}
    )
    assert response.status_code == 200
    assert captured["headers"]["model"] == "s2-pro"


def test_new_env_names_win_and_old_names_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FISH_TTS_MODEL", "FISH_MODEL", "FISH_SPEED", "FISH_SPEED_SCALE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FISH_MODEL", "s2-pro")
    monkeypatch.setenv("FISH_SPEED_SCALE", "1.25")
    old = load_settings().defaults
    assert (old.tts_model, old.speed) == ("s2-pro", 1.25)
    monkeypatch.setenv("FISH_TTS_MODEL", "s1")
    monkeypatch.setenv("FISH_SPEED", "0.75")
    new = load_settings().defaults
    assert (new.tts_model, new.speed) == ("s1", 0.75)
    assert load_settings().tts_aliases["tts-1"] == "s1"


def test_text_opt_ins_default_off_and_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FISH_MOOD_LEAD", "FISH_DROP_NARRATION", "FISH_TTS_DIALOGUE_ONLY"):
        monkeypatch.delenv(name, raising=False)
    off = load_settings()
    assert (off.tts_mood_lead, off.tts_drop_narration, off.tts_dialogue_only) == (False,) * 3
    monkeypatch.setenv("FISH_MOOD_LEAD", "1")
    monkeypatch.setenv("FISH_DROP_NARRATION", "yes")
    on = load_settings()
    assert on.tts_mood_lead is True
    assert on.tts_drop_narration is True


def test_tts_text_is_not_logged_at_info_unless_asked(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "a very private sentence for the log test"
    fresh_app_env(monkeypatch)
    with caplog.at_level("INFO", logger="fish-audio-suite-proxy"):
        response, _ = post_speech(monkeypatch, {"input": secret})
    assert response.status_code == 200
    assert "private sentence" not in caplog.text
    assert "raw_len=" in caplog.text
    caplog.clear()
    fresh_app_env(monkeypatch, FISH_PROXY_LOG_TEXT="1")
    with caplog.at_level("INFO", logger="fish-audio-suite-proxy"):
        post_speech(monkeypatch, {"input": secret})
    assert "private sentence" in caplog.text


def test_the_silent_mp3_is_one_whole_frame() -> None:
    header = int.from_bytes(SILENT_MP3[:4], "big")
    assert header >> 21 == 0x7FF  # frame sync
    assert (header >> 19) & 3 == 3  # MPEG-1
    assert (header >> 17) & 3 == 1  # Layer III
    bitrate = (None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)[
        (header >> 12) & 15
    ]
    sample_rate = (44100, 48000, 32000)[(header >> 10) & 3]
    padding = (header >> 9) & 1
    assert bitrate is not None
    assert 144 * bitrate * 1000 // sample_rate + padding == len(SILENT_MP3)


def test_fish_send_stops_at_the_attempt_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(503, b'{"message": "down", "status": 503}') for _ in range(3)],
        policy=RetryPolicy(attempts=3),
    )
    assert out.status_code == 503
    assert client.sends == 3
    assert sleeps.await_count == 2


def test_fish_send_does_not_repeat_a_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [httpx.ReadTimeout("slow"), FakeUpstream(200)],
    )
    assert out.status_code == 504
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    out, _client, sleeps = run_fish_send(
        monkeypatch,
        [
            FakeUpstream(429, b'{"message": "slow", "status": 429}', {"retry-after": "7"}),
            FakeUpstream(200),
        ],
    )
    assert out.status_code == 200
    assert sleeps.await_args is not None
    assert sleeps.await_args.args[0] >= 7


def test_retry_after_s_reads_seconds_only() -> None:
    assert retry_after_s({"retry-after": " 3 "}) == 3.0
    assert retry_after_s({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert retry_after_s({}) is None
    assert retry_after_s(None) is None


def test_fish_send_gives_up_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(503, b'{"message": "down", "status": 503}') for _ in range(5)],
        policy=RetryPolicy(attempts=5, deadline_s=0.01),
    )
    assert out.status_code == 503
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_fish_send_stops_when_the_caller_hung_up(monkeypatch: pytest.MonkeyPatch) -> None:
    async def gone() -> bool:
        return True

    out, client, sleeps = run_fish_send(
        monkeypatch,
        [FakeUpstream(503, b'{"message": "down", "status": 503}'), FakeUpstream(200)],
        is_disconnected=gone,
    )
    assert out.status_code == 499
    assert client.sends == 1
    sleeps.assert_not_awaited()


def test_a_filename_cannot_carry_a_path_or_grow_without_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("../../etc/" + "n" * 400 + ".wav", b"RIFF", "audio/wav")},
        )
    assert r.status_code == 200
    name = captured["files"]["audio"][0]
    assert "/" not in name
    assert len(name) <= 255
