from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import ormsgpack
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from proxy_helpers import (
    AsrJson,
    FakeUpstream,
    asr_body,
    capture_upstream,
    fresh_app_env,
    post_speech,
    run_fish_send,
)

from fish_audio_suite_proxy.fields import (
    SILENT_MP3,
)
from fish_audio_suite_proxy.limits import _declared_too_large
from fish_audio_suite_proxy.models import catalog_ids, resolve_tts_model
from fish_audio_suite_proxy.server import _uvicorn_run_kwargs, app
from fish_audio_suite_proxy.settings import ProxySettings, SettingsError, load_settings
from fish_audio_suite_proxy.transcribe import transcription_body
from fish_audio_suite_proxy.upstream import FishHttp, RetryPolicy, fish_send


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
    fresh_app_env(monkeypatch, FISH_PROXY_TTS_ALIASES="my-voice=s2-pro, broken ,=x")
    with TestClient(app) as client:
        data: list[dict[str, Any]] = client.get("/v1/models").json()["data"]
    assert data
    assert all((m["object"], m["owned_by"]) == ("model", "fish-audio") for m in data)
    assert all(isinstance(m["created"], int) for m in data)
    ids = {m["id"] for m in data}
    assert {"tts-1", "my-voice", "whisper-1", "fish-audio/s2.1-pro"} <= ids


def test_custom_alias_resolves_on_the_speech_route(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch, FISH_PROXY_TTS_ALIASES="my-voice=s2-pro")
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


@dataclass(frozen=True)
class _Renamed:
    new: str
    old: str
    value: str
    other: str
    read: Callable[[ProxySettings], object]
    expected: object


# value: what is set; other: a different valid value for the losing name.
_RENAMED_ENV = [
    _Renamed(
        "FISH_TTS_MODEL", "FISH_MODEL", "s2-pro", "s1", lambda s: s.defaults.tts_model, "s2-pro"
    ),
    _Renamed("FISH_SPEED", "FISH_SPEED_SCALE", "1.25", "0.75", lambda s: s.defaults.speed, 1.25),
    _Renamed("FISH_TTS_FORMAT", "FISH_FORMAT", "wav", "opus", lambda s: s.tts_format, "wav"),
    _Renamed(
        "FISH_PROXY_TTS_ALIASES",
        "FISH_TTS_ALIASES",
        "my-voice=s2-pro",
        "my-voice=s1",
        lambda s: s.tts_aliases.get("my-voice"),
        "s2-pro",
    ),
    _Renamed(
        "FISH_PROXY_ASR_STRIP_SPEAKERS",
        "FISH_ASR_STRIP_SPEAKERS",
        "1",
        "0",
        lambda s: s.asr_strip_speakers,
        True,
    ),
    _Renamed(
        "FISH_PROXY_ASR_STRIP_CUES",
        "FISH_ASR_STRIP_CUES",
        "1",
        "0",
        lambda s: s.asr_strip_cues,
        True,
    ),
    _Renamed("FISH_TTS_MOOD_LEAD", "FISH_MOOD_LEAD", "1", "0", lambda s: s.tts_mood_lead, True),
    _Renamed(
        "FISH_TTS_DROP_NARRATION",
        "FISH_DROP_NARRATION",
        "1",
        "0",
        lambda s: s.tts_drop_narration,
        True,
    ),
]


def _clear_renamed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for case in _RENAMED_ENV:
        monkeypatch.delenv(case.new, raising=False)
        monkeypatch.delenv(case.old, raising=False)


@pytest.mark.parametrize("case", _RENAMED_ENV, ids=lambda case: case.old)
def test_an_old_env_name_still_works_and_logs_its_replacement(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: _Renamed
) -> None:
    _clear_renamed_env(monkeypatch)
    assert case.read(load_settings()) != case.expected
    monkeypatch.setenv(case.old, case.value)
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"):
        settings = load_settings()
    assert case.read(settings) == case.expected
    assert f"{case.old} is deprecated; use {case.new}" in caplog.text


def test_an_old_env_name_is_reported_once_however_often_settings_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # FISH_FORMAT feeds two fields, and startup loads the settings twice.
    _clear_renamed_env(monkeypatch)
    monkeypatch.setenv("FISH_FORMAT", "wav")
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"):
        load_settings()
        load_settings()
    assert caplog.text.count("FISH_FORMAT is deprecated; use FISH_TTS_FORMAT") == 1


@pytest.mark.parametrize("case", _RENAMED_ENV, ids=lambda case: case.new)
def test_the_new_env_name_wins_over_the_old_one_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: _Renamed
) -> None:
    _clear_renamed_env(monkeypatch)
    monkeypatch.setenv(case.new, case.value)
    monkeypatch.setenv(case.old, case.other)
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"):
        settings = load_settings()
    assert case.read(settings) == case.expected
    assert "deprecated" not in caplog.text


def test_health_shows_the_tts_keys_under_new_and_old_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_renamed_env(monkeypatch)
    fresh_app_env(monkeypatch, FISH_TTS_MODEL="s1", FISH_TTS_FORMAT="opus", FISH_SPEED="1.5")
    with TestClient(app) as client:
        health = client.get("/health").json()["defaults"]
    assert (health["tts_model"], health["tts_format"], health["tts_speed"]) == ("s1", "opus", 1.5)
    # The old keys stay, with the same values, for one minor release.
    assert (health["model"], health["format"], health["speed_scale"]) == ("s1", "opus", 1.5)


def test_text_opt_ins_default_off_and_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FISH_TTS_MOOD_LEAD",
        "FISH_MOOD_LEAD",
        "FISH_TTS_DROP_NARRATION",
        "FISH_DROP_NARRATION",
        "FISH_TTS_DIALOGUE_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)
    off = load_settings()
    assert (off.tts_mood_lead, off.tts_drop_narration, off.tts_dialogue_only) == (False,) * 3
    monkeypatch.setenv("FISH_TTS_MOOD_LEAD", "1")
    monkeypatch.setenv("FISH_TTS_DROP_NARRATION", "yes")
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


@pytest.mark.parametrize("hint", ["-5", "nan", "inf", "Wed, 21 Oct 2026 07:28:00 GMT", ""])
def test_fish_send_ignores_an_unusable_retry_after(
    monkeypatch: pytest.MonkeyPatch, hint: str
) -> None:
    out, _client, sleeps = run_fish_send(
        monkeypatch,
        [
            FakeUpstream(429, b'{"message": "slow", "status": 429}', {"retry-after": hint}),
            FakeUpstream(200),
        ],
    )
    assert out.status_code == 200
    assert sleeps.await_args is not None
    pause = sleeps.await_args.args[0]
    assert 0 <= pause < 60


def test_fish_send_retries_a_429_and_stops_on_an_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, client, _sleeps = run_fish_send(
        monkeypatch,
        [
            FakeUpstream(429, b'{"message": "slow", "status": 429}'),
            FakeUpstream(401, b'{"message": "bad key", "status": 401}'),
            FakeUpstream(200),
        ],
    )
    assert out.status_code == 401
    assert client.sends == 2


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


class _StalledClient:
    sends = 0

    def build_request(self, method: str, url: str, **_kwargs: Any) -> dict[str, str]:
        return {"method": method, "url": url}

    async def send(self, request: Any, *, stream: bool = False) -> Any:
        del request, stream
        self.sends += 1
        await asyncio.Event().wait()


@pytest.mark.perf
def test_the_deadline_cuts_off_a_stalled_request() -> None:
    client = _StalledClient()

    async def run() -> Any:
        return await fish_send(
            cast(FishHttp, client),
            stream=False,
            policy=RetryPolicy(attempts=5, deadline_s=0.05),
            method="POST",
            url="https://api.fish.audio/v1/tts",
        )

    started = time.monotonic()
    out = asyncio.run(run())
    assert out.status_code == 504
    assert client.sends == 1
    assert time.monotonic() - started < 2


class _BrokenBody(FakeUpstream):
    closed = False

    async def aread(self) -> bytes:
        raise httpx.ReadError("connection dropped")

    async def aclose(self) -> None:
        self.closed = True


def test_a_failed_error_body_read_closes_the_response_and_is_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _BrokenBody(503)
    out, client, _sleeps = run_fish_send(monkeypatch, [broken], policy=RetryPolicy(attempts=1))
    assert broken.closed is True
    assert client.sends == 1
    assert out.status_code >= 500


def test_a_failed_error_body_read_still_retries_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _BrokenBody(503)
    out, client, _sleeps = run_fish_send(
        monkeypatch, [broken, FakeUpstream(200)], policy=RetryPolicy(attempts=3)
    )
    assert broken.closed is True
    assert client.sends == 2
    assert out.status_code == 200


def test_a_huge_content_length_is_too_large_not_an_error() -> None:
    assert _declared_too_large("9" * 5000, 1000) is True
    assert _declared_too_large("1001", 1000) is True
    assert _declared_too_large("1000", 1000) is False
    assert _declared_too_large("0" * 5000 + "5", 1000) is False
    assert _declared_too_large("\u00b2", 1000) is False
    assert _declared_too_large("", 1000) is False
    assert _declared_too_large("abc", 1000) is False


def test_an_unsupported_transcription_format_is_a_400_and_never_calls_fish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_upstream(monkeypatch, AsrJson())
    with TestClient(app) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"xx", "audio/wav")},
            data={"response_format": "xml"},
        )
        ok = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", b"xx", "audio/wav")},
            data={"response_format": "JSON"},
        )
    assert r.status_code == 400
    assert "xml" in r.json()["error"]["message"]
    assert "verbose_json" in r.json()["error"]["message"]
    assert ok.status_code == 200
    assert captured


def test_word_rows_are_scrubbed_like_the_transcript_and_empty_ones_dropped() -> None:
    data = asr_body(
        {
            "words": [
                {"word": "<|speaker:0|>Hello", "start": 0.0, "end": 0.4},
                {"word": "[laughter]", "start": 0.4, "end": 0.9},
                {"word": "ok\ud800", "start": 0.9, "end": 1.2},
                {"word": "   ", "start": 1.2, "end": 1.3},
            ]
        }
    )
    stripped = transcription_body(
        "verbose_json",
        "Hello ok",
        [],
        data,
        language="en",
        granularities=["word"],
        strip_speakers=True,
        strip_cues=True,
    )
    kept = transcription_body(
        "verbose_json", "Hello ok", [], data, language="en", granularities=["word"]
    )
    assert isinstance(stripped, dict)
    assert isinstance(kept, dict)
    assert [w["word"] for w in stripped["words"]] == ["Hello", "ok?"]
    assert "[laughter]" in [w["word"] for w in kept["words"]]
    json.dumps(stripped)


def test_catalog_ids_list_each_id_once_when_an_alias_shadows_a_native_id() -> None:
    ids = catalog_ids({"s2-pro": "s2.1-pro", "my-voice": "s2-pro"})
    assert ids == list(dict.fromkeys(ids))
    assert "my-voice" in ids


def test_an_alias_target_with_the_prefix_reaches_fish_without_it() -> None:
    table = {"my-voice": "fish-audio/s2-pro"}
    assert resolve_tts_model("my-voice", "s2.1-pro", table) == "s2-pro"


def test_a_transport_error_text_never_reaches_the_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "connect to 10.0.0.5:443 refused"
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"):
        out, _client, _sleeps = run_fish_send(
            monkeypatch,
            [httpx.ConnectError(secret) for _ in range(2)],
            policy=RetryPolicy(attempts=2),
        )
    assert out.status_code == 502
    assert b"10.0.0.5" not in out.body
    assert "10.0.0.5" in caplog.text


def test_a_timeout_is_still_a_504_with_the_fixed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, _client, _sleeps = run_fish_send(
        monkeypatch, [httpx.ReadTimeout("slow upstream 10.0.0.5")]
    )
    assert out.status_code == 504
    assert b"10.0.0.5" not in out.body


def test_api_keys_never_appear_in_the_settings_repr() -> None:
    settings = ProxySettings(api_keys=("sk-PROXY-SECRET-1", "sk-PROXY-SECRET-2"))
    assert settings.auth_required is True
    for shown in (repr(settings), str(settings), f"{settings!r}", repr([settings])):
        assert "sk-PROXY-SECRET" not in shown
    assert "sk-PROXY-SECRET" not in json.dumps(settings.health())


@pytest.mark.parametrize("raw", [" , ", ",", ",,  ,", " , , "])
def test_a_key_list_with_no_key_fails_closed(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("FISH_PROXY_API_KEYS", raw)
    with pytest.raises(SettingsError, match="FISH_PROXY_API_KEYS"):
        load_settings()


def test_startup_stops_on_a_key_list_with_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_app_env(monkeypatch, FISH_PROXY_API_KEYS=" , ")
    with pytest.raises(SettingsError), TestClient(app):
        pass


@pytest.mark.parametrize("raw", ["", "   "])
def test_an_empty_key_list_still_means_auth_is_off(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("FISH_PROXY_API_KEYS", raw)
    settings = load_settings()
    assert settings.api_keys == ()
    assert settings.auth_required is False


def test_keys_are_split_and_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_PROXY_API_KEYS", " one , two ,, three ")
    assert load_settings().api_keys == ("one", "two", "three")


def test_startup_warns_once_about_a_cleartext_fish_base(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fresh_app_env(monkeypatch, FISH_BASE="http://user:pw@10.0.0.5:8080")
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"), TestClient(app):
        pass
    warnings = [r for r in caplog.records if "cleartext" in r.getMessage()]
    assert len(warnings) == 1
    assert "10.0.0.5" in warnings[0].getMessage()
    assert "pw" not in warnings[0].getMessage()


@pytest.mark.parametrize("base", ["https://api.fish.audio", "http://127.0.0.1:8080"])
def test_startup_does_not_warn_for_https_or_loopback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, base: str
) -> None:
    fresh_app_env(monkeypatch, FISH_BASE=base)
    with caplog.at_level("WARNING", logger="fish-audio-suite-proxy"), TestClient(app):
        pass
    assert not [r for r in caplog.records if "cleartext" in r.getMessage()]


def _cut_off_response(status: int) -> tuple[httpx.AsyncClient, list[int]]:
    class CutOff(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            raise httpx.ReadError("connection dropped mid-body on 10.0.0.5")
            yield b""  # pragma: no cover

    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, headers={"retry-after": "2"}, stream=CutOff())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


@pytest.mark.parametrize(
    ("status", "retried"), [(401, False), (400, False), (429, True), (503, True)]
)
def test_an_unreadable_error_body_keeps_the_status_fish_sent(
    monkeypatch: pytest.MonkeyPatch, status: int, retried: bool
) -> None:
    client, calls = _cut_off_response(status)
    monkeypatch.setattr("fish_audio_suite_proxy.upstream.asyncio.sleep", AsyncMock())

    async def run() -> tuple[int, bytes]:
        async with client:
            out = await fish_send(
                client,
                stream=True,
                method="POST",
                url="https://fish.test/v1/tts",
                policy=RetryPolicy(attempts=3),
            )
            assert isinstance(out, JSONResponse)
            return out.status_code, bytes(out.body)

    code, body = asyncio.run(run())
    assert code == status
    assert len(calls) == (3 if retried else 1)
    assert b"error body could not be read" in body
    assert b"10.0.0.5" not in body
    assert b"connection dropped" not in body
