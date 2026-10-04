from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fish_audio_suite_proxy.server import _uvicorn_run_kwargs, app


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
