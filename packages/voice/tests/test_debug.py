from __future__ import annotations

import io
import logging
import os
import sys

import pytest

from fish_audio_suite_voice.debug import (
    configure_voice_logging,
    debug,
    env_debug,
    header_meta,
    public_meta,
    write_reply_token,
    ws_event_view,
)


def test_configure_voice_logging_replaces_library_handlers() -> None:
    previous = os.environ.get("FISH_VOICE_DEBUG")
    try:
        os.environ.pop("FISH_VOICE_DEBUG", None)
        configure_voice_logging(debug=False)
        configure_voice_logging(debug=False)
        httpx_log = logging.getLogger("httpx")
        assert httpx_log.level == logging.WARNING
        assert len(httpx_log.handlers) == 1
        configure_voice_logging(debug=True)
        assert logging.getLogger("httpx").level == logging.INFO
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert len(logging.getLogger("asyncio").handlers) == 1
    finally:
        configure_voice_logging(debug=False)
        if previous is None:
            os.environ.pop("FISH_VOICE_DEBUG", None)
        else:
            os.environ["FISH_VOICE_DEBUG"] = previous


def test_debug_follows_a_replaced_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_voice_logging(debug=True)
    caught = io.StringIO()
    monkeypatch.setattr(sys, "stderr", caught)
    debug("after-redirect")
    assert "after-redirect" in caught.getvalue()


def test_debug_closes_reply_before_the_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_voice_logging(debug=True)
    write_reply_token("[calm] hey")
    debug("llm.done")
    captured = capsys.readouterr()
    assert captured.out == "[calm] hey\n"
    assert "llm.done" in captured.err
    assert not captured.err.startswith("[calm]")


def test_env_debug_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_DEBUG", raising=False)
    assert not env_debug()
    monkeypatch.setenv("FISH_VOICE_DEBUG", "1")
    assert env_debug()
    monkeypatch.setenv("FISH_VOICE_DEBUG", "true")
    assert env_debug()
    monkeypatch.setenv("FISH_VOICE_DEBUG", "0")
    assert not env_debug()


def test_ws_event_view_strips_audio() -> None:
    view = ws_event_view({"event": "audio", "audio": b"\x00" * 12, "time": 1.5})
    assert view == {"event": "audio", "audio_bytes": 12, "time": 1.5}


def test_public_meta_drops_text_and_secrets() -> None:
    meta = public_meta(
        {
            "text": "hello there friend",
            "duration": 1.2,
            "language": "en",
            "Authorization": "Bearer secret",
            "segments": [{}, {}],
        }
    )
    assert meta["text_chars"] == 18
    assert meta["duration"] == 1.2
    assert meta["language"] == "en"
    assert meta["segments_len"] == 2
    assert "Authorization" not in meta


def test_header_meta_keeps_x_headers() -> None:
    got = header_meta(
        {
            "Authorization": "Bearer x",
            "x-request-id": "abc",
            "Content-Type": "application/json",
        }
    )
    assert got["x-request-id"] == "abc"
    assert "Authorization" not in got
    assert got["Content-Type"] == "application/json"
