from __future__ import annotations

import io
import logging
import os
import sys

import pytest

from fish_audio_suite_voice.debug import (
    configure_voice_logging,
    conversation,
    debug,
    debug_level,
    env_debug,
    header_meta,
    heartbeat_due,
    mark_turn,
    public_meta,
    short_model,
    trace,
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
        assert logging.getLogger("httpx").level == logging.WARNING
        configure_voice_logging(debug=2)
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
    debug("llm.stream_end finish=stop")
    captured = capsys.readouterr()
    assert captured.out == "[calm] hey\n"
    assert "llm" in captured.err
    assert "stream_end finish=stop" in captured.err
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


def test_debug_level_and_trace_gating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("FISH_VOICE_DEBUG", raising=False)
    assert debug_level() == 0
    monkeypatch.setenv("FISH_VOICE_DEBUG", "1")
    assert debug_level() == 1
    monkeypatch.setenv("FISH_VOICE_DEBUG", "trace")
    assert debug_level() == 2
    monkeypatch.setenv("FISH_VOICE_DEBUG", "1")
    configure_voice_logging(debug=1)
    trace("barge.mic heartbeat")
    debug("barge.arm ready")
    err = capsys.readouterr().err
    assert "heartbeat" not in err
    assert "arm ready" in err
    assert heartbeat_due(20, 20) is False
    configure_voice_logging(debug=2)
    trace("barge.mic heartbeat")
    assert "heartbeat" in capsys.readouterr().err
    assert heartbeat_due(20, 20) is True


def test_debug_lines_are_tagged_and_offset_from_the_turn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_voice_logging(debug=1)
    debug("tts.say {!r} ({} chars)", "Hello there.", 12)
    first = capsys.readouterr().err
    assert "tts" in first
    assert "say 'Hello there.' (12 chars)" in first
    mark_turn()
    capsys.readouterr()
    debug("asr.done status=200")
    later = capsys.readouterr().err
    assert "+0." in later
    assert "asr" in later


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
    hidden = header_meta({"x-api-key": "sk-secret", "x-request-id": "abc"})
    assert "x-api-key" not in hidden
    assert hidden["x-request-id"] == "abc"
    body = public_meta({"api_key": "sk-secret", "language": "en"})
    assert "api_key" not in body
    assert body["language"] == "en"


def test_debug_flag_does_not_write_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FISH_VOICE_DEBUG", raising=False)
    configure_voice_logging(debug=True)
    assert env_debug()
    assert "FISH_VOICE_DEBUG" not in os.environ
    configure_voice_logging(debug=False)
    assert not env_debug()


def test_conversation_is_plain_on_stdout_without_debug(
    capsys: pytest.CaptureFixture[str],
) -> None:
    conversation("you", "Hey there")
    conversation("llm", "[happy] Hello!")
    captured = capsys.readouterr()
    assert captured.out == "you \u25b8 Hey there\nllm \u25b8 [happy] Hello!\n"
    assert captured.err == ""


def test_conversation_joins_the_timed_log_with_debug(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_voice_logging(debug=1)
    mark_turn()
    capsys.readouterr()
    conversation("you", "Hey there")
    debug("llm.first_token 700ms")
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert "you \u25b8" in lines[0]
    assert "Hey there" in lines[0]
    assert "+0." in lines[0]
    assert lines[0][2] == ":"
    assert "first_token" in lines[1]


def test_short_model_drops_only_the_vendor_prefix() -> None:
    assert short_model("cognitivecomputations/dolphin-mistral-24b-venice-edition") == (
        "dolphin-mistral-24b-venice-edition"
    )
    assert short_model("openai/gpt-4o:nitro") == "gpt-4o:nitro"
    assert short_model("local-model") == "local-model"
    assert short_model(None) == ""
