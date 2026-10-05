"""``hear_line`` with Deepgram selected: each way a streamed turn can end, and the way back."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from session_fakes import make_ctx

from fish_audio_suite_voice.duplex_state import EXIT_FATAL, DuplexContext
from fish_audio_suite_voice.events import EVENTS, Event, Heard
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.streaming import StreamedTurn, StreamFallback
from fish_audio_suite_voice.tune import SttTune


def _ctx(provider: str = "deepgram") -> DuplexContext:
    ctx = make_ctx()
    ctx.config = replace(ctx.config, stt=SttTune(provider=provider, deepgram_key="k"))
    return ctx


def _stream_returns(monkeypatch: pytest.MonkeyPatch, outcome: object) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> object:
        calls.append(kwargs)
        return outcome

    monkeypatch.setattr("fish_audio_suite_voice.hearing.stream_turn", fake)
    return calls


def _turn(text: str = "what is the weather") -> StreamedTurn:
    return StreamedTurn(text=text, asr_ms=240.0, started=123.0, trace_id="a" * 32)


def test_a_streamed_turn_becomes_a_line_and_is_announced(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stream_returns(monkeypatch, _turn())
    seen: list[Event] = []
    stop = EVENTS.subscribe(seen.append)
    try:
        heard = asyncio.run(hear_line(_ctx(), ""))
    finally:
        stop()
    assert heard == HeardLine(
        "line", text="what is the weather", asr_ms=240.0, started=123.0, trace_id="a" * 32
    )
    assert Heard("what is the weather", 240.0) in seen
    assert calls[0]["stt"].deepgram_key == "k"  # the settings go down; nothing below reads the env


def test_the_default_provider_never_touches_deepgram(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stream_returns(monkeypatch, _turn())
    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", lambda *_a, **_k: b"")
    assert asyncio.run(hear_line(_ctx("fish"), "")) == HeardLine("noise")  # the batch path ran
    assert calls == []


def test_when_deepgram_cannot_be_reached_the_turn_is_heard_the_usual_way_from_the_speech_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = b"\x05\x00" * 200
    _stream_returns(monkeypatch, StreamFallback(captured))
    recorded: list[bytes] = []

    def record(*_args: Any, prefix: bytes = b"", **_kwargs: Any) -> bytes:
        recorded.append(prefix)
        return b""

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", record)
    assert asyncio.run(hear_line(_ctx(), "")) == HeardLine("noise")
    assert recorded == [captured]  # the batch path carries on from what was already said


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("fatal", HeardLine("fatal", code=EXIT_FATAL)),
        ("again", HeardLine("again")),
        ("noise", HeardLine("noise")),
        ("stopped", HeardLine("noise")),  # a typed line or mute cut it short
    ],
)
def test_each_way_a_streamed_turn_can_fail_maps_to_what_the_loop_does_next(
    monkeypatch: pytest.MonkeyPatch, outcome: str, expected: HeardLine
) -> None:
    _stream_returns(monkeypatch, outcome)
    assert asyncio.run(hear_line(_ctx(), "")) == expected


def test_quitting_during_a_streamed_turn_says_bye(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx()
    _stream_returns(monkeypatch, "stopped")
    ctx.session.quit_requested.set()
    assert asyncio.run(hear_line(ctx, "")) == HeardLine("bye")
    ctx2 = _ctx()
    _stream_returns(monkeypatch, _turn())
    ctx2.session.quit_requested.set()  # quit arrived while the answer was coming back
    assert asyncio.run(hear_line(ctx2, "")) == HeardLine("bye")


def test_a_streamed_transcript_is_judged_like_a_batch_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _stream_returns(monkeypatch, _turn("goodbye"))
    assert asyncio.run(hear_line(_ctx(), "")) == HeardLine("bye")  # a quit word ends the session
    _stream_returns(monkeypatch, _turn("mm-hmm"))
    ctx = _ctx()
    ctx.barge_prefix = b"\x01\x00" * 100  # said over the reply, so a lone "mm-hmm" means "go on"
    assert asyncio.run(hear_line(ctx, "")) == HeardLine("noise")
    _stream_returns(monkeypatch, _turn("what is the weather"))
    assert asyncio.run(hear_line(_ctx(), "what is the weather")).kind == "noise"  # a stale copy


def test_the_barge_prefix_is_handed_to_the_stream_and_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stream_returns(monkeypatch, _turn())
    ctx = _ctx()
    ctx.barge_prefix = b"\x02\x00" * 50
    asyncio.run(hear_line(ctx, ""))
    assert calls[0]["prefix"] == b"\x02\x00" * 50
    assert ctx.barge_prefix == b""
