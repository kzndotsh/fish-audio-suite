"""``hear_line`` with Deepgram selected: each way a streamed turn can end, and the way back."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
from session_fakes import FakeBackend, ScriptedAsr, ScriptedRecordings, barge_clip, make_ctx

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.events import EVENTS, Event, ExitCode, Heard
from fish_audio_suite_voice.heard import HeardLine
from fish_audio_suite_voice.hearing import hear_line
from fish_audio_suite_voice.recognised import NoTurn, StreamFallback, Transcript
from fish_audio_suite_voice.tune import DeepgramTune, SttProvider, SttTune


def _ctx(asr: ScriptedAsr | None = None, provider: SttProvider = "deepgram") -> DuplexContext:
    ctx = make_ctx(asr=asr)
    return replace(
        ctx,
        config=replace(ctx.config, stt=SttTune(provider=provider, deepgram=DeepgramTune(key="k"))),
    )


def _stream_returns(monkeypatch: pytest.MonkeyPatch, outcome: object) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> object:
        calls.append(kwargs)
        return outcome

    monkeypatch.setattr("fish_audio_suite_voice.hearing.stream_turn", fake)
    return calls


def _turn(text: str = "what is the weather") -> Transcript:
    return Transcript(text, asr_ms=240.0, started=123.0, trace_id="a" * 32)


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
    assert (
        calls[0]["settings"].stt.deepgram.key == "k"
    )  # the settings go down; nothing below reads the env


def test_the_default_provider_never_touches_deepgram(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stream_returns(monkeypatch, _turn())
    ScriptedRecordings(monkeypatch)  # hears nothing
    heard = asyncio.run(hear_line(_ctx(provider="fish"), ""))
    assert heard == HeardLine("noise")  # the batch path ran
    assert calls == []


def test_when_deepgram_cannot_be_reached_the_turn_is_heard_the_usual_way_from_the_speech_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = b"\x05\x00" * 200
    _stream_returns(monkeypatch, StreamFallback(((captured, True),), None))
    recordings = ScriptedRecordings(monkeypatch)
    assert asyncio.run(hear_line(_ctx(), "")) == HeardLine("noise")
    # The batch path carries on from what was already said.
    assert [call["carried"].pcm for call in recordings.calls] == [captured]


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (NoTurn("fatal"), HeardLine("fatal", code=ExitCode.FATAL)),
        (NoTurn("again"), HeardLine("again")),
        (NoTurn("noise"), HeardLine("noise")),
        (NoTurn("stopped"), HeardLine("noise")),  # a typed line or mute cut it short
    ],
)
def test_each_way_a_streamed_turn_can_fail_maps_to_what_the_loop_does_next(
    monkeypatch: pytest.MonkeyPatch, outcome: NoTurn, expected: HeardLine
) -> None:
    _stream_returns(monkeypatch, outcome)
    assert asyncio.run(hear_line(_ctx(), "")) == expected


def test_quitting_during_a_streamed_turn_says_bye(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx()
    _stream_returns(monkeypatch, NoTurn("stopped"))
    ctx.turn_input.session.quit_requested.set()
    assert asyncio.run(hear_line(ctx, "")) == HeardLine("bye")
    ctx2 = _ctx()
    _stream_returns(monkeypatch, _turn())
    ctx2.turn_input.session.quit_requested.set()  # quit arrived while the answer was coming back
    assert asyncio.run(hear_line(ctx2, "")) == HeardLine("bye")


def test_a_streamed_transcript_is_judged_like_a_batch_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _stream_returns(monkeypatch, _turn("goodbye"))
    assert asyncio.run(hear_line(_ctx(), "")) == HeardLine("bye")  # a quit word ends the session
    _stream_returns(monkeypatch, _turn("mm-hmm"))
    ctx = _ctx()
    ctx.turn_input.carry.keep(
        barge_clip(b"\x01\x00" * 100), ""
    )  # said over the reply: "mm-hmm" is "go on"
    assert asyncio.run(hear_line(ctx, "")) == HeardLine("noise")
    _stream_returns(monkeypatch, _turn("what is the weather"))
    assert asyncio.run(hear_line(_ctx(), "what is the weather")).kind == "noise"  # a stale copy


def test_the_barge_prefix_is_handed_to_the_stream_and_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stream_returns(monkeypatch, _turn())
    ctx = _ctx()
    clip = barge_clip(b"\x02\x00" * 50, since=42)
    ctx.turn_input.carry.keep(clip, "")
    asyncio.run(hear_line(ctx, ""))
    assert calls[0]["carried"] == clip
    assert ctx.turn_input.carry.clip is None


def _eager_ctx() -> DuplexContext:
    ctx = _ctx()
    ctx = replace(
        ctx, config=replace(ctx.config, stt=replace(ctx.config.stt, eager_eot_threshold=0.5))
    )
    ctx.chat.history.messages.append({"role": "user", "content": "earlier"})
    ctx.chat.history.messages.append({"role": "assistant", "content": "reply"})
    return ctx


def test_without_an_eager_threshold_no_reply_is_started_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stream_returns(monkeypatch, _turn())
    asyncio.run(hear_line(_ctx(), ""))
    assert calls[0]["on_eager"] is None
    assert calls[0]["on_resumed"] is None


def test_an_early_end_of_turn_starts_a_reply_that_the_speaker_resuming_throws_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _eager_ctx()
    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> object:
        kwargs["on_eager"]("what is the weather")
        seen["started"] = ctx.chat.speculation
        kwargs["on_resumed"]()
        seen["after_resume"] = ctx.chat.speculation
        kwargs["on_eager"]("what is the weather today")
        return _turn("what is the weather today")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.stream_turn", fake)
    heard = asyncio.run(hear_line(ctx, ""))
    assert heard.kind == "line"
    assert seen["started"] is not None
    assert seen["after_resume"] is None
    assert ctx.chat.speculation is not None  # the confirmed turn keeps the latest one for the reply
    assert ctx.chat.speculation.text == "what is the weather today"
    assert [m["content"] for m in ctx.chat.history.messages] == [
        "be brief",
        "earlier",
        "reply",
    ]  # not touched


def test_a_reply_started_early_is_dropped_when_the_turn_turns_out_to_be_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _eager_ctx()

    async def fake(**kwargs: Any) -> object:
        kwargs["on_eager"]("hmm")
        return NoTurn("noise")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.stream_turn", fake)
    assert asyncio.run(hear_line(ctx, "")) == HeardLine("noise")
    assert ctx.chat.speculation is None


def _fallback_then_asr(
    monkeypatch: pytest.MonkeyPatch, said: str
) -> tuple[ScriptedRecordings, DuplexContext]:
    fallback = StreamFallback(((b"\x05\x00" * 480, True),), since=77)
    _stream_returns(monkeypatch, fallback)
    recordings = ScriptedRecordings(monkeypatch, b"wav")
    ctx = _ctx(ScriptedAsr(monkeypatch, said))
    return recordings, ctx


def test_speech_handed_back_by_a_failed_stream_is_a_fresh_turn_not_an_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fallback used to ride on the barge-in prefix, so "yeah" was taken for "go on" and
    # skipped, and the batch listen read on from "now", losing the frames in between.
    recordings, ctx = _fallback_then_asr(monkeypatch, "Yeah.")
    heard = asyncio.run(hear_line(ctx, ""))
    assert (heard.kind, heard.text) == ("line", "Yeah.")
    assert (
        recordings.calls[0]["carried"].since == 77
    )  # the batch listen reads on from just after the speech


def test_speech_handed_back_by_a_failed_stream_can_still_be_a_stale_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _carried, ctx = _fallback_then_asr(monkeypatch, "What is the weather?")
    assert asyncio.run(hear_line(ctx, "what is the weather")) == HeardLine("noise")


def test_a_reply_started_early_never_edits_an_unanswered_line_in_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The last user line got no reply, so the new one is merged into it. The early reply's
    # request did that merge in the session's own message, and the confirmed line was
    # merged a second time on top.
    ctx = _eager_ctx()
    ctx.chat.history.messages.append({"role": "user", "content": "are you there"})
    asked: list[list[ChatMessage]] = []

    async def tokens(messages: list[ChatMessage], **_kwargs: object) -> AsyncIterator[str]:
        asked.append([message.copy() for message in messages])
        yield "Hi."

    ctx.chat.backend = FakeBackend(tokens)

    async def fake(**kwargs: Any) -> object:
        kwargs["on_eager"]("what is the weather")
        await asyncio.sleep(0.01)
        return _turn("what is the weather")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.stream_turn", fake)
    heard = asyncio.run(hear_line(ctx, ""))
    assert heard.kind == "line"
    assert ctx.chat.history.messages[-1] == {"role": "user", "content": "are you there"}
    assert asked[0][-1]["content"] == "are you there\nwhat is the weather"
