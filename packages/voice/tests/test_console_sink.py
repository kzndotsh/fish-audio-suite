"""The plain terminal display, driven by events, and the session it prints."""

from __future__ import annotations

import asyncio
import re

import pytest
from session_fakes import (
    FakeBackend,
    hello_tokens,
    quick_config,
    run_session_with,
    speak_with_fakes,
)
from voice_fakes import install_audio

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.console_sink import ConsoleSink
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.events import (
    EVENTS,
    Bye,
    Event,
    EventBus,
    Heard,
    Listening,
    Notice,
    ReplyEnd,
    ReplyToken,
    TurnEnded,
)
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.speaker import FishSpeaker


def _emit(bus: EventBus, *events: Event) -> None:
    for event in events:
        bus.emit(event)


def test_a_streamed_reply_prints_on_one_line_and_closes_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, ReplyToken("Hello "), ReplyToken("there."), ReplyEnd("Hello there."))
    assert capsys.readouterr().out == "llm ▸ Hello there.\n"


def test_an_empty_reply_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, ReplyEnd(""))
    assert capsys.readouterr().out == ""


def test_with_debug_on_the_reply_is_held_and_printed_whole(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.console_sink.debug_enabled", lambda: True)
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, Listening(), ReplyToken("Hello "), ReplyToken("there."), ReplyEnd("Hello there."))
    assert capsys.readouterr().out == "llm ▸ Hello there.\n"


def test_each_event_prints_its_line(capsys: pytest.CaptureFixture[str]) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(
        bus,
        Listening(),
        Heard("tell me a story", 120.0),
        Notice("[llm 429, retrying in 4s]"),
        TurnEnded(LatencySnapshot(asr_ms=120.0, first_audio_ms=900.0)),
        Bye(),
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["listening…", "you ▸ tell me a story", "  [llm 429, retrying in 4s]"]
    assert re.fullmatch(r"  ↳ first audio .*asr .*", lines[3])
    assert lines[4:] == ["", "bye"]


def test_with_debug_on_the_summary_goes_to_the_log_not_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logged: list[str] = []
    monkeypatch.setattr("fish_audio_suite_voice.console_sink.debug_enabled", lambda: True)
    monkeypatch.setattr(
        "fish_audio_suite_voice.console_sink.debug", lambda message, *_a: logged.append(message)
    )
    monkeypatch.setattr("fish_audio_suite_voice.console_sink.trace", lambda *_a: None)
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, Listening(), TurnEnded(LatencySnapshot(first_audio_ms=900.0)))
    assert capsys.readouterr().out == ""
    assert logged == ["turn.summary {}"]


def test_a_closed_sink_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    bus = EventBus()
    sink = ConsoleSink(bus)
    sink.close()
    _emit(bus, Listening(), Bye())
    assert capsys.readouterr().out == ""


class _OneTurn:
    """The real mic path for the first turn, then goodbye."""

    def __init__(self, *, real: bool) -> None:
        self._real = real
        self._turns = 0

    async def next_turn(self, ctx: DuplexContext, last_user: str) -> HeardLine:
        self._turns += 1
        if self._turns > 1:
            return HeardLine("bye")
        if self._real:
            return await hear_line(ctx, last_user)
        return HeardLine("line", text="hi")


def test_a_whole_turn_prints_the_same_transcript_as_before(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    speak_with_fakes(monkeypatch, tts)

    async def asr(*_args: object, **_kwargs: object) -> str:
        return "tell me something fun"

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", lambda *_a, **_k: b"w")
    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", asr)
    assert run_session_with(quick_config(), tts, hello_tokens, _OneTurn(real=True)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["listening…", "you ▸ tell me something fun", "llm ▸ Hello there friend."]
    assert out[3].startswith("  ↳ ")
    assert out[4:] == ["", "bye"]


def test_a_display_that_draws_its_own_screen_leaves_the_console_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    seen: list[Event] = []
    unsubscribe = EVENTS.subscribe(seen.append)
    try:
        code = asyncio.run(
            asyncio.wait_for(
                duplex_turns(
                    quick_config(),
                    tts,
                    None,
                    FakeBackend(hello_tokens),
                    source=_OneTurn(real=False),
                    console=False,
                ),
                5,
            )
        )
    finally:
        unsubscribe()
    assert code == 0
    assert capsys.readouterr().out == ""
    assert any(isinstance(e, Bye) for e in seen)
