"""The plain terminal display, driven by events, and the session it prints."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator

import pytest
from loguru import logger
from session_fakes import (
    MIC,
    FakeBackend,
    ScriptedAsr,
    ScriptedRecordings,
    ScriptedTurns,
    hello_tokens,
    quick_config,
    run_session,
    speak_with_fakes,
)
from voice_fakes import install_gate

from fish_audio_suite_voice.console_sink import ConsoleSink
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.events import (
    EVENTS,
    Bye,
    Event,
    EventBus,
    ExitCode,
    Heard,
    Listening,
    Notice,
    ReplyEnd,
    ReplyToken,
    TurnEnded,
)
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.latency import LatencySnapshot
from fish_audio_suite_voice.speaker import FishSpeaker


def _emit(bus: EventBus, *events: Event) -> None:
    for event in events:
        bus.emit(event)


@pytest.fixture
def debug_log(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Turn debug on, as ``FISH_VOICE_DEBUG=1`` does, and collect the log lines."""
    monkeypatch.setenv("FISH_VOICE_DEBUG", "1")
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(message.record["message"]), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(sink)


def test_a_streamed_reply_prints_on_one_line_and_closes_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, ReplyToken("Hello "), ReplyToken("there."), ReplyEnd("Hello there."))
    assert capsys.readouterr().out == "llm ▸ Hello there.\n"


def test_a_warning_mid_reply_closes_the_line_and_the_reply_goes_on_under_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    with ConsoleSink(bus):
        _emit(bus, ReplyToken("Hello "))
        warn("[tts] slow")
        _emit(bus, ReplyToken("there."), ReplyEnd("Hello there."))
    # The warning goes to stderr (or the log), between the two halves of the reply.
    assert capsys.readouterr().out == "llm ▸ Hello \nthere.\n"


def test_an_empty_reply_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, ReplyEnd(""))
    assert capsys.readouterr().out == ""


def test_with_debug_on_the_reply_is_held_and_printed_whole(
    capsys: pytest.CaptureFixture[str], debug_log: list[str]
) -> None:
    del debug_log
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
    assert lines[4] == "session 1 turns · first audio p50 0.90s p90 0.90s"
    assert lines[5:] == ["", "bye"]


def test_with_debug_on_the_summary_goes_to_the_log_not_stdout(
    capsys: pytest.CaptureFixture[str], debug_log: list[str]
) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, Listening(), TurnEnded(LatencySnapshot(first_audio_ms=900.0)))
    assert capsys.readouterr().out == ""
    (line,) = debug_log
    assert line.startswith("turn.summary ")
    assert "0.90s" in line


def test_a_closed_sink_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    bus = EventBus()
    sink = ConsoleSink(bus)
    sink.close()
    _emit(bus, Listening(), Bye())
    assert capsys.readouterr().out == ""


def test_a_whole_turn_prints_the_same_transcript_as_before(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)

    ScriptedRecordings(monkeypatch, b"w")
    ScriptedAsr(monkeypatch, "tell me something fun").serve_sessions()
    assert run_session(quick_config(), tts, hello_tokens, source=ScriptedTurns(MIC)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["listening…", "you ▸ tell me something fun", "llm ▸ Hello there friend."]
    assert out[3].startswith("  ↳ ")
    assert out[4].startswith("session 1 turns · ")
    assert out[5:] == ["", "bye"]


def test_a_display_that_draws_its_own_screen_leaves_the_console_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
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
                    source=ScriptedTurns(HeardLine("line", text="hi")),
                ),
                5,
            )
        )
    finally:
        unsubscribe()
    assert code == 0
    assert capsys.readouterr().out == ""
    assert any(isinstance(e, Bye) for e in seen)


def _turn(since_stop: float, first_token: float) -> TurnEnded:
    return TurnEnded(
        LatencySnapshot(
            llm_first_token_ms=first_token, first_audio_ms=since_stop - 400, end_wait_ms=400
        )
    )


def test_the_session_ends_with_the_typical_and_the_slow_wait(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    waits = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 5000]
    _emit(bus, *(_turn(wait, wait / 2) for wait in waits), Bye(ExitCode.OK))
    out = capsys.readouterr().out
    summary = next(line for line in out.splitlines() if line.startswith("session"))
    # The middle of ten turns, and the slow one in ten (the 9th of 10 sorted).
    assert summary == (
        "session 10 turns · since you stopped p50 1.80s p90 2.60s"
        " · first audio p50 1.40s p90 2.20s · llm first token p50 0.90s p90 1.30s"
    )
    assert out.rstrip().endswith("bye")


def test_a_session_with_no_turns_prints_no_summary(capsys: pytest.CaptureFixture[str]) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, Bye(ExitCode.OK))
    assert "session" not in capsys.readouterr().out


def test_the_session_summary_goes_to_the_log_with_debug_on(
    capsys: pytest.CaptureFixture[str], debug_log: list[str]
) -> None:
    bus = EventBus()
    ConsoleSink(bus)
    _emit(bus, _turn(1000, 500), Bye(ExitCode.FATAL))
    assert any(line.startswith("session.summary 1 turns") for line in debug_log)
    assert "session" not in capsys.readouterr().out
