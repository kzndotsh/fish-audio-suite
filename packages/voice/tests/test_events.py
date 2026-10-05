"""The event bus, and the events a session reports as it runs."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest
from test_duplex import _ctx  # pyright: ignore[reportPrivateUsage]
from test_duplex_turns import (
    _hello,  # pyright: ignore[reportPrivateUsage]
    _line,  # pyright: ignore[reportPrivateUsage]
    _Loop,  # pyright: ignore[reportPrivateUsage]
    _quick,  # pyright: ignore[reportPrivateUsage]
    _run,  # pyright: ignore[reportPrivateUsage]
    _spoken_by,  # pyright: ignore[reportPrivateUsage]
)
from voice_fakes import install_audio

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
    notice,
)
from fish_audio_suite_voice.hearing import hear_line
from fish_audio_suite_voice.speaker import FishSpeaker


@pytest.fixture
def seen() -> Iterator[list[Event]]:
    events: list[Event] = []
    unsubscribe = EVENTS.subscribe(events.append)
    yield events
    unsubscribe()


def test_subscribers_get_every_event_in_order_until_they_unsubscribe() -> None:
    bus = EventBus()
    first: list[Event] = []
    second: list[Event] = []
    stop_first = bus.subscribe(first.append)
    bus.subscribe(second.append)
    bus.emit(Listening())
    bus.emit(ReplyToken("hi"))
    stop_first()
    stop_first()  # twice is harmless
    bus.emit(Bye())
    assert first == [Listening(), ReplyToken("hi")]
    assert second == [Listening(), ReplyToken("hi"), Bye()]


def test_a_failing_subscriber_does_not_stop_the_others_or_the_emitter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    got: list[Event] = []

    def broken(_event: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(broken)
    bus.subscribe(got.append)
    bus.emit(Listening())
    assert got == [Listening()]
    assert "a subscriber failed on Listening" in capsys.readouterr().err


def test_events_can_be_emitted_from_other_threads() -> None:
    bus = EventBus()
    got: list[Event] = []
    bus.subscribe(got.append)
    threads = [
        threading.Thread(target=lambda n=n: [bus.emit(ReplyToken(f"{n}")) for _ in range(50)])
        for n in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(got) == 200


def test_a_notice_prints_its_line_and_reports_it_trimmed(
    seen: list[Event], capsys: pytest.CaptureFixture[str]
) -> None:
    notice("  [llm cut off, continuing]")
    assert capsys.readouterr().out == "  [llm cut off, continuing]\n"
    assert seen == [Notice("[llm cut off, continuing]")]


def test_hear_line_reports_listening_and_the_transcript(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    async def asr(*_args: object, **_kwargs: object) -> str:
        return "tell me something fun"

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", lambda *_a, **_k: b"w")
    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", asr)
    heard = asyncio.run(hear_line(_ctx(), ""))
    assert heard.kind == "line"
    assert seen[0] == Listening()
    assert isinstance(seen[1], Heard)
    assert seen[1].text == "tell me something fun"
    assert seen[1].asr_ms >= 0


def test_a_turn_reports_the_reply_as_it_streams_then_the_timings_then_bye(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    _spoken_by(monkeypatch, tts)
    _Loop(monkeypatch, _line("hi there"))
    assert _run(_quick(), tts, _hello) == 0
    assert [type(e) for e in seen] == [ReplyToken, ReplyToken, ReplyEnd, TurnEnded, Bye]
    assert [e.text for e in seen if isinstance(e, ReplyToken)] == ["Hello ", "there friend."]
    assert ReplyEnd("Hello there friend.") in seen
    ended = next(e for e in seen if isinstance(e, TurnEnded))
    assert ended.snapshot.tts_first_audio_ms is not None
