"""The event bus, and the events a session reports as it runs."""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import threading
import time
import typing
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger
from test_duplex import _ctx  # pyright: ignore[reportPrivateUsage]
from test_duplex_turns import (
    _hello,  # pyright: ignore[reportPrivateUsage]
    _line,  # pyright: ignore[reportPrivateUsage]
    _Loop,  # pyright: ignore[reportPrivateUsage]
    _quick,  # pyright: ignore[reportPrivateUsage]
    _run,  # pyright: ignore[reportPrivateUsage]
    _spoken_by,  # pyright: ignore[reportPrivateUsage]
)
from voice_fakes import install_audio, install_vad, make_result, set_tts

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice import events as events_module
from fish_audio_suite_voice.barge import FRAME_BYTES, MIC_LEVEL_EVERY_FRAMES, BargeGate
from fish_audio_suite_voice.cli import _quit_line  # pyright: ignore[reportPrivateUsage]
from fish_audio_suite_voice.debug import configure_voice_logging
from fish_audio_suite_voice.events import (
    EVENTS,
    BargedIn,
    Bye,
    Event,
    EventBus,
    EventQueue,
    Heard,
    Listening,
    LogLine,
    MicLevel,
    Notice,
    ReplyEnd,
    ReplyToken,
    SessionState,
    Speaking,
    StateChanged,
    StateTracker,
    TurnEnded,
    available_actions,
    forward_logs,
    next_state,
    notice,
)
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.listen import _Listen  # pyright: ignore[reportPrivateUsage]
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tune import ListenTune


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


def test_a_notice_is_reported_trimmed_and_printed_by_nothing_itself(
    seen: list[Event], capsys: pytest.CaptureFixture[str]
) -> None:
    notice("  [llm cut off, continuing]")
    assert capsys.readouterr().out == ""
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


def test_events_carry_a_timestamp_that_is_not_part_of_equality() -> None:
    before = time.monotonic()
    first = Heard("hi", 3.0)
    time.sleep(0.01)
    second = Heard("hi", 3.0)
    assert before <= first.at < second.at
    assert first == second


@pytest.mark.parametrize(
    ("state", "event", "expected"),
    [
        (SessionState.IDLE, Listening(), SessionState.LISTENING),
        (SessionState.LISTENING, Heard("hi", 1.0), SessionState.THINKING),
        (SessionState.THINKING, Speaking(), SessionState.SPEAKING),
        (SessionState.SPEAKING, BargedIn(), SessionState.LISTENING),
        (SessionState.SPEAKING, TurnEnded(LatencySnapshot()), SessionState.IDLE),
        (SessionState.LISTENING, Bye(), SessionState.IDLE),
        (SessionState.THINKING, ReplyToken("x"), SessionState.THINKING),
        (SessionState.LISTENING, MicLevel(1.0, 2.0, "listen"), SessionState.LISTENING),
    ],
)
def test_the_session_state_follows_the_events(
    state: SessionState, event: Event, expected: SessionState
) -> None:
    assert next_state(state, event) is expected


def test_a_state_tracker_announces_each_change_once() -> None:
    bus = EventBus()
    got: list[Event] = []
    tracker = StateTracker(bus)
    bus.subscribe(got.append)
    for event in (
        Listening(),
        MicLevel(1.0, 2.0, "listen"),
        Listening(),
        Heard("hi", 1.0),
        Speaking(),
    ):
        bus.emit(event)
    changes = [e.state for e in got if isinstance(e, StateChanged)]
    assert changes == [SessionState.LISTENING, SessionState.THINKING, SessionState.SPEAKING]
    assert tracker.state is SessionState.SPEAKING
    tracker.close()
    bus.emit(TurnEnded(LatencySnapshot()))
    assert tracker.state is SessionState.SPEAKING


def test_a_queue_keeps_every_conversation_event_and_drops_old_mic_levels() -> None:
    bus = EventBus()
    queue = EventQueue(bus, max_droppable=3)
    bus.emit(Heard("hi", 1.0))
    for n in range(10):
        bus.emit(MicLevel(float(n), 1.0, "listen"))
        bus.emit(ReplyToken(str(n)))
    events = queue.drain()
    levels = [e.rms for e in events if isinstance(e, MicLevel)]
    assert levels == [7.0, 8.0, 9.0]
    assert [e.text for e in events if isinstance(e, ReplyToken)] == [str(n) for n in range(10)]
    assert Heard("hi", 1.0) in events
    assert queue.dropped == 7
    assert queue.get(timeout=0) is None


def test_a_queue_hands_events_to_a_waiting_thread_and_wakes_its_loop() -> None:
    bus = EventBus()
    woken: list[int] = []
    queue = EventQueue(bus, wake=lambda: woken.append(1))
    got: list[Event | None] = []
    reader = threading.Thread(target=lambda: got.append(queue.get(timeout=5)))
    reader.start()
    bus.emit(Listening())
    reader.join(timeout=5)
    assert got == [Listening()]
    assert woken == [1]
    queue.close()
    assert queue.get(timeout=5) is None  # closed, so it does not wait
    bus.emit(Listening())
    assert queue.drain() == []


def test_closing_a_queue_wakes_a_blocked_reader() -> None:
    queue = EventQueue(EventBus())
    got: list[Event | None] = []
    reader = threading.Thread(target=lambda: got.append(queue.get()))
    reader.start()
    time.sleep(0.05)
    queue.close()
    reader.join(timeout=5)
    assert got == [None]


def test_forwarded_logs_become_events_and_stay_off_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    got: list[Event] = []
    bus.subscribe(got.append)
    configure_voice_logging(debug=1, to_stderr=False)
    stop = forward_logs(bus)
    try:
        logger.debug("tts.start voice=abc")
        logger.warning("something odd")
    finally:
        stop()
        stop()  # twice is harmless
        configure_voice_logging(debug=False)
    assert got == [
        LogLine("DEBUG", "tts", "start voice=abc"),
        LogLine("WARNING", "warn", "something odd"),
    ]
    assert capsys.readouterr().err == ""


def test_a_subscriber_that_fails_on_log_lines_cannot_loop_forever(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    calls: list[Event] = []

    def broken(event: Event) -> None:
        calls.append(event)
        raise RuntimeError("boom")

    bus.subscribe(broken)
    configure_voice_logging(debug=False, to_stderr=False)
    stop = forward_logs(bus)
    try:
        bus.emit(Listening())
    finally:
        stop()
        configure_voice_logging(debug=False)
    # The failure is logged once; that log line reaches the subscriber, which
    # fails again but is not reported a second time.
    assert [type(e).__name__ for e in calls] == ["Listening", "LogLine"]


def _loud(level: int = 500) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


class _SpeechVad:
    def __init__(self, mode: int = 0) -> None:
        self.mode = mode

    def is_speech(self, frame: bytes, rate: int) -> bool:
        del frame, rate
        return True


def test_the_barge_gate_reports_the_mic_level_and_the_interruption(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    def frames(
        device: object,
        stop: threading.Event | None,
        *,
        timeout: float,
        aec: object = None,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout, aec
        for _ in range(10):
            yield _loud()

    install_vad(monkeypatch, _SpeechVad)
    monkeypatch.setattr("fish_audio_suite_voice.barge.mic_frames", frames)
    gate = BargeGate(hit_frames=MIC_LEVEL_EVERY_FRAMES + 2, min_rms=1.0, bleed_delay_s=0)
    cancel = threading.Event()
    gate.watch(cancel)
    assert cancel.is_set()
    kinds = [type(e) for e in seen]
    assert kinds == [MicLevel, BargedIn]
    level = seen[0]
    assert isinstance(level, MicLevel)
    assert (level.source, level.rms > level.need) == ("barge", True)


def test_listening_reports_the_mic_level_every_few_frames(seen: list[Event]) -> None:
    heard = _Listen(ListenTune(), _SpeechVad())
    for idle in range(1, MIC_LEVEL_EVERY_FRAMES * 2 + 1):
        heard.take(_loud(100), idle)
    levels = [e for e in seen if isinstance(e, MicLevel)]
    assert len(levels) == 2
    assert all(e.source == "listen" for e in levels)


def test_the_first_audio_of_a_reply_is_reported_as_speaking(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)

    def speak(text: str, sink: object, cancel: object = None, on_first_audio: Any = None) -> Any:
        on_first_audio()
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    set_tts(monkeypatch, tts, speak=speak)
    _Loop(monkeypatch, _line("hi there"))
    assert _run(_quick(), tts, _hello) == 0
    kinds = [type(e) for e in seen]
    assert kinds.index(Speaking) > kinds.index(ReplyEnd)
    assert kinds.index(Speaking) < kinds.index(TurnEnded)


def test_the_same_callback_can_subscribe_twice_and_each_unsubscribes_alone() -> None:
    bus = EventBus()
    got: list[Event] = []
    first = bus.subscribe(got.append)
    bus.subscribe(got.append)
    bus.emit(Listening())
    assert len(got) == 2
    first()
    first()  # the second call must not remove the other subscription
    bus.emit(Listening())
    assert len(got) == 3


@pytest.mark.parametrize(
    ("heard", "code"), [(HeardLine("fatal", code=2), 2), (HeardLine("bye"), 0)]
)
def test_bye_is_sent_once_with_the_exit_code_however_the_session_ends(
    monkeypatch: pytest.MonkeyPatch,
    seen: list[Event],
    capsys: pytest.CaptureFixture[str],
    heard: HeardLine,
    code: int,
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    _Loop(monkeypatch, heard)
    assert _run(_quick(), tts, _hello) == code
    assert [e for e in seen if isinstance(e, Bye)] == [Bye(code)]
    assert next(e for e in seen if isinstance(e, Bye)).code == code
    assert ("bye" in capsys.readouterr().out) is (code == 0)


def test_bye_is_sent_when_the_session_crashes(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    async def broken(_ctx: object, _last: str) -> HeardLine:
        raise RuntimeError("boom")

    monkeypatch.setattr("fish_audio_suite_voice.duplex.hear_line", broken)
    with pytest.raises(RuntimeError, match="boom"):
        _run(_quick(), FishSpeaker(api_key="k", voice_id="voice"), _hello)
    assert [e for e in seen if isinstance(e, Bye)] == [Bye(1)]


def test_a_second_ctrl_c_still_prints_the_quit_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert _quit_line() == 0
    assert capsys.readouterr().out == "\nbye\n"


def _event_classes() -> tuple[type, ...]:
    return typing.get_args(Event.__value__)


def test_every_event_is_an_immutable_timestamped_record_that_is_exported() -> None:
    """A display receives events on another thread, so each must be a frozen record."""
    classes = _event_classes()
    assert len(classes) == len({c.__name__ for c in classes}) > 0
    for cls in classes:
        name = cls.__name__
        assert dataclasses.is_dataclass(cls), f"{name} must be a dataclass"
        params = getattr(cls, "__dataclass_params__")  # noqa: B009 - not in the stubs
        assert params.frozen, f"{name} must be frozen"
        fields = {f.name: f for f in dataclasses.fields(cls)}
        assert "at" in fields, f"{name} needs a timestamp"
        assert fields["at"].compare is False, f"{name}.at must not affect equality"
        assert name in events_module.__all__, f"export {name} from events"


def test_the_events_that_may_be_dropped_are_real_event_names() -> None:
    names = {c.__name__ for c in _event_classes()}
    dropped = set(events_module._DROPPABLE)  # pyright: ignore[reportPrivateUsage]
    assert dropped <= names, f"unknown event names in _DROPPABLE: {dropped - names}"
    assert "ReplyToken" not in dropped
    assert "Heard" not in dropped


def test_an_event_cannot_be_changed_after_it_is_sent() -> None:
    event = Heard("hi", 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("state", "muted", "expected"),
    [
        (SessionState.IDLE, False, {"send", "quit", "mute"}),
        (SessionState.LISTENING, False, {"send", "quit", "mute"}),
        (SessionState.LISTENING, True, {"send", "quit", "unmute"}),
        (SessionState.THINKING, False, {"send", "quit", "mute", "interrupt"}),
        (SessionState.SPEAKING, True, {"send", "quit", "unmute", "interrupt"}),
    ],
)
def test_only_the_actions_that_make_sense_now_are_offered(
    state: SessionState, muted: bool, expected: set[str]
) -> None:
    assert {str(a) for a in available_actions(state, muted=muted)} == expected
