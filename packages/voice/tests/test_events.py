"""The event bus, and the events a session reports as it runs."""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import dataclasses
import threading
import time
import typing
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger
from session_fakes import (
    FakeBackend,
    ScriptedAsr,
    ScriptedRecordings,
    ScriptedTurns,
    heard_line,
    hello_tokens,
    make_ctx,
    quick_config,
    run_session,
    speak_with_fakes,
)
from voice_fakes import ScriptedMic, install_gate, install_vad, make_result, set_tts

from fish_audio_suite_voice import events as events_module
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.console_sink import ConsoleSink
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import TurnInput
from fish_audio_suite_voice.events import (
    DEFAULT_KEYS,
    EVENTS,
    BargedIn,
    Bye,
    Event,
    EventBus,
    EventQueue,
    ExitCode,
    Heard,
    Interim,
    Listening,
    LogLine,
    MicLevel,
    Notice,
    OutputLevel,
    ReplyEnd,
    ReplyToken,
    SessionAction,
    SessionState,
    Speaking,
    TurnEnded,
    available_actions,
    next_state,
    notice,
)
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.latency import LatencySnapshot
from fish_audio_suite_voice.listen import record_utterance
from fish_audio_suite_voice.log_setup import configure_voice_logging, forward_logs
from fish_audio_suite_voice.pcm import FRAME_BYTES, MIC_LEVEL_EVERY_FRAMES
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tune import BargeTune


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
    ScriptedRecordings(monkeypatch, b"w")
    ctx = make_ctx(asr=ScriptedAsr(monkeypatch, "tell me something fun"))
    heard = asyncio.run(hear_line(ctx, ""))
    assert heard.kind == "line"
    assert seen[0] == Listening()
    assert isinstance(seen[1], Heard)
    assert seen[1].text == "tell me something fun"
    assert seen[1].asr_ms >= 0


def test_a_turn_reports_the_reply_as_it_streams_then_the_timings_then_bye(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    source = ScriptedTurns(heard_line("hi there"))
    assert run_session(quick_config(), tts, hello_tokens, source=source) == 0
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
        (SessionState.SPEAKING, OutputLevel(1000.0), SessionState.SPEAKING),
    ],
)
def test_the_session_state_follows_the_events(
    state: SessionState, event: Event, expected: SessionState
) -> None:
    assert next_state(state, event) is expected


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
    assert queue.drain() == []


def test_a_queue_wakes_its_consumer_from_the_emitting_thread_until_closed() -> None:
    bus = EventBus()
    woken: list[str] = []
    queue = EventQueue(bus, wake=lambda: woken.append(threading.current_thread().name))
    emitter = threading.Thread(target=lambda: bus.emit(Listening()), name="emitter")
    emitter.start()
    emitter.join(timeout=5)
    assert woken == ["emitter"]
    assert queue.drain() == [Listening()]
    queue.close()
    bus.emit(Listening())
    assert queue.drain() == []
    assert woken == ["emitter"]


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
    def frames(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        for _ in range(10):
            yield _loud()

    install_vad(monkeypatch, _SpeechVad)
    gate = BargeGate(
        mic=ScriptedMic(frames),
        tune=BargeTune(hit_frames=MIC_LEVEL_EVERY_FRAMES + 2, min_rms=1.0, bleed_delay_s=0),
    )
    cancel = threading.Event()
    gate.watch(cancel)
    assert cancel.is_set()
    kinds = [type(e) for e in seen]
    assert kinds == [MicLevel, BargedIn]
    level = seen[0]
    assert isinstance(level, MicLevel)
    assert (level.source, level.rms > level.need) == ("barge", True)


def test_listening_reports_the_mic_level_every_few_frames(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    install_vad(monkeypatch, _SpeechVad)
    room = ScriptedMic([bytes(FRAME_BYTES)] * (MIC_LEVEL_EVERY_FRAMES * 3))
    clip = record_utterance(room, give_up_frames=MIC_LEVEL_EVERY_FRAMES * 2)
    assert clip is None
    levels = [e for e in seen if isinstance(e, MicLevel)]
    assert len(levels) == 2
    assert all(e.source == "listen" for e in levels)


def test_the_first_audio_of_a_reply_is_reported_as_speaking(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)

    def speak(
        text: str,
        sink: object,
        cancel: object = None,
        on_first_audio: Any = None,
        trace_headers: object = None,
    ) -> Any:
        on_first_audio()
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    set_tts(monkeypatch, tts, speak=speak)
    source = ScriptedTurns(heard_line("hi there"))
    assert run_session(quick_config(), tts, hello_tokens, source=source) == 0
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
    ("heard", "code"),
    [(HeardLine("fatal", code=ExitCode.FATAL), ExitCode.FATAL), (HeardLine("bye"), ExitCode.OK)],
)
def test_bye_is_sent_once_with_the_exit_code_however_the_session_ends(
    monkeypatch: pytest.MonkeyPatch,
    seen: list[Event],
    capsys: pytest.CaptureFixture[str],
    heard: HeardLine,
    code: ExitCode,
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    assert run_session(quick_config(), tts, hello_tokens, source=ScriptedTurns(heard)) == code
    assert [e for e in seen if isinstance(e, Bye)] == [Bye(code)]
    assert next(e for e in seen if isinstance(e, Bye)).code == code
    assert ("bye" in capsys.readouterr().out) is (code == 0)


def test_bye_is_sent_when_the_session_crashes(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event]
) -> None:
    def broken(_turn: TurnInput) -> HeardLine:
        raise RuntimeError("boom")

    tts = FishSpeaker(api_key="k", voice_id="voice")
    with pytest.raises(RuntimeError, match="boom"):
        run_session(quick_config(), tts, hello_tokens, source=ScriptedTurns(broken))
    assert [e for e in seen if isinstance(e, Bye)] == [Bye(ExitCode.CRASH)]


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


_ONE_OF_EACH: tuple[Event, ...] = (
    Listening(),
    Heard("hi", 1.0),
    Interim("hi"),
    ReplyToken("x"),
    ReplyEnd("x"),
    Speaking(),
    BargedIn(),
    MicLevel(1.0, 2.0, "listen"),
    OutputLevel(1.0),
    LogLine("DEBUG", "tts", "x"),
    Notice("x"),
    TurnEnded(LatencySnapshot()),
    Bye(),
)


def test_a_queue_that_falls_behind_drops_only_levels_interim_words_and_log_lines() -> None:
    assert {type(e) for e in _ONE_OF_EACH} == set(_event_classes()), "add the new event here"
    bus = EventBus()
    queue = EventQueue(bus, max_droppable=1)
    for event in _ONE_OF_EACH:
        for _ in range(3):
            bus.emit(event)
    kept = collections.Counter(type(e) for e in queue.drain())
    dropped = {type(e) for e in _ONE_OF_EACH if kept[type(e)] < 3}
    assert dropped == {MicLevel, OutputLevel, Interim, LogLine}


def test_an_event_cannot_be_changed_after_it_is_sent() -> None:
    event = Heard("hi", 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("state", "muted", "expected"),
    [
        (SessionState.IDLE, False, {"mute"}),
        (SessionState.LISTENING, False, {"mute"}),
        (SessionState.LISTENING, True, {"unmute"}),
        (SessionState.THINKING, False, {"mute", "interrupt"}),
        (SessionState.SPEAKING, True, {"unmute", "interrupt"}),
    ],
)
def test_only_the_actions_that_make_sense_now_are_offered(
    state: SessionState, muted: bool, expected: set[str]
) -> None:
    assert {str(a) for a in available_actions(state, muted=muted)} == expected


def test_a_cancelled_session_still_sends_bye_and_stops_the_console(
    monkeypatch: pytest.MonkeyPatch, seen: list[Event], capsys: pytest.CaptureFixture[str]
) -> None:
    """A UI runs the session as a worker, and a worker is cancelled when the app exits."""
    listening = asyncio.Event()

    async def wait_forever(_turn: TurnInput) -> HeardLine:
        listening.set()
        await asyncio.sleep(3600)
        return HeardLine("bye")

    async def printed_session() -> int:
        with ConsoleSink(EVENTS):  # as ``fish-voice`` runs it
            return await duplex_turns(
                quick_config(),
                FishSpeaker(api_key="k", voice_id="voice"),
                None,
                FakeBackend(hello_tokens),
                source=ScriptedTurns(wait_forever),
            )

    async def scenario() -> None:
        task = asyncio.ensure_future(printed_session())
        await listening.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert [e for e in seen if isinstance(e, Bye)] == [Bye(ExitCode.CRASH)]
    capsys.readouterr()  # drop what the session itself printed
    # The console subscriber was removed with the session, so later events print nothing.
    EVENTS.emit(Listening())
    EVENTS.emit(Heard("late", 1.0))
    assert capsys.readouterr().out == ""


_UNIVERSAL_KEYS = {"enter", "escape", "space", "tab", *(f"f{n}" for n in range(1, 11))}
_NOT_SAFE = {"ctrl+c", "ctrl+i", "ctrl+j", "ctrl+m", "ctrl+h", "ctrl+z", "ctrl+s", "ctrl+d"}


def test_every_action_has_a_default_key_a_terminal_really_sends() -> None:
    assert set(DEFAULT_KEYS) == set(SessionAction)
    for action, key in DEFAULT_KEYS.items():
        assert key == key.lower(), f"{action}: write keys in lower case"
        assert not any(w in key for w in ("cmd", "option", "alt", "super", "win", "meta")), (
            f"{action}: {key} is a modifier terminals often swallow"
        )
        assert key not in _NOT_SAFE, f"{action}: {key} is another key or a terminal shortcut"
        is_ctrl_letter = key.startswith("ctrl+") and len(key) == len("ctrl+x")
        assert key in _UNIVERSAL_KEYS or is_ctrl_letter, f"{action}: {key} may not reach the app"


def test_default_keys_do_not_clash_except_the_mute_toggle() -> None:
    assert DEFAULT_KEYS[SessionAction.MUTE] == DEFAULT_KEYS[SessionAction.UNMUTE]
    keys = [k for a, k in DEFAULT_KEYS.items() if a is not SessionAction.UNMUTE]
    assert len(keys) == len(set(keys))
    # A printable key would be typed into the input line, so none is a plain letter.
    assert all(len(k) > 1 for k in keys)


def test_a_queue_drops_old_output_levels_but_never_the_words() -> None:
    bus = EventBus()
    queue = EventQueue(bus, max_droppable=2)
    bus.emit(Heard("hi", 1.0))
    for n in range(6):
        bus.emit(OutputLevel(float(n)))
    bus.emit(ReplyToken("x"))
    events = queue.drain()
    assert [e.rms for e in events if isinstance(e, OutputLevel)] == [4.0, 5.0]
    assert Heard("hi", 1.0) in events
    assert ReplyToken("x") in events
    assert queue.dropped == 4
