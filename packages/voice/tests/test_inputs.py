"""``LiveInput``: typed lines and mute, over a fake mic."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable

import pytest
from session_fakes import (
    ScriptedAsr,
    ScriptedRecordings,
    ScriptedTurns,
    barge_clip,
    hello_tokens,
    make_ctx,
    quick_config,
    run_session,
    speak_with_fakes,
)
from voice_fakes import ScriptedMic, install_gate

from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.duplex_state import DuplexContext, TurnInput
from fish_audio_suite_voice.events import EVENTS, Event, Heard
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.speaker import FishSpeaker


class _Mic:
    """A listen that waits for its stop flag, or returns a scripted result."""

    def __init__(self, result: HeardLine | None = None) -> None:
        self.result = result
        self.opened = threading.Event()
        self.stopped = threading.Event()
        self.calls = 0

    async def hear(self, stop: StopFlag) -> HeardLine:
        self.calls += 1
        self.opened.set()
        if self.result is not None:
            return self.result
        while not stop.is_set():
            await asyncio.to_thread(time.sleep, 0.005)
        self.stopped.set()
        return HeardLine("noise")

    def then(self, *steps: Callable[[], object]) -> threading.Thread:
        """Once the recording is open, run each step in a thread, in order."""

        def run() -> None:
            self.opened.wait(5)
            for step in steps:
                step()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread


async def _no_listen(_stop: StopFlag) -> HeardLine:
    raise AssertionError("the mic was not expected to listen")


def _turn(source: LiveInput, ctx: DuplexContext, mic: _Mic | None = None) -> HeardLine:
    hear = mic.hear if mic is not None else _no_listen
    return asyncio.run(asyncio.wait_for(source.next_turn(ctx.turn_input, hear), 5))


def test_a_typed_line_is_answered_like_a_spoken_one() -> None:
    ctx = make_ctx()
    ctx.turn_input.carry.keep(barge_clip(), "")
    seen: list[Event] = []
    unsubscribe = EVENTS.subscribe(seen.append)
    source = LiveInput()
    source.submit("  hello there  ")
    try:
        heard = _turn(source, ctx)
    finally:
        unsubscribe()
    assert (heard.kind, heard.text) == ("line", "hello there")
    assert ctx.turn_input.carry.clip is None
    assert seen == [Heard("hello there", 0.0)]


def test_blank_lines_are_ignored_and_lines_keep_their_order() -> None:
    source = LiveInput()
    source.submit("   ")
    source.submit("one")
    source.submit("two")
    ctx = make_ctx()
    assert _turn(source, ctx).text == "one"
    assert _turn(source, ctx).text == "two"


def test_a_typed_line_stops_the_mic_and_drops_what_it_caught() -> None:
    mic = _Mic()
    source = LiveInput()
    mic.then(lambda: source.submit("typed instead"))
    heard = _turn(source, make_ctx(), mic)
    assert mic.stopped.is_set()
    assert (heard.kind, heard.text) == ("line", "typed instead")


def test_a_mic_result_is_returned_when_nothing_interrupts() -> None:
    spoken = HeardLine("line", text="from the mic")
    mic = _Mic(spoken)
    assert _turn(LiveInput(), make_ctx(), mic) is spoken
    assert mic.calls == 1


def test_a_muted_session_keeps_the_mic_closed_until_unmuted_or_typed() -> None:
    mic = _Mic(HeardLine("line", text="from the mic"))
    source = LiveInput()
    source.mute()
    assert source.muted
    timer = threading.Timer(0.15, source.unmute)
    timer.start()
    start = time.monotonic()
    try:
        heard = _turn(source, make_ctx(), mic)
    finally:
        timer.cancel()
    assert time.monotonic() - start >= 0.1
    assert heard.text == "from the mic"
    assert not source.muted
    muted = LiveInput()
    muted.mute()
    muted.submit("typed while muted")
    mic.calls = 0
    assert _turn(muted, make_ctx(), mic).text == "typed while muted"
    assert mic.calls == 0


def test_muting_during_a_recording_stops_it() -> None:
    mic = _Mic()
    source = LiveInput()
    stopped_by_mute: list[bool] = []
    mic.then(
        source.mute,
        lambda: stopped_by_mute.append(mic.stopped.wait(5)),
        lambda: source.submit("now typing"),
    )
    heard = _turn(source, make_ctx(), mic)
    assert stopped_by_mute == [True]
    assert heard.text == "now typing"


def test_quit_ends_a_muted_wait_and_a_recording() -> None:
    mic = _Mic()
    ctx = make_ctx()
    source = LiveInput()
    source.mute()
    timer = threading.Timer(0.05, ctx.turn_input.session.request_quit)
    timer.start()
    try:
        assert _turn(source, ctx, mic).kind == "bye"
    finally:
        timer.cancel()
    assert _turn(LiveInput(), ctx, mic).kind == "bye"  # quit already set


def test_the_loop_answers_a_typed_line_through_a_source(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts)
    source = LiveInput()
    source.submit("tell me something")
    # The second turn finds nothing typed, so the mic listens and hears goodbye.
    ScriptedRecordings(monkeypatch, b"wav")
    ScriptedAsr(monkeypatch, "goodbye").serve_sessions()
    assert run_session(quick_config(), tts, hello_tokens, source=source) == 0
    assert said == ["[calm] Hello there friend."]


def _bound_turn(ctx: DuplexContext) -> tuple[threading.Event, asyncio.Event]:
    cancel, llm_cancel = threading.Event(), asyncio.Event()
    ctx.turn_input.session.turn.bind(cancel, llm_cancel)
    return cancel, llm_cancel


def test_a_typed_line_stops_the_reply_that_is_playing() -> None:
    ctx = make_ctx()
    source = LiveInput()
    source.submit("first")
    _turn(source, ctx)  # the source now knows the session
    cancel, _ = _bound_turn(ctx)
    source.submit("over the reply")
    assert cancel.is_set()
    assert _turn(source, ctx).text == "over the reply"


def test_a_typed_line_can_wait_for_the_reply_to_finish() -> None:
    ctx = make_ctx()
    source = LiveInput()
    source.submit("first")
    _turn(source, ctx)
    cancel, _ = _bound_turn(ctx)
    source.submit("after the reply", interrupt=False)
    assert not cancel.is_set()
    assert _turn(source, ctx).text == "after the reply"


def test_a_line_typed_before_the_first_turn_has_nothing_to_stop() -> None:
    source = LiveInput()
    source.submit("early")  # no session seen yet: must not fail
    assert _turn(source, make_ctx()).text == "early"


def test_toggling_mute_flips_the_switch_and_reports_the_new_state() -> None:
    source = LiveInput()
    assert source.toggle_mute() is True
    assert source.muted
    assert source.toggle_mute() is False
    assert not source.muted


class _LockThatPausesEachThreadOnce:
    """A lock that, the first time each thread lets go of it, waits for one other thread.

    Two threads that each toggle once then interleave as tightly as the lock allows: a
    toggle that let go of the lock between reading and writing the switch loses one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._meet = threading.Barrier(2, timeout=2)
        self._seen: set[int] = set()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()
        me = threading.get_ident()
        if me not in self._seen:
            self._seen.add(me)
            with contextlib.suppress(threading.BrokenBarrierError):
                self._meet.wait()


def test_two_toggles_at_once_are_both_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    source = LiveInput()
    monkeypatch.setattr(source, "_lock", _LockThatPausesEachThreadOnce())
    threads = [threading.Thread(target=source.toggle_mute) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not source.muted  # muted, then unmuted


def test_stopping_the_reply_cancels_the_turn_and_does_nothing_before_a_session() -> None:
    LiveInput().stop_reply()  # no session seen yet: nothing to stop, and no error
    ctx = make_ctx()
    source = LiveInput()
    source.submit("first")
    _turn(source, ctx)
    cancel, llm_cancel = _bound_turn(ctx)
    source.stop_reply()
    assert cancel.is_set()
    assert asyncio.run(_llm_cancel_is_set(llm_cancel))


async def _llm_cancel_is_set(event: asyncio.Event) -> bool:
    await asyncio.sleep(0)  # the cancel crosses threads through the loop
    return event.is_set()


def test_the_context_takes_the_mute_switch_so_the_barge_in_watch_can_follow_it() -> None:
    assert not make_ctx().turn_input.muted()  # a context with no mute switch is never muted
    seen: list[bool] = []

    def flip_the_switch(turn: TurnInput) -> HeardLine:
        seen.append(turn.muted())
        source.muted = True
        seen.append(turn.muted())
        source.muted = False
        seen.append(turn.muted())
        return HeardLine("bye")

    source = ScriptedTurns(flip_the_switch)
    tts = FishSpeaker(api_key="k", voice_id="voice")
    assert run_session(quick_config(), tts, hello_tokens, source=source) == 0
    assert seen == [False, True, False]


def test_muting_closes_the_session_mic_at_once_and_while_muted() -> None:
    ctx = make_ctx()
    mic = ctx.turn_input.mic
    assert isinstance(mic, ScriptedMic)
    source = LiveInput()
    source.submit("hi")
    _turn(source, ctx)  # the source now knows the session's mic
    source.mute()
    assert mic.released == 1  # closed straight away, even with a reply still playing
    source.submit("typed while muted")
    assert _turn(source, ctx).text == "typed while muted"
    source.toggle_mute()  # unmuted
    assert mic.released == 1
    source.toggle_mute()  # muted again
    assert mic.released == 2


def test_a_session_muted_before_its_first_turn_keeps_the_mic_closed() -> None:
    ctx = make_ctx()
    mic = ctx.turn_input.mic
    assert isinstance(mic, ScriptedMic)
    source = LiveInput()
    source.mute()
    threading.Timer(0.05, lambda: source.submit("typed")).start()
    assert _turn(source, ctx).text == "typed"
    assert mic.released >= 1
