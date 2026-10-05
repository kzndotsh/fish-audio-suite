"""``LiveInput``: typed lines and mute, over a fake mic."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from test_duplex import _ctx  # pyright: ignore[reportPrivateUsage]
from test_duplex_turns import (
    _hello,  # pyright: ignore[reportPrivateUsage]
    _quick,  # pyright: ignore[reportPrivateUsage]
    _run_with,  # pyright: ignore[reportPrivateUsage]
    _spoken_by,  # pyright: ignore[reportPrivateUsage]
)
from voice_fakes import install_audio

from fish_audio_suite_voice.barge import StopFlag
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.events import EVENTS, Event, Heard
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.speaker import FishSpeaker


class _Mic:
    """A fake ``hear_line`` that waits for its stop flag, or returns a scripted result."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, result: HeardLine | None = None) -> None:
        self.result = result
        self.opened = threading.Event()
        self.calls = 0
        monkeypatch.setattr("fish_audio_suite_voice.inputs.hear_line", self._hear)

    async def _hear(
        self, ctx: DuplexContext, last_user: str, *, stop: StopFlag | None = None
    ) -> HeardLine:
        del ctx, last_user
        self.calls += 1
        self.opened.set()
        if self.result is not None:
            return self.result
        assert stop is not None
        while not stop.is_set():
            await asyncio.to_thread(time.sleep, 0.005)
        return HeardLine("noise")


def _turn(source: LiveInput, ctx: DuplexContext) -> HeardLine:
    return asyncio.run(asyncio.wait_for(source.next_turn(ctx, ""), 5))


def test_a_typed_line_is_answered_like_a_spoken_one() -> None:
    ctx = _ctx()
    ctx.barge_prefix = b"clip"
    seen: list[Event] = []
    unsubscribe = EVENTS.subscribe(seen.append)
    source = LiveInput()
    source.submit("  hello there  ")
    try:
        heard = _turn(source, ctx)
    finally:
        unsubscribe()
    assert (heard.kind, heard.text) == ("line", "hello there")
    assert ctx.barge_prefix == b""
    assert seen == [Heard("hello there", 0.0)]


def test_blank_lines_are_ignored_and_lines_keep_their_order() -> None:
    source = LiveInput()
    source.submit("   ")
    source.submit("one")
    source.submit("two")
    ctx = _ctx()
    assert _turn(source, ctx).text == "one"
    assert _turn(source, ctx).text == "two"


def test_a_typed_line_stops_the_mic_and_drops_what_it_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic = _Mic(monkeypatch)
    source = LiveInput()
    timer = threading.Timer(0.05, source.submit, args=("typed instead",))
    timer.start()
    try:
        heard = _turn(source, _ctx())
    finally:
        timer.cancel()
    assert mic.opened.is_set()
    assert (heard.kind, heard.text) == ("line", "typed instead")


def test_a_mic_result_is_returned_when_nothing_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spoken = HeardLine("line", text="from the mic")
    mic = _Mic(monkeypatch, spoken)
    assert _turn(LiveInput(), _ctx()) is spoken
    assert mic.calls == 1


def test_a_muted_session_keeps_the_mic_closed_until_unmuted_or_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic = _Mic(monkeypatch, HeardLine("line", text="from the mic"))
    source = LiveInput()
    source.mute()
    assert source.muted
    timer = threading.Timer(0.15, source.unmute)
    timer.start()
    start = time.monotonic()
    try:
        heard = _turn(source, _ctx())
    finally:
        timer.cancel()
    assert time.monotonic() - start >= 0.1
    assert heard.text == "from the mic"
    assert not source.muted
    muted = LiveInput()
    muted.mute()
    muted.submit("typed while muted")
    mic.calls = 0
    assert _turn(muted, _ctx()).text == "typed while muted"
    assert mic.calls == 0


def test_muting_during_a_recording_stops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _Mic(monkeypatch)
    source = LiveInput()
    ctx = _ctx()
    timers = [
        threading.Timer(0.05, source.mute),
        threading.Timer(0.2, source.submit, args=("now typing",)),
    ]
    for timer in timers:
        timer.start()
    try:
        heard = _turn(source, ctx)
    finally:
        for timer in timers:
            timer.cancel()
    assert heard.text == "now typing"


def test_quit_ends_a_muted_wait_and_a_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    _Mic(monkeypatch)
    ctx = _ctx()
    source = LiveInput()
    source.mute()
    timer = threading.Timer(0.05, ctx.session.quit_requested.set)
    timer.start()
    try:
        assert _turn(source, ctx).kind == "bye"
    finally:
        timer.cancel()
    assert _turn(LiveInput(), ctx).kind == "bye"  # quit already set


def test_the_loop_answers_a_typed_line_through_a_source(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    said = _spoken_by(monkeypatch, tts)
    source = LiveInput()
    source.submit("tell me something")
    # The second turn finds nothing typed, so the fake mic says goodbye.
    _Mic(monkeypatch, HeardLine("bye"))
    assert _run_with(_quick(), tts, _hello, source) == 0
    assert said == ["Hello there friend."]


def _bound_turn(ctx: DuplexContext) -> tuple[threading.Event, asyncio.Event]:
    cancel, llm_cancel = threading.Event(), asyncio.Event()
    ctx.session.turn.bind(cancel, llm_cancel)
    return cancel, llm_cancel


def test_a_typed_line_stops_the_reply_that_is_playing() -> None:
    ctx = _ctx()
    source = LiveInput()
    source.submit("first")
    _turn(source, ctx)  # the source now knows the session
    cancel, _ = _bound_turn(ctx)
    source.submit("over the reply")
    assert cancel.is_set()
    assert _turn(source, ctx).text == "over the reply"


def test_a_typed_line_can_wait_for_the_reply_to_finish() -> None:
    ctx = _ctx()
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
    assert _turn(source, _ctx()).text == "early"


def test_toggling_mute_flips_the_switch_and_reports_the_new_state() -> None:
    source = LiveInput()
    assert source.toggle_mute() is True
    assert source.muted
    assert source.toggle_mute() is False
    assert not source.muted


def test_stopping_the_reply_cancels_the_turn_and_does_nothing_before_a_session() -> None:
    LiveInput().stop_reply()  # no session seen yet: nothing to stop, and no error
    ctx = _ctx()
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
