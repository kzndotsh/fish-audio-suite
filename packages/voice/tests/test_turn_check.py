"""The end-of-turn check on the Fish path, driven with a scripted mic and recogniser."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from session_fakes import (
    WAIT_FOR_STOP,
    AsrReply,
    Recording,
    ScriptedAsr,
    ScriptedRecordings,
    barge_clip,
    make_ctx,
)

from fish_audio_suite_voice.asr.outcome import HeardLine
from fish_audio_suite_voice.asr.pipeline import hear_line
from fish_audio_suite_voice.asr.turn_check import (
    OpenTurn,
    continuation_tune,
    first_clip_tune,
    wait_frames,
)
from fish_audio_suite_voice.capture.pcm import FRAME_MS
from fish_audio_suite_voice.config.tune import ListenTune, TurnTune
from fish_audio_suite_voice.session.events import EVENTS, Event, Heard
from fish_audio_suite_voice.session.state import DuplexContext


def _mic(monkeypatch: pytest.MonkeyPatch, *continuations: Recording) -> ScriptedRecordings:
    """Record ``clip1`` first, then each continuation, then keep listening until stopped.

    A continuation is the WAV it returns, None for a recording that heard nothing, or
    ``WAIT_FOR_STOP`` for a mic that keeps listening until the caller sets its stop flag.
    """
    return ScriptedRecordings(monkeypatch, b"clip1", *continuations, then=WAIT_FOR_STOP)


def _asr(
    monkeypatch: pytest.MonkeyPatch, said: dict[bytes, AsrReply | Callable[[], AsrReply]]
) -> ScriptedAsr:
    # The mic keeps listening while each clip is transcribed.
    return ScriptedAsr(monkeypatch, by_clip=said, delay_s=0.03)


def _ctx(asr: ScriptedAsr, **turn: Any) -> DuplexContext:
    ctx = make_ctx(asr=asr)
    return replace(ctx, config=replace(ctx.config, turn=TurnTune(enabled=True, **turn)))


def _hear(
    ctx: DuplexContext, *, last_user: str = "", stop: Any = None
) -> tuple[HeardLine, list[Event]]:
    seen: list[Event] = []
    unsubscribe = EVENTS.subscribe(seen.append)
    try:
        heard = asyncio.run(asyncio.wait_for(hear_line(ctx, last_user, stop=stop), 5))
    finally:
        unsubscribe()
    return heard, seen


def _said(seen: list[Event]) -> list[str]:
    return [event.text for event in seen if isinstance(event, Heard)]


def test_a_finished_clip_is_answered_on_the_short_silence_and_the_mic_is_closed_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "What time is it?"})
    mic = _mic(monkeypatch)
    heard, seen = _hear(_ctx(asr))
    assert (heard.kind, heard.text) == ("line", "What time is it?")
    assert asr.clips == [b"clip1"]
    assert _said(seen) == ["What time is it?"]
    first, reopened = mic.calls
    assert first["tune"].end_silence_frames == 12
    assert reopened["tune"].end_silence_frames == 40
    assert reopened["give_up_frames"] == 28
    assert reopened["stop"].is_set()  # nothing is left listening when the reply starts


def test_the_silence_waited_out_is_counted_from_the_clip_that_ended_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_ms = 12 * FRAME_MS
    _mic(monkeypatch)
    heard, _ = _hear(_ctx(_asr(monkeypatch, {b"clip1": "What time is it?"})))
    assert heard.end_wait_ms == first_ms
    asr = _asr(
        monkeypatch, {b"clip1": "I was going to ask, but", b"clip2": "never mind, it is fine."}
    )
    _mic(monkeypatch, b"clip2")
    joined, _ = _hear(_ctx(asr))
    assert joined.end_wait_ms == 40 * FRAME_MS  # the continuation's longer silence


def test_the_mic_is_already_listening_for_more_while_the_first_clip_is_transcribed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic = _mic(monkeypatch)
    started_during_asr: list[bool] = []

    def heard() -> AsrReply:
        started_during_asr.append(len(mic.calls) == 2)
        return "What time is it?"

    _hear(_ctx(ScriptedAsr(monkeypatch, heard, delay_s=0.05)))
    assert started_during_asr == [True]


def test_an_unfinished_clip_is_joined_with_its_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(
        monkeypatch, {b"clip1": "I was going to ask, but", b"clip2": "never mind, it is fine."}
    )
    _mic(monkeypatch, b"clip2")
    heard, seen = _hear(_ctx(asr))
    assert heard.kind == "line"
    assert heard.text == "I was going to ask, but never mind, it is fine."
    assert asr.clips == [b"clip1", b"clip2"]
    assert _said(seen) == [heard.text]  # announced once, as the whole line


def test_the_first_words_are_kept_when_nothing_more_is_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "Tell me about the"})
    mic = _mic(monkeypatch, None)
    heard, _ = _hear(_ctx(asr))
    assert (heard.kind, heard.text) == ("line", "Tell me about the")
    assert len(mic.calls) == 2


def test_a_turn_is_extended_no_more_than_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    said: dict[bytes, AsrReply | Callable[[], AsrReply]] = {
        b"clip1": "First, and",
        b"clip2": "second, and",
        b"clip3": "third, and",
    }
    asr = _asr(monkeypatch, said)
    one = _mic(monkeypatch, b"clip2", b"clip3")
    heard, _ = _hear(_ctx(asr, max_extensions=1))
    assert heard.text == "First, and second, and"
    assert asr.clips == [b"clip1", b"clip2"]
    assert len(one.calls) == 2  # no mic is opened for a third clip
    two = _mic(monkeypatch, b"clip2", b"clip3")
    heard, _ = _hear(_ctx(_asr(monkeypatch, said), max_extensions=2))
    assert heard.text == "First, and second, and third, and"
    assert len(two.calls) == 3


def test_a_continuation_that_is_noise_is_dropped_and_the_first_words_stand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "I think that,", b"clip2": "Thanks for watching."})
    _mic(monkeypatch, b"clip2")
    heard, _ = _hear(_ctx(asr))
    assert heard.text == "I think that,"


def test_a_failed_continuation_keeps_the_first_words(monkeypatch: pytest.MonkeyPatch) -> None:
    asr = _asr(monkeypatch, {b"clip1": "Let me think,", b"clip2": 503})
    _mic(monkeypatch, b"clip2")
    heard, _ = _hear(_ctx(asr))
    assert (heard.kind, heard.text) == ("line", "Let me think,")


def test_a_failed_first_clip_is_listened_for_again(monkeypatch: pytest.MonkeyPatch) -> None:
    asr = _asr(monkeypatch, {b"clip1": 503})
    _mic(monkeypatch)
    heard, _ = _hear(_ctx(asr))
    assert heard.kind == "again"


def test_a_filler_the_speaker_goes_on_from_is_not_dropped_as_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "Um,", b"clip2": "can you open the door?"})
    _mic(monkeypatch, b"clip2")
    heard, _ = _hear(_ctx(asr))
    assert (heard.kind, heard.text) == ("line", "Um, can you open the door?")


def test_a_quit_word_the_speaker_trails_off_on_still_ends_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "Goodbye,"})
    mic = _mic(monkeypatch, None)
    heard, _ = _hear(_ctx(asr))
    assert heard.kind == "bye"
    assert len(mic.calls) == 2  # it waited for more first


def test_quit_while_waiting_for_a_continuation_is_a_bye_and_nothing_is_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(_asr(monkeypatch, {b"clip1": "I wanted to say that,"}))
    mic = _mic(monkeypatch, WAIT_FOR_STOP)
    timer = threading.Timer(0.2, ctx.turn_input.session.quit_requested.set)
    timer.start()
    try:
        heard, seen = _hear(ctx)
    finally:
        timer.cancel()
    assert heard.kind == "bye"
    assert _said(seen) == []
    assert mic.calls[1]["stop"].is_set()


def test_a_mute_or_typed_line_while_waiting_drops_the_turn_without_announcing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "I wanted to say that,"})
    _mic(monkeypatch, WAIT_FOR_STOP)
    interrupt = threading.Event()
    timer = threading.Timer(0.2, interrupt.set)
    timer.start()
    try:
        heard, seen = _hear(_ctx(asr), stop=interrupt)
    finally:
        timer.cancel()
    assert heard.kind == "noise"
    assert _said(seen) == []


def test_a_muted_mic_is_not_reopened_for_a_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(_asr(monkeypatch, {b"clip1": "I wanted to say that,"}))
    ctx = replace(ctx, turn_input=replace(ctx.turn_input, muted=lambda: True))
    mic = _mic(monkeypatch)
    heard, _ = _hear(ctx)
    assert (heard.kind, heard.text) == ("line", "I wanted to say that,")
    assert len(mic.calls) == 1


def test_a_barge_in_clip_goes_to_the_first_recording_only(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(_asr(monkeypatch, {b"clip1": "Wait,", b"clip2": "stop that please."}))
    clip = barge_clip(b"pcm")
    ctx.turn_input.carry.keep(clip, "")
    mic = _mic(monkeypatch, b"clip2")
    heard, _ = _hear(ctx)
    assert heard.text == "Wait, stop that please."
    assert mic.calls[0]["carried"] == clip
    assert "carried" not in mic.calls[1]
    assert ctx.turn_input.carry.clip is None


def test_a_joined_line_is_not_taken_for_a_stale_copy_of_the_last_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = _asr(monkeypatch, {b"clip1": "Tell me a story,", b"clip2": "tell me a story."})
    _mic(monkeypatch, b"clip2")
    heard, _ = _hear(_ctx(asr), last_user="Tell me a story, tell me a story.")
    assert heard.kind == "line"


def test_the_check_off_keeps_one_clip_with_the_ordinary_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx(asr=_asr(monkeypatch, {b"clip1": "I was going to say that, but"}))
    mic = _mic(monkeypatch)
    heard, _ = _hear(ctx)
    assert (heard.kind, heard.text) == ("line", "I was going to say that, but")
    assert len(mic.calls) == 1
    assert mic.calls[0]["tune"].end_silence_frames == 40


def test_a_failed_mic_reopen_is_fatal_like_a_failed_first_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_audio_suite_voice.capture.portaudio import PortAudioMissingError

    asr = _asr(monkeypatch, {b"clip1": "I wanted to say that,"})
    ScriptedRecordings(monkeypatch, b"clip1", PortAudioMissingError("gone"))
    heard = asyncio.run(asyncio.wait_for(hear_line(_ctx(asr), ""), 5))
    assert heard.kind == "fatal"


def test_open_turn_joins_clips_keeps_tone_once_and_counts_clips() -> None:
    turn = OpenTurn().add("I was saying,", ("laughter",)).add("and then", ("laughter", "tired"))
    assert turn.text == "I was saying, and then"
    assert turn.tone == ("laughter", "tired")
    assert turn.clips == 2


@pytest.mark.parametrize(
    ("cap", "clips", "room"), [(1, 0, True), (1, 1, False), (2, 1, True), (2, 2, False)]
)
def test_open_turn_has_room_for_more_until_the_cap(cap: int, clips: int, room: bool) -> None:
    turn = OpenTurn(text="x", clips=clips)
    assert turn.room_for_more(TurnTune(enabled=True, max_extensions=cap)) is room


def test_the_first_clip_never_waits_longer_than_the_ordinary_silence() -> None:
    listen = ListenTune(end_silence_frames=40)
    on = TurnTune(enabled=True, silence_frames=12)
    assert first_clip_tune(listen, on).end_silence_frames == 12
    assert first_clip_tune(ListenTune(end_silence_frames=8), on).end_silence_frames == 8
    assert first_clip_tune(listen, TurnTune(silence_frames=12)) is listen


def test_a_continuation_waits_the_longer_silence_less_what_was_spent_and_needs_less_voice() -> None:
    turn = TurnTune(enabled=True, silence_frames=12, extend_frames=40)
    assert wait_frames(turn) == 28
    assert wait_frames(TurnTune(silence_frames=30, extend_frames=20)) == 1
    tune = continuation_tune(ListenTune(min_voiced_frames=12), turn)
    assert (tune.end_silence_frames, tune.min_voiced_frames) == (40, 6)
    assert continuation_tune(ListenTune(min_voiced_frames=3), turn).min_voiced_frames == 3
