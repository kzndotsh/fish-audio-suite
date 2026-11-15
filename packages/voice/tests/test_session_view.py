"""Folding events into a snapshot a display can show."""

from __future__ import annotations

import dataclasses
from functools import reduce

import pytest
from hypothesis import given
from hypothesis import strategies as st
from real_mic import BLOCK_S, QUIET_BELOW, RMS, SPEECH_FROM

from fish_audio_suite_voice.cli.session_view import (
    RECENT_TURNS,
    SessionView,
    SpeechScale,
    bar_dots,
    reduce_view,
    turn_timings,
)
from fish_audio_suite_voice.session.conversation import Conversation, HistoryView
from fish_audio_suite_voice.session.events import (
    BargedIn,
    Bye,
    ChatShown,
    Event,
    ExitCode,
    Heard,
    Listening,
    LogLine,
    MicLevel,
    Notice,
    ReplyEnd,
    ReplyToken,
    SessionState,
    Speaking,
    TurnEnded,
)
from fish_audio_suite_voice.session.latency import LatencySnapshot


def _fold(*events: Event, start: SessionView | None = None) -> SessionView:
    return reduce(reduce_view, events, start or SessionView())


def test_a_whole_turn_folds_into_the_view_a_screen_shows() -> None:
    snapshot = LatencySnapshot(asr_ms=120.0, first_audio_ms=900.0)
    view = _fold(
        Listening(),
        MicLevel(300.0, 200.0, "listen"),
        Heard("tell me a story", 120.0),
        ReplyToken("Once "),
        ReplyToken("upon a time."),
        ReplyEnd("Once upon a time."),
        Speaking(),
        TurnEnded(snapshot),
    )
    assert view == SessionView(
        state=SessionState.IDLE,
        reply="Once upon a time.",
        recent_turns=(snapshot,),
        turns_timed=1,
    )
    assert view.last_turn == snapshot


def test_the_view_keeps_the_recent_turns_and_counts_them_all() -> None:
    snapshots = [LatencySnapshot(first_audio_ms=float(n)) for n in range(RECENT_TURNS + 5)]
    view = _fold(*(TurnEnded(snapshot) for snapshot in snapshots))
    assert view.recent_turns == tuple(snapshots[-RECENT_TURNS:])
    assert view.turns_timed == RECENT_TURNS + 5
    assert SessionView().last_turn is None


def test_the_view_says_how_much_of_the_chat_the_model_is_sent() -> None:
    record = Conversation("be brief")
    for line in ("one", "two", "three"):
        record.add_user_line(line)
        record.add_reply("ok.", interrupted=False)
    record.set_hidden(record.turns[0].id, hidden=True)
    sent = frozenset(turn.id for turn in HistoryView(10).sent(record))
    view = _fold(ChatShown(record.id, tuple(record.turns), sent))
    assert (view.sent_turns, view.chat_turns) == (2, 3)


def test_the_view_follows_the_state_through_a_turn() -> None:
    states = [
        _fold(*events).state
        for events in (
            (Listening(),),
            (Listening(), Heard("hi", 1.0)),
            (Listening(), Heard("hi", 1.0), Speaking()),
            (Listening(), Heard("hi", 1.0), Speaking(), BargedIn()),
        )
    ]
    assert states == [
        SessionState.LISTENING,
        SessionState.THINKING,
        SessionState.SPEAKING,
        SessionState.LISTENING,
    ]


def test_the_reply_grows_token_by_token_and_a_new_line_starts_it_over() -> None:
    view = _fold(Heard("one", 1.0), ReplyToken("Hel"), ReplyToken("lo"))
    assert view.reply == "Hello"
    assert _fold(Heard("two", 1.0), start=view).reply == ""


def test_the_end_of_the_reply_replaces_what_was_streamed() -> None:
    view = _fold(ReplyToken("[calm] Hi"), ReplyEnd("[calm] Hi there"))
    assert view.reply == "[calm] Hi there"


def test_the_end_of_the_session_is_kept() -> None:
    view = _fold(Notice("[llm 429, retrying in 4s]"), Bye(ExitCode.FATAL))
    assert view.exit_code == 2
    assert view.state is SessionState.IDLE
    assert SessionView().exit_code is None


def test_log_lines_do_not_change_the_view() -> None:
    view = _fold(Listening())
    assert reduce_view(view, LogLine("DEBUG", "tts", "start")) == view


def test_reducing_never_changes_the_view_it_was_given() -> None:
    before = SessionView(reply="so far")
    after = reduce_view(before, ReplyToken(" more"))
    assert before.reply == "so far"
    assert after.reply == "so far more"
    with pytest.raises(dataclasses.FrozenInstanceError):
        before.reply = "changed"  # type: ignore[misc]


def test_the_same_mic_level_twice_gives_an_equal_view_so_a_screen_need_not_refresh() -> None:
    first = _fold(MicLevel(250.0, 200.0, "listen"))
    assert _fold(MicLevel(250.0, 200.0, "listen"), start=first) == first


def _dots(char: str) -> int:
    return (ord(char) - 0x2800).bit_count()


def test_bar_dots_grow_from_the_bottom_and_never_vanish() -> None:
    heights = [_dots(bar_dots([level / 10] * 2)) for level in range(11)]
    assert heights[0] == 2  # silence: one dot per bar, a thin line along the bottom
    assert heights[-1] == 8  # the whole cell
    assert heights == sorted(heights)
    assert bar_dots([-1.0, 5.0]) == bar_dots([0.0, 1.0])  # out of range is clamped
    assert len(bar_dots([0.5] * 5)) == 3  # two bars to a character, the odd one on its own
    assert bar_dots([]) == ""


def test_a_turns_wait_is_split_into_asr_llm_and_the_rest_and_adds_up() -> None:
    timings = turn_timings(
        LatencySnapshot(
            asr_ms=300.0, llm_first_token_ms=700.0, first_audio_ms=1500.0, voice_to_voice_ms=6000.0
        )
    )
    assert timings is not None
    assert timings.wait_ms == 1500.0
    assert timings.total_ms == 6000.0
    assert timings.stages == (("asr", 300.0), ("llm", 700.0), ("tts", 500.0))
    assert sum(ms for _, ms in timings.stages) == timings.wait_ms


def test_a_turn_without_asr_or_without_a_wait_is_handled() -> None:
    typed = turn_timings(LatencySnapshot(llm_first_token_ms=400.0, first_audio_ms=900.0))
    assert typed is not None
    assert [name for name, _ in typed.stages] == ["llm", "tts"]  # a typed line has no asr
    assert turn_timings(LatencySnapshot(asr_ms=100.0)) is None  # nothing to break down
    # Stages that overlap the wait entirely leave nothing for tts, never a negative bar.
    over = turn_timings(
        LatencySnapshot(asr_ms=500.0, llm_first_token_ms=600.0, first_audio_ms=900.0)
    )
    assert over is not None
    assert all(ms > 0 for _, ms in over.stages)
    assert "tts" not in dict(over.stages)


def _replay(rms: tuple[float, ...] | list[float], gain: float = 1.0) -> list[float]:
    scale = SpeechScale()
    return [scale.update(value * gain, step * BLOCK_S) for step, value in enumerate(rms)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_a_real_quiet_mic_fills_most_of_the_bar_when_its_owner_speaks_and_is_empty_otherwise() -> (
    None
):
    levels = _replay(RMS)
    speaking = [level for level, rms in zip(levels, RMS, strict=True) if rms > SPEECH_FROM]
    quiet = [level for level, rms in zip(levels, RMS, strict=True) if rms < QUIET_BELOW]
    assert _mean(speaking) > 0.6  # the fixed -24 dB scale left this at 0.36
    assert max(quiet) < 0.25
    assert _mean(quiet) < 0.1


def test_the_same_voice_draws_the_same_height_on_a_quiet_mic_and_a_hot_one() -> None:
    quiet = _replay(RMS, gain=0.5)
    reference = _replay(RMS)
    hot = _replay(RMS, gain=8.0)  # 18 dB louder, with the room louder too

    def speaking(levels: list[float]) -> float:
        # Skip the first seconds, while the scale is still learning the mic.
        return _mean([lv for lv, r in zip(levels[30:], RMS[30:], strict=True) if r > SPEECH_FROM])

    assert speaking(quiet) == pytest.approx(speaking(reference), abs=0.12)
    assert speaking(hot) == pytest.approx(speaking(reference), abs=0.12)


def test_one_loud_bump_does_not_shrink_the_speech_that_follows_it() -> None:
    bumped = list(RMS)
    bumped[40] = 25000.0  # a bump in the quiet gap after the first sentence
    after = slice(60, None)

    def speaking(levels: list[float], source: tuple[float, ...] | list[float]) -> float:
        return _mean(
            [lv for lv, r in zip(levels[after], source[after], strict=True) if r > SPEECH_FROM]
        )

    assert speaking(_replay(bumped), bumped) >= 0.9 * speaking(_replay(RMS), RMS)
    assert _replay(bumped)[40] == 1.0  # the bump itself is drawn, just not allowed to rescale


def test_a_room_with_nobody_speaking_never_looks_like_speech() -> None:
    room = [
        32768.0 * 10 ** (-55.0 / 20) * (1.0 + 0.3 * ((step * 7) % 5 - 2) / 2) for step in range(400)
    ]
    assert max(_replay(room)) < 0.3  # not stretched, even after thirty-six seconds of it


def test_the_speech_level_is_kept_through_a_long_pause() -> None:
    talk = list(RMS[20:60])  # the first sentence
    silence = [30.0] * 200  # eighteen seconds of the room
    levels = _replay(talk + silence + talk)
    first = _mean([lv for lv, r in zip(levels[:40], talk, strict=True) if r > SPEECH_FROM])
    again = _mean([lv for lv, r in zip(levels[240:], talk, strict=True) if r > SPEECH_FROM])
    assert again == pytest.approx(first, abs=0.15)  # it did not re-learn from the room


def test_a_steady_loud_voice_with_digital_silence_between_words_still_shows_its_shape() -> None:
    # TTS: loud and even, with exact silence in the pauses.
    voice = [
        32768.0 * 10 ** (-12.0 / 20) * (0.55 + 0.45 * abs((step * 5) % 7 - 3) / 3)
        for step in range(90)
    ]
    for step in range(5, 90, 9):
        voice[step] = voice[step + 1] = 0.0
    levels = _replay(voice)
    pauses = [lv for lv, v in zip(levels, voice, strict=True) if v == 0.0]
    words = [lv for lv, v in zip(levels[20:], voice[20:], strict=True) if v > 0.0]
    assert max(pauses) == 0.0
    assert _mean(words) > 0.6
    assert max(words) - min(words) > 0.2  # not a flat block


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=-100.0, max_value=1e6) | st.just(float("nan")),
            st.floats(min_value=0.0, max_value=2.0),
        ),
        max_size=120,
    )
)
def test_a_speech_scale_always_answers_between_empty_and_full(
    blocks: list[tuple[float, float]],
) -> None:
    scale = SpeechScale()
    clock = 0.0
    for rms, step in blocks:
        clock += step
        assert 0.0 <= scale.update(rms, clock) <= 1.0


def test_a_scale_keeps_the_noise_floor_it_measured_through_a_gap_in_the_audio() -> None:
    room = 32768.0 * 10 ** (-45.0 / 20)  # a noisy room
    speech = 32768.0 * 10 ** (-25.0 / 20)
    scale = SpeechScale()
    for step in range(130):
        scale.update(speech if step % 8 < 4 else room, step * BLOCK_S)
    # Twenty seconds with no audio at all (the model thinking and speaking), then the room again.
    first_blocks = [scale.update(room, 40.0 + step * BLOCK_S) for step in range(3)]
    assert max(first_blocks) < 0.15  # not a blip while it relearns the room
    forgetful = SpeechScale()  # the same, but it was started again after the gap
    assert forgetful.update(room, 40.0) > max(first_blocks)  # what it did before this fix


def test_the_turn_summary_leads_with_the_wait_since_the_last_word() -> None:
    from fish_audio_suite_voice.session.latency import turn_summary

    line = turn_summary(LatencySnapshot(first_audio_ms=900.0, end_wait_ms=360.0))
    assert line.startswith("since you stopped 1.26s · first audio 0.90s")
    assert "since you stopped" not in turn_summary(LatencySnapshot(first_audio_ms=900.0))
