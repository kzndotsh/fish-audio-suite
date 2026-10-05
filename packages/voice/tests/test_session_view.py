"""Folding events into a snapshot a display can show."""

from __future__ import annotations

import dataclasses
from functools import reduce

import pytest
from hypothesis import given
from hypothesis import strategies as st
from real_mic import BLOCK_S, QUIET_BELOW, RMS, SPEECH_FROM

from fish_audio_suite_kit import LatencySnapshot
from fish_audio_suite_voice.events import (
    BargedIn,
    Bye,
    Event,
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
    TurnEnded,
)
from fish_audio_suite_voice.session_view import (
    SessionView,
    SpeechScale,
    level_fraction,
    reduce_view,
    turn_timings,
    wave_dots,
)


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
        heard="tell me a story",
        reply="Once upon a time.",
        last_turn=snapshot,
        turns=1,
        mic_rms=300.0,
        mic_need=200.0,
    )


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


def test_notices_and_the_end_of_the_session_are_kept() -> None:
    view = _fold(Notice("[llm 429, retrying in 4s]"), Bye(2))
    assert view.last_notice == "[llm 429, retrying in 4s]"
    assert view.exit_code == 2
    assert view.state is SessionState.IDLE
    assert SessionView().exit_code is None


@pytest.mark.parametrize(
    "event", [LogLine("DEBUG", "tts", "start"), StateChanged(SessionState.IDLE)]
)
def test_log_lines_and_announced_state_changes_do_not_change_the_view(event: Event) -> None:
    view = _fold(Listening())
    assert reduce_view(view, event) == view


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


def test_the_meter_is_empty_when_silent_full_at_the_peak_and_never_outside_zero_to_one() -> None:
    assert level_fraction(0.0) == 0.0
    assert level_fraction(-5.0) == 0.0
    assert level_fraction(5e-324) == 0.0  # too small to divide: still empty, not an error
    assert level_fraction(float("nan")) == 0.0
    assert level_fraction(0.5) == 0.0  # far below the quietest level shown
    assert level_fraction(32768.0) == 1.0
    assert level_fraction(1e9) == 1.0


def test_the_meter_rises_with_loudness_and_shows_speech_well_above_empty() -> None:
    levels = [level_fraction(r) for r in (60, 150, 400, 1500, 4000, 40000)]
    assert levels == sorted(levels)
    assert len(set(levels)) == len(levels)
    # Ordinary speech, a few hundred out of 32768, would be invisible on a linear bar.
    assert level_fraction(300.0) > 0.25  # quiet speech is a quarter of the bar
    assert level_fraction(1000.0) > 0.5  # a normal speaking level is over half
    assert level_fraction(9000.0) < 1.0  # and a loud syllable on a hot mic keeps its shape
    assert level_fraction(300.0) > 30 * (300.0 / 32768.0)  # a linear bar would show under 1%


def test_a_view_gives_the_meter_and_its_threshold_marker() -> None:
    view = _fold(MicLevel(1000.0, 200.0, "listen"))
    assert view.mic_fraction == pytest.approx(level_fraction(1000.0))
    assert view.mic_need_fraction == pytest.approx(level_fraction(200.0))
    assert view.mic_fraction > view.mic_need_fraction
    assert SessionView().mic_fraction == 0.0


def _dots(char: str) -> int:
    return (ord(char) - 0x2800).bit_count()


def test_wave_dots_are_mirrored_never_empty_and_grow_with_the_level() -> None:
    assert wave_dots([0.0, 0.0], 1) == ["\u2836"]  # a thin dotted line through the middle
    assert wave_dots([1.0, 1.0], 1) == ["\u28ff"]  # every dot
    for rows in (1, 2, 4):
        silent = wave_dots([0.0] * 6, rows)
        assert len(silent) == rows
        assert all(len(line) == 3 for line in silent)  # two bars to a character
        # Silence is two dots tall per bar: one above the middle and one below it.
        assert sum(_dots(ch) for line in silent for ch in line) == 6 * 2
        totals = [
            sum(_dots(ch) for line in wave_dots([level / 20] * 4, rows) for ch in line)
            for level in range(21)
        ]
        assert totals == sorted(totals)  # louder never draws fewer dots
        loud = wave_dots([0.5, 1.0, 0.2, 0.9], rows)
        for line, mirror in zip(loud, reversed(loud), strict=True):
            assert [_dots(a) for a in line] == [_dots(b) for b in mirror]


def test_wave_dots_clamps_levels_and_copes_with_odd_and_empty_input() -> None:
    assert wave_dots([-5.0, 9.0], 2) == wave_dots([0.0, 1.0], 2)  # out of range is clamped
    assert wave_dots([0.5], 1) == ["\u2806"]  # an odd bar fills only the left dots of its character
    assert wave_dots([], 3) == ["", "", ""]
    assert len(wave_dots([0.5, 0.5, 0.5], 0)) == 1  # a height under 1 still draws


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
    fresh = SpeechScale()
    for step in range(130):
        fresh.update(speech if step % 8 < 4 else room, step * BLOCK_S)
    forgetful = SpeechScale()  # the same, but it was started again after the gap
    assert forgetful.update(room, 40.0) > max(first_blocks)  # what it did before this fix
