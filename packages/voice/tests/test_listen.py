"""Listening for one turn: when speech starts and ends, what a clip must hold to be a turn, and the clip a listen carries on from."""

from __future__ import annotations

import collections.abc
import io
import itertools
import threading
import wave

import pytest
from voice_fakes import ScriptedMic, install_detector, level_frame

from fish_audio_suite_voice.asr.result import BargeIn, StreamFallback
from fish_audio_suite_voice.capture.listen import (
    listen_reject_reason,
    record_utterance,
    spike_start_allowed,
    start_frames_needed,
    start_hit,
    stream_utterance,
    trailing_start_hits,
    voice_needed,
)
from fish_audio_suite_voice.capture.pcm import FRAME_BYTES, SAMPLE_RATE
from fish_audio_suite_voice.config.tune import (
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_RMS,
    DEFAULT_END_SILENCE_FRAMES,
    DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MIN_VOICED_FRAMES,
    IMPULSE_START_EXTRA,
    MAX_UTTERANCE_FRAMES,
    ListenTune,
)
from fish_audio_suite_voice.session.cancel import StopFlag


def _stream(
    frames: list[bytes], tune: ListenTune, carried: BargeIn | None = None
) -> tuple[list[tuple[bytes, bool]], bool, int]:
    """Run ``stream_utterance`` over ``frames``.

    Returns what it passed on (each frame with whether it was speech), whether the
    silence ended it, and how many mic frames it read.
    """
    kept: list[tuple[bytes, bool]] = []
    pulled = 0

    def script(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        nonlocal pulled
        for frame in frames:
            pulled += 1
            yield frame

    ended = stream_utterance(
        ScriptedMic(script),
        None,
        lambda frame, hit: kept.append((frame, hit)),
        carried=carried,
        tune=tune,
    )
    return kept, ended, pulled


def _hits(kept: list[tuple[bytes, bool]]) -> list[bool]:
    return [hit for _frame, hit in kept]


def _wav_frames(wav: bytes) -> list[bytes]:
    with wave.open(io.BytesIO(wav)) as wf:
        pcm = wf.readframes(wf.getnframes())
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


def test_listen_tune_keeps_vad_and_pre_pad_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_VAD", "9")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD_FRAMES", "-1")
    tune = ListenTune.from_env()
    assert tune.vad_aggressiveness == 1
    assert tune.pre_pad_frames == 20
    monkeypatch.setenv("FISH_VOICE_VAD", "3")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD_FRAMES", "0")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "0")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED_FRAMES", "-1")
    tune = ListenTune.from_env()
    assert tune.vad_aggressiveness == 3
    assert tune.pre_pad_frames == tune.start_speech_frames + IMPULSE_START_EXTRA
    assert tune.start_speech_frames == 4
    assert tune.min_voiced_frames == 12
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED_FRAMES", "6")
    tune = ListenTune.from_env()
    assert tune.start_speech_frames == 2
    assert tune.min_voiced_frames == 6
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "-5")
    assert ListenTune.from_env().min_speech_rms == DEFAULT_MIN_SPEECH_RMS
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "0")
    assert ListenTune.from_env().end_silence_frames == DEFAULT_END_SILENCE_FRAMES
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "-3")
    assert ListenTune.from_env().end_silence_frames == DEFAULT_END_SILENCE_FRAMES
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED_FRAMES", str(MAX_UTTERANCE_FRAMES + 1))
    assert ListenTune.from_env().min_voiced_frames == DEFAULT_MIN_VOICED_FRAMES
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED_FRAMES", "40")
    assert ListenTune.from_env().min_voiced_frames == 40


def test_pre_pad_zero_still_starts_on_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_PRE_PAD_FRAMES", "0")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "200")
    install_detector(monkeypatch, lambda _frame: True)
    tune = ListenTune.from_env()
    kept, _ended, _pulled = _stream([level_frame(500)] * tune.pre_pad_frames, tune)
    assert kept  # speech started, so its frames were passed on


def test_loud_frames_do_not_raise_the_floor_above_the_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 30 frames at 300 that the detector calls noise, then the same level voiced. Had the
    # loud noise taught the floor (300 x 2.5), the voice could never start a turn.
    asked = itertools.count(1)
    install_detector(monkeypatch, lambda _frame: next(asked) > 30)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=4,
        min_speech_rms=200.0,
        pre_pad_frames=10,
        min_voiced_frames=12,
    )
    kept, _ended, _pulled = _stream([level_frame(300)] * 40, tune)
    assert kept


def test_a_gap_before_the_start_still_counts_as_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    install_detector(monkeypatch, any)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=2,
        start_speech_frames=4,
        min_speech_rms=200.0,
        pre_pad_frames=20,
        min_voiced_frames=12,
    )
    loud, quiet = level_frame(500), level_frame(0)
    frames = [loud] * 3 + [quiet] + [loud] * 4 + [loud] * 5 + [quiet] * 2
    kept, ended, pulled = _stream(frames, tune)
    assert ended
    assert pulled == len(frames)
    # The voiced frames before the gap are part of the clip and count toward its voice.
    assert _hits(kept) == [True] * 3 + [False] + [True] * 9 + [False] * 2
    assert record_utterance(ScriptedMic(frames), tune=tune) is not None


def test_a_dip_after_the_length_cap_does_not_end_the_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_detector(monkeypatch, any)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=4,
        min_voiced_frames=12,
    )
    frames = [level_frame(500)] * MAX_UTTERANCE_FRAMES + [level_frame(0)] * tune.end_silence_frames
    _kept, ended, pulled = _stream(frames, tune)
    assert ended
    quiet_read = pulled - MAX_UTTERANCE_FRAMES
    # At the cap a short quiet run ends the turn, but not a single dip inside a word.
    assert 1 < quiet_read < tune.end_silence_frames


def test_a_long_utterance_keeps_only_the_recent_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    install_detector(monkeypatch, any)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=4,
        min_voiced_frames=12,
    )
    first, rest, quiet = level_frame(500), level_frame(800), level_frame(0)
    frames = [first] + [rest] * (MAX_UTTERANCE_FRAMES + 20) + [quiet] * tune.end_silence_frames
    wav = record_utterance(ScriptedMic(frames), tune=tune)
    assert wav is not None
    kept = _wav_frames(wav)
    assert len(kept) == MAX_UTTERANCE_FRAMES
    assert first not in kept


def test_quiet_vad_frame_holds_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    end = level_frame(1)  # the one frame the detector calls silence
    install_detector(monkeypatch, lambda frame: frame != end)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=2,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=20,
        min_voiced_frames=12,
    )
    hesitation = level_frame(0)  # too quiet to be speech, but the detector still hears a voice
    frames = [level_frame(500)] + [hesitation] * 5 + [end] * 2
    kept, ended, pulled = _stream(frames, tune)
    # The hesitation is not silence, so only the two silent frames end the turn,
    # and it is not counted as speech either.
    assert ended
    assert pulled == len(frames)
    assert _hits(kept) == [True] + [False] * 7


def test_start_hit_requires_vad_and_full_floor() -> None:
    assert not start_hit(200.0, 200.0, vad_speech=False)
    assert start_hit(200.0, 200.0, vad_speech=True)
    assert not start_hit(121.0, 200.0, vad_speech=True)
    assert not start_hit(111.0, 200.0, vad_speech=True)


def test_start_frames_needed_raises_on_spike() -> None:
    assert start_frames_needed(200.0, 200.0, 4) == 4
    assert start_frames_needed(1005.0, 200.0, 4) == 4
    assert start_frames_needed(1600.0, 200.0, 4) == 10


def test_trailing_start_hits_ignores_older_scored_frames() -> None:
    ring: collections.deque[tuple[bytes, bool]] = collections.deque(
        [(b"", True), (b"", True), (b"", False), (b"", True), (b"", True)]
    )
    assert trailing_start_hits(ring) == 2


def test_spike_start_blocks_decaying_bang() -> None:
    assert spike_start_allowed(200.0, 200.0, 200.0)
    assert not spike_start_allowed(1923.0, 671.0, 200.0)
    assert spike_start_allowed(1923.0, 1000.0, 200.0)


def test_listen_reject_cough_and_impulse() -> None:
    def reason(voiced: int, hits: int, peak: float) -> str | None:
        return listen_reject_reason(
            clip_frames=voiced,
            speech_hits=hits,
            peak_rms=peak,
            min_voiced_frames=12,
            min_speech_rms=200.0,
        )

    assert reason(48, 8, 346.0) == "too_little_voice"
    assert reason(54, 14, 2220.0) == "impulse"
    assert reason(46, 16, 1005.0) is None
    assert reason(80, 18, 1600.0) == "impulse"
    assert reason(80, 20, 400.0) is None
    assert reason(120, 68, 1486.0) is None
    assert reason(2, 2, 100.0) == "too_short"


def _install_listen_fakes(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[bytes],
) -> tuple[ScriptedMic, ListenTune]:
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "200")
    install_detector(monkeypatch, lambda frame: frame[:2] != b"\x00\x00")

    def fake_mic(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        yield from frames

    return ScriptedMic(fake_mic), ListenTune.from_env()


def test_record_utterance_writes_a_wav_for_a_held_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [level_frame(500), level_frame(500), level_frame(0), level_frame(0)]
    mic, tune = _install_listen_fakes(monkeypatch, frames)
    wav = record_utterance(mic, tune=tune)
    assert wav is not None
    # 16 kHz mono int16, as Fish ASR is sent it, holding every frame the listen kept.
    with wave.open(io.BytesIO(wav)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.readframes(wf.getnframes()) == b"".join(frames)


def test_record_utterance_drops_a_finished_clip_when_quit_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = level_frame(500)
    quiet = level_frame(0)
    mic, tune = _install_listen_fakes(monkeypatch, [loud, loud, quiet, quiet])
    stop = threading.Event()
    stop.set()
    assert record_utterance(mic, quit_requested=stop, tune=tune) is None


def test_record_utterance_drops_a_clip_that_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic, _tune = _install_listen_fakes(monkeypatch, [level_frame(0), level_frame(0)])
    assert record_utterance(mic) is None


def test_record_utterance_gives_up_when_speech_has_not_started_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = level_frame(500)
    quiet = level_frame(0)
    mic, tune = _install_listen_fakes(monkeypatch, [quiet] * 6 + [loud] * 6 + [quiet] * 2)
    assert record_utterance(mic, tune=tune, give_up_frames=3) is None
    # Once speech has started, the limit no longer applies.
    assert record_utterance(mic, tune=tune, give_up_frames=8) is not None
    assert record_utterance(mic, tune=tune) is not None


def _barge_carried_listen(
    monkeypatch: pytest.MonkeyPatch, *, min_speech_rms: float = 200.0
) -> ListenTune:
    """A listen whose detector hears no speech on the mic, so only a carried clip counts."""
    install_detector(monkeypatch, lambda _frame: False)
    return ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=1,
        start_speech_frames=4,
        min_speech_rms=min_speech_rms,
        pre_pad_frames=20,
        min_voiced_frames=DEFAULT_MIN_VOICED_FRAMES,
    )


def test_trimmed_barge_credit_does_not_keep_a_short_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    end = level_frame(1)  # the one frame the detector calls silence
    install_detector(monkeypatch, lambda frame: frame != end)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=4,
        min_voiced_frames=12,
    )
    loud, hesitation = level_frame(800), level_frame(0)
    barge = BargeIn(loud * 4, None, DEFAULT_BARGE_RMS)
    # Straight after the barge-in, four more voiced frames are enough: the clip is credited.
    short = [loud] * 4 + [end] * tune.end_silence_frames
    assert record_utterance(ScriptedMic(short), carried=barge, tune=tune) is not None
    # After a hesitation as long as the cap, the barge-in clip has left the clip, and so
    # has its credit: four voiced frames are a short tail, not a turn.
    trimmed = [hesitation] * MAX_UTTERANCE_FRAMES + [loud] * 4 + [end] * tune.end_silence_frames
    assert record_utterance(ScriptedMic(trimmed), carried=barge, tune=tune) is None


def test_barge_prefix_meets_the_listen_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    tune = _barge_carried_listen(monkeypatch)
    quiet = [level_frame(0)]
    clip = BargeIn(level_frame(1000) * DEFAULT_BARGE_HIT_FRAMES, None, DEFAULT_BARGE_RMS)
    kept, _ended, _pulled = _stream(quiet, tune, carried=clip)
    assert _hits(kept)[:DEFAULT_BARGE_HIT_FRAMES] == [True] * DEFAULT_BARGE_HIT_FRAMES
    wav = record_utterance(ScriptedMic(quiet), carried=clip, tune=tune)
    assert wav is not None
    assert wav.startswith(b"RIFF")
    # A loud barge-in clip is kept too, although the same clip on its own is an impulse.
    bang = BargeIn(level_frame(4000) * DEFAULT_BARGE_HIT_FRAMES, None, DEFAULT_BARGE_RMS)
    assert record_utterance(ScriptedMic(quiet), carried=bang, tune=tune) is not None
    assert (
        listen_reject_reason(
            clip_frames=14,
            speech_hits=14,
            peak_rms=4000,
            min_voiced_frames=DEFAULT_MIN_VOICED_FRAMES,
            min_speech_rms=200.0,
        )
        == "impulse"
    )


def test_barge_prefix_counts_under_a_higher_listen_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=800.0)
    speech = level_frame(256)
    quiet = level_frame(1)
    clip = BargeIn(quiet + speech * DEFAULT_BARGE_HIT_FRAMES, None, DEFAULT_BARGE_RMS)
    kept, _ended, _pulled = _stream([level_frame(0)], tune, carried=clip)
    assert (
        _hits(kept)[: DEFAULT_BARGE_HIT_FRAMES + 1] == [False] + [True] * DEFAULT_BARGE_HIT_FRAMES
    )
    assert record_utterance(ScriptedMic([level_frame(0)]), carried=clip, tune=tune) is not None


def test_a_barge_clip_starts_the_listen_and_passes_on_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_detector(monkeypatch, lambda _frame: False)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=4,
        min_speech_rms=200.0,
        pre_pad_frames=20,
        min_voiced_frames=12,
    )
    loud = b"\x00\x10" * 480
    quiet = b"\x01\x00" * 480
    after = level_frame(0)
    kept, _ended, _pulled = _stream([after], tune, carried=BargeIn(quiet + loud, None, 220.0))
    # The clip goes first, scored at its gate's level, and the next mic frame follows at
    # once: the listen started from the clip and did not wait for speech to start.
    assert kept == [(quiet, False), (loud, True), (after, False)]


@pytest.mark.parametrize(("gate_rms", "credited"), [(100.0, 10), (DEFAULT_BARGE_RMS, 0)])
def test_a_barge_clip_is_scored_at_the_level_its_own_gate_admitted_it(
    monkeypatch: pytest.MonkeyPatch, gate_rms: float, credited: int
) -> None:
    # FISH_VOICE_BARGE_RMS=100 let a soft interruption at 150 stop the reply. Scored at the
    # default 220 instead, none of it counted and the words that stopped the reply were lost.
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=DEFAULT_MIN_SPEECH_RMS)
    clip = BargeIn(level_frame(150) * 10, None, gate_rms)
    kept, _ended, _pulled = _stream([level_frame(0)], tune, carried=clip)
    assert sum(_hits(kept)) == credited
    wav = record_utterance(ScriptedMic([level_frame(0)]), carried=clip, tune=tune)
    assert (wav is not None) is (credited > 0)


def test_a_fallback_clip_goes_on_as_ordinary_speech_with_no_barge_in_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=DEFAULT_MIN_SPEECH_RMS)
    voiced = (level_frame(600), True)
    unvoiced = (level_frame(600), False)
    # Speech already started, so the first words are kept and the listen reads on from
    # the position after them; the mic alone never starts a turn here.
    mic = ScriptedMic([level_frame(0)])
    full = StreamFallback((voiced,) * DEFAULT_MIN_VOICED_FRAMES, 7)
    wav = record_utterance(mic, carried=full, tune=tune)
    assert wav is not None
    assert _wav_frames(wav)[:DEFAULT_MIN_VOICED_FRAMES] == [voiced[0]] * DEFAULT_MIN_VOICED_FRAMES
    assert mic.since == 7
    # Each frame counts as the listen gate judged it when it was captured, and a fallback
    # is not an interruption: one frame short of the voice any turn needs is not a turn.
    short = StreamFallback((voiced,) * (DEFAULT_MIN_VOICED_FRAMES - 1) + (unvoiced,), 7)
    assert record_utterance(ScriptedMic([level_frame(0)]), carried=short, tune=tune) is None


def test_a_fallback_from_a_barged_turn_keeps_the_credit_of_the_barge_in_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=DEFAULT_MIN_SPEECH_RMS)
    barge = BargeIn(level_frame(600) * 4, 3, DEFAULT_BARGE_RMS)
    frames = ((level_frame(600), True),) * 4 + ((level_frame(600), True), (level_frame(5), False))
    barged = StreamFallback(frames, 9, barge=barge)
    # 5 voiced frames: enough after a barge-in, too few for a turn that began on its own.
    assert record_utterance(ScriptedMic([level_frame(0)]), carried=barged, tune=tune) is not None
    alone = StreamFallback(frames, 9)
    assert record_utterance(ScriptedMic([level_frame(0)]), carried=alone, tune=tune) is None


@pytest.mark.parametrize(
    ("min_voiced", "credited", "need"), [(12, 0, 12), (12, 4, 4), (12, 20, 12), (3, 4, 3)]
)
def test_the_voice_a_clip_needs_counts_a_barge_in_in_full(
    min_voiced: int, credited: int, need: int
) -> None:
    assert voice_needed(min_voiced, credited) == need
    assert (
        listen_reject_reason(
            clip_frames=40,
            speech_hits=need,
            peak_rms=300.0,
            min_voiced_frames=min_voiced,
            min_speech_rms=200.0,
            credited_hits=credited,
        )
        is None
    )
