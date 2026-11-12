from __future__ import annotations

import collections.abc
import io
import itertools
import sys
import threading
import types
import wave
from collections.abc import Callable

import pytest
from voice_fakes import ScriptedMic, install_vad

from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import (
    BARGE_MISS_DECAY_FRAMES,
    BargeGate,
    barge_rms_need,
)
from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.listen import (
    cooldown_report,
    listen_reject_reason,
    record_utterance,
    spike_start_allowed,
    start_frames_needed,
    start_hit,
    stream_utterance,
    trailing_start_hits,
    voice_needed,
)
from fish_audio_suite_voice.pcm import FRAME_BYTES, SAMPLE_RATE, take_frames
from fish_audio_suite_voice.recognised import BargeIn, StreamFallback
from fish_audio_suite_voice.tune import (
    DEFAULT_AEC_BLEED_DELAY_S,
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_PLAYING_GAIN,
    DEFAULT_BARGE_RMS,
    DEFAULT_BLEED_DELAY_S,
    DEFAULT_END_SILENCE_FRAMES,
    DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MIN_VOICED_FRAMES,
    DEFAULT_POST_SPEAK_COOLDOWN_S,
    IMPULSE_START_EXTRA,
    MAX_UTTERANCE_FRAMES,
    AecTune,
    BargeTune,
    ListenTune,
)


def _pcm(level: int) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


def _vad(monkeypatch: pytest.MonkeyPatch, is_speech: Callable[[bytes], bool]) -> None:
    """Make the default detector call a frame speech when ``is_speech`` says so."""

    class Vad:
        def __init__(self, mode: int) -> None:
            del mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del rate
            return is_speech(frame)

    install_vad(monkeypatch, Vad)


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


class _RecordingCancel(threading.Event):
    """A cancel flag whose waits return at once and record their timeout, as a fake clock."""

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        return self.is_set()


def _bleed_of(gate: BargeGate) -> float | None:
    """The bleed delay ``start_after_bleed`` waits out before it watches."""
    cancel = _RecordingCancel()
    gate.start_after_bleed(cancel).join(timeout=5)
    return cancel.waits[0]


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
    _vad(monkeypatch, lambda _frame: True)
    tune = ListenTune.from_env()
    kept, _ended, _pulled = _stream([_pcm(500)] * tune.pre_pad_frames, tune)
    assert kept  # speech started, so its frames were passed on


def test_loud_frames_do_not_raise_the_floor_above_the_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 30 frames at 300 that the detector calls noise, then the same level voiced. Had the
    # loud noise taught the floor (300 x 2.5), the voice could never start a turn.
    asked = itertools.count(1)
    _vad(monkeypatch, lambda _frame: next(asked) > 30)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=4,
        min_speech_rms=200.0,
        pre_pad_frames=10,
        min_voiced_frames=12,
    )
    kept, _ended, _pulled = _stream([_pcm(300)] * 40, tune)
    assert kept


def test_a_gap_before_the_start_still_counts_as_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    _vad(monkeypatch, any)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=2,
        start_speech_frames=4,
        min_speech_rms=200.0,
        pre_pad_frames=20,
        min_voiced_frames=12,
    )
    loud, quiet = _pcm(500), _pcm(0)
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
    _vad(monkeypatch, any)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=4,
        min_voiced_frames=12,
    )
    frames = [_pcm(500)] * MAX_UTTERANCE_FRAMES + [_pcm(0)] * tune.end_silence_frames
    _kept, ended, pulled = _stream(frames, tune)
    assert ended
    quiet_read = pulled - MAX_UTTERANCE_FRAMES
    # At the cap a short quiet run ends the turn, but not a single dip inside a word.
    assert 1 < quiet_read < tune.end_silence_frames


def test_a_long_utterance_keeps_only_the_recent_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    _vad(monkeypatch, any)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=4,
        min_voiced_frames=12,
    )
    first, rest, quiet = _pcm(500), _pcm(800), _pcm(0)
    frames = [first] + [rest] * (MAX_UTTERANCE_FRAMES + 20) + [quiet] * tune.end_silence_frames
    wav = record_utterance(ScriptedMic(frames), tune=tune)
    assert wav is not None
    kept = _wav_frames(wav)
    assert len(kept) == MAX_UTTERANCE_FRAMES
    assert first not in kept


def test_trimmed_barge_credit_does_not_keep_a_short_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    end = _pcm(1)  # the one frame the detector calls silence
    _vad(monkeypatch, lambda frame: frame != end)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=40,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=4,
        min_voiced_frames=12,
    )
    loud, hesitation = _pcm(800), _pcm(0)
    barge = BargeIn(loud * 4, None, DEFAULT_BARGE_RMS)
    # Straight after the barge-in, four more voiced frames are enough: the clip is credited.
    short = [loud] * 4 + [end] * tune.end_silence_frames
    assert record_utterance(ScriptedMic(short), carried=barge, tune=tune) is not None
    # After a hesitation as long as the cap, the barge-in clip has left the clip, and so
    # has its credit: four voiced frames are a short tail, not a turn.
    trimmed = [hesitation] * MAX_UTTERANCE_FRAMES + [loud] * 4 + [end] * tune.end_silence_frames
    assert record_utterance(ScriptedMic(trimmed), carried=barge, tune=tune) is None


def test_bleed_wait_ends_when_the_turn_is_cancelled() -> None:
    gate = BargeGate(mic=ScriptedMic(), tune=BargeTune(bleed_delay_s=30))
    cancel = threading.Event()
    thread = gate.start_after_bleed(cancel)
    cancel.set()
    # A 30 s bleed that ended inside the join timeout proves cancel cut it short.
    thread.join(timeout=1)
    assert thread.is_alive() is False


def test_the_barge_watch_scores_speech_with_the_detector_listen_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[int] = []

    class Vad:
        def __init__(self, strictness: int) -> None:
            opened.append(strictness)

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return True

    plugin = types.ModuleType("barge_vad_plugin")
    vars(plugin)["Vad"] = Vad
    monkeypatch.setitem(sys.modules, "barge_vad_plugin", plugin)
    gate = BargeGate(
        mic=ScriptedMic([(3000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)] * 2),
        tune=BargeTune(hit_frames=2, min_rms=1.0),
        listen=ListenTune(vad_aggressiveness=3, vad_engine="barge_vad_plugin:Vad"),
    )
    cancel = threading.Event()
    gate.watch(cancel)
    assert cancel.is_set()
    assert opened == [3]


def test_barge_tune_reads_env_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "1.5")
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "9")
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "500")
    gate = BargeGate(mic=ScriptedMic(), tune=BargeTune.from_env())
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "2")
    assert gate.tune.bleed_delay_s == 1.5
    assert gate.tune.hit_frames == 9
    assert gate.tune.min_rms == 500.0


def test_barge_bleed_delay_cannot_be_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "-1")
    no_aec = EchoCanceller(AecTune(enabled=False))
    gate = BargeGate(mic=ScriptedMic(aec=no_aec), tune=BargeTune.from_env())
    assert gate.tune.bleed_delay_s == DEFAULT_BLEED_DELAY_S
    assert _bleed_of(gate) == DEFAULT_BLEED_DELAY_S
    with pytest.raises(ValueError, match="bleed_delay_s"):
        BargeTune(bleed_delay_s=-0.5)
    assert _bleed_of(BargeGate(mic=ScriptedMic(aec=no_aec), tune=BargeTune(bleed_delay_s=0.0))) == 0


def test_barge_hit_frames_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "0")
    assert BargeTune.from_env().hit_frames == DEFAULT_BARGE_HIT_FRAMES
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "-2")
    assert BargeTune.from_env().hit_frames == DEFAULT_BARGE_HIT_FRAMES
    with pytest.raises(ValueError, match="hit_frames"):
        BargeTune(hit_frames=0)
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "0")
    assert BargeTune.from_env().min_rms == DEFAULT_BARGE_RMS
    with pytest.raises(ValueError, match="min_rms"):
        BargeTune(min_rms=-1)
    assert BargeGate(mic=ScriptedMic(), tune=BargeTune(min_rms=10)).tune.min_rms == 10


def test_barge_keeps_the_lookback_that_tripped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Distinct frames: 22 under the gate's level, then 3 loud ones that trip it.
    frames = [_pcm(n) for n in range(1, 23)] + [_pcm(1000 + n) for n in range(3)]
    gate, cancel = _watch_with(
        monkeypatch,
        frames,
        aec=EchoCanceller(AecTune(enabled=False)),
        tune=BargeTune(hit_frames=3, min_rms=200.0),
    )
    gate.watch(cancel)
    assert cancel.is_set()
    # The last 20 frames (600 ms) are kept for the next listen, the tripping ones included.
    assert gate.captured == b"".join(frames[-20:])


def _barge_carried_listen(
    monkeypatch: pytest.MonkeyPatch, *, min_speech_rms: float = 200.0
) -> ListenTune:
    """A listen whose detector hears no speech on the mic, so only a carried clip counts."""
    _vad(monkeypatch, lambda _frame: False)
    return ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=1,
        start_speech_frames=4,
        min_speech_rms=min_speech_rms,
        pre_pad_frames=20,
        min_voiced_frames=DEFAULT_MIN_VOICED_FRAMES,
    )


def test_barge_prefix_meets_the_listen_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    tune = _barge_carried_listen(monkeypatch)
    quiet = [_pcm(0)]
    clip = BargeIn(_pcm(1000) * DEFAULT_BARGE_HIT_FRAMES, None, DEFAULT_BARGE_RMS)
    kept, _ended, _pulled = _stream(quiet, tune, carried=clip)
    assert _hits(kept)[:DEFAULT_BARGE_HIT_FRAMES] == [True] * DEFAULT_BARGE_HIT_FRAMES
    wav = record_utterance(ScriptedMic(quiet), carried=clip, tune=tune)
    assert wav is not None
    assert wav.startswith(b"RIFF")
    # A loud barge-in clip is kept too, although the same clip on its own is an impulse.
    bang = BargeIn(_pcm(4000) * DEFAULT_BARGE_HIT_FRAMES, None, DEFAULT_BARGE_RMS)
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
    speech = _pcm(256)
    quiet = _pcm(1)
    clip = BargeIn(quiet + speech * DEFAULT_BARGE_HIT_FRAMES, None, DEFAULT_BARGE_RMS)
    kept, _ended, _pulled = _stream([_pcm(0)], tune, carried=clip)
    assert (
        _hits(kept)[: DEFAULT_BARGE_HIT_FRAMES + 1] == [False] + [True] * DEFAULT_BARGE_HIT_FRAMES
    )
    assert record_utterance(ScriptedMic([_pcm(0)]), carried=clip, tune=tune) is not None


def test_a_barge_clip_starts_the_listen_and_passes_on_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _vad(monkeypatch, lambda _frame: False)
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
    after = _pcm(0)
    kept, _ended, _pulled = _stream([after], tune, carried=BargeIn(quiet + loud, None, 220.0))
    # The clip goes first, scored at its gate's level, and the next mic frame follows at
    # once: the listen started from the clip and did not wait for speech to start.
    assert kept == [(quiet, False), (loud, True), (after, False)]


def test_barge_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_BLEED_DELAY", raising=False)
    monkeypatch.delenv("FISH_VOICE_BARGE_FRAMES", raising=False)
    monkeypatch.delenv("FISH_VOICE_COOLDOWN", raising=False)
    gate = BargeGate(mic=ScriptedMic(), tune=BargeTune.from_env())
    assert gate.tune.bleed_delay_s == DEFAULT_BLEED_DELAY_S
    assert gate.tune.hit_frames == DEFAULT_BARGE_HIT_FRAMES
    assert BargeTune.from_env().cooldown_s == 0.8
    monkeypatch.setenv("FISH_VOICE_COOLDOWN", "0.3")
    assert BargeTune.from_env().cooldown_s == 0.3
    monkeypatch.setenv("FISH_VOICE_COOLDOWN", "0")
    assert BargeTune.from_env().cooldown_s == 0.0
    monkeypatch.setenv("FISH_VOICE_COOLDOWN", "-1")
    assert BargeTune.from_env().cooldown_s == DEFAULT_POST_SPEAK_COOLDOWN_S


def test_barge_over_speaker_raises_need() -> None:
    assert barge_rms_need(220.0, far_playing=False, playing_gain=2.2) == 220.0
    assert barge_rms_need(220.0, far_playing=True, playing_gain=2.2, aec_on=False) == pytest.approx(
        220.0 * 2.2
    )
    assert barge_rms_need(220.0, far_playing=True, playing_gain=2.2, aec_on=True) == 220.0
    with pytest.raises(ValueError, match="playing_gain"):
        BargeTune(playing_gain=0)
    assert barge_rms_need(
        220.0, far_playing=True, playing_gain=BargeTune().playing_gain, aec_on=False
    ) == pytest.approx(220.0 * DEFAULT_BARGE_PLAYING_GAIN)
    assert barge_rms_need(220.0, far_playing=True, playing_gain=1, aec_on=False) == 220.0


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


def test_mic_chunks_join_when_short_and_split_when_long() -> None:
    pending = bytearray()
    half = b"\x01" * (FRAME_BYTES // 2)
    assert take_frames(pending, half, FRAME_BYTES) == []
    assert take_frames(pending, half, FRAME_BYTES) == [b"\x01" * FRAME_BYTES]
    assert pending == bytearray()
    long = b"\x02" * (FRAME_BYTES * 2 + 3)
    frames = take_frames(pending, long, FRAME_BYTES)
    assert frames == [b"\x02" * FRAME_BYTES, b"\x02" * FRAME_BYTES]
    rest = b"\x03" * (FRAME_BYTES - 3)
    done = take_frames(pending, rest, FRAME_BYTES)
    assert done == [b"\x02" * 3 + rest]
    assert pending == bytearray()


def _trip_frame(monkeypatch: pytest.MonkeyPatch, frames: list[bytes]) -> int | None:
    """The frame (counted from 1) on which a three-hit gate tripped, or None."""
    gate, cancel = _watch_with(
        monkeypatch,
        frames,
        aec=EchoCanceller(AecTune(enabled=False)),
        tune=BargeTune(hit_frames=3, min_rms=200.0),
    )
    gate.watch(cancel)
    return gate.resume_at  # ScriptedMic is always at position 0


def test_barge_hits_decay_after_three_misses_and_then_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud, quiet = _pcm(500), _pcm(0)
    short_gap = [loud, loud] + [quiet] * (BARGE_MISS_DECAY_FRAMES - 1) + [loud]
    # A gap shorter than the decay keeps both hits, so one more trips the gate.
    assert _trip_frame(monkeypatch, short_gap) == len(short_gap)
    long_gap = [loud, loud] + [quiet] * BARGE_MISS_DECAY_FRAMES + [loud, loud, loud]
    # The third miss takes one hit back (not all of them), so it takes two more to trip.
    assert _trip_frame(monkeypatch, long_gap) == 2 + BARGE_MISS_DECAY_FRAMES + 2


def test_barge_watch_logs_a_mic_failure_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SpeechVad:
        def __init__(self, mode: int) -> None:
            self.mode = mode

    def boom(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        raise OSError("no device")

    install_vad(monkeypatch, SpeechVad)
    gate = BargeGate(
        mic=ScriptedMic(boom), tune=BargeTune(hit_frames=2, min_rms=1.0, bleed_delay_s=0)
    )
    cancel = threading.Event()
    gate.watch(cancel)
    assert not cancel.is_set()
    assert gate.captured == b""
    err = capsys.readouterr().err
    assert "no device" in err
    assert "off for this reply" in err
    assert isinstance(gate.failure, OSError)


def test_bleed_thread_skips_the_mic_when_already_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _vad(monkeypatch, lambda _frame: False)
    mic = ScriptedMic()
    gate = BargeGate(mic=mic, tune=BargeTune(bleed_delay_s=0))
    cancel = threading.Event()
    cancel.set()
    gate.start_after_bleed(cancel).join(timeout=1)
    assert mic.opened == 0
    cancel.clear()
    gate.start_after_bleed(cancel).join(timeout=1)
    assert mic.opened == 1
    assert gate.failure is None


def test_quiet_vad_frame_holds_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    end = _pcm(1)  # the one frame the detector calls silence
    _vad(monkeypatch, lambda frame: frame != end)
    tune = ListenTune(
        vad_aggressiveness=1,
        end_silence_frames=2,
        start_speech_frames=1,
        min_speech_rms=200.0,
        pre_pad_frames=20,
        min_voiced_frames=12,
    )
    hesitation = _pcm(0)  # too quiet to be speech, but the detector still hears a voice
    frames = [_pcm(500)] + [hesitation] * 5 + [end] * 2
    kept, ended, pulled = _stream(frames, tune)
    # The hesitation is not silence, so only the two silent frames end the turn,
    # and it is not counted as speech either.
    assert ended
    assert pulled == len(frames)
    assert _hits(kept) == [True] + [False] * 7


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
    _vad(monkeypatch, lambda frame: frame[:2] != b"\x00\x00")

    def fake_mic(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        yield from frames

    return ScriptedMic(fake_mic), ListenTune.from_env()


def test_record_utterance_writes_a_wav_for_a_held_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [_pcm(500), _pcm(500), _pcm(0), _pcm(0)]
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
    loud = _pcm(500)
    quiet = _pcm(0)
    mic, tune = _install_listen_fakes(monkeypatch, [loud, loud, quiet, quiet])
    stop = threading.Event()
    stop.set()
    assert record_utterance(mic, quit_requested=stop, tune=tune) is None


def test_record_utterance_drops_a_clip_that_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic, _tune = _install_listen_fakes(monkeypatch, [_pcm(0), _pcm(0)])
    assert record_utterance(mic) is None


def test_record_utterance_gives_up_when_speech_has_not_started_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = _pcm(500)
    quiet = _pcm(0)
    mic, tune = _install_listen_fakes(monkeypatch, [quiet] * 6 + [loud] * 6 + [quiet] * 2)
    assert record_utterance(mic, tune=tune, give_up_frames=3) is None
    # Once speech has started, the limit no longer applies.
    assert record_utterance(mic, tune=tune, give_up_frames=8) is not None
    assert record_utterance(mic, tune=tune) is not None


def _watch_with(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[bytes],
    *,
    aec: EchoCanceller,
    tune: BargeTune,
    vad_speech: bool = True,
) -> tuple[BargeGate, threading.Event]:
    _vad(monkeypatch, lambda _frame: vad_speech)
    return BargeGate(mic=ScriptedMic(frames, aec=aec), tune=tune), threading.Event()


def test_barge_watch_aligns_the_far_end_before_reading_the_mic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    order: list[str] = []
    monkeypatch.setattr(aec, "align", lambda: order.append("align"))
    _vad(monkeypatch, lambda _frame: False)

    def mic(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        order.append("mic")
        yield _pcm(0)

    gate = BargeGate(mic=ScriptedMic(mic, aec=aec), tune=BargeTune())
    gate.watch(threading.Event())
    assert order == ["align", "mic"]


def test_barge_trips_after_the_configured_number_of_loud_voiced_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    tune = BargeTune(hit_frames=3, min_rms=200.0)
    gate, cancel = _watch_with(monkeypatch, [_pcm(500)] * 5, aec=aec, tune=tune)
    gate.watch(cancel)
    assert cancel.is_set()
    assert gate.captured


def test_barge_ignores_loud_frames_the_vad_calls_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    tune = BargeTune(hit_frames=2, min_rms=200.0)
    gate, cancel = _watch_with(monkeypatch, [_pcm(900)] * 6, aec=aec, tune=tune, vad_speech=False)
    gate.watch(cancel)
    assert not cancel.is_set()


def test_barge_floor_follows_the_room_noise_above_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    tune = BargeTune(hit_frames=2, min_rms=100.0)
    # Without any quiet history a 150 voice is above the 100 seed and trips.
    gate, cancel = _watch_with(monkeypatch, [_pcm(150)] * 3, aec=aec, tune=tune)
    gate.watch(cancel)
    assert cancel.is_set()
    # After 40 frames of 90 hiss the floor is about 90 * 2.5, so 150 no longer trips.
    hissy = [*[_pcm(90)] * 40, *[_pcm(150)] * 3]
    gate2, cancel2 = _watch_with(monkeypatch, hissy, aec=aec, tune=tune)
    gate2.watch(cancel2)
    assert not cancel2.is_set()
    # A voice well above that floor still trips.
    loud = [*[_pcm(90)] * 40, *[_pcm(600)] * 3]
    gate3, cancel3 = _watch_with(monkeypatch, loud, aec=aec, tune=tune)
    gate3.watch(cancel3)
    assert cancel3.is_set()


class _ScriptedFarEnd(EchoCanceller):
    """No echo cancellation, and the speaker plays while ``playing`` says so."""

    def __init__(self) -> None:
        super().__init__(AecTune(enabled=False))
        self.playing = True

    def far_end_playing(self, window_s: float = 0.0) -> bool:
        del window_s
        return self.playing


def test_speech_between_the_floor_and_the_boosted_need_does_not_raise_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    far = _ScriptedFarEnd()

    def mic(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        # While the speaker plays, with AEC off, the need is 220 x 2.2 = 484: speech at 300
        # is under it but over the 220 floor, so it is not the room's hiss.
        yield from [_pcm(300)] * 40
        far.playing = False
        yield from [_pcm(400)] * 3

    _vad(monkeypatch, lambda _frame: True)
    gate = BargeGate(
        mic=ScriptedMic(mic, aec=far), tune=BargeTune(hit_frames=3, min_rms=220.0, playing_gain=2.2)
    )
    cancel = threading.Event()
    gate.watch(cancel)
    # Had those 40 frames taught the floor (300 x 2.5), 400 could not trip it afterwards.
    assert cancel.is_set()


def test_a_gate_without_an_aec_runs_without_one() -> None:
    gate = BargeGate(mic=ScriptedMic())
    assert gate.aec.available() is False
    assert _bleed_of(gate) == gate.tune.bleed_delay_s


def test_with_echo_cancellation_the_aec_bleed_delay_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gate(aec_bleed: float, barge_bleed: float = DEFAULT_BLEED_DELAY_S) -> BargeGate:
        aec = EchoCanceller(AecTune(enabled=True))
        monkeypatch.setattr(aec, "available", lambda: True)
        return BargeGate(
            mic=ScriptedMic(aec=aec),
            tune=BargeTune(bleed_delay_s=barge_bleed, aec_bleed_delay_s=aec_bleed),
        )

    assert _bleed_of(gate(0.3)) == 0.3
    assert _bleed_of(gate(1.5)) == 1.5
    assert _bleed_of(gate(0.0)) == 0.0
    assert _bleed_of(gate(0.3, barge_bleed=1.5)) == 0.3
    with pytest.raises(ValueError, match="bleed_delay_s"):
        BargeTune(aec_bleed_delay_s=-1.0)


def test_the_bleed_wait_follows_whether_echo_cancellation_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    off = EchoCanceller(AecTune(enabled=False))
    assert _bleed_of(BargeGate(mic=ScriptedMic(aec=off), tune=BargeTune(bleed_delay_s=0.9))) == 0.9
    on = EchoCanceller(AecTune())
    monkeypatch.setattr(on, "load", object)  # a processor that loaded
    assert (
        _bleed_of(BargeGate(mic=ScriptedMic(aec=on), tune=BargeTune(bleed_delay_s=0.9)))
        == DEFAULT_AEC_BLEED_DELAY_S
    )


def test_a_small_floor_window_still_warms_up() -> None:
    from fish_audio_suite_voice.floor import AdaptiveFloor

    floor = AdaptiveFloor(100.0, window=10)
    assert floor.value() == 100.0
    for _ in range(10):
        floor.observe(50.0, quiet=True)
    assert floor.value() == 125.0
    # A zero window cannot hold a sample, so it is raised to one.
    tiny = AdaptiveFloor(100.0, window=0)
    tiny.observe(60.0, quiet=True)
    assert tiny.value() == 150.0


class _BrokenAec(EchoCanceller):
    """An echo canceller whose processor fails as the watcher asks about it."""

    def available(self) -> bool:
        raise RuntimeError("aec failed to load")


def test_a_setup_failure_in_the_watcher_thread_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = BargeGate(
        mic=ScriptedMic(aec=_BrokenAec(AecTune.off())),
        tune=BargeTune(hit_frames=2, min_rms=1.0, bleed_delay_s=0),
    )
    cancel = threading.Event()
    thread = gate.start_after_bleed(cancel)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(gate.failure, RuntimeError)
    assert not cancel.is_set()
    err = capsys.readouterr().err
    assert "aec failed to load" in err
    assert "off for this reply" in err


def test_a_reply_that_starts_muted_is_not_watched_and_the_mic_is_not_opened() -> None:
    mic = ScriptedMic()
    gate = BargeGate(mic=mic, tune=BargeTune(bleed_delay_s=0), muted=lambda: True)
    gate.start_after_bleed(threading.Event()).join(timeout=1)
    assert mic.opened == 0
    assert gate.failure is None


def test_muting_during_a_reply_ends_the_watch_without_interrupting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    muted = threading.Event()
    read: list[int] = []

    def mic(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        assert stop is not None
        while not stop.is_set():  # the stream closes as soon as the stop flag is set
            read.append(1)
            if len(read) == 2:
                muted.set()
            yield _pcm(0)

    _vad(monkeypatch, lambda _frame: True)
    aec = EchoCanceller(AecTune(enabled=False))
    gate = BargeGate(mic=ScriptedMic(mic, aec=aec), tune=BargeTune(), muted=muted.is_set)
    cancel = threading.Event()
    gate.watch(cancel)
    assert len(read) == 2  # it stopped reading when muted
    assert not cancel.is_set()  # and the reply was not interrupted


def test_a_gate_with_no_mute_switch_is_always_watching(monkeypatch: pytest.MonkeyPatch) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    gate, cancel = _watch_with(
        monkeypatch, [_pcm(500)] * 5, aec=aec, tune=BargeTune(hit_frames=3, min_rms=200.0)
    )
    gate.watch(cancel)
    assert cancel.is_set()


def test_a_trip_records_where_the_next_listen_reads_on_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Positioned(ScriptedMic):
        @property
        def position(self) -> int:
            return 100

    _vad(monkeypatch, lambda _frame: True)
    aec = EchoCanceller(AecTune(enabled=False))
    source = _Positioned([_pcm(500)] * 5, aec=aec)
    gate = BargeGate(mic=source, tune=BargeTune(hit_frames=3, min_rms=200.0))
    cancel = threading.Event()
    gate.watch(cancel)
    assert cancel.is_set()
    assert source.since == 100  # read from where the mic was when the watch began
    assert gate.resume_at == 103  # the trip came on the third frame


@pytest.mark.parametrize(("gate_rms", "credited"), [(100.0, 10), (DEFAULT_BARGE_RMS, 0)])
def test_a_barge_clip_is_scored_at_the_level_its_own_gate_admitted_it(
    monkeypatch: pytest.MonkeyPatch, gate_rms: float, credited: int
) -> None:
    # FISH_VOICE_BARGE_RMS=100 let a soft interruption at 150 stop the reply. Scored at the
    # default 220 instead, none of it counted and the words that stopped the reply were lost.
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=DEFAULT_MIN_SPEECH_RMS)
    clip = BargeIn(_pcm(150) * 10, None, gate_rms)
    kept, _ended, _pulled = _stream([_pcm(0)], tune, carried=clip)
    assert sum(_hits(kept)) == credited
    wav = record_utterance(ScriptedMic([_pcm(0)]), carried=clip, tune=tune)
    assert (wav is not None) is (credited > 0)


def test_a_fallback_clip_goes_on_as_ordinary_speech_with_no_barge_in_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=DEFAULT_MIN_SPEECH_RMS)
    voiced = (_pcm(600), True)
    unvoiced = (_pcm(600), False)
    # Speech already started, so the first words are kept and the listen reads on from
    # the position after them; the mic alone never starts a turn here.
    mic = ScriptedMic([_pcm(0)])
    full = StreamFallback((voiced,) * DEFAULT_MIN_VOICED_FRAMES, 7)
    wav = record_utterance(mic, carried=full, tune=tune)
    assert wav is not None
    assert _wav_frames(wav)[:DEFAULT_MIN_VOICED_FRAMES] == [voiced[0]] * DEFAULT_MIN_VOICED_FRAMES
    assert mic.since == 7
    # Each frame counts as the listen gate judged it when it was captured, and a fallback
    # is not an interruption: one frame short of the voice any turn needs is not a turn.
    short = StreamFallback((voiced,) * (DEFAULT_MIN_VOICED_FRAMES - 1) + (unvoiced,), 7)
    assert record_utterance(ScriptedMic([_pcm(0)]), carried=short, tune=tune) is None


def test_a_fallback_from_a_barged_turn_keeps_the_credit_of_the_barge_in_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tune = _barge_carried_listen(monkeypatch, min_speech_rms=DEFAULT_MIN_SPEECH_RMS)
    barge = BargeIn(_pcm(600) * 4, 3, DEFAULT_BARGE_RMS)
    frames = ((_pcm(600), True),) * 4 + ((_pcm(600), True), (_pcm(5), False))
    barged = StreamFallback(frames, 9, barge=barge)
    # 5 voiced frames: enough after a barge-in, too few for a turn that began on its own.
    assert record_utterance(ScriptedMic([_pcm(0)]), carried=barged, tune=tune) is not None
    alone = StreamFallback(frames, 9)
    assert record_utterance(ScriptedMic([_pcm(0)]), carried=alone, tune=tune) is None


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


class _CountingVad:
    """Calls every frame speech and counts how often it was asked about each one."""

    def __init__(self) -> None:
        self.asked: collections.Counter[bytes] = collections.Counter()

    def is_speech(self, frame: bytes, rate: int) -> bool:
        del rate
        self.asked[frame] += 1
        return True


def _counting_engine(monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[_CountingVad]]:
    opened: list[_CountingVad] = []

    def make(strictness: int) -> _CountingVad:
        del strictness
        opened.append(_CountingVad())
        return opened[-1]

    plugin = types.ModuleType("counting_vad_plugin")
    vars(plugin)["make"] = make
    monkeypatch.setitem(sys.modules, "counting_vad_plugin", plugin)
    return "counting_vad_plugin:make", opened


def test_each_gate_asks_its_detector_about_a_frame_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Distinct levels, so each frame is its own key: quiet ones under every floor, loud ones over.
    quiet = [_pcm(n) for n in range(1, 31)]
    loud = [_pcm(1000 + n) for n in range(4)]
    frames = [*quiet, *loud, *quiet[:5]]
    engine, opened = _counting_engine(monkeypatch)
    listen = ListenTune(vad_engine=engine, end_silence_frames=40)
    record_utterance(ScriptedMic(frames), tune=listen)
    cooldown = _CountingVad()
    cooldown_report(frames, listen, cooldown)
    gate = BargeGate(
        mic=ScriptedMic(frames, aec=EchoCanceller(AecTune(enabled=False))),
        tune=BargeTune(hit_frames=len(frames) + 1),
        listen=listen,
    )
    gate.watch(threading.Event())
    listened, barged = opened
    # A listen and the cooldown log hear every frame once, as a stateful detector needs.
    assert listened.asked == collections.Counter(frames)
    assert cooldown.asked == collections.Counter(frames)
    # Barge-in asks only about frames at its level, once each: quiet frames are never heard.
    assert barged.asked == collections.Counter(loud)
