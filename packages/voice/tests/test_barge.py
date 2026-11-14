from __future__ import annotations

import collections.abc
import sys
import threading
import types

import pytest
from voice_fakes import ScriptedMic, install_detector, install_vad, level_frame

from fish_audio_suite_voice.asr.result import BargeIn
from fish_audio_suite_voice.capture.aec import EchoCanceller
from fish_audio_suite_voice.capture.barge_in import (
    BARGE_MISS_DECAY_FRAMES,
    BargeGate,
    barge_rms_need,
)
from fish_audio_suite_voice.capture.listen import (
    cooldown_report,
    record_utterance,
)
from fish_audio_suite_voice.capture.pcm import FRAME_BYTES
from fish_audio_suite_voice.config.tune import (
    DEFAULT_AEC_BLEED_DELAY_S,
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_PLAYING_GAIN,
    DEFAULT_BARGE_RMS,
    DEFAULT_BLEED_DELAY_S,
    DEFAULT_POST_SPEAK_COOLDOWN_S,
    AecTune,
    BargeTune,
    ListenTune,
)
from fish_audio_suite_voice.session.cancel import StopFlag


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
    frames = [level_frame(n) for n in range(1, 23)] + [level_frame(1000 + n) for n in range(3)]
    gate, cancel = _watch_with(
        monkeypatch,
        frames,
        aec=EchoCanceller(AecTune(enabled=False)),
        tune=BargeTune(hit_frames=3, min_rms=200.0),
    )
    assert gate.clip() is None  # nothing to carry before a trip
    gate.watch(cancel)
    assert cancel.is_set()
    # The last 20 frames (600 ms) are kept for the next listen, the tripping ones included.
    assert gate.captured == b"".join(frames[-20:])
    # What the next listen starts from: the clip, where to read on, and this gate's level.
    assert gate.clip() == BargeIn(gate.captured, gate.resume_at, 200.0)


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
    loud, quiet = level_frame(500), level_frame(0)
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
    install_detector(monkeypatch, lambda _frame: False)
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


def _watch_with(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[bytes],
    *,
    aec: EchoCanceller,
    tune: BargeTune,
    vad_speech: bool = True,
) -> tuple[BargeGate, threading.Event]:
    install_detector(monkeypatch, lambda _frame: vad_speech)
    return BargeGate(mic=ScriptedMic(frames, aec=aec), tune=tune), threading.Event()


def test_barge_watch_aligns_the_far_end_before_reading_the_mic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    order: list[str] = []
    monkeypatch.setattr(aec, "align", lambda: order.append("align"))
    install_detector(monkeypatch, lambda _frame: False)

    def mic(stop: StopFlag | None) -> collections.abc.Iterator[bytes]:
        del stop
        order.append("mic")
        yield level_frame(0)

    gate = BargeGate(mic=ScriptedMic(mic, aec=aec), tune=BargeTune())
    gate.watch(threading.Event())
    assert order == ["align", "mic"]


def test_barge_trips_after_the_configured_number_of_loud_voiced_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    tune = BargeTune(hit_frames=3, min_rms=200.0)
    gate, cancel = _watch_with(monkeypatch, [level_frame(500)] * 5, aec=aec, tune=tune)
    gate.watch(cancel)
    assert cancel.is_set()
    assert gate.captured


def test_barge_ignores_loud_frames_the_vad_calls_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    tune = BargeTune(hit_frames=2, min_rms=200.0)
    gate, cancel = _watch_with(
        monkeypatch, [level_frame(900)] * 6, aec=aec, tune=tune, vad_speech=False
    )
    gate.watch(cancel)
    assert not cancel.is_set()


def test_barge_floor_follows_the_room_noise_above_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    tune = BargeTune(hit_frames=2, min_rms=100.0)
    # Without any quiet history a 150 voice is above the 100 seed and trips.
    gate, cancel = _watch_with(monkeypatch, [level_frame(150)] * 3, aec=aec, tune=tune)
    gate.watch(cancel)
    assert cancel.is_set()
    # After 40 frames of 90 hiss the floor is about 90 * 2.5, so 150 no longer trips.
    hissy = [*[level_frame(90)] * 40, *[level_frame(150)] * 3]
    gate2, cancel2 = _watch_with(monkeypatch, hissy, aec=aec, tune=tune)
    gate2.watch(cancel2)
    assert not cancel2.is_set()
    # A voice well above that floor still trips.
    loud = [*[level_frame(90)] * 40, *[level_frame(600)] * 3]
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
        yield from [level_frame(300)] * 40
        far.playing = False
        yield from [level_frame(400)] * 3

    install_detector(monkeypatch, lambda _frame: True)
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
    from fish_audio_suite_voice.capture.energy_floor import AdaptiveFloor

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
            yield level_frame(0)

    install_detector(monkeypatch, lambda _frame: True)
    aec = EchoCanceller(AecTune(enabled=False))
    gate = BargeGate(mic=ScriptedMic(mic, aec=aec), tune=BargeTune(), muted=muted.is_set)
    cancel = threading.Event()
    gate.watch(cancel)
    assert len(read) == 2  # it stopped reading when muted
    assert not cancel.is_set()  # and the reply was not interrupted


def test_a_gate_with_no_mute_switch_is_always_watching(monkeypatch: pytest.MonkeyPatch) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    gate, cancel = _watch_with(
        monkeypatch, [level_frame(500)] * 5, aec=aec, tune=BargeTune(hit_frames=3, min_rms=200.0)
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

    install_detector(monkeypatch, lambda _frame: True)
    aec = EchoCanceller(AecTune(enabled=False))
    source = _Positioned([level_frame(500)] * 5, aec=aec)
    gate = BargeGate(mic=source, tune=BargeTune(hit_frames=3, min_rms=200.0))
    cancel = threading.Event()
    gate.watch(cancel)
    assert cancel.is_set()
    assert source.since == 100  # read from where the mic was when the watch began
    assert gate.resume_at == 103  # the trip came on the third frame


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
    quiet = [level_frame(n) for n in range(1, 31)]
    loud = [level_frame(1000 + n) for n in range(4)]
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
