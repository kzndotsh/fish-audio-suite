from __future__ import annotations

import sys
import wave
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from wire_helpers import FakeOutputStream, open_sounddevice_sink

from fish_audio_suite_voice.aec import EchoCanceller, FarEndTap, resample_int16
from fish_audio_suite_voice.floor import AdaptiveFloor
from fish_audio_suite_voice.pcm import pcm_rms
from fish_audio_suite_voice.playback import (
    SounddeviceSink,
    dac_slice_bytes,
    iter_pcm_slices,
    make_sink,
)
from fish_audio_suite_voice.tune import AecTune


def test_resample_44100_to_16000_length() -> None:
    n_src = 441
    pcm = b"\x00\x10" * n_src
    out = resample_int16(pcm, 44100, 16000)
    assert len(out) == 160 * 2
    assert resample_int16(pcm, 16000, 16000) == pcm


def test_far_end_pop_empty_until_enough() -> None:
    tap = FarEndTap()
    tap.push(b"\x00\x00" * 80, 16000)
    assert tap.pop(480 * 2) == b""
    tap.push(b"\x10\x00" * 400, 16000)
    chunk = tap.pop(100 * 2)
    assert len(chunk) == 200


def test_clean_mic_passthrough_when_every_stage_is_off() -> None:
    aec = EchoCanceller(AecTune.off())
    near = b"\x01\x00" * 480
    assert aec.clean(near) == near
    assert not aec.available()


def test_echo_canceller_instances_do_not_share_a_tap() -> None:
    first = EchoCanceller()
    second = EchoCanceller()
    first.tap_playback(b"\x00\x10" * 800, 16000)
    assert first.far_end_playing(0.0)
    assert not second.far_end_playing(0.0)


def test_disabled_aec_still_marks_playback() -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    aec.tap_playback(b"\x00\x10" * 1600, 16000)
    assert aec.far_end_playing(0.0)
    assert aec.tap.pop(2) == b""


def test_keep_last_drops_old_far_end() -> None:
    tap = FarEndTap()
    tap.push(b"\x01\x00" * 16000, 16000)
    tap.keep_last(0.25)
    assert len(tap.pop(4000 * 2)) == 8000
    assert tap.pop(2) == b""


def test_align_keeps_latency_plus_margin() -> None:
    aec = EchoCanceller()
    aec.tap.push(b"\x01\x00" * 16000, 16000)
    aec.output_latency_s = 0.2
    aec.align()
    kept = aec.tap.pop(16000 * 2)
    assert kept == b""
    assert len(aec.tap.pop(4160 * 2)) == 8320


def test_adaptive_floor_stays_default_until_window() -> None:
    floor = AdaptiveFloor(200.0, window=80)
    for _ in range(10):
        floor.observe(40.0, quiet=True)
    assert floor.value() == 200.0
    for _ in range(30):
        floor.observe(40.0, quiet=True)
    assert floor.value() == 200.0
    for _ in range(30):
        floor.observe(200.0, quiet=True)
    assert floor.value() >= 200.0
    loud_seed = AdaptiveFloor(600.0, window=30)
    for _ in range(25):
        loud_seed.observe(10.0, quiet=True)
    assert loud_seed.value() == 600.0
    capped = AdaptiveFloor(200.0, window=30)
    for _ in range(25):
        capped.observe(400.0, quiet=True)
    assert capped.value() == 450.0


def test_far_end_playing_covers_marked_duration() -> None:
    tap = FarEndTap()
    assert not tap.playing_recently(0.0)
    tap.mark_playing(0.3)
    assert tap.playing_recently(0.0)


def test_iter_pcm_slices_caps_30ms() -> None:
    step = dac_slice_bytes(44100)
    assert step == 2646
    pcm = b"\x00\x00" * 3000
    parts = list(iter_pcm_slices(pcm, step))
    assert sum(len(p) for p in parts) == len(pcm)
    assert all(len(p) <= step for p in parts)


class _RecordingAec(EchoCanceller):
    """Records what the sink taps as far end and how often it is cleared."""

    def __init__(self) -> None:
        super().__init__()
        self.taps: list[tuple[bytes, int]] = []
        self.cleared = 0

    def tap_playback(self, pcm: bytes, sample_rate: int) -> None:
        self.taps.append((pcm, sample_rate))

    def clear(self) -> None:
        self.cleared += 1


def test_sounddevice_sink_taps_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    aec = _RecordingAec()
    stream = FakeOutputStream()
    sink = open_sounddevice_sink(monkeypatch, stream, aec=aec)
    assert aec.cleared == 1  # a new turn starts with no far end
    pcm = b"\x00\x00" * 3000
    sink.write(pcm)
    step = dac_slice_bytes(44100)
    # Each slice is tapped as it goes to the device, so the reference is what plays.
    assert [p for p, _sr in aec.taps] == stream.written
    assert all(len(p) <= step for p, _sr in aec.taps)
    assert sum(len(p) for p, _sr in aec.taps) == len(pcm)
    sink.finish()
    assert aec.cleared == 2


def _loud(n_samples: int) -> bytes:
    return (2_000).to_bytes(2, "little", signed=True) * n_samples


def _install_processor(monkeypatch: pytest.MonkeyPatch, make: Callable[..., object]) -> None:
    """Make ``pywebrtc_audio.AudioProcessor`` build what ``make`` returns."""
    module = ModuleType("pywebrtc_audio")
    vars(module)["AudioProcessor"] = make
    monkeypatch.setitem(sys.modules, "pywebrtc_audio", module)


def test_clean_mic_ignores_a_bad_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    near = _loud(8)
    calls: list[str] = []

    class Processor:
        def process(self, near_a: np.ndarray, far_a: np.ndarray) -> np.ndarray:
            calls.append("process")
            if len(calls) == 1:
                raise RuntimeError("aec down")
            return near_a[:2]

    _install_processor(monkeypatch, lambda **_kwargs: Processor())
    aec = EchoCanceller()
    aec.tap_playback(_loud(16), 16000)  # the speaker is playing
    assert aec.clean(near) == near
    assert aec.clean(near) == near
    assert calls == ["process", "process"]


def test_clean_mic_mixes_when_the_processor_returns_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    near = _loud(8)

    class Processor:
        def process(self, near_a: np.ndarray, far_a: np.ndarray) -> np.ndarray:
            return np.zeros_like(near_a)

    _install_processor(monkeypatch, lambda **_kwargs: Processor())
    aec = EchoCanceller(AecTune(wet=0.5))
    aec.tap_playback(_loud(8), 16000)
    mixed = np.frombuffer(aec.clean(near), dtype=np.int16)
    assert mixed.tolist() == [1_000] * 8


def test_with_the_speaker_quiet_every_frame_is_still_processed_against_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    near = _loud(8)
    fars: list[list[int]] = []

    class Processor:
        def process(self, near_a: np.ndarray, far_a: np.ndarray) -> np.ndarray:
            fars.append(far_a.tolist())
            return near_a * 2  # the gain stage at work

    _install_processor(monkeypatch, lambda **_kwargs: Processor())
    aec = EchoCanceller(AecTune(wet=0.5))
    aec.tap_playback(b"\x05\x00" * 8, 16000)  # under the silence floor
    # No echo to cancel, so none of the raw mic is mixed back in.
    assert np.frombuffer(aec.clean(near), dtype=np.int16).tolist() == [4_000] * 8
    assert fars == [[0] * 8]


def test_with_echo_cancellation_off_gain_and_noise_suppression_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    near = _loud(8)
    calls: list[tuple[int, ...]] = []

    class Processor:
        def process(self, *arrays: np.ndarray) -> np.ndarray:
            calls.append(tuple(a.size for a in arrays))
            return arrays[0] * 2

    _install_processor(monkeypatch, lambda **_kwargs: Processor())
    aec = EchoCanceller(AecTune(enabled=False))
    assert np.frombuffer(aec.clean(near), dtype=np.int16).tolist() == [4_000] * 8
    assert calls == [(8,)]  # no far end at all
    assert not aec.available()  # so barge-in keeps the long bleed delay


def test_sounddevice_sink_rejoins_odd_pcm_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeOutputStream()
    sink = open_sounddevice_sink(monkeypatch, stream)
    sink.write(b"\x01\x02\x03")
    sink.write(b"\x04\x05\x06")
    assert b"".join(stream.written) == b"\x01\x02\x03\x04\x05\x06"
    assert sink.bytes_played() == 6


def test_sounddevice_sink_records_output_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    aec = EchoCanceller()

    class Stream:
        latency = 0.04

        def start(self) -> None:
            return None

    class Sd:
        @staticmethod
        def RawOutputStream(**_kwargs: object) -> Stream:  # noqa: N802
            return Stream()

    monkeypatch.setattr("fish_audio_suite_voice.playback.load_sounddevice", lambda: Sd)
    sink = SounddeviceSink(aec=aec)
    sink.start()
    assert aec.output_latency_s == 0.04


def test_the_sounddevice_sink_asks_for_the_buffer_size_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[dict[str, object]] = []

    class Stream:
        latency = 0.2

        def start(self) -> None:
            return None

    class Sd:
        @staticmethod
        def RawOutputStream(**kwargs: object) -> Stream:  # noqa: N802
            asked.append(kwargs)
            return Stream()

    monkeypatch.setattr("fish_audio_suite_voice.playback.load_sounddevice", lambda: Sd)
    for latency in ("low", "high"):
        sink = make_sink("sounddevice", latency=latency)
        assert isinstance(sink, SounddeviceSink)
        sink.start()
    assert [call["latency"] for call in asked] == ["low", "high"]


def _quiet_speech_frames() -> list[bytes]:
    # Three seconds of espeak-ng speech at about -50 dBFS, a quiet talker.
    with wave.open(str(Path(__file__).parent / "data" / "quiet_speech.wav"), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
    return [pcm[i : i + 960] for i in range(0, len(pcm) - 959, 960)]


def _level(frames: list[bytes]) -> float:
    return pcm_rms(b"".join(frames))


def test_the_default_cleanup_lifts_a_quiet_voice_and_off_leaves_it_alone() -> None:
    pytest.importorskip("pywebrtc_audio")
    frames = _quiet_speech_frames()
    untouched = EchoCanceller(AecTune.off())
    settle = 30  # the gain rises over the first second
    for lifted in (EchoCanceller(), EchoCanceller(AecTune(enabled=False))):
        out = [lifted.clean(frame) for frame in frames][settle:]
        assert _level(out) > 4 * _level(frames[settle:])  # more than 12 dB louder
    assert [untouched.clean(frame) for frame in frames] == frames


@pytest.mark.parametrize(("raw", "expected"), [("12", 12), ("0", 0), ("10", 18), ("x", 18)])
def test_noise_suppression_takes_only_the_processor_steps(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("FISH_VOICE_NOISE_SUPPRESSION", raw)
    assert AecTune.from_env().noise_suppression_db == expected


def test_gain_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_AGC", "0")
    assert AecTune.from_env().auto_gain is False


def test_a_processor_that_fails_to_start_leaves_the_mic_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(**_kwargs: object) -> object:
        raise RuntimeError("bad native build")

    _install_processor(monkeypatch, broken)
    aec = EchoCanceller()
    near = _loud(8)
    assert aec.clean(near) == near
    assert aec.load() is None
