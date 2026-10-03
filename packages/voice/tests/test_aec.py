from __future__ import annotations

import numpy as np
import pytest

from fish_audio_suite_voice.aec import EchoCanceller, FarEndTap, resample_int16
from fish_audio_suite_voice.floor import AdaptiveFloor
from fish_audio_suite_voice.playback import SounddeviceSink, dac_slice_bytes, iter_pcm_slices
from fish_audio_suite_voice.tune import DEFAULT_AEC_BLEED_S, AecTune


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


def test_clean_mic_passthrough_when_aec_off() -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    near = b"\x01\x00" * 480
    assert aec.clean(near) == near
    assert not aec.available()


def test_effective_bleed_without_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    off = EchoCanceller(AecTune(enabled=False))
    assert off.effective_bleed_s(0.9) == 0.9
    assert off.effective_bleed_s(-1) == 0.0
    on = EchoCanceller(AecTune(bleed_delay_s=DEFAULT_AEC_BLEED_S))
    monkeypatch.setattr(on, "load", object)
    assert on.effective_bleed_s(0.9) == DEFAULT_AEC_BLEED_S
    negative = EchoCanceller(AecTune(bleed_delay_s=-1.0))
    monkeypatch.setattr(negative, "load", object)
    assert negative.effective_bleed_s(0.9) == DEFAULT_AEC_BLEED_S


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


def test_sounddevice_sink_taps_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    taps: list[tuple[bytes, int]] = []
    cleared = {"n": 0}
    aec = EchoCanceller()
    monkeypatch.setattr(aec, "tap_playback", lambda pcm, sr: taps.append((pcm, sr)))
    monkeypatch.setattr(aec, "clear", lambda: cleared.__setitem__("n", cleared["n"] + 1))
    sink = SounddeviceSink(aec=aec)

    class Fake:
        def write(self, chunk: bytes) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    sink._stream = Fake()
    pcm = b"\x00\x00" * 3000
    sink.write(pcm)
    step = dac_slice_bytes(44100)
    assert taps
    assert all(len(p) <= step for p, _sr in taps)
    assert sum(len(p) for p, _sr in taps) == len(pcm)
    sink.finish()
    assert cleared["n"] == 1


def _loud(n_samples: int) -> bytes:
    return (2_000).to_bytes(2, "little", signed=True) * n_samples


def test_clean_mic_ignores_a_bad_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    near = _loud(8)
    calls: list[str] = []

    class Processor:
        def process(self, near_a: np.ndarray, far_a: np.ndarray) -> np.ndarray:
            calls.append("process")
            if len(calls) == 1:
                raise RuntimeError("aec down")
            return near_a[:2]

    aec = EchoCanceller()
    monkeypatch.setattr(aec, "load", Processor)
    monkeypatch.setattr(aec.tap, "pop", lambda _n: _loud(8))
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

    aec = EchoCanceller(AecTune(wet=0.5))
    monkeypatch.setattr(aec, "load", Processor)
    monkeypatch.setattr(aec.tap, "pop", lambda _n: _loud(8))
    mixed = np.frombuffer(aec.clean(near), dtype=np.int16)
    assert mixed.tolist() == [1_000] * 8


def test_clean_mic_skips_a_quiet_far_end(monkeypatch: pytest.MonkeyPatch) -> None:
    near = _loud(8)
    called = {"n": 0}

    class Processor:
        def process(self, near_a: np.ndarray, far_a: np.ndarray) -> np.ndarray:
            called["n"] += 1
            return np.zeros_like(near_a)

    aec = EchoCanceller()
    monkeypatch.setattr(aec, "load", Processor)
    monkeypatch.setattr(aec.tap, "pop", lambda _n: b"\x00\x00" * 8)
    assert aec.clean(near) == near
    assert called["n"] == 0


def test_sounddevice_sink_rejoins_odd_pcm_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[bytes] = []
    sink = SounddeviceSink()

    class Fake:
        def write(self, chunk: bytes) -> None:
            written.append(chunk)

    sink._stream = Fake()
    sink.write(b"\x01\x02\x03")
    sink.write(b"\x04\x05\x06")
    assert b"".join(written) == b"\x01\x02\x03\x04\x05\x06"
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
