from __future__ import annotations

import pytest

from fish_audio_suite_voice.aec import (
    DEFAULT_AEC_BLEED_S,
    AdaptiveFloor,
    FarEndTap,
    clean_mic_frame,
    effective_bleed_s,
    resample_int16,
)
from fish_audio_suite_voice.playback import SounddeviceSink


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


def test_clean_mic_passthrough_when_aec_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_AEC", "0")
    near = b"\x01\x00" * 480
    assert clean_mic_frame(near) == near


def test_effective_bleed_without_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_AEC", "0")
    assert effective_bleed_s(0.9) == 0.9
    monkeypatch.setenv("FISH_VOICE_AEC", "1")

    def fake_proc() -> object:
        return object()

    monkeypatch.setattr("fish_audio_suite_voice.aec.load_processor", fake_proc)
    monkeypatch.setenv("FISH_VOICE_AEC_BLEED", str(DEFAULT_AEC_BLEED_S))
    assert effective_bleed_s(0.9) == DEFAULT_AEC_BLEED_S


def test_adaptive_floor_stays_default_until_window() -> None:
    floor = AdaptiveFloor(200.0, window=80)
    for _ in range(10):
        floor.observe(40.0, quiet=True)
    assert floor.value() == 200.0
    for _ in range(30):
        floor.observe(40.0, quiet=True)
    assert 80.0 <= floor.value() <= 200.0


def test_sounddevice_sink_taps_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    taps: list[tuple[bytes, int]] = []
    cleared = {"n": 0}
    monkeypatch.setattr(
        "fish_audio_suite_voice.playback.tap_playback",
        lambda pcm, sr: taps.append((pcm, sr)),
    )
    monkeypatch.setattr(
        "fish_audio_suite_voice.playback.tap_clear",
        lambda: cleared.__setitem__("n", cleared["n"] + 1),
    )
    sink = SounddeviceSink()

    class Fake:
        def write(self, chunk: bytes) -> None:
            return None

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    sink._stream = Fake()
    pcm = b"\x00\x00" * 8
    sink.write(pcm)
    assert taps == [(pcm, 44100)]
    sink.finish()
    assert cleared["n"] == 1
