from __future__ import annotations

import collections

import pytest

from fish_audio_suite_voice.barge import (
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BLEED_DELAY_S,
    BargeGate,
    barge_rms_need,
    listen_reject_reason,
    post_speak_cooldown_s,
    spike_start_allowed,
    start_frames_needed,
    start_hit,
    trailing_start_hits,
)


def test_barge_gate_reads_env_at_construct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "1.5")
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "9")
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "500")
    gate = BargeGate()
    assert gate.bleed_delay_s == 1.5
    assert gate.hit_frames == 9
    assert gate.min_rms == 500.0


def test_barge_gate_explicit_kwargs_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "9.9")
    gate = BargeGate(bleed_delay_s=0.2, hit_frames=3, min_rms=10.0)
    assert gate.bleed_delay_s == 0.2
    assert gate.hit_frames == 3
    assert gate.min_rms == 10.0


def test_barge_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_BLEED_DELAY", raising=False)
    monkeypatch.delenv("FISH_VOICE_BARGE_FRAMES", raising=False)
    monkeypatch.delenv("FISH_VOICE_COOLDOWN", raising=False)
    gate = BargeGate()
    assert gate.bleed_delay_s == DEFAULT_BLEED_DELAY_S
    assert gate.hit_frames == DEFAULT_BARGE_HIT_FRAMES
    assert post_speak_cooldown_s() == 0.8
    monkeypatch.setenv("FISH_VOICE_COOLDOWN", "0.3")
    assert post_speak_cooldown_s() == 0.3


def test_barge_over_speaker_raises_need() -> None:
    assert barge_rms_need(220.0, far_playing=False, over=2.2) == 220.0
    assert barge_rms_need(220.0, far_playing=True, over=2.2, aec_on=False) == pytest.approx(
        220.0 * 2.2
    )
    assert barge_rms_need(220.0, far_playing=True, over=2.2, aec_on=True) == 220.0


def test_start_hit_requires_vad_and_full_floor() -> None:
    assert not start_hit(200.0, False, 200.0)
    assert start_hit(200.0, True, 200.0)
    assert not start_hit(121.0, True, 200.0)
    assert not start_hit(111.0, True, 200.0)


def test_start_frames_needed_raises_on_spike() -> None:
    assert start_frames_needed(200.0, 200.0, 4) == 4
    assert start_frames_needed(964.0, 200.0, 4) == 10


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
    assert (
        listen_reject_reason(
            voiced_frames=48,
            speech_hits=8,
            peak_rms=346.0,
            min_voiced=12,
            min_speech_rms=200.0,
        )
        == "too_little_voice"
    )
    assert (
        listen_reject_reason(
            voiced_frames=54,
            speech_hits=14,
            peak_rms=2220.0,
            min_voiced=12,
            min_speech_rms=200.0,
        )
        == "impulse"
    )
    assert (
        listen_reject_reason(
            voiced_frames=80,
            speech_hits=18,
            peak_rms=1047.0,
            min_voiced=12,
            min_speech_rms=200.0,
        )
        == "impulse"
    )
    assert (
        listen_reject_reason(
            voiced_frames=80,
            speech_hits=20,
            peak_rms=400.0,
            min_voiced=12,
            min_speech_rms=200.0,
        )
        is None
    )
    assert (
        listen_reject_reason(
            voiced_frames=120,
            speech_hits=68,
            peak_rms=1486.0,
            min_voiced=12,
            min_speech_rms=200.0,
        )
        is None
    )
    assert (
        listen_reject_reason(
            voiced_frames=2,
            speech_hits=2,
            peak_rms=100.0,
            min_voiced=12,
            min_speech_rms=200.0,
        )
        == "too_short"
    )
