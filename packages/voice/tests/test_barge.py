from __future__ import annotations

import pytest

from fish_audio_suite_voice.barge import (
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BLEED_DELAY_S,
    BargeGate,
    post_speak_cooldown_s,
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
