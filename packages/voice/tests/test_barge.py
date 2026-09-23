from __future__ import annotations

import collections
import io
import wave

import pytest

from fish_audio_suite_voice.barge import (
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BLEED_DELAY_S,
    BargeGate,
    barge_rms_need,
    post_speak_cooldown_s,
)
from fish_audio_suite_voice.listen import (
    _encode_wav,
    _Listen,
    _listen_tune,
    _ListenTune,
    listen_reject_reason,
    prime_listen,
    spike_start_allowed,
    start_frames_needed,
    start_hit,
    trailing_start_hits,
)


def test_listen_tune_keeps_vad_and_pre_pad_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_VAD", "9")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD", "-1")
    tune = _listen_tune()
    assert tune.vad_aggressiveness == 1
    assert tune.pre_pad_frames == 20
    monkeypatch.setenv("FISH_VOICE_VAD", "3")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD", "0")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "0")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED", "-1")
    tune = _listen_tune()
    assert tune.vad_aggressiveness == 3
    assert tune.pre_pad_frames == 0
    assert tune.speech_frames_start == 4
    assert tune.min_voiced == 12
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED", "6")
    tune = _listen_tune()
    assert tune.speech_frames_start == 2
    assert tune.min_voiced == 6


def test_encode_wav_keeps_pcm_samples() -> None:
    pcm = b"\x01\x00\x02\x00"
    with wave.open(io.BytesIO(_encode_wav(pcm))) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        assert wf.readframes(wf.getnframes()) == pcm


def test_barge_gate_reads_env_at_construct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "1.5")
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "9")
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "500")
    gate = BargeGate()
    assert gate.bleed_delay_s == 1.5
    assert gate.hit_frames == 9
    assert gate.min_rms == 500.0


def test_barge_bleed_delay_cannot_be_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_AEC", "0")
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "-1")
    gate = BargeGate()
    assert gate.bleed_delay_s == DEFAULT_BLEED_DELAY_S
    assert gate._bleed_wait() == DEFAULT_BLEED_DELAY_S
    assert BargeGate(bleed_delay_s=-0.5)._bleed_wait() == DEFAULT_BLEED_DELAY_S
    assert BargeGate(bleed_delay_s=0.0)._bleed_wait() == 0.0


def test_barge_hit_frames_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "0")
    assert BargeGate().hit_frames == DEFAULT_BARGE_HIT_FRAMES
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "-2")
    assert BargeGate().hit_frames == DEFAULT_BARGE_HIT_FRAMES
    assert BargeGate(hit_frames=0).hit_frames == DEFAULT_BARGE_HIT_FRAMES


def test_barge_gate_explicit_kwargs_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "9.9")
    gate = BargeGate(bleed_delay_s=0.2, hit_frames=3, min_rms=10.0)
    assert gate.bleed_delay_s == 0.2
    assert gate.hit_frames == 3
    assert gate.min_rms == 10.0


def test_barge_keeps_the_lookback_that_tripped() -> None:
    gate = BargeGate()
    frame = b"\x01\x00" * 480
    for n in range(25):
        gate._heard.append(frame[:-1] + bytes([n]))
    gate.captured = b"".join(gate._heard)
    assert len(gate._heard) == 20
    assert gate.captured.startswith(frame[:-1] + bytes([5]))
    assert gate.captured.endswith(frame[:-1] + bytes([24]))


def test_prime_listen_counts_loud_prefix_frames() -> None:
    tune = _ListenTune(1, 40, 4, 200.0, 20, 12)
    heard = _Listen(tune, object())
    loud = b"\x00\x10" * 480
    quiet = b"\x01\x00" * 480
    prime_listen(heard, quiet + loud)
    assert heard.triggered
    assert len(heard.voiced) == 2
    assert heard.speech_hits == 1
    assert heard.silence == 0


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


def test_quiet_vad_frame_holds_the_turn() -> None:
    tune = _ListenTune(1, 2, 4, 200.0, 20, 12)
    heard = _Listen(tune, object())
    assert heard._hold(b"\x00\x00", False, vad_speech=True) is False
    assert heard.silence == 0
    assert heard.speech_hits == 0
    assert heard._hold(b"\x00\x00", False, vad_speech=False) is False
    assert heard.silence == 1
    assert heard._hold(b"\x00\x00", False, vad_speech=False) is True


def test_listen_reject_cough_and_impulse() -> None:
    def reason(voiced: int, hits: int, peak: float) -> str | None:
        return listen_reject_reason(
            voiced_frames=voiced,
            speech_hits=hits,
            peak_rms=peak,
            min_voiced=12,
            min_speech_rms=200.0,
        )

    assert reason(48, 8, 346.0) == "too_little_voice"
    assert reason(54, 14, 2220.0) == "impulse"
    assert reason(46, 16, 1005.0) is None
    assert reason(80, 18, 1600.0) == "impulse"
    assert reason(80, 20, 400.0) is None
    assert reason(120, 68, 1486.0) is None
    assert reason(2, 2, 100.0) == "too_short"
