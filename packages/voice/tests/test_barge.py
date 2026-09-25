from __future__ import annotations

import collections
import io
import sys
import threading
import time
import wave

import pytest

from fish_audio_suite_voice.barge import (
    _BARGE_VAD,
    BARGE_MISS_DECAY_FRAMES,
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_OVER,
    DEFAULT_BARGE_RMS,
    DEFAULT_BLEED_DELAY_S,
    DEFAULT_POST_SPEAK_COOLDOWN_S,
    FRAME_BYTES,
    BargeGate,
    _barge_step,
    _take_full_frames,
    barge_rms_need,
    post_speak_cooldown_s,
)
from fish_audio_suite_voice.listen import (
    _CAP_QUIET_FRAMES,
    DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MIN_VOICED_FRAMES,
    DEFAULT_SILENCE_FRAMES_END,
    DEFAULT_VAD_AGGRESSIVENESS,
    IMPULSE_START_EXTRA,
    MAX_UTTERANCE_FRAMES,
    _clip_wav,
    _encode_wav,
    _Listen,
    _listen_tune,
    _ListenTune,
    listen_reject_reason,
    prime_listen,
    record_utterance,
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
    assert tune.pre_pad_frames == tune.speech_frames_start + IMPULSE_START_EXTRA
    assert tune.speech_frames_start == 4
    assert tune.min_voiced == 12
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED", "6")
    tune = _listen_tune()
    assert tune.speech_frames_start == 2
    assert tune.min_voiced == 6
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "-5")
    assert _listen_tune().min_speech_rms == DEFAULT_MIN_SPEECH_RMS
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "0")
    assert _listen_tune().silence_frames_end == DEFAULT_SILENCE_FRAMES_END
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "-3")
    assert _listen_tune().silence_frames_end == DEFAULT_SILENCE_FRAMES_END
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED", str(MAX_UTTERANCE_FRAMES + 1))
    assert _listen_tune().min_voiced == DEFAULT_MIN_VOICED_FRAMES
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED", "40")
    assert _listen_tune().min_voiced == 40


def test_pre_pad_zero_still_starts_on_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_PRE_PAD", "0")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "200")

    class Vad:
        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return True

    tune = _listen_tune()
    heard = _Listen(tune, Vad())
    frame = _pcm(500)
    for idle in range(tune.pre_pad_frames):
        assert heard.take(frame, idle) is False
    assert heard.triggered


def test_loud_frames_do_not_raise_the_floor_above_the_voice() -> None:
    class Vad:
        on: bool = False

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return self.on

    tune = _ListenTune(1, 40, 4, 200.0, 10, 12)
    vad = Vad()
    heard = _Listen(tune, vad)
    frame = _pcm(300)
    for idle in range(30):
        assert heard.take(frame, idle) is False
    assert heard.min_speech_now == 200.0
    vad.on = True
    for idle in range(30, 40):
        heard.take(frame, idle)
    assert heard.triggered


def test_a_gap_before_the_start_still_counts_as_voice() -> None:
    class Vad:
        def is_speech(self, frame: bytes, rate: int) -> bool:
            del rate
            return any(frame)

    tune = _ListenTune(1, 2, 4, 200.0, 20, 12)
    heard = _Listen(tune, Vad())
    loud = _pcm(500)
    quiet = _pcm(0)
    for _ in range(3):
        assert heard.take(loud, 0) is False
    assert heard.take(quiet, 0) is False
    for _ in range(4):
        heard.take(loud, 0)
    assert heard.triggered
    assert heard.speech_hits == 7
    for _ in range(5):
        assert heard.take(loud, 0) is False
    assert heard.take(quiet, 0) is False
    assert heard.take(quiet, 0) is True
    assert _clip_wav(heard, tune) is not None


def test_a_dip_after_the_length_cap_does_not_end_the_utterance() -> None:
    class Vad:
        speech = True

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return self.speech

    tune = _ListenTune(1, 40, 1, 200.0, 4, 12)
    heard = _Listen(tune, Vad())
    loud = _pcm(500)
    for _ in range(MAX_UTTERANCE_FRAMES):
        assert heard.take(loud, 0) is False
    assert len(heard.voiced) >= MAX_UTTERANCE_FRAMES
    Vad.speech = False
    quiet = _pcm(0)
    assert heard.take(quiet, 0) is False
    for _ in range(_CAP_QUIET_FRAMES - 2):
        assert heard.take(quiet, 0) is False
    assert heard.take(quiet, 0) is True


def test_a_long_utterance_keeps_only_the_recent_frames() -> None:
    class Vad:
        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return True

    tune = _ListenTune(1, 40, 1, 200.0, 4, 12)
    heard = _Listen(tune, Vad())
    first = _pcm(500)
    rest = _pcm(800)
    heard.take(first, 0)
    for _ in range(MAX_UTTERANCE_FRAMES + 20):
        assert heard.take(rest, 0) is False
    assert len(heard.voiced) == MAX_UTTERANCE_FRAMES
    assert first not in heard.voiced
    assert heard.speech_hits == MAX_UTTERANCE_FRAMES


def test_trimmed_barge_credit_does_not_keep_a_short_tail() -> None:
    class Vad:
        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return True

    tune = _ListenTune(1, 40, 1, 200.0, 4, 12)
    heard = _Listen(tune, Vad())
    loud = _pcm(800)
    prime_listen(heard, loud * 4)
    assert heard.primed_hits == 4
    quiet = _pcm(0)
    for _ in range(MAX_UTTERANCE_FRAMES):
        assert heard.take(quiet, 0) is False
    assert heard.primed_hits == 0
    assert heard.speech_hits == 0
    for _ in range(4):
        assert heard.take(loud, 0) is False
    assert heard.speech_hits == 4
    assert heard.primed_hits == 0
    assert _clip_wav(heard, tune) is None


def test_encode_wav_keeps_pcm_samples() -> None:
    pcm = b"\x01\x00\x02\x00"
    with wave.open(io.BytesIO(_encode_wav(pcm))) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16_000
        assert wf.readframes(wf.getnframes()) == pcm


def test_bleed_wait_ends_when_the_turn_is_cancelled() -> None:
    gate = BargeGate(bleed_delay_s=30)
    cancel = threading.Event()
    started = time.monotonic()
    thread = gate.start_after_bleed(cancel)
    cancel.set()
    thread.join(timeout=1)
    assert thread.is_alive() is False
    assert time.monotonic() - started < 1


def test_barge_vad_matches_listen() -> None:
    assert _BARGE_VAD == DEFAULT_VAD_AGGRESSIVENESS


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
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "0")
    assert BargeGate().min_rms == DEFAULT_BARGE_RMS
    assert BargeGate(min_rms=-1).min_rms == DEFAULT_BARGE_RMS
    assert BargeGate(min_rms=10).min_rms == 10


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


def test_barge_prefix_meets_the_listen_minimum() -> None:
    tune = _ListenTune(1, 1, 4, 200.0, 20, DEFAULT_MIN_VOICED_FRAMES)
    heard = _Listen(tune, object())
    loud = (1000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
    prime_listen(heard, loud * DEFAULT_BARGE_HIT_FRAMES)
    assert heard.speech_hits == DEFAULT_BARGE_HIT_FRAMES
    wav = _clip_wav(heard, tune)
    assert wav is not None
    assert wav.startswith(b"RIFF")
    spike = (4000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
    bang = _Listen(tune, object())
    prime_listen(bang, spike * DEFAULT_BARGE_HIT_FRAMES)
    kept = _clip_wav(bang, tune)
    assert kept is not None
    assert kept.startswith(b"RIFF")
    assert (
        listen_reject_reason(
            voiced_frames=14,
            speech_hits=14,
            peak_rms=4000,
            min_voiced=DEFAULT_MIN_VOICED_FRAMES,
            min_speech_rms=200.0,
        )
        == "impulse"
    )


def test_barge_prefix_counts_under_a_higher_listen_floor() -> None:
    tune = _ListenTune(1, 40, 4, 800.0, 20, DEFAULT_MIN_VOICED_FRAMES)
    heard = _Listen(tune, object())
    speech = (256).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
    quiet = b"\x01\x00" * (FRAME_BYTES // 2)
    prime_listen(heard, quiet + speech * DEFAULT_BARGE_HIT_FRAMES)
    assert heard.speech_hits == DEFAULT_BARGE_HIT_FRAMES
    assert _clip_wav(heard, tune) is not None


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
    monkeypatch.setenv("FISH_VOICE_COOLDOWN", "0")
    assert post_speak_cooldown_s() == 0.0
    monkeypatch.setenv("FISH_VOICE_COOLDOWN", "-1")
    assert post_speak_cooldown_s() == DEFAULT_POST_SPEAK_COOLDOWN_S


def test_barge_over_speaker_raises_need() -> None:
    assert barge_rms_need(220.0, far_playing=False, over=2.2) == 220.0
    assert barge_rms_need(220.0, far_playing=True, over=2.2, aec_on=False) == pytest.approx(
        220.0 * 2.2
    )
    assert barge_rms_need(220.0, far_playing=True, over=2.2, aec_on=True) == 220.0
    assert barge_rms_need(220.0, far_playing=True, over=0, aec_on=False) == pytest.approx(
        220.0 * DEFAULT_BARGE_OVER
    )
    assert barge_rms_need(220.0, far_playing=True, over=1, aec_on=False) == 220.0


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


def test_mic_chunks_join_when_short_and_split_when_long() -> None:
    pending = bytearray()
    half = b"\x01" * (FRAME_BYTES // 2)
    assert _take_full_frames(pending, half, FRAME_BYTES) == []
    assert _take_full_frames(pending, half, FRAME_BYTES) == [b"\x01" * FRAME_BYTES]
    assert pending == bytearray()
    long = b"\x02" * (FRAME_BYTES * 2 + 3)
    frames = _take_full_frames(pending, long, FRAME_BYTES)
    assert frames == [b"\x02" * FRAME_BYTES, b"\x02" * FRAME_BYTES]
    rest = b"\x03" * (FRAME_BYTES - 3)
    done = _take_full_frames(pending, rest, FRAME_BYTES)
    assert done == [b"\x02" * 3 + rest]
    assert pending == bytearray()


def test_barge_hits_decay_after_three_misses_and_then_trip() -> None:
    hit, miss, tripped = 2, 0, False
    for _ in range(BARGE_MISS_DECAY_FRAMES - 1):
        hit, miss, tripped = _barge_step(
            hit=hit, miss=miss, voiced=False, rms=0, need=200, far=False, want=3
        )
        assert tripped is False
        assert hit == 2
    hit, miss, tripped = _barge_step(
        hit=hit, miss=miss, voiced=False, rms=0, need=200, far=False, want=3
    )
    assert (hit, miss, tripped) == (1, 0, False)
    hit, miss, tripped = _barge_step(
        hit=hit, miss=miss, voiced=True, rms=400, need=200, far=False, want=3
    )
    assert tripped is False
    hit, miss, tripped = _barge_step(
        hit=hit, miss=miss, voiced=True, rms=400, need=200, far=False, want=3
    )
    assert tripped is True


def test_barge_watch_logs_a_mic_failure_without_cancelling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SpeechVad:
        def __init__(self, mode: int) -> None:
            self.mode = mode

    class Mod:
        Vad = SpeechVad

    def boom(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout
        raise OSError("no device")

    monkeypatch.setitem(sys.modules, "webrtcvad", Mod)
    monkeypatch.setattr("fish_audio_suite_voice.barge.mic_frames", boom)
    gate = BargeGate(hit_frames=2, min_rms=1.0, bleed_delay_s=0)
    cancel = threading.Event()
    gate.watch(cancel)
    assert not cancel.is_set()
    assert gate.captured == b""
    assert "no device" in capsys.readouterr().err


def test_bleed_thread_skips_the_mic_when_already_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = BargeGate(bleed_delay_s=0)
    watched: list[bool] = []

    def watch(cancel: threading.Event) -> None:
        watched.append(cancel.is_set())

    monkeypatch.setattr(gate, "watch", watch)
    cancel = threading.Event()
    cancel.set()
    gate.start_after_bleed(cancel).join(timeout=1)
    assert watched == []
    cancel.clear()
    gate.start_after_bleed(cancel).join(timeout=1)
    assert watched == [False]


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


def _pcm(level: int) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


def _install_listen_fakes(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[bytes],
) -> None:
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED", "2")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "200")

    class SpeechVad:
        def __init__(self, mode: int) -> None:
            self.mode = mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            return frame[:2] != b"\x00\x00"

    class Mod:
        Vad = SpeechVad

    monkeypatch.setitem(sys.modules, "webrtcvad", Mod)

    def fake_mic(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout
        yield from frames

    monkeypatch.setattr("fish_audio_suite_voice.listen.mic_frames", fake_mic)


def test_record_utterance_writes_a_wav_for_a_held_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = _pcm(500)
    quiet = _pcm(0)
    _install_listen_fakes(monkeypatch, [loud, loud, quiet, quiet])
    wav = record_utterance()
    assert wav is not None
    assert wav.startswith(b"RIFF")
    assert len(wav) > 44


def test_record_utterance_drops_a_finished_clip_when_quit_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = _pcm(500)
    quiet = _pcm(0)
    _install_listen_fakes(monkeypatch, [loud, loud, quiet, quiet])
    stop = threading.Event()
    stop.set()
    assert record_utterance(stop=stop) is None


def test_record_utterance_drops_a_clip_that_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_listen_fakes(monkeypatch, [_pcm(0), _pcm(0)])
    assert record_utterance() is None
