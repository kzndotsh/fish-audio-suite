from __future__ import annotations

import collections.abc
import io
import threading
import wave

import pytest
from voice_fakes import install_vad

from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import (
    _BARGE_VAD,
    BARGE_MISS_DECAY_FRAMES,
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_PLAYING_GAIN,
    DEFAULT_BARGE_RMS,
    DEFAULT_BLEED_DELAY_S,
    DEFAULT_POST_SPEAK_COOLDOWN_S,
    FRAME_BYTES,
    BargeGate,
    _barge_step,
    _take_full_frames,
    barge_rms_need,
)
from fish_audio_suite_voice.listen import (
    _CAP_QUIET_FRAMES,
    _clip_wav,
    _encode_wav,
    _Listen,
    _prime_listen,
    listen_reject_reason,
    record_utterance,
    spike_start_allowed,
    start_frames_needed,
    start_hit,
    trailing_start_hits,
)
from fish_audio_suite_voice.tune import (
    DEFAULT_END_SILENCE_FRAMES,
    DEFAULT_MIN_SPEECH_RMS,
    DEFAULT_MIN_VOICED_FRAMES,
    DEFAULT_VAD_AGGRESSIVENESS,
    IMPULSE_START_EXTRA,
    MAX_UTTERANCE_FRAMES,
    AecTune,
    BargeTune,
    ListenTune,
)


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

    class Vad:
        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return True

    tune = ListenTune.from_env()
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

    tune = ListenTune(1, 40, 4, 200.0, 10, 12)
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

    tune = ListenTune(1, 2, 4, 200.0, 20, 12)
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

    tune = ListenTune(1, 40, 1, 200.0, 4, 12)
    heard = _Listen(tune, Vad())
    loud = _pcm(500)
    for _ in range(MAX_UTTERANCE_FRAMES):
        assert heard.take(loud, 0) is False
    assert len(heard.clip_frames) >= MAX_UTTERANCE_FRAMES
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

    tune = ListenTune(1, 40, 1, 200.0, 4, 12)
    heard = _Listen(tune, Vad())
    first = _pcm(500)
    rest = _pcm(800)
    heard.take(first, 0)
    for _ in range(MAX_UTTERANCE_FRAMES + 20):
        assert heard.take(rest, 0) is False
    assert len(heard.clip_frames) == MAX_UTTERANCE_FRAMES
    assert first not in heard.clip_frames
    assert heard.speech_hits == MAX_UTTERANCE_FRAMES


def test_trimmed_barge_credit_does_not_keep_a_short_tail() -> None:
    class Vad:
        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return True

    tune = ListenTune(1, 40, 1, 200.0, 4, 12)
    heard = _Listen(tune, Vad())
    loud = _pcm(800)
    _prime_listen(heard, loud * 4)
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
    thread = gate.start_after_bleed(cancel)
    cancel.set()
    # A 30 s bleed that ended inside the join timeout proves cancel cut it short.
    thread.join(timeout=1)
    assert thread.is_alive() is False


def test_barge_vad_matches_listen() -> None:
    assert _BARGE_VAD == DEFAULT_VAD_AGGRESSIVENESS


def test_barge_tune_reads_env_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "1.5")
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "9")
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "500")
    gate = BargeGate(tune=BargeTune.from_env())
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "2")
    assert gate.bleed_delay_s == 1.5
    assert gate.hit_frames == 9
    assert gate.min_rms == 500.0


def test_barge_bleed_delay_cannot_be_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "-1")
    no_aec = EchoCanceller(AecTune(enabled=False))
    gate = BargeGate(tune=BargeTune.from_env(), aec=no_aec)
    assert gate.bleed_delay_s == DEFAULT_BLEED_DELAY_S
    assert gate._bleed_wait() == DEFAULT_BLEED_DELAY_S
    assert BargeGate(bleed_delay_s=-0.5, aec=no_aec)._bleed_wait() == DEFAULT_BLEED_DELAY_S
    assert BargeGate(bleed_delay_s=0.0, aec=no_aec)._bleed_wait() == 0.0


def test_barge_hit_frames_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "0")
    assert BargeTune.from_env().hit_frames == DEFAULT_BARGE_HIT_FRAMES
    monkeypatch.setenv("FISH_VOICE_BARGE_FRAMES", "-2")
    assert BargeTune.from_env().hit_frames == DEFAULT_BARGE_HIT_FRAMES
    assert BargeGate(hit_frames=0).hit_frames == DEFAULT_BARGE_HIT_FRAMES
    monkeypatch.setenv("FISH_VOICE_BARGE_RMS", "0")
    assert BargeTune.from_env().min_rms == DEFAULT_BARGE_RMS
    assert BargeGate(min_rms=-1).min_rms == DEFAULT_BARGE_RMS
    assert BargeGate(min_rms=10).min_rms == 10


def test_barge_gate_explicit_kwargs_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_BLEED_DELAY", "9.9")
    gate = BargeGate(tune=BargeTune.from_env(), bleed_delay_s=0.2, hit_frames=3, min_rms=10.0)
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
    tune = ListenTune(1, 1, 4, 200.0, 20, DEFAULT_MIN_VOICED_FRAMES)
    heard = _Listen(tune, object())
    loud = (1000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
    _prime_listen(heard, loud * DEFAULT_BARGE_HIT_FRAMES)
    assert heard.speech_hits == DEFAULT_BARGE_HIT_FRAMES
    wav = _clip_wav(heard, tune)
    assert wav is not None
    assert wav.startswith(b"RIFF")
    spike = (4000).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
    bang = _Listen(tune, object())
    _prime_listen(bang, spike * DEFAULT_BARGE_HIT_FRAMES)
    kept = _clip_wav(bang, tune)
    assert kept is not None
    assert kept.startswith(b"RIFF")
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


def test_barge_prefix_counts_under_a_higher_listen_floor() -> None:
    tune = ListenTune(1, 40, 4, 800.0, 20, DEFAULT_MIN_VOICED_FRAMES)
    heard = _Listen(tune, object())
    speech = (256).to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)
    quiet = b"\x01\x00" * (FRAME_BYTES // 2)
    _prime_listen(heard, quiet + speech * DEFAULT_BARGE_HIT_FRAMES)
    assert heard.speech_hits == DEFAULT_BARGE_HIT_FRAMES
    assert _clip_wav(heard, tune) is not None


def test_prime_listen_counts_loud_prefix_frames() -> None:
    tune = ListenTune(1, 40, 4, 200.0, 20, 12)
    heard = _Listen(tune, object())
    loud = b"\x00\x10" * 480
    quiet = b"\x01\x00" * 480
    _prime_listen(heard, quiet + loud)
    assert heard.triggered
    assert len(heard.clip_frames) == 2
    assert heard.speech_hits == 1
    assert heard.silence == 0


def test_barge_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_BLEED_DELAY", raising=False)
    monkeypatch.delenv("FISH_VOICE_BARGE_FRAMES", raising=False)
    monkeypatch.delenv("FISH_VOICE_COOLDOWN", raising=False)
    gate = BargeGate(tune=BargeTune.from_env())
    assert gate.bleed_delay_s == DEFAULT_BLEED_DELAY_S
    assert gate.hit_frames == DEFAULT_BARGE_HIT_FRAMES
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
    assert barge_rms_need(220.0, far_playing=True, playing_gain=0, aec_on=False) == pytest.approx(
        220.0 * DEFAULT_BARGE_PLAYING_GAIN
    )
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

    def boom(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
        aec: object = None,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout, aec
        raise OSError("no device")

    install_vad(monkeypatch, SpeechVad)
    monkeypatch.setattr("fish_audio_suite_voice.barge.mic_frames", boom)
    gate = BargeGate(hit_frames=2, min_rms=1.0, bleed_delay_s=0)
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
    tune = ListenTune(1, 2, 4, 200.0, 20, 12)
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


def _pcm(level: int) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


def _install_listen_fakes(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[bytes],
) -> ListenTune:
    monkeypatch.setenv("FISH_VOICE_SILENCE_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_VOICED_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_SPEECH_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_PRE_PAD_FRAMES", "2")
    monkeypatch.setenv("FISH_VOICE_MIN_RMS", "200")

    class SpeechVad:
        def __init__(self, mode: int) -> None:
            self.mode = mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            return frame[:2] != b"\x00\x00"

    install_vad(monkeypatch, SpeechVad)

    def fake_mic(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
        aec: object = None,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout, aec
        yield from frames

    monkeypatch.setattr("fish_audio_suite_voice.listen.mic_frames", fake_mic)
    return ListenTune.from_env()


def test_record_utterance_writes_a_wav_for_a_held_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = _pcm(500)
    quiet = _pcm(0)
    tune = _install_listen_fakes(monkeypatch, [loud, loud, quiet, quiet])
    wav = record_utterance(tune=tune)
    assert wav is not None
    assert wav.startswith(b"RIFF")
    assert len(wav) > 44


def test_record_utterance_drops_a_finished_clip_when_quit_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loud = _pcm(500)
    quiet = _pcm(0)
    tune = _install_listen_fakes(monkeypatch, [loud, loud, quiet, quiet])
    stop = threading.Event()
    stop.set()
    assert record_utterance(quit_requested=stop, tune=tune) is None


def test_record_utterance_drops_a_clip_that_never_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_listen_fakes(monkeypatch, [_pcm(0), _pcm(0)])
    assert record_utterance() is None


def _watch_with(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[bytes],
    *,
    aec: EchoCanceller,
    tune: BargeTune,
    vad_speech: bool = True,
) -> tuple[BargeGate, threading.Event]:
    class Vad:
        def __init__(self, mode: int) -> None:
            self.mode = mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return vad_speech

    install_vad(monkeypatch, Vad)

    def fake_mic(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
        aec: object = None,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout, aec
        yield from frames

    monkeypatch.setattr("fish_audio_suite_voice.barge.mic_frames", fake_mic)
    return BargeGate(tune=tune, aec=aec), threading.Event()


def test_barge_watch_aligns_the_far_end_before_reading_the_mic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=False))
    order: list[str] = []
    monkeypatch.setattr(aec, "align", lambda: order.append("align"))
    gate, cancel = _watch_with(monkeypatch, [_pcm(0)], aec=aec, tune=BargeTune())

    def mic(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
        aec: object = None,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout, aec
        order.append("mic")
        yield _pcm(0)

    monkeypatch.setattr("fish_audio_suite_voice.barge.mic_frames", mic)
    gate.watch(cancel)
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


def _constant_frame(level: int) -> bytes:
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


def test_speech_between_the_floor_and_the_boosted_need_does_not_raise_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[float, bool]] = []

    class Floor:
        def __init__(self, default: float) -> None:
            self.default = default

        def value(self) -> float:
            return self.default

        def observe(self, rms: float, *, quiet: bool) -> None:
            seen.append((rms, quiet))

    class SilentVad:
        def __init__(self, mode: int) -> None:
            del mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del frame, rate
            return False

    def mic(
        device: str | int | None,
        stop: threading.Event | None,
        *,
        timeout: float,
        aec: object = None,
    ) -> collections.abc.Iterator[bytes]:
        del device, stop, timeout, aec
        yield from [_constant_frame(300)] * 5

    install_vad(monkeypatch, SilentVad)
    monkeypatch.setattr("fish_audio_suite_voice.barge.mic_frames", mic)
    monkeypatch.setattr("fish_audio_suite_voice.barge.AdaptiveFloor", Floor)
    aec = EchoCanceller(AecTune(enabled=False))
    # The speaker is playing and AEC is off, so ``need`` is 220 x 2.2 = 484.
    aec.tap_playback(b"\x00\x00" * 16000, 16000)
    gate = BargeGate(tune=BargeTune(min_rms=220.0, playing_gain=2.2), aec=aec)
    gate.watch(threading.Event())
    assert len(seen) == 5
    assert all(not quiet for _, quiet in seen)


def test_a_gate_without_an_aec_runs_without_one() -> None:
    gate = BargeGate()
    assert gate.aec.available() is False
    assert gate._bleed_wait() == gate.tune.bleed_delay_s


def test_an_explicit_bleed_delay_wins_over_the_aec_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aec = EchoCanceller(AecTune(enabled=True, bleed_delay_s=0.3))
    monkeypatch.setattr(aec, "available", lambda: True)
    assert BargeGate(aec=aec)._bleed_wait() == 0.3
    assert BargeGate(aec=aec, bleed_delay_s=1.5)._bleed_wait() == 1.5
    assert BargeGate(aec=aec, bleed_delay_s=0.0)._bleed_wait() == 0.0
    assert BargeGate(aec=aec, bleed_delay_s=-1.0)._bleed_wait() == 0.3


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


def test_a_setup_failure_in_the_watcher_thread_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = BargeGate(hit_frames=2, min_rms=1.0, bleed_delay_s=0)

    def broken_bleed() -> float:
        raise RuntimeError("aec failed to load")

    monkeypatch.setattr(gate, "_bleed_wait", broken_bleed)
    cancel = threading.Event()
    thread = gate.start_after_bleed(cancel)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(gate.failure, RuntimeError)
    assert not cancel.is_set()
    err = capsys.readouterr().err
    assert "aec failed to load" in err
    assert "off for this reply" in err
