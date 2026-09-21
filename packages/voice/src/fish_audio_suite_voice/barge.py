"""Mic barge-in gate: lookback, energy, min-speech, speaker-bleed delay."""

from __future__ import annotations

import collections
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from typing import Any

import numpy as np

from fish_audio_suite_voice.aec import (
    AdaptiveFloor,
    clean_mic_frame,
    effective_bleed_s,
    far_end_playing,
)
from fish_audio_suite_voice.debug import env_debug, logger
from fish_audio_suite_voice.playback import load_sounddevice

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
MAX_UTTERANCE_FRAMES = 500
DEFAULT_VAD_AGGRESSIVENESS = 1
DEFAULT_SILENCE_FRAMES_END = 22
DEFAULT_SPEECH_FRAMES_START = 4
DEFAULT_MIN_SPEECH_RMS = 200.0
DEFAULT_PRE_PAD_FRAMES = 20
HOLD_RMS_RATIO = 0.55
MIN_UTTERANCE_FRAMES = 4
LISTEN_HEARTBEAT_FRAMES = 20
DEFAULT_MIN_VOICED_FRAMES = 12
IMPULSE_PEAK_RATIO = 4.0
IMPULSE_EXTRA_VOICED = 12
IMPULSE_START_EXTRA = 6
DEFAULT_BARGE_HIT_FRAMES = 10
DEFAULT_BARGE_RMS = 220.0
DEFAULT_BARGE_OVER = 2.2
DEFAULT_BLEED_DELAY_S = 0.9
DEFAULT_POST_SPEAK_COOLDOWN_S = 0.8


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def barge_rms_need(min_rms: float, *, far_playing: bool, over: float) -> float:
    """While the DAC is live, demand louder cleaned-mic energy than residual echo."""
    if far_playing:
        return min_rms * over
    return min_rms


def post_speak_cooldown_s() -> float:
    return _env_float("FISH_VOICE_COOLDOWN", DEFAULT_POST_SPEAK_COOLDOWN_S)


def pcm_rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) + 1e-9)


def start_hit(rms: float, vad_speech: bool, min_rms: float) -> bool:
    """Count a listen-start frame. Score VAD once per capture; do not replay the ring.

    Require VAD and full min RMS. The hold ratio is only for staying in an
    utterance after it has already started.
    """
    if not vad_speech:
        return False
    return rms >= min_rms


def listen_reject_reason(
    *,
    voiced_frames: int,
    speech_hits: int,
    peak_rms: float,
    min_voiced: int,
    min_speech_rms: float,
) -> str | None:
    """Drop coughs and spikes. None means send the clip to ASR."""
    if voiced_frames < MIN_UTTERANCE_FRAMES:
        return "too_short"
    if speech_hits < min_voiced:
        return "too_little_voice"
    if (
        peak_rms >= min_speech_rms * IMPULSE_PEAK_RATIO
        and speech_hits <= min_voiced + IMPULSE_EXTRA_VOICED
    ):
        return "impulse"
    return None


def start_frames_needed(peak_rms: float, min_rms: float, speech_frames_start: int) -> int:
    """A 4x spike in the pre-pad ring is a chair pop until more VAD hits pile up."""
    if peak_rms >= min_rms * IMPULSE_PEAK_RATIO:
        return speech_frames_start + IMPULSE_START_EXTRA
    return speech_frames_start


class BargeGate:
    """VAD-energy + min-speech. Apps that own the mic can import this."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        bleed_delay_s: float | None = None,
        hit_frames: int | None = None,
        min_rms: float | None = None,
    ) -> None:
        self.device = device
        self._bleed_override = bleed_delay_s
        self.bleed_delay_s = (
            _env_float("FISH_VOICE_BLEED_DELAY", DEFAULT_BLEED_DELAY_S)
            if bleed_delay_s is None
            else bleed_delay_s
        )
        self.hit_frames = (
            _env_int("FISH_VOICE_BARGE_FRAMES", DEFAULT_BARGE_HIT_FRAMES)
            if hit_frames is None
            else hit_frames
        )
        self.min_rms = (
            _env_float("FISH_VOICE_BARGE_RMS", DEFAULT_BARGE_RMS) if min_rms is None else min_rms
        )

    def watch(self, cancel: threading.Event) -> None:
        sd = load_sounddevice()
        import webrtcvad

        vad = webrtcvad.Vad(3)
        hit = 0
        over = _env_float("FISH_VOICE_BARGE_OVER", DEFAULT_BARGE_OVER)
        q: queue.Queue[bytes] = queue.Queue()
        if env_debug():
            logger.debug(
                "barge.arm delay_s={} hit_frames={} min_rms={} over={} vad=3",
                self.bleed_delay_s,
                self.hit_frames,
                self.min_rms,
                over,
            )

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                device=self.device,
                callback=callback,
            ):
                while not cancel.is_set():
                    try:
                        frame = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if len(frame) < FRAME_BYTES:
                        continue
                    frame = frame[:FRAME_BYTES]
                    frame = clean_mic_frame(frame)
                    rms = pcm_rms(frame)
                    far = far_end_playing()
                    need = barge_rms_need(self.min_rms, far_playing=far, over=over)
                    if rms < need:
                        if hit and env_debug():
                            logger.debug(
                                "barge.decay hit={} rms={:.0f} need={:.0f} far={}",
                                hit,
                                rms,
                                need,
                                far,
                            )
                        hit = max(0, hit - 1)
                        continue
                    if vad.is_speech(frame, SAMPLE_RATE):
                        hit += 1
                        if env_debug():
                            logger.debug("barge.hit {}/{} rms={:.0f}", hit, self.hit_frames, rms)
                        if hit >= self.hit_frames:
                            if env_debug():
                                logger.debug("barge.interrupt rms={:.0f} hits={}", rms, hit)
                            cancel.set()
                            return
                    else:
                        hit = max(0, hit - 1)
        except Exception as e:
            print(f"[barge-in] {e}", file=sys.stderr)

    def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
        def _run() -> None:
            delay = (
                self._bleed_override
                if self._bleed_override is not None
                else effective_bleed_s(_env_float("FISH_VOICE_BLEED_DELAY", DEFAULT_BLEED_DELAY_S))
            )
            self.bleed_delay_s = delay
            if env_debug():
                logger.debug("barge.bleed sleep_s={}", delay)
            time.sleep(delay)
            if cancel.is_set():
                if env_debug():
                    logger.debug("barge.bleed skipped (already cancelled)")
                return
            self.watch(cancel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


def record_utterance(
    device: str | int | None = None,
    stop: threading.Event | None = None,
) -> bytes | None:
    """Block until one VAD utterance. Returns WAV bytes (16 kHz mono) or None."""
    sd = load_sounddevice()
    import webrtcvad

    vad_aggressiveness = _env_int("FISH_VOICE_VAD", DEFAULT_VAD_AGGRESSIVENESS)
    silence_frames_end = _env_int("FISH_VOICE_SILENCE_FRAMES", DEFAULT_SILENCE_FRAMES_END)
    speech_frames_start = _env_int("FISH_VOICE_SPEECH_FRAMES", DEFAULT_SPEECH_FRAMES_START)
    min_speech_rms = _env_float("FISH_VOICE_MIN_RMS", DEFAULT_MIN_SPEECH_RMS)
    pre_pad_frames = _env_int("FISH_VOICE_PRE_PAD", DEFAULT_PRE_PAD_FRAMES)
    min_voiced = _env_int("FISH_VOICE_MIN_VOICED", DEFAULT_MIN_VOICED_FRAMES)

    vad = webrtcvad.Vad(vad_aggressiveness)
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        q.put(bytes(indata))

    voiced: list[bytes] = []
    ring: collections.deque[tuple[bytes, bool]] = collections.deque(maxlen=pre_pad_frames)
    triggered = False
    silence = 0
    idle_frames = 0
    window_peak = 0.0
    clip_peak = 0.0
    speech_hits = 0
    floor = AdaptiveFloor(min_speech_rms)
    min_speech_now = min_speech_rms
    if env_debug():
        logger.debug(
            "listen.open vad={} start_frames={} min_rms={} min_voiced={} pre_pad={} silence_end={}",
            vad_aggressiveness,
            speech_frames_start,
            min_speech_rms,
            min_voiced,
            pre_pad_frames,
            silence_frames_end,
        )

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        device=device,
        callback=callback,
    ):
        while True:
            if stop is not None and stop.is_set():
                return None
            try:
                frame = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if len(frame) < FRAME_BYTES:
                continue
            frame = frame[:FRAME_BYTES]
            frame = clean_mic_frame(frame)
            rms = pcm_rms(frame)
            vad_speech = bool(vad.is_speech(frame, SAMPLE_RATE))
            min_speech_now = floor.value()
            if not triggered:
                floor.observe(rms, quiet=True)
            hold_rms = min_speech_now * HOLD_RMS_RATIO
            need = min_speech_now if not triggered else hold_rms
            is_speech = rms >= need and vad_speech
            idle_frames += 1
            window_peak = max(window_peak, rms)
            clip_peak = max(clip_peak, rms)
            if env_debug() and idle_frames % LISTEN_HEARTBEAT_FRAMES == 0:
                logger.debug(
                    "listen.mic frames={} peak_rms={:.0f} last_rms={:.0f} floor={:.0f} vad={} triggered={} hits={}",
                    idle_frames,
                    window_peak,
                    rms,
                    min_speech_now,
                    vad_speech,
                    triggered,
                    sum(1 for _, counted in ring if counted) if not triggered else speech_hits,
                )
                window_peak = 0.0

            if not triggered:
                ring.append((frame, start_hit(rms, vad_speech, min_speech_now)))
                hits_now = sum(1 for _, counted in ring if counted)
                peak_ring = max((pcm_rms(pcm) for pcm, _ in ring), default=0.0)
                need_start = start_frames_needed(peak_ring, min_speech_now, speech_frames_start)
                if hits_now >= need_start:
                    triggered = True
                    voiced.extend(pcm for pcm, _ in ring)
                    speech_hits = hits_now
                    if env_debug():
                        logger.debug(
                            "listen.speech_start rms={:.0f} vad={} prepad_frames={} voiced_hits={} need_start={}",
                            rms,
                            vad_speech,
                            len(voiced),
                            speech_hits,
                            need_start,
                        )
                    ring.clear()
                    silence = 0
            else:
                voiced.append(frame)
                if is_speech:
                    speech_hits += 1
                    silence = 0
                else:
                    silence += 1
                    if silence >= silence_frames_end:
                        break
                if len(voiced) >= MAX_UTTERANCE_FRAMES:
                    break

    why = listen_reject_reason(
        voiced_frames=len(voiced),
        speech_hits=speech_hits,
        peak_rms=clip_peak,
        min_voiced=min_voiced,
        min_speech_rms=min_speech_now if triggered else min_speech_rms,
    )
    if why is not None:
        if env_debug():
            logger.debug(
                "listen.reject {} frames={} voiced_hits={} peak_rms={:.0f} min_voiced={}",
                why,
                len(voiced),
                speech_hits,
                clip_peak,
                min_voiced,
            )
        return None

    pcm = b"".join(voiced)
    if env_debug():
        logger.debug(
            "listen.end frames={} wav_bytes={} silence={} voiced_hits={} peak_rms={:.0f} duration_ms={}",
            len(voiced),
            len(pcm) + 44,
            silence,
            speech_hits,
            clip_peak,
            len(voiced) * FRAME_MS,
        )
    with tempfile.SpooledTemporaryFile(max_size=2_000_000) as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        buf.seek(0)
        return buf.read()
