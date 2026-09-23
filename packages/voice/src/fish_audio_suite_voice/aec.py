"""Optional in-process AEC3. Far-end is speaker PCM; near-end is the mic.

Missing extra, FISH_VOICE_AEC=0, empty far-end, or FileSink: return mic unchanged.
Bleed delay stays the fallback. PipeWire echo-cancel is a host trick, not this module.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from fish_audio_suite_kit import env_bool, env_float
from fish_audio_suite_voice.debug import debug

AEC_RATE = 16_000
SAMPLE_BYTES = 2
_FAR_HOLD_S = 2
FAR_HOLD_SAMPLES = AEC_RATE * _FAR_HOLD_S
_FAR_RECENT_S = 0.4
_RMS_FLOOR = 1e-9
FAR_SILENCE_RMS = 40.0
DEFAULT_AEC_BLEED_S = 0.3
DEFAULT_AEC_WET = 0.85
_FULL_WET = 0.999


class _ProcHolder:
    proc: Any = None
    tried: bool = False
    import_fail: bool = False


_HOLDER = _ProcHolder()
_PROC_LOCK = threading.Lock()


def aec_wanted() -> bool:
    return env_bool("FISH_VOICE_AEC", default=True)


def even_pcm(pcm: bytes) -> bytes:
    extra = len(pcm) % SAMPLE_BYTES
    return pcm if extra == 0 else pcm[:-extra]


def _realign(buf: bytearray) -> None:
    extra = len(buf) % SAMPLE_BYTES
    if extra:
        del buf[:extra]


_INT16_LO = -32_768
_INT16_HI = 32_767


def _int16_bytes(samples: Any) -> bytes:
    return np.clip(samples, _INT16_LO, _INT16_HI).astype(np.int16).tobytes()


def resample_int16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if not pcm or src_rate <= 0 or dst_rate <= 0:
        return b""
    aligned = even_pcm(pcm)
    if src_rate == dst_rate:
        return aligned
    samples = np.frombuffer(aligned, dtype=np.int16)
    if samples.size == 0:
        return b""
    n_dst = max(1, round(samples.size * dst_rate / src_rate))
    src_t = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    out = np.interp(dst_t, src_t, samples.astype(np.float64))
    return _int16_bytes(out)


class FarEndTap:
    """16 kHz int16 ring of what we actually wrote to the DAC."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pcm = bytearray()
        self._playing_until = 0.0

    def clear(self) -> None:
        with self._lock:
            self._pcm.clear()
            self._playing_until = 0.0

    def _note_playing(self, duration_s: float) -> None:
        end = time.monotonic() + max(0.0, duration_s)
        self._playing_until = max(self._playing_until, end)

    def mark_playing(self, duration_s: float) -> None:
        with self._lock:
            self._note_playing(duration_s)

    def push(self, pcm: bytes, sample_rate: int) -> None:
        chunk = resample_int16(pcm, sample_rate, AEC_RATE)
        if not chunk:
            return
        duration_s = len(even_pcm(pcm)) / SAMPLE_BYTES / sample_rate
        with self._lock:
            self._pcm.extend(chunk)
            extra = len(self._pcm) - FAR_HOLD_SAMPLES * SAMPLE_BYTES
            if extra > 0:
                self._drop_front(extra)
            self._note_playing(duration_s)

    def _drop_front(self, n: int) -> bytes:
        out = bytes(self._pcm[:n])
        del self._pcm[:n]
        _realign(self._pcm)
        return out

    def pop(self, n_bytes: int) -> bytes:
        if n_bytes <= 0:
            return b""
        with self._lock:
            if len(self._pcm) < n_bytes:
                return b""
            return self._drop_front(n_bytes)

    def playing_recently(self, window_s: float = _FAR_RECENT_S) -> bool:
        return time.monotonic() < (self._playing_until + window_s)


TAP = FarEndTap()


def tap_playback(pcm: bytes, sample_rate: int) -> None:
    if not aec_wanted():
        return
    TAP.push(pcm, sample_rate)


def tap_clear() -> None:
    TAP.clear()


def far_end_playing(window_s: float = _FAR_RECENT_S) -> bool:
    return TAP.playing_recently(window_s)


def pcm_rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) + _RMS_FLOOR)


def load_processor() -> Any | None:
    if not aec_wanted():
        return None
    with _PROC_LOCK:
        if _HOLDER.tried:
            return _HOLDER.proc
        _HOLDER.tried = True
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError:
            _HOLDER.import_fail = True
            debug("aec.skip extra pywebrtc-audio not installed")
            return None
        _HOLDER.proc = AudioProcessor(
            sample_rate=AEC_RATE,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=False,
            high_pass_filter=True,
            auto_gain_control=False,
            stream_delay_ms=0,
        )
        debug(
            "aec.on AEC3 16k wet={}",
            env_float("FISH_VOICE_AEC_WET", DEFAULT_AEC_WET),
        )
        return _HOLDER.proc


def aec_available() -> bool:
    return load_processor() is not None


def effective_bleed_s(fallback_s: float) -> float:
    if not aec_available():
        return fallback_s if fallback_s >= 0 else 0.0
    bleed = env_float("FISH_VOICE_AEC_BLEED", DEFAULT_AEC_BLEED_S)
    return bleed if bleed >= 0 else DEFAULT_AEC_BLEED_S


def clean_mic_frame(near: bytes) -> bytes:
    """AEC when far-end has energy; otherwise return near (no AEC on silence)."""
    proc = load_processor()
    if proc is None or len(near) < SAMPLE_BYTES:
        return near
    far = TAP.pop(len(near))
    if not far or pcm_rms(far) < FAR_SILENCE_RMS:
        return near
    wet = env_float("FISH_VOICE_AEC_WET", DEFAULT_AEC_WET)
    wet = min(1.0, max(0.0, wet))
    near_a = np.frombuffer(near, dtype=np.int16)
    far_a = np.frombuffer(far, dtype=np.int16)
    if near_a.size != far_a.size:
        return near
    try:
        clean = proc.process(near_a, far_a)
    except Exception as exc:
        debug("aec.fail {}", exc)
        return near
    if wet >= _FULL_WET:
        return _int16_bytes(clean)
    mixed = (1.0 - wet) * near_a.astype(np.float32) + wet * np.asarray(clean, dtype=np.float32)
    return _int16_bytes(mixed)
