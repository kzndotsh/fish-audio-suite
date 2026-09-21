"""Optional in-process AEC3. Far-end is speaker PCM; near-end is the mic.

Missing extra, FISH_VOICE_AEC=0, empty far-end, or FileSink: return mic unchanged.
Bleed delay stays the fallback. PipeWire echo-cancel is a host trick, not this module.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any

import numpy as np

from fish_audio_suite_voice.debug import env_debug, logger

AEC_RATE = 16_000
FAR_HOLD_SAMPLES = AEC_RATE * 2
FAR_SILENCE_RMS = 40.0
DEFAULT_AEC_BLEED_S = 0.1
DEFAULT_AEC_WET = 0.85
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


class _ProcHolder:
    proc: Any = None
    tried: bool = False
    import_fail: bool = False


_HOLDER = _ProcHolder()
_PROC_LOCK = threading.Lock()


def aec_wanted() -> bool:
    raw = os.environ.get("FISH_VOICE_AEC", "1").strip().lower()
    if raw in _FALSY:
        return False
    return raw in _TRUTHY or raw == ""


def resample_int16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if not pcm or src_rate <= 0 or dst_rate <= 0:
        return b""
    if src_rate == dst_rate:
        return pcm if len(pcm) % 2 == 0 else pcm[:-1]
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype=np.int16)
    if samples.size == 0:
        return b""
    n_dst = max(1, round(samples.size * dst_rate / src_rate))
    src_t = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    dst_t = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    out = np.interp(dst_t, src_t, samples.astype(np.float64))
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


class FarEndTap:
    """16 kHz int16 ring of what we actually wrote to the DAC."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pcm = bytearray()
        self._last_energy = 0.0

    def clear(self) -> None:
        with self._lock:
            self._pcm.clear()

    def push(self, pcm: bytes, sample_rate: int) -> None:
        chunk = resample_int16(pcm, sample_rate, AEC_RATE)
        if not chunk:
            return
        rms = _rms(chunk)
        with self._lock:
            self._pcm.extend(chunk)
            extra = len(self._pcm) - FAR_HOLD_SAMPLES * 2
            if extra > 0:
                del self._pcm[:extra]
            if rms >= FAR_SILENCE_RMS:
                self._last_energy = time.monotonic()

    def pop(self, n_bytes: int) -> bytes:
        if n_bytes <= 0:
            return b""
        with self._lock:
            if len(self._pcm) < n_bytes:
                return b""
            out = bytes(self._pcm[:n_bytes])
            del self._pcm[:n_bytes]
            return out

    def playing_recently(self, window_s: float = 0.35) -> bool:
        return (time.monotonic() - self._last_energy) < window_s


TAP = FarEndTap()


def tap_playback(pcm: bytes, sample_rate: int) -> None:
    if not aec_wanted():
        return
    TAP.push(pcm, sample_rate)


def tap_clear() -> None:
    TAP.clear()


def _rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) + 1e-9)


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
            if env_debug():
                logger.debug("aec.skip extra pywebrtc-audio not installed")
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
        if env_debug():
            logger.debug(
                "aec.on AEC3 16k wet={}",
                os.environ.get("FISH_VOICE_AEC_WET", str(DEFAULT_AEC_WET)),
            )
        return _HOLDER.proc


def aec_available() -> bool:
    return load_processor() is not None


def effective_bleed_s(fallback_s: float) -> float:
    if not aec_available():
        return fallback_s
    return float(os.environ.get("FISH_VOICE_AEC_BLEED", str(DEFAULT_AEC_BLEED_S)))


def clean_mic_frame(near: bytes) -> bytes:
    """AEC when far-end has energy; otherwise return near (no AEC on silence)."""
    proc = load_processor()
    if proc is None or len(near) < 2:
        return near
    far = TAP.pop(len(near))
    if not far or _rms(far) < FAR_SILENCE_RMS:
        return near
    wet = float(os.environ.get("FISH_VOICE_AEC_WET", str(DEFAULT_AEC_WET)))
    wet = min(1.0, max(0.0, wet))
    near_a = np.frombuffer(near, dtype=np.int16)
    far_a = np.frombuffer(far, dtype=np.int16)
    if near_a.size != far_a.size:
        return near
    try:
        clean = proc.process(near_a, far_a)
    except Exception as exc:
        if env_debug():
            logger.debug("aec.fail {}", exc)
        return near
    if wet >= 0.999:
        return np.asarray(clean, dtype=np.int16).tobytes()
    mixed = (1.0 - wet) * near_a.astype(np.float32) + wet * np.asarray(clean, dtype=np.float32)
    return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()


class AdaptiveFloor:
    """Quiet-percentile RMS gate. FISH_VOICE_MIN_RMS is the seed until the window fills."""

    def __init__(
        self,
        default: float,
        *,
        window: int = 80,
        percentile: float = 20.0,
        gain: float = 2.5,
        lo: float = 80.0,
        hi: float = 450.0,
    ) -> None:
        self.default = default
        self.percentile = percentile
        self.gain = gain
        self.lo = lo
        self.hi = hi
        self.window: collections.deque[float] = collections.deque(maxlen=window)

    def observe(self, rms: float, *, quiet: bool) -> None:
        if quiet:
            self.window.append(rms)

    def value(self) -> float:
        if len(self.window) < 25:
            return self.default
        quiet = np.fromiter(self.window, dtype=np.float64)
        est = float(np.percentile(quiet, self.percentile) * self.gain)
        return float(min(self.hi, max(self.lo, est)))
