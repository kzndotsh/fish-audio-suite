"""Optional in-process AEC3. Far-end is speaker PCM; near-end is the mic.

Missing extra, ``AecTune.enabled`` False, empty far-end, or FileSink: return mic
unchanged. Bleed delay stays the fallback. PipeWire echo-cancel is a host trick,
not this module.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Final

import numpy as np

from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.tune import DEFAULT_AEC_BLEED_S, AecTune

__all__ = [
    "AEC_RATE",
    "SAMPLE_BYTES",
    "EchoCanceller",
    "FarEndTap",
    "even_pcm",
    "pcm_rms",
    "resample_int16",
]

AEC_RATE: Final = 16_000
SAMPLE_BYTES: Final = 2
_FAR_HOLD_S: Final = 2
FAR_HOLD_SAMPLES: Final = AEC_RATE * _FAR_HOLD_S
_FAR_RECENT_S: Final = 0.4
_RMS_FLOOR: Final = 1e-9
FAR_SILENCE_RMS: Final = 40.0
_FULL_WET: Final = 0.999


def even_pcm(pcm: bytes) -> bytes:
    """Drop a trailing odd byte so the buffer stays int16-aligned."""
    extra = len(pcm) % SAMPLE_BYTES
    return pcm if extra == 0 else pcm[:-extra]


def _realign(buf: bytearray) -> None:
    extra = len(buf) % SAMPLE_BYTES
    if extra:
        del buf[:extra]


_INT16_LO: Final = -32_768
_INT16_HI: Final = 32_767


def _int16_bytes(samples: Any) -> bytes:
    return np.clip(samples, _INT16_LO, _INT16_HI).astype(np.int16).tobytes()


def resample_int16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linearly resample mono int16 PCM. Empty or non-positive rates return empty bytes."""
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
        """Drop stored far-end audio and the playing-until mark."""
        with self._lock:
            self._pcm.clear()
            self._playing_until = 0.0

    def _note_playing(self, duration_s: float) -> None:
        end = time.monotonic() + max(0.0, duration_s)
        self._playing_until = max(self._playing_until, end)

    def mark_playing(self, duration_s: float) -> None:
        """Extend the playing-until mark by this chunk's duration."""
        with self._lock:
            self._note_playing(duration_s)

    def push(self, pcm: bytes, sample_rate: int) -> None:
        """Append resampled far-end PCM and keep only the recent hold window."""
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
        """Take ``n_bytes`` from the front, or empty bytes when the ring is shorter."""
        if n_bytes <= 0:
            return b""
        with self._lock:
            if len(self._pcm) < n_bytes:
                return b""
            return self._drop_front(n_bytes)

    def keep_last(self, seconds: float) -> None:
        """Drop all but the newest ``seconds`` of far-end audio.

        Parameters
        ----------
        seconds : float
            Audio to keep, at the 16 kHz far-end rate.
        """
        keep = max(0, int(seconds * AEC_RATE)) * SAMPLE_BYTES
        with self._lock:
            extra = len(self._pcm) - keep
            if extra > 0:
                self._drop_front(extra)

    def playing_recently(self, window_s: float = _FAR_RECENT_S) -> bool:
        """Return whether playback ended inside the recent window."""
        return time.monotonic() < (self._playing_until + window_s)


def pcm_rms(frame: bytes) -> float:
    """Return RMS of an int16 frame, floored so silence is not zero."""
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) + _RMS_FLOOR)


_ALIGN_MIN_S: Final = 0.06
_ALIGN_MARGIN_S: Final = 0.06
_DEFAULT_OUTPUT_LATENCY_S: Final = 0.0


class EchoCanceller:
    """One session's far-end tap, AEC3 processor, and tuning.

    Parameters
    ----------
    tune : AecTune or None, optional
        Echo settings. None uses the defaults.

    Notes
    -----
    The processor loads on first use. A missing ``pywebrtc-audio`` extra is
    remembered for this instance and the mic passes through unchanged.
    """

    def __init__(self, tune: AecTune | None = None) -> None:
        self.tune: AecTune = tune or AecTune()
        self.tap: FarEndTap = FarEndTap()
        self.output_latency_s: float = _DEFAULT_OUTPUT_LATENCY_S
        self._proc: Any = None
        self._tried = False
        self._lock = threading.Lock()

    def load(self) -> Any | None:
        """Load the optional AEC3 processor once.

        Returns
        -------
        Any or None
            The processor, or None when AEC is off or the extra is missing.
        """
        if not self.tune.enabled:
            return None
        with self._lock:
            if self._tried:
                return self._proc
            self._tried = True
            try:
                from pywebrtc_audio import AudioProcessor
            except ImportError:
                debug("aec.skip extra pywebrtc-audio not installed")
                return None
            self._proc = AudioProcessor(
                sample_rate=AEC_RATE,
                num_channels=1,
                echo_cancellation=True,
                noise_suppression=False,
                high_pass_filter=True,
                auto_gain_control=False,
                stream_delay_ms=0,
            )
            debug("aec.on AEC3 16k wet={}", self.tune.wet)
            return self._proc

    def available(self) -> bool:
        """Return whether AEC3 loaded and is enabled."""
        return self.load() is not None

    def tap_playback(self, pcm: bytes, sample_rate: int) -> None:
        """Record speaker PCM as far-end, or only the playing mark when AEC is off.

        Parameters
        ----------
        pcm : bytes
            Mono int16 PCM about to reach the speaker.
        sample_rate : int
            Rate of ``pcm``.
        """
        if self.tune.enabled:
            self.tap.push(pcm, sample_rate)
        elif sample_rate > 0:
            self.tap.mark_playing(len(even_pcm(pcm)) / SAMPLE_BYTES / sample_rate)

    def clear(self) -> None:
        """Drop stored far-end audio."""
        self.tap.clear()

    def far_end_playing(self, window_s: float = _FAR_RECENT_S) -> bool:
        """Return whether the speaker was still playing inside ``window_s``."""
        return self.tap.playing_recently(window_s)

    def align(self) -> None:
        """Keep only far-end audio that is about to be heard.

        Notes
        -----
        The sink pushes PCM when it hands it to PortAudio. It reaches the air
        ``output_latency_s`` later. The barge gate starts reading the ring
        after the bleed delay, so without this the reference would trail the
        mic by that whole delay for the rest of the reply.
        """
        keep_s = max(self.output_latency_s, _ALIGN_MIN_S) + _ALIGN_MARGIN_S
        self.tap.keep_last(keep_s)

    def effective_bleed_s(self, fallback_s: float) -> float:
        """Seconds to ignore the mic after TTS starts, so speaker bleed is not speech.

        Parameters
        ----------
        fallback_s : float
            Delay used when AEC3 is not loaded.

        Returns
        -------
        float
            ``AecTune.bleed_s`` when AEC3 is loaded, otherwise ``fallback_s``.
            Negative values become 0 or the AEC default.
        """
        if not self.available():
            return fallback_s if fallback_s >= 0 else 0.0
        bleed = self.tune.bleed_s
        return bleed if bleed >= 0 else DEFAULT_AEC_BLEED_S

    def clean(self, near: bytes) -> bytes:
        """Cancel echo when far-end has energy; otherwise return ``near``.

        Parameters
        ----------
        near : bytes
            One 16 kHz int16 mic frame.

        Returns
        -------
        bytes
            Cleaned frame, or ``near`` when AEC is off, far-end is silent, or
            the processor failed.
        """
        proc = self.load()
        if proc is None or len(near) < SAMPLE_BYTES:
            return near
        far = self.tap.pop(len(near))
        if not far or pcm_rms(far) < FAR_SILENCE_RMS:
            return near
        wet = min(1.0, max(0.0, self.tune.wet))
        near_a = np.frombuffer(near, dtype=np.int16)
        far_a = np.frombuffer(far, dtype=np.int16)
        if near_a.size != far_a.size:
            return near
        try:
            clean = proc.process(near_a, far_a)
        except Exception as exc:  # noqa: BLE001 - native AEC boundary, any failure falls back to the raw mic
            debug("aec.fail {}", exc)
            return near
        clean_a = np.asarray(clean)
        # A short or wide result would change the mic frame size and break VAD.
        if clean_a.shape != near_a.shape:
            debug("aec.fail shape {} != {}", clean_a.shape, near_a.shape)
            return near
        if wet >= _FULL_WET:
            return _int16_bytes(clean_a)
        mixed = (1.0 - wet) * near_a.astype(np.float32) + wet * clean_a.astype(np.float32)
        return _int16_bytes(mixed)
