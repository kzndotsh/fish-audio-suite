"""The audio the voice package records, streams and saves: 16 kHz mono int16 PCM, in 30 ms frames.

The mic, the voice detector, the echo canceller and the streaming recognisers all take this one
format, so its numbers and the small helpers every module needs for it live here.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import BinaryIO, Final

import numpy as np

from fish_audio_suite_kit import MS_PER_S, SuiteDefaults

__all__ = [
    "BYTES_PER_S",
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "LISTEN_HEARTBEAT_FRAMES",
    "MIC_LEVEL_EVERY_FRAMES",
    "SAMPLE_BYTES",
    "SAMPLE_DTYPE",
    "SAMPLE_RATE",
    "even_pcm",
    "pcm_rms",
    "take_frames",
    "wav_rate",
    "write_mono_wav",
]

# WebRTC's echo canceller and voice detector both run at this rate.
SAMPLE_RATE: Final = 16_000
SAMPLE_BYTES: Final = 2
# Little-endian int16, as NumPy reads the samples.
SAMPLE_DTYPE: Final = "<i2"
BYTES_PER_S: Final = SAMPLE_RATE * SAMPLE_BYTES
# The unit of every FISH_VOICE_*_FRAMES setting.
FRAME_MS: Final = 30
FRAME_SAMPLES: Final = SAMPLE_RATE * FRAME_MS // MS_PER_S
FRAME_BYTES: Final = FRAME_SAMPLES * SAMPLE_BYTES
# How often a mic reader reports while it listens: one heartbeat trace line every this many
# frames (about 600 ms), and one MicLevel event every this many (about 90 ms), so a display
# can draw a meter.
LISTEN_HEARTBEAT_FRAMES: Final = 20
MIC_LEVEL_EVERY_FRAMES: Final = 3

_MONO: Final = 1
_RMS_FLOOR: Final = 1e-9
_DEFAULT_RATE: Final = SuiteDefaults().sample_rate
# wave stores the rate as an unsigned 32-bit field. Larger values raise while
# the samples are already in memory, so the file is never written.
_WAV_RATE_HI: Final = 2**32 - 1


def even_pcm(pcm: bytes) -> bytes:
    """Drop a trailing odd byte so the buffer stays int16-aligned.

    Parameters
    ----------
    pcm : bytes
        Int16 samples, perhaps with half a sample at the end.

    Returns
    -------
    bytes
        Whole samples only.
    """
    extra = len(pcm) % SAMPLE_BYTES
    return pcm if extra == 0 else pcm[:-extra]


def pcm_rms(frame: bytes) -> float:
    """Return the RMS of an int16 frame, floored so silence is not zero.

    Parameters
    ----------
    frame : bytes
        Int16 samples.

    Returns
    -------
    float
        The level in int16 units. 0 for an empty frame.
    """
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) + _RMS_FLOOR)


def take_frames(pending: bytearray, chunk: bytes, frame_bytes: int = FRAME_BYTES) -> list[bytes]:
    """Add ``chunk`` to ``pending`` and take out every whole frame.

    Parameters
    ----------
    pending : bytearray
        What is left over from earlier chunks. The part of a frame still missing stays in it.
    chunk : bytes
        The audio that just arrived.
    frame_bytes : int, optional
        The size of one frame. Below 1, nothing is taken or kept.

    Returns
    -------
    list of bytes
        The whole frames, oldest first. Pass a fresh ``bytearray()`` to cut one buffer into
        frames and drop its tail.
    """
    # PortAudio may deliver a short buffer or more than one block. A short
    # buffer has to wait for the next callback, and the tail of a long buffer
    # has to be kept. Dropping either cuts a hole in the utterance.
    if frame_bytes < 1:
        return []
    pending.extend(chunk)
    frames: list[bytes] = []
    while len(pending) >= frame_bytes:
        frames.append(bytes(pending[:frame_bytes]))
        del pending[:frame_bytes]
    return frames


def wav_rate(sample_rate: int) -> int:
    """Return ``sample_rate``, or the default rate when a WAV header cannot store it.

    Parameters
    ----------
    sample_rate : int
        Samples per second.

    Returns
    -------
    int
        The rate unchanged from 1 to 2**32 - 1. Anything else, which the WAV writer would
        reject after the audio was collected, is the default Fish output rate.
    """
    # 0 and negative rates raise "sampling rate not specified".
    if sample_rate < 1 or sample_rate > _WAV_RATE_HI:
        return _DEFAULT_RATE
    return sample_rate


def write_mono_wav(target: str | Path | BinaryIO, pcm: bytes, sample_rate: int) -> None:
    """Write mono int16 PCM as a WAV file.

    Parameters
    ----------
    target : str or Path or BinaryIO
        A path, or a writable binary file object.
    pcm : bytes
        Mono int16 samples.
    sample_rate : int
        Samples per second. A value the WAV header cannot store becomes the default rate.
    """
    with wave.open(str(target) if isinstance(target, Path) else target, "wb") as wf:
        wf.setnchannels(_MONO)
        wf.setsampwidth(SAMPLE_BYTES)
        wf.setframerate(wav_rate(sample_rate))
        wf.writeframes(pcm)
