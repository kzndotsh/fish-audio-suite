"""The microphone: 30 ms frames at 16 kHz, cleaned by the session's echo canceller."""

from __future__ import annotations

import queue
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from fish_audio_suite_kit import MS_PER_S
from fish_audio_suite_voice.aec import AEC_RATE, SAMPLE_BYTES, EchoCanceller
from fish_audio_suite_voice.playback import load_sounddevice, pcm_stream_kwargs
from fish_audio_suite_voice.tune import AecTune

__all__ = [
    "FRAME_BYTES",
    "FRAME_MS",
    "SAMPLE_RATE",
    "Mic",
    "SounddeviceMic",
    "StopFlag",
]

SAMPLE_RATE: Final = AEC_RATE
FRAME_MS: Final = 30
FRAME_SAMPLES: Final = SAMPLE_RATE * FRAME_MS // MS_PER_S
FRAME_BYTES: Final = FRAME_SAMPLES * SAMPLE_BYTES
# How long a read waits for audio before it checks the stop flag again.
_POLL_S: Final = 0.2


class StopFlag(Protocol):
    """Anything with ``is_set``, such as a ``threading.Event``, that ends a mic stream."""

    def is_set(self) -> bool:
        """Return whether the stream should stop."""
        ...


class Mic(Protocol):
    """Where the listen, streaming and barge-in readers get their audio.

    Notes
    -----
    ``aec`` is the session's echo canceller. ``frames`` has already cleaned each
    frame with it, and barge-in reads it to know whether the speaker is playing.
    """

    @property
    def aec(self) -> EchoCanceller:
        """The echo canceller that cleans these frames."""
        ...

    def frames(self, stop: StopFlag | None) -> Iterator[bytes]:
        """Yield 30 ms int16 frames until ``stop`` is set.

        Parameters
        ----------
        stop : StopFlag or None
            Ends the stream when set. None reads until the caller stops iterating.

        Returns
        -------
        Iterator
            One frame of ``FRAME_BYTES`` bytes at a time.
        """
        ...


def _no_aec() -> EchoCanceller:
    return EchoCanceller(AecTune(enabled=False))


def _take_full_frames(pending: bytearray, chunk: bytes, frame_bytes: int) -> list[bytes]:
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


@dataclass(frozen=True, slots=True)
class SounddeviceMic:
    """A PortAudio input opened for each read.

    Parameters
    ----------
    device : str or int or None, optional
        PortAudio input. None uses the host default.
    aec : EchoCanceller, optional
        Cleans each frame against the far-end tap. The default is off.
    """

    device: str | int | None = None
    aec: EchoCanceller = field(default_factory=_no_aec)

    def frames(self, stop: StopFlag | None) -> Iterator[bytes]:
        """Open the input and yield cleaned frames until ``stop`` is set.

        Parameters
        ----------
        stop : StopFlag or None
            Ends the stream when set.

        Yields
        ------
        bytes
            One 30 ms int16 frame.
        """
        sd = load_sounddevice()
        audio: queue.Queue[bytes] = queue.Queue()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            audio.put(bytes(indata))

        with sd.RawInputStream(
            **pcm_stream_kwargs(SAMPLE_RATE, self.device),
            blocksize=FRAME_SAMPLES,
            callback=callback,
        ):
            pending = bytearray()
            while stop is None or not stop.is_set():
                try:
                    chunk = audio.get(timeout=_POLL_S)
                except queue.Empty:
                    continue
                for frame in _take_full_frames(pending, chunk, FRAME_BYTES):
                    yield self.aec.clean(frame)
