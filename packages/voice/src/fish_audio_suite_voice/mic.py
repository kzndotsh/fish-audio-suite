"""The microphone: 30 ms frames at 16 kHz, cleaned by the session's echo canceller."""

from __future__ import annotations

import contextlib
import itertools
import queue
import threading
from collections import deque
from collections.abc import Iterator
from typing import Any, Final, Protocol, Self

from fish_audio_suite_kit import MS_PER_S
from fish_audio_suite_voice.aec import AEC_RATE, SAMPLE_BYTES, EchoCanceller
from fish_audio_suite_voice.playback import load_sounddevice, pcm_stream_kwargs
from fish_audio_suite_voice.tune import AecTune

__all__ = [
    "FRAME_BYTES",
    "FRAME_MS",
    "SAMPLE_RATE",
    "Mic",
    "SessionMic",
    "StopFlag",
]

SAMPLE_RATE: Final = AEC_RATE
FRAME_MS: Final = 30
FRAME_SAMPLES: Final = SAMPLE_RATE * FRAME_MS // MS_PER_S
FRAME_BYTES: Final = FRAME_SAMPLES * SAMPLE_BYTES
# How long a read waits for audio before it checks the stop flag again.
_POLL_S: Final = 0.2
# About two seconds of PortAudio blocks. A reader further behind than this is
# hearing the past, so the oldest block makes room for the newest.
_MAX_BACKLOG: Final = 2 * MS_PER_S // FRAME_MS


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

    @property
    def position(self) -> int:
        """The number of the next frame to arrive."""
        ...

    def frames(self, stop: StopFlag | None, since: int | None = None) -> Iterator[bytes]:
        """Yield 30 ms int16 frames until ``stop`` is set.

        Parameters
        ----------
        stop : StopFlag or None
            Ends the stream when set. None reads until the caller stops iterating.
        since : int or None, optional
            The ``position`` to read from, so a reader can pick up where another
            stopped. None reads from now.

        Returns
        -------
        Iterator
            One frame of ``FRAME_BYTES`` bytes at a time.
        """
        ...

    def release(self) -> None:
        """Close the device and forget what it heard, as mute does. A later read reopens it."""
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


class SessionMic:
    """One PortAudio input for a whole session, with every frame numbered.

    Parameters
    ----------
    device : str or int or None, optional
        PortAudio input. None uses the host default.
    aec : EchoCanceller or None, optional
        Cleans each frame against the far-end tap as it arrives. None is off.

    Notes
    -----
    The input opens on the first read and stays open until ``close``. One
    thread cleans every frame in order and keeps the last two seconds, so a
    reader that starts late, or picks up where another stopped, misses
    nothing. Opening and closing it per read lost the audio between readers,
    such as the start of the words that tripped a barge-in.
    """

    def __init__(self, device: str | int | None = None, aec: EchoCanceller | None = None) -> None:
        self.device = device
        self.aec: EchoCanceller = aec if aec is not None else _no_aec()
        self._ring: deque[bytes] = deque(maxlen=_MAX_BACKLOG)
        self._next = 0
        self._ready = threading.Condition()
        self._failure: Exception | None = None
        self._opening = threading.Lock()
        self._closed = threading.Event()
        self._stream: Any = None
        self._reader: threading.Thread | None = None
        self._running = threading.Event()

    def __enter__(self) -> Self:
        """Return the mic. It opens on the first read."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the input."""
        self.close()

    @property
    def position(self) -> int:
        """The number of the next frame to arrive. ``frames(since=...)`` reads from one."""
        with self._ready:
            return self._next

    def frames(self, stop: StopFlag | None, since: int | None = None) -> Iterator[bytes]:
        """Yield cleaned frames until ``stop`` is set.

        Parameters
        ----------
        stop : StopFlag or None
            Ends the stream when set.
        since : int or None, optional
            The ``position`` to read from. None reads from now. A frame older than
            the two seconds kept is skipped.

        Yields
        ------
        bytes
            One 30 ms int16 frame.

        Raises
        ------
        Exception
            Whatever stopped the input, such as a lost device.
        """
        with self._ready:
            index = self._next if since is None else since
        # Opened after the start is fixed, so a first read misses nothing the
        # device delivers while it opens.
        self._open()
        while stop is None or not stop.is_set():
            with self._ready:
                if self._failure is not None:
                    raise self._failure
                if index >= self._next:
                    self._ready.wait(_POLL_S)
                    continue
                oldest = self._next - len(self._ring)
                index = max(index, oldest)
                batch = list(itertools.islice(self._ring, index - oldest, None))
            for frame in batch:
                if stop is not None and stop.is_set():
                    return
                index += 1
                yield frame

    def release(self) -> None:
        """Close the device and drop the audio kept, as mute does. A later read reopens it.

        Notes
        -----
        Positions keep counting across a release, so an old ``since`` is simply
        older than anything kept.
        """
        with self._opening:
            self._running.clear()
            stream, self._stream = self._stream, None
            reader, self._reader = self._reader, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()
        if reader is not None:
            reader.join(timeout=_POLL_S * 5)
        with self._ready:
            self._ring.clear()

    def close(self) -> None:
        """Release the device for good. Safe to call more than once."""
        self._closed.set()
        self.release()

    def _open(self) -> None:
        with self._opening:
            if self._stream is not None or self._closed.is_set():
                return
            sd = load_sounddevice()
            blocks: queue.Queue[bytes] = queue.Queue(maxsize=_MAX_BACKLOG)

            def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
                del frames, time_info, status
                # PortAudio's thread is the only writer, so after one block is taken
                # out there is room again.
                if blocks.full():
                    with contextlib.suppress(queue.Empty):
                        blocks.get_nowait()
                blocks.put_nowait(bytes(indata))

            stream = sd.RawInputStream(
                **pcm_stream_kwargs(SAMPLE_RATE, self.device),
                blocksize=FRAME_SAMPLES,
                callback=callback,
            )
            stream.start()
            self._stream = stream
            with self._ready:
                self._failure = None
            running = threading.Event()
            running.set()
            self._running = running
            self._reader = threading.Thread(
                target=self._read, args=(blocks, running), name="mic", daemon=True
            )
            self._reader.start()

    def _read(self, blocks: queue.Queue[bytes], running: threading.Event) -> None:
        pending = bytearray()
        try:
            while running.is_set():
                try:
                    chunk = blocks.get(timeout=_POLL_S)
                except queue.Empty:
                    continue
                for frame in _take_full_frames(pending, chunk, FRAME_BYTES):
                    # Cleaned as it arrives, so the echo canceller sees the mic and
                    # the speaker in step even while no one is reading.
                    clean = self.aec.clean(frame)
                    with self._ready:
                        self._ring.append(clean)
                        self._next += 1
                        self._ready.notify_all()
        except Exception as exc:  # noqa: BLE001 - mic-thread boundary, handed to the readers
            with self._ready:
                self._failure = exc
                self._ready.notify_all()
