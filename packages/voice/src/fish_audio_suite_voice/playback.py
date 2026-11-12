"""Playback sinks for Fish audio: the sound card, a file, stdout, or mpv for mp3."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from fish_audio_suite_kit import MS_PER_S, AudioFormat, SuiteDefaults
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.events import EVENTS, OutputLevel
from fish_audio_suite_voice.pcm import SAMPLE_BYTES, even_pcm, pcm_rms, wav_rate, write_mono_wav
from fish_audio_suite_voice.playback_kind import PlaybackKind, parse_playback
from fish_audio_suite_voice.portaudio import load_sounddevice, pcm_stream_kwargs

__all__ = [
    "FileSink",
    "MpvSink",
    "PlaybackSink",
    "SounddeviceSink",
    "StdoutSink",
    "audio_format_for",
    "dac_slice_bytes",
    "duplex_playback_problem",
    "iter_pcm_slices",
    "make_sink",
]

_DEFAULT_RATE: Final = SuiteDefaults().sample_rate


@runtime_checkable
class PlaybackSink(Protocol):
    """Where one TTS turn writes audio.

    ``start`` opens the device or file. ``write`` accepts encoded or PCM
    chunks. ``finish`` closes them; ``kill=True`` means barge-in, so mpv
    should stop instead of playing out the buffer. ``bytes_played`` is what
    the sink accepted for playback, which ``spoken_so_far`` is cut to.
    """

    def start(self) -> None:
        """Open the sink. Called once per turn."""

    def write(self, chunk: bytes) -> None:
        """Append one audio chunk.

        Parameters
        ----------
        chunk : bytes
            PCM or encoded audio, depending on the sink.
        """

    def finish(self, *, kill: bool = False) -> None:
        """Close the sink.

        Parameters
        ----------
        kill : bool, optional
            True on barge-in or Ctrl+C. The sink should drop unplayed audio.
        """

    def bytes_played(self) -> int:
        """Return how many bytes were accepted for playback.

        Returns
        -------
        int
            Used to trim ``spoken_so_far`` when the turn was cancelled mid-word.
        """
        ...

    @property
    def output_latency_s(self) -> float:
        """Seconds of audio the device buffers ahead of the speaker, or 0 when unknown.

        Returns
        -------
        float
            Used to discount audio that was written but not yet heard when
            ``spoken_so_far`` is estimated.
        """
        ...


OUTPUT_LEVEL_EVERY_S: Final = (
    0.09  # one level per this much audio: as often as the mic's, so a strip scrolls evenly
)


class _LevelTap:
    """Reports how loud the audio being played is, a few times a second.

    Parameters
    ----------
    sample_rate : int
        Samples per second of the mono int16 audio fed in.
    """

    def __init__(self, sample_rate: int) -> None:
        self._block = max(SAMPLE_BYTES, round(sample_rate * OUTPUT_LEVEL_EVERY_S) * SAMPLE_BYTES)
        self._pending = b""

    def feed(self, pcm: bytes) -> None:
        """Add audio that is going to the speaker, and report each full block of it."""
        self._pending += pcm
        while len(self._pending) >= self._block:
            block, self._pending = self._pending[: self._block], self._pending[self._block :]
            EVENTS.emit(OutputLevel(pcm_rms(block)))

    def clear(self) -> None:
        """Forget a part-filled block, as at the start of a turn."""
        self._pending = b""


class _Played:
    def __init__(self) -> None:
        self._played = 0
        self.output_latency_s: float = 0.0

    def _reset_played(self) -> None:
        self._played = 0

    def _count(self, chunk: bytes) -> None:
        self._played += len(chunk)

    def bytes_played(self) -> int:
        """Return bytes accepted by this sink since the last start."""
        return self._played


class FileSink(_Played):
    """Collect a turn and write it as WAV or raw PCM on finish."""

    def __init__(self, path: Path, *, sample_rate: int = _DEFAULT_RATE, wav: bool = True) -> None:
        super().__init__()
        self.path: Path = path
        self.sample_rate: int = wav_rate(sample_rate)
        self.wav: bool = wav
        self._buf = bytearray()

    def start(self) -> None:
        """Clear the buffer for a new turn."""
        self._buf.clear()
        self._reset_played()

    def write(self, chunk: bytes) -> None:
        """Append one encoded or PCM chunk. It is not played until finish."""
        if not chunk:
            return
        self._buf.extend(chunk)

    def finish(self, *, kill: bool = False) -> None:
        """Write the buffer unless it is empty or barge-in asked to drop it."""
        # An empty or cancelled turn must not replace the last file, and
        # history must not record audio the file never received.
        if kill or not self._buf:
            return
        # A trailing odd byte is half an int16 sample. The data chunk would
        # not match the block align, and a strict reader rejects the file.
        pcm = even_pcm(bytes(self._buf))
        if not pcm:
            return
        # Count only after the file is on disk, so a full disk that raises
        # does not leave history saying the user heard the reply.
        if self.wav:
            write_mono_wav(str(self.path), pcm, self.sample_rate)
        else:
            self.path.write_bytes(pcm)
        self._played = len(pcm)


class StdoutSink(_Played):
    """Write raw PCM chunks to stdout."""

    def __init__(self) -> None:
        super().__init__()
        self._odd = b""

    def start(self) -> None:
        """Reset the played-byte counter and any held half-sample."""
        self._reset_played()
        self._odd = b""

    def write(self, chunk: bytes) -> None:
        """Write aligned PCM, holding a trailing odd byte for the next chunk."""
        if not chunk:
            return
        # An odd chunk shifts every later int16 sample by one byte.
        data = self._odd + chunk
        self._odd = b""
        if len(data) % SAMPLE_BYTES:
            self._odd = data[-1:]
            data = data[:-1]
        if not data:
            return
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        self._count(data)

    def finish(self, *, kill: bool = False) -> None:
        """Drop a held half-sample. It is not a complete int16."""
        del kill
        self._odd = b""


DAC_SLICE_MS: Final = 30
# A normal finish lets mpv play out what it holds, up to this long.
_MPV_WAIT_S: Final = 90
_MPV_KILL_S: Final = 2
# A small buffer keeps the first words quick.
_MPV_BUFFER_S: Final = 0.2


def dac_slice_bytes(sample_rate: int, frame_ms: int = DAC_SLICE_MS) -> int:
    """Return an even int16 byte count for one DAC slice."""
    return max(1, sample_rate * frame_ms // MS_PER_S) * SAMPLE_BYTES


def iter_pcm_slices(chunk: bytes, slice_bytes: int) -> Iterator[bytes]:
    """Yield even PCM pieces of ``slice_bytes``, or the whole chunk when that is 0."""
    if slice_bytes <= 0:
        if chunk:
            yield chunk
        return
    for i in range(0, len(chunk), slice_bytes):
        piece = even_pcm(chunk[i : i + slice_bytes])
        if piece:
            yield piece


class SounddeviceSink(_Played):
    """Play PCM through PortAudio in about 30 ms slices.

    Notes
    -----
    ``output_latency_s`` is the device buffer reported by PortAudio, so the
    spoken-prefix estimate can discount audio that has not been heard yet.
    The far-end tap used by AEC (``aec``) is filled before each blocking write,
    so the reference matches what is about to hit the speaker. The stream's
    output latency is recorded on it so the barge gate can align the reference.
    Without ``aec`` nothing is tapped. Missing PortAudio
    raises ``PortAudioMissingError`` when the stream opens, not at import.
    """

    output_latency_s: float

    def __init__(
        self,
        *,
        sample_rate: int = _DEFAULT_RATE,
        device: str | int | None = None,
        cancel: threading.Event | None = None,
        aec: EchoCanceller | None = None,
        latency: str = "low",
    ) -> None:
        super().__init__()
        self.latency = latency
        self.sample_rate: int = wav_rate(sample_rate)
        self.device: str | int | None = device
        self._cancel = cancel
        self._aec = aec
        self._stream: Any = None
        self._odd = b""
        self._tap = _LevelTap(self.sample_rate)
        self._wrote_at = 0.0  # when the last slice went to the device
        self._chunk_idle_ms = 0.0

    def start(self) -> None:
        """Open a PortAudio output stream and clear the far-end tap."""
        sd = load_sounddevice()

        self._reset_played()
        self._odd = b""
        self._tap.clear()
        self._wrote_at = 0.0
        if self._aec is not None:
            self._aec.clear()
        self._stream = sd.RawOutputStream(
            **pcm_stream_kwargs(self.sample_rate, self.device, latency=self.latency)
        )
        self._stream.start()
        self.output_latency_s = _stream_latency_s(self._stream)
        if self._aec is not None:
            self._aec.output_latency_s = self.output_latency_s

    def write(self, chunk: bytes) -> None:
        """Play PCM in short slices, tapping far-end before each blocking write."""
        # Writes run on a worker thread, so finish() may clear the stream
        # between slices. Hold the stream this write started with.
        stream = self._stream
        if not chunk or stream is None:
            return
        data = self._odd + chunk
        self._odd = b""
        if len(data) % SAMPLE_BYTES:
            self._odd = data[-1:]
            data = data[:-1]
        step = dac_slice_bytes(self.sample_rate)
        # How long the device was left without a new chunk: audio that arrived late.
        arrived = time.perf_counter()
        self._chunk_idle_ms = (arrived - self._wrote_at) * 1000 if self._wrote_at else 0.0
        for piece in iter_pcm_slices(data, step):
            if self._stream is not stream or (self._cancel is not None and self._cancel.is_set()):
                self._odd = b""
                return
            # Tap before the blocking DAC write so AEC has far-end while this slice plays.
            if self._aec is not None:
                self._aec.tap_playback(piece, self.sample_rate)
            # PortAudio reports True when the device ran dry before this write,
            # which plays as a click or a gap.
            before = time.perf_counter()
            underflowed = stream.write(piece)
            took_ms = (time.perf_counter() - before) * 1000
            if underflowed:
                # chunk_wait is how long since the last slice when this chunk came in, in_python
                # how long this thread spent between two writes, and write how long this write
                # call took, which includes waiting to get the GIL back after it.
                debug(
                    "tts.underrun after {} kB played "
                    "(chunk_wait {:.0f} ms, in_python {:.0f} ms, write {:.0f} ms)",
                    self._played // 1000,
                    self._chunk_idle_ms,
                    (before - self._wrote_at) * 1000 if self._wrote_at else 0.0,
                    took_ms,
                )
            self._wrote_at = time.perf_counter()
            self._count(piece)
            self._tap.feed(piece)

    def finish(self, *, kill: bool = False) -> None:
        """Stop or abort the PortAudio stream and close it."""
        self._odd = b""
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        with contextlib.suppress(Exception):
            if kill:
                stream.abort()
            else:
                stream.stop()
        with contextlib.suppress(Exception):
            stream.close()
        if self._aec is not None:
            self._aec.clear()


def _stream_latency_s(stream: Any) -> float:
    # sounddevice reports one float for a stream. Some builds report a pair.
    latency = getattr(stream, "latency", 0.0)
    if isinstance(latency, (tuple, list)):
        latency = latency[-1] if latency else 0.0
    try:
        value = float(latency)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 else 0.0


def _reap(proc: subprocess.Popen[bytes], *, timeout: float) -> None:
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)


class MpvSink(_Played):
    """Optional mp3 stdin player. Not required on PATH for other sinks."""

    def __init__(self) -> None:
        super().__init__()
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Kill any previous mpv and start a new stdin player."""
        self.finish(kill=True)
        self._reset_played()
        mpv = shutil.which("mpv")
        if mpv is None:
            raise FileNotFoundError(
                "mpv is not on PATH. Install mpv or pick another sink with --playback."
            )
        # A fixed argument list with the resolved binary and no shell.
        self.proc = subprocess.Popen(  # noqa: S603
            [
                mpv,
                "--no-terminal",
                "--really-quiet",
                f"--audio-buffer={_MPV_BUFFER_S}",
                "--demuxer-lavf-format=mp3",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, chunk: bytes) -> None:
        """Write one mp3 chunk to mpv stdin."""
        if not self.proc or not self.proc.stdin or not chunk:
            return
        try:
            self.proc.stdin.write(chunk)
            self.proc.stdin.flush()
            self._count(chunk)
        except BrokenPipeError:
            pass

    def finish(self, *, kill: bool = False) -> None:
        """Close mpv stdin, or kill the process on barge-in."""
        proc = self.proc
        if proc is None:
            return
        # Closing stdin is EOF. mpv would play the buffered mp3 before exit.
        # Barge-in has to kill the process while the pipe is still open.
        if kill:
            _reap(proc, timeout=_MPV_KILL_S)
            self.proc = None
            return
        with contextlib.suppress(Exception):
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        try:
            proc.wait(timeout=_MPV_WAIT_S)
        except subprocess.TimeoutExpired:
            _reap(proc, timeout=_MPV_KILL_S)
        self.proc = None


def duplex_playback_problem(name: str) -> str | None:
    """Why this sink cannot play a duplex reply. None means it can."""
    kind = parse_playback(name)
    if kind is None:
        return f"unknown playback sink {name!r}"
    if kind is PlaybackKind.FILE:
        return "file playback needs a path"
    if kind is PlaybackKind.MPV and shutil.which("mpv") is None:
        return "mpv is not on PATH"
    return None


def audio_format_for(playback: str) -> AudioFormat:
    """Fish format for a sink.

    Parameters
    ----------
    playback : str
        Sink name.

    Returns
    -------
    str
        ``mp3`` for mpv. ``pcm`` for sounddevice, file, and stdout.
    """
    if parse_playback(playback) is PlaybackKind.MPV:
        return "mp3"
    return "pcm"


def make_sink(
    name: str,
    *,
    path: Path | None = None,
    sample_rate: int = _DEFAULT_RATE,
    device: str | int | None = None,
    cancel: threading.Event | None = None,
    aec: EchoCanceller | None = None,
    latency: str = "low",
) -> PlaybackSink:
    """Build the sink named by ``FISH_VOICE_PLAYBACK`` or ``--playback``.

    Parameters
    ----------
    name : str
        A ``PlaybackKind`` value, such as ``sounddevice``, ``file``, ``stdout`` or ``mpv``.
    path : Path or None, optional
        Required for ``file``.
    sample_rate : int, optional
        PCM rate. Ignored by mpv, which receives MP3.
    device : str or int or None, optional
        PortAudio device.
    cancel : threading.Event or None, optional
        Passed to the sounddevice sink so a barge-in can stop the stream.
    aec : EchoCanceller or None, optional
        Far-end tap filled by the sounddevice sink.
    latency : str, optional
        The sounddevice sink's buffer size, ``low`` or ``high``.

    Returns
    -------
    PlaybackSink
        The concrete sink.

    Raises
    ------
    ValueError
        ``file`` without ``path``, or a name this function does not know.
    """
    kind = parse_playback(name)
    if kind is PlaybackKind.FILE:
        if path is None:
            raise ValueError("file sink requires path")
        return FileSink(path, sample_rate=sample_rate, wav=path.suffix.lower() == ".wav")
    if kind is PlaybackKind.STDOUT:
        return StdoutSink()
    if kind is PlaybackKind.MPV:
        return MpvSink()
    if kind is not None and kind.is_speaker:
        return SounddeviceSink(
            sample_rate=sample_rate, device=device, cancel=cancel, aec=aec, latency=latency
        )
    raise ValueError(f"unknown playback sink {name!r}")
