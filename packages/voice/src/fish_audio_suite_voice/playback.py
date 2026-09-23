"""Playback sinks for Fish PCM (and optional mpv mp3)."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fish_audio_suite_kit import MS_PER_S, SuiteDefaults
from fish_audio_suite_voice.aec import SAMPLE_BYTES, even_pcm, tap_clear, tap_playback

PORTAUDIO_HINT = """sounddevice needs the PortAudio C library (the Python wheel does not ship it).
  Debian/Ubuntu: sudo apt install libportaudio2
  Fedora: sudo dnf install portaudio
  macOS: brew install portaudio
  NixOS: nix run .#fish-audio-suite-voice   # flake prefixes LD_LIBRARY_PATH
          or ./packages/voice/dev.sh"""


class PortAudioMissingError(OSError):
    """sounddevice imported, but libportaudio is not on the loader path."""


def missing_portaudio() -> PortAudioMissingError:
    """Build the error raised when libportaudio is not loadable."""
    return PortAudioMissingError(PORTAUDIO_HINT)


_MONO = 1
_DEFAULT_RATE = SuiteDefaults().sample_rate


def pcm_stream_kwargs(sample_rate: int, device: str | int | None) -> dict[str, Any]:
    """Return RawStream kwargs for mono int16 PCM."""
    return {
        "samplerate": sample_rate,
        "channels": _MONO,
        "dtype": "int16",
        "device": device,
    }


def load_sounddevice() -> Any:
    """Import sounddevice, mapping a missing library to PortAudioMissingError."""
    try:
        import sounddevice as sd
    except OSError as exc:
        raise missing_portaudio() from exc
    return sd


@runtime_checkable
class PlaybackSink(Protocol):
    """Where one TTS turn writes audio.

    ``start`` opens the device or file. ``write`` accepts encoded or PCM
    chunks. ``finish`` closes them; ``kill=True`` means barge-in, so mpv
    should stop instead of playing out the buffer. ``bytes_played`` is what
    actually reached the device, which ``spoken_so_far`` is cut to.
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


class _Played:
    def __init__(self) -> None:
        self._played = 0

    def _reset_played(self) -> None:
        self._played = 0

    def _count(self, chunk: bytes) -> None:
        self._played += len(chunk)

    def bytes_played(self) -> int:
        """Return bytes accepted by this sink since the last start."""
        return self._played


def write_mono_wav(target: Any, pcm: bytes, sample_rate: int) -> None:
    """Write mono int16 PCM as a WAV file."""
    with wave.open(target, "wb") as wf:
        wf.setnchannels(_MONO)
        wf.setsampwidth(SAMPLE_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


class FileSink(_Played):
    """Collect a turn and write it as WAV or raw PCM on finish."""

    def __init__(self, path: Path, *, sample_rate: int = _DEFAULT_RATE, wav: bool = True) -> None:
        super().__init__()
        self.path = path
        self.sample_rate = sample_rate
        self.wav = wav
        self._buf = bytearray()

    def start(self) -> None:
        """Clear the buffer for a new turn."""
        self._buf.clear()
        self._reset_played()

    def write(self, chunk: bytes) -> None:
        """Append one encoded or PCM chunk."""
        if not chunk:
            return
        self._buf.extend(chunk)
        self._count(chunk)

    def finish(self, *, kill: bool = False) -> None:
        """Write the buffer unless barge-in asked to drop it."""
        if kill:
            return
        if self.wav:
            write_mono_wav(str(self.path), bytes(self._buf), self.sample_rate)
        else:
            self.path.write_bytes(bytes(self._buf))


class StdoutSink(_Played):
    """Write raw chunks to stdout."""

    def start(self) -> None:
        """Reset the played-byte counter."""
        self._reset_played()

    def write(self, chunk: bytes) -> None:
        """Write one chunk to stdout and count it."""
        if not chunk:
            return
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        self._count(chunk)

    def finish(self, *, kill: bool = False) -> None:
        """Stdout has nothing to close."""
        return


DAC_SLICE_MS = 30
_MPV_WAIT_S = 90
_MPV_KILL_S = 2
_MPV_BUFFER_S = 0.2
_SPEAKER_SINKS = frozenset({"sounddevice", "speakers", "pcm"})


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
    The far-end tap used by AEC is filled before each blocking write, so the
    reference matches what is about to hit the speaker. Missing PortAudio
    raises ``PortAudioMissingError`` when the stream opens, not at import.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _DEFAULT_RATE,
        device: str | int | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        self._cancel = cancel
        self._stream: Any = None
        self._odd = b""

    def start(self) -> None:
        """Open a PortAudio output stream and clear the far-end tap."""
        sd = load_sounddevice()

        self._reset_played()
        self._odd = b""
        tap_clear()
        self._stream = sd.RawOutputStream(**pcm_stream_kwargs(self.sample_rate, self.device))
        self._stream.start()

    def write(self, chunk: bytes) -> None:
        """Play PCM in short slices, tapping far-end before each blocking write."""
        if not chunk or self._stream is None:
            return
        data = self._odd + chunk
        self._odd = b""
        if len(data) % SAMPLE_BYTES:
            self._odd = data[-1:]
            data = data[:-1]
        step = dac_slice_bytes(self.sample_rate)
        for piece in iter_pcm_slices(data, step):
            if self._cancel is not None and self._cancel.is_set():
                self._odd = b""
                return
            # Tap before the blocking DAC write so AEC has far-end while this slice plays.
            tap_playback(piece, self.sample_rate)
            self._stream.write(piece)
            self._count(piece)

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
        tap_clear()


def _reap(proc: subprocess.Popen[bytes], *, timeout: float) -> None:
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)


class MpvSink(_Played):
    """Optional legacy mp3 stdin player. Not required on PATH for other sinks."""

    def __init__(self) -> None:
        super().__init__()
        self.proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Kill any previous mpv and start a new stdin player."""
        self.finish(kill=True)
        self._reset_played()
        self.proc = subprocess.Popen(
            [
                "mpv",
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
        with contextlib.suppress(Exception):
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        if kill:
            _reap(proc, timeout=_MPV_KILL_S)
        else:
            try:
                proc.wait(timeout=_MPV_WAIT_S)
            except subprocess.TimeoutExpired:
                _reap(proc, timeout=_MPV_KILL_S)
        self.proc = None


DEFAULT_PLAYBACK = "sounddevice"


def playback_key(name: str) -> str:
    """Normalize a playback name.

    Parameters
    ----------
    name : str
        ``FISH_PLAYBACK`` or ``--playback``.

    Returns
    -------
    str
        Stripped, lowercased name. Unknown names are returned as written so
        ``duplex_playback_problem`` can reject them.
    """
    return name.strip().lower()


def duplex_playback_problem(name: str) -> str | None:
    """Why this sink cannot play a duplex reply. None means it can."""
    key = playback_key(name)
    if key == "file":
        return "file playback needs a path"
    if key == "mpv" and shutil.which("mpv") is None:
        return "mpv is not on PATH"
    if key in {"stdout", "mpv"} or key in _SPEAKER_SINKS:
        return None
    return f"unknown playback sink {name!r}"


def audio_format_for(playback: str) -> str:
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
    if playback_key(playback) == "mpv":
        return "mp3"
    return "pcm"


def make_sink(
    name: str,
    *,
    path: Path | None = None,
    sample_rate: int = _DEFAULT_RATE,
    device: str | int | None = None,
    cancel: threading.Event | None = None,
) -> PlaybackSink:
    """Build the sink named by ``FISH_PLAYBACK`` or ``--playback``.

    Parameters
    ----------
    name : str
        ``sounddevice``, ``file``, ``stdout``, or ``mpv``.
    path : Path or None, optional
        Required for ``file``.
    sample_rate : int, optional
        PCM rate. Ignored by mpv, which receives MP3.
    device : str or int or None, optional
        PortAudio device.
    cancel : threading.Event or None, optional
        Passed to the sounddevice sink so a barge-in can stop the stream.

    Returns
    -------
    PlaybackSink
        The concrete sink.

    Raises
    ------
    ValueError
        ``file`` without ``path``, or a name this function does not know.
    """
    key = playback_key(name)
    if key == "file":
        if path is None:
            raise ValueError("file sink requires path")
        return FileSink(path, sample_rate=sample_rate, wav=path.suffix.lower() == ".wav")
    if key == "stdout":
        return StdoutSink()
    if key == "mpv":
        return MpvSink()
    if key in _SPEAKER_SINKS:
        return SounddeviceSink(sample_rate=sample_rate, device=device, cancel=cancel)
    raise ValueError(f"unknown playback sink {name!r}")
