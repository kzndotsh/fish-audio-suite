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
    return PortAudioMissingError(PORTAUDIO_HINT)


_MONO = 1
_DEFAULT_RATE = SuiteDefaults().sample_rate


def pcm_stream_kwargs(sample_rate: int, device: str | int | None) -> dict[str, Any]:
    return {
        "samplerate": sample_rate,
        "channels": _MONO,
        "dtype": "int16",
        "device": device,
    }


def load_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except OSError as exc:
        raise missing_portaudio() from exc
    return sd


@runtime_checkable
class PlaybackSink(Protocol):
    def start(self) -> None: ...
    def write(self, chunk: bytes) -> None: ...
    def finish(self, *, kill: bool = False) -> None: ...
    def bytes_played(self) -> int: ...


class _Played:
    def __init__(self) -> None:
        self._played = 0

    def _reset_played(self) -> None:
        self._played = 0

    def _count(self, chunk: bytes) -> None:
        self._played += len(chunk)

    def bytes_played(self) -> int:
        return self._played


def write_mono_wav(target: Any, pcm: bytes, sample_rate: int) -> None:
    with wave.open(target, "wb") as wf:
        wf.setnchannels(_MONO)
        wf.setsampwidth(SAMPLE_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


class FileSink(_Played):
    def __init__(self, path: Path, *, sample_rate: int = _DEFAULT_RATE, wav: bool = True) -> None:
        super().__init__()
        self.path = path
        self.sample_rate = sample_rate
        self.wav = wav
        self._buf = bytearray()

    def start(self) -> None:
        self._buf.clear()
        self._reset_played()

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buf.extend(chunk)
        self._count(chunk)

    def finish(self, *, kill: bool = False) -> None:
        if kill:
            return
        if self.wav:
            write_mono_wav(str(self.path), bytes(self._buf), self.sample_rate)
        else:
            self.path.write_bytes(bytes(self._buf))


class StdoutSink(_Played):
    def start(self) -> None:
        self._reset_played()

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        self._count(chunk)

    def finish(self, *, kill: bool = False) -> None:
        return


DAC_SLICE_MS = 30
_MPV_WAIT_S = 90
_MPV_KILL_S = 2
_MPV_BUFFER_S = 0.2
_SPEAKER_SINKS = frozenset({"sounddevice", "speakers", "pcm"})


def dac_slice_bytes(sample_rate: int, frame_ms: int = DAC_SLICE_MS) -> int:
    return max(1, sample_rate * frame_ms // MS_PER_S) * SAMPLE_BYTES


def iter_pcm_slices(chunk: bytes, slice_bytes: int) -> Iterator[bytes]:
    if slice_bytes <= 0:
        if chunk:
            yield chunk
        return
    for i in range(0, len(chunk), slice_bytes):
        piece = even_pcm(chunk[i : i + slice_bytes])
        if piece:
            yield piece


class SounddeviceSink(_Played):
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
        sd = load_sounddevice()

        self._reset_played()
        self._odd = b""
        tap_clear()
        self._stream = sd.RawOutputStream(**pcm_stream_kwargs(self.sample_rate, self.device))
        self._stream.start()

    def write(self, chunk: bytes) -> None:
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
        if not self.proc or not self.proc.stdin or not chunk:
            return
        try:
            self.proc.stdin.write(chunk)
            self.proc.stdin.flush()
            self._count(chunk)
        except BrokenPipeError:
            pass

    def finish(self, *, kill: bool = False) -> None:
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
