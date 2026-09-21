"""Playback sinks for Fish PCM (and optional mpv mp3)."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class PlaybackSink(Protocol):
    def start(self) -> None: ...
    def write(self, chunk: bytes) -> None: ...
    def finish(self, *, kill: bool = False) -> None: ...
    def bytes_played(self) -> int: ...


class FileSink:
    def __init__(self, path: Path, *, sample_rate: int = 44100, wav: bool = True) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.wav = wav
        self._buf = bytearray()
        self._played = 0

    def start(self) -> None:
        self._buf.clear()
        self._played = 0

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._buf.extend(chunk)
        self._played += len(chunk)

    def finish(self, *, kill: bool = False) -> None:
        if kill:
            return
        if self.wav:
            with wave.open(str(self.path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(bytes(self._buf))
        else:
            self.path.write_bytes(bytes(self._buf))

    def bytes_played(self) -> int:
        return self._played


class StdoutSink:
    def __init__(self) -> None:
        self._played = 0

    def start(self) -> None:
        self._played = 0

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        self._played += len(chunk)

    def finish(self, *, kill: bool = False) -> None:
        return

    def bytes_played(self) -> int:
        return self._played


class SounddeviceSink:
    def __init__(self, *, sample_rate: int = 44100, device: str | int | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._stream: object | None = None
        self._played = 0

    def start(self) -> None:
        import sounddevice as sd

        self._played = 0
        self._stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
        )
        self._stream.start()

    def write(self, chunk: bytes) -> None:
        if not chunk or self._stream is None:
            return
        self._stream.write(chunk)
        self._played += len(chunk)

    def finish(self, *, kill: bool = False) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            if kill:
                stream.abort()
            else:
                stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def bytes_played(self) -> int:
        return self._played


class MpvSink:
    """Optional legacy mp3 stdin player. Not required on PATH for other sinks."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self._played = 0

    def start(self) -> None:
        self.finish(kill=True)
        self._played = 0
        self.proc = subprocess.Popen(
            [
                "mpv",
                "--no-terminal",
                "--really-quiet",
                "--audio-buffer=0.2",
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
            self._played += len(chunk)
        except BrokenPipeError:
            pass

    def finish(self, *, kill: bool = False) -> None:
        if not self.proc:
            return
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        if kill:
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                pass
        else:
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
        self.proc = None

    def bytes_played(self) -> int:
        return self._played


def make_sink(
    name: str,
    *,
    path: Path | None = None,
    sample_rate: int = 44100,
    device: str | int | None = None,
) -> PlaybackSink:
    key = name.strip().lower()
    if key == "file":
        if path is None:
            raise ValueError("file sink requires path")
        return FileSink(path, sample_rate=sample_rate, wav=path.suffix.lower() == ".wav")
    if key == "stdout":
        return StdoutSink()
    if key == "mpv":
        return MpvSink()
    if key in {"sounddevice", "speakers", "pcm"}:
        return SounddeviceSink(sample_rate=sample_rate, device=device)
    raise ValueError(f"unknown playback sink {name!r}")
