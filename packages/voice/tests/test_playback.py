from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fishaudio.exceptions import AuthenticationError, RateLimitError

from fish_audio_suite_voice.live import (
    IsolatedFishTts,
    IsolatedResult,
    _anext_chunk,
    _classify_fish_exc,
    _is_cancel_noise,
    _spoken_prefix,
    _wait_task,
)
from fish_audio_suite_voice.playback import (
    FileSink,
    PortAudioMissingError,
    SounddeviceSink,
    StdoutSink,
    make_sink,
    missing_portaudio,
)


def test_file_sink_writes_wav(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"
    sink = FileSink(path, sample_rate=44100, wav=True)
    sink.start()
    pcm = b"\x00\x00" * 100
    sink.write(pcm)
    assert sink.bytes_played() == len(pcm)
    sink.finish(kill=False)
    assert path.is_file()
    assert path.stat().st_size > 44


def test_file_sink_kill_skips_write(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"
    sink = FileSink(path, sample_rate=44100, wav=True)
    sink.start()
    sink.write(b"\x00\x00" * 10)
    sink.finish(kill=True)
    assert not path.exists()


def test_make_sink_file(tmp_path: Path) -> None:
    path = tmp_path / "raw.pcm"
    sink = make_sink("file", path=path, sample_rate=44100)
    sink.start()
    sink.write(b"abc")
    sink.finish()
    assert path.read_bytes() == b"abc"


def test_stdout_sink_counts() -> None:
    sink = StdoutSink()
    sink.start()
    sink.write(b"xx")
    assert sink.bytes_played() == 2
    sink.finish()


def test_spoken_prefix_omits_before_audio() -> None:
    assert (
        _spoken_prefix(
            "hello",
            bytes_played=0,
            sample_rate=44100,
            audio_format="pcm",
            got_audio=False,
            cancelled=True,
        )
        == ""
    )


def test_spoken_prefix_full_when_complete() -> None:
    assert (
        _spoken_prefix(
            "hello there",
            bytes_played=100,
            sample_rate=44100,
            audio_format="pcm",
            got_audio=True,
            cancelled=False,
        )
        == "hello there"
    )


def test_classify_fish_exc_retries_429_not_401() -> None:
    retry, status, message = _classify_fish_exc(RateLimitError(429, "slow down", None))
    assert retry is True
    assert status == 429
    assert message == "slow down"
    retry, status, message = _classify_fish_exc(AuthenticationError(401, "Invalid Token", None))
    assert retry is False
    assert status == 401
    assert message == "Invalid Token"


def test_cancel_scope_runtime_error_is_noise() -> None:
    err = RuntimeError("Attempted to exit cancel scope in a different task than it was entered in")
    retry, status, _message = _classify_fish_exc(err)
    assert retry is False
    assert status is None
    assert _is_cancel_noise(err)


def test_missing_portaudio_hint() -> None:
    err = missing_portaudio(OSError("PortAudio library not found"))
    assert isinstance(err, PortAudioMissingError)
    assert "libportaudio2" in str(err)
    assert "nix run" in str(err)


def test_sounddevice_sink_skips_write_when_cancelled() -> None:
    cancel = threading.Event()
    sink = SounddeviceSink(cancel=cancel)

    class Boom:
        def write(self, chunk: bytes) -> None:
            raise AssertionError("cancelled sink must not write")

    sink._stream = Boom()
    cancel.set()
    sink.write(b"\x00\x00" * 8)
    assert sink.bytes_played() == 0


def test_speak_isolated_works_inside_asyncio_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_speak(
        self: IsolatedFishTts,
        text: str,
        sink: FileSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        sink.start()
        sink.write(b"\x00\x00" * 64)
        sink.finish()
        return IsolatedResult("ok", 128, True, False, 1.0, 1.0)

    monkeypatch.setattr(IsolatedFishTts, "speak", fake_speak)
    tts = IsolatedFishTts(api_key="k", voice_id="v")
    sink = FileSink(tmp_path / "t.wav", sample_rate=44100, wav=True)

    async def outer() -> IsolatedResult:
        return tts.speak_isolated("[clear] hi", sink)

    result = asyncio.run(outer())
    assert result.got_audio
    assert result.bytes_played == 128


def test_wait_task_does_not_cancel_slow_anext() -> None:
    async def slow() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.35)
        yield b"pcm"

    async def run() -> None:
        it = aiter(slow())
        pending = asyncio.create_task(_anext_chunk(it))
        assert not await _wait_task(pending, 0.08)
        assert not pending.done()
        assert await _wait_task(pending, 0.5)
        assert pending.result() == b"pcm"

    asyncio.run(run())
