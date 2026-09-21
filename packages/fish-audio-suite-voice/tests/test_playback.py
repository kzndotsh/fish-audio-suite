from __future__ import annotations

from pathlib import Path

from fish_audio_suite_voice.live import _spoken_prefix
from fish_audio_suite_voice.playback import FileSink, StdoutSink, make_sink


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
