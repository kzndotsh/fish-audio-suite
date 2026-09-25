from __future__ import annotations

import asyncio
import threading
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fishaudio.exceptions import AuthenticationError, RateLimitError

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_voice.live import IsolatedFishTts, IsolatedResult, is_cancel_noise
from fish_audio_suite_voice.playback import (
    FileSink,
    MpvSink,
    PortAudioMissingError,
    SounddeviceSink,
    StdoutSink,
    audio_format_for,
    duplex_playback_problem,
    make_sink,
    missing_portaudio,
)
from fish_audio_suite_voice.wire import (
    _anext_chunk,
    _classify_fish_exc,
    _spoken_prefix,
    _wait_task,
    _word_prefix,
)


def test_zero_sample_rate_still_writes_a_wav(tmp_path: Path) -> None:
    pcm = b"\x00\x01\x02\x03"
    for rate in (0, 2**32):
        path = tmp_path / f"{rate}.wav"
        sink = FileSink(path, sample_rate=rate, wav=True)
        sink.start()
        sink.write(pcm)
        sink.finish()
        with wave.open(str(path)) as wf:
            assert wf.getframerate() == SuiteDefaults().sample_rate
            assert wf.readframes(wf.getnframes()) == pcm


def test_file_sink_writes_wav(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"
    sink = FileSink(path, sample_rate=44100, wav=True)
    sink.start()
    pcm = b"\x00\x00" * 100
    sink.write(pcm)
    assert sink.bytes_played() == 0
    sink.finish(kill=False)
    assert sink.bytes_played() == len(pcm)
    assert path.is_file()
    assert path.stat().st_size > 44


def test_file_sink_drops_a_trailing_odd_byte(tmp_path: Path) -> None:
    path = tmp_path / "odd.wav"
    sink = FileSink(path, sample_rate=16000, wav=True)
    sink.start()
    sink.write(b"\x01\x00")
    sink.write(b"\x02")
    sink.finish(kill=False)
    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 1
        assert wf.readframes(1) == b"\x01\x00"
    assert path.read_bytes()[40:44] == (2).to_bytes(4, "little")
    assert sink.bytes_played() == 2
    raw = tmp_path / "odd.pcm"
    pcm_sink = FileSink(raw, sample_rate=16000, wav=False)
    pcm_sink.start()
    pcm_sink.write(b"\x01\x00\x02")
    pcm_sink.finish(kill=False)
    assert raw.read_bytes() == b"\x01\x00"


def test_file_sink_empty_finish_keeps_the_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"
    path.write_bytes(b"previous")
    sink = FileSink(path, sample_rate=44100, wav=True)
    sink.start()
    sink.finish(kill=False)
    assert path.read_bytes() == b"previous"
    missing = tmp_path / "fresh.wav"
    FileSink(missing, sample_rate=44100, wav=True).finish(kill=False)
    assert not missing.exists()


def test_file_sink_does_not_count_a_failed_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def full(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    path = tmp_path / "out.wav"
    sink = FileSink(path, sample_rate=16000, wav=True)
    sink.start()
    sink.write(b"\x00\x01" * 8)
    monkeypatch.setattr("fish_audio_suite_voice.playback.write_mono_wav", full)
    with pytest.raises(OSError, match="No space left"):
        sink.finish(kill=False)
    assert sink.bytes_played() == 0
    assert not path.exists()


def test_file_sink_kill_skips_write(tmp_path: Path) -> None:
    path = tmp_path / "out.wav"
    sink = FileSink(path, sample_rate=44100, wav=True)
    sink.start()
    sink.write(b"\x00\x00" * 10)
    sink.finish(kill=True)
    assert not path.exists()
    assert sink.bytes_played() == 0


def test_duplex_playback_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    assert duplex_playback_problem("sounddevice") is None
    assert duplex_playback_problem("file") == "file playback needs a path"
    assert duplex_playback_problem("nope") == "unknown playback sink 'nope'"
    monkeypatch.setattr("fish_audio_suite_voice.playback.shutil.which", lambda _name: None)
    assert duplex_playback_problem("mpv") == "mpv is not on PATH"
    monkeypatch.setattr("fish_audio_suite_voice.playback.shutil.which", lambda _name: "/bin/mpv")
    assert duplex_playback_problem("MPV") is None


def test_mpv_kill_does_not_close_stdin_first() -> None:
    class Stdin:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Proc:
        def __init__(self) -> None:
            self.stdin = Stdin()
            self.order: list[str] = []

        def kill(self) -> None:
            self.order.append("kill")

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.order.append("wait")
            return 0

    proc = Proc()
    sink = MpvSink()
    sink.proc = proc
    sink.finish(kill=True)
    assert proc.order == ["kill", "wait"]
    assert proc.stdin.closed is False
    assert sink.proc is None

    proc = Proc()
    sink.proc = proc
    sink.finish(kill=False)
    assert proc.order == ["wait"]
    assert proc.stdin.closed is True


def test_mpv_playback_uses_mp3() -> None:
    assert audio_format_for(" MPV ") == "mp3"
    assert audio_format_for("speakers") == "pcm"
    assert isinstance(make_sink("MPV"), MpvSink)


def test_make_sink_file(tmp_path: Path) -> None:
    path = tmp_path / "raw.pcm"
    sink = make_sink("file", path=path, sample_rate=44100)
    sink.start()
    sink.write(b"abcd")
    sink.finish()
    assert path.read_bytes() == b"abcd"


def test_stdout_sink_counts() -> None:
    sink = StdoutSink()
    sink.start()
    sink.write(b"xx")
    assert sink.bytes_played() == 2
    sink.finish()


def test_stdout_sink_rejoins_odd_pcm_chunks(capsysbinary: pytest.CaptureFixture[bytes]) -> None:
    sink = StdoutSink()
    sink.start()
    sink.write(b"\x01\x02\x03")
    assert sink.bytes_played() == 2
    sink.write(b"")
    sink.write(b"\x04\x05\x06")
    sink.finish()
    assert capsysbinary.readouterr().out == b"\x01\x02\x03\x04\x05\x06"
    assert sink.bytes_played() == 6


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


def test_spoken_prefix_ignores_audio_the_sink_did_not_play() -> None:
    assert (
        _spoken_prefix(
            "hello there",
            bytes_played=0,
            sample_rate=44100,
            audio_format="mp3",
            got_audio=True,
            cancelled=False,
        )
        == ""
    )


def test_played_line_keeps_the_finished_words_before_the_next_line() -> None:
    line = "Hello there friend.\nThe next sentence is longer today."
    # 23 ends on "The" and the next character is a space, so that word was played.
    assert _word_prefix(line, 23) == "Hello there friend. The"
    # One character into "next" is not a finished word.
    assert _word_prefix(line, 25).split() == ["Hello", "there", "friend.", "The"]
    assert _word_prefix("Hello\nthere friend today please", 6) == "Hello"


def test_spoken_prefix_keeps_played_cjk_without_spaces() -> None:
    # 8000 bytes at 16 kHz int16 is 0.25 s, about four characters.
    heard = _spoken_prefix(
        "我们今天见面",
        bytes_played=8000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert heard == "我们今天"
    assert (
        _spoken_prefix(
            "hello there friend",
            bytes_played=6000,
            sample_rate=16_000,
            audio_format="pcm",
            got_audio=True,
            cancelled=True,
        )
        == ""
    )


def test_spoken_prefix_does_not_spend_the_barge_budget_on_a_cue() -> None:
    # 16000 bytes at 16 kHz is half a second, about eight characters.
    # Those characters used to be "[clear] ", so history stored the tag
    # and the next turn said "hello" again.
    heard = _spoken_prefix(
        "[clear] hello there friend",
        bytes_played=16_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert heard == "hello"


def test_finished_turn_does_not_store_the_clear_tag() -> None:
    heard = _spoken_prefix(
        "[clear] hello there friend",
        bytes_played=160_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=False,
    )
    assert heard == "hello there friend"
    mood = _spoken_prefix(
        "[excited] hello there friend",
        bytes_played=160_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=False,
    )
    assert mood == "[excited] hello there friend"


def test_spoken_prefix_keeps_a_played_index() -> None:
    # 40000 bytes at 16 kHz is 1.25 s, about twenty characters, which
    # reaches the index. Those brackets are speech, not a Fish cue.
    heard = _spoken_prefix(
        "Use the key a[i][j] in the code today friend.",
        bytes_played=40_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert "a[i][j]" in heard


def test_spoken_prefix_keeps_the_word_before_a_nonbreaking_space() -> None:
    assert _word_prefix("Hello\u00a0there friend today", 8) == "Hello"
    assert _word_prefix("Hello there friend", 8) == "Hello"


def test_word_prefix_keeps_a_word_that_ends_on_the_cut() -> None:
    # The next character is the space, so "there" was played. Dropping it
    # made the next turn say "there" again.
    assert _word_prefix("Hello there friend", 11) == "Hello there"
    assert _word_prefix("Hello there friend", 10) == "Hello"
    assert _word_prefix("Hello there friend", 5) == "Hello"
    assert _word_prefix("Hello there friend", 3) == ""
    assert _word_prefix("Привет друг сегодня", len("Привет")) == "Привет"
    assert _word_prefix("Hello there. Friend today", 11) == "Hello there"
    assert _word_prefix("Hello there, friend today", 11) == "Hello there"
    # The next letter is still part of "друг". Recording "дру" made the
    # next turn skip the rest of the word.
    assert _word_prefix("Привет друг сегодня", len("Привет дру")) == "Привет"
    assert _word_prefix("Привет друг сегодня", len("Привет друг")) == "Привет друг"
    assert _word_prefix("Hello 你好朋友", 7) == "Hello 你"
    assert _word_prefix("Hello 你好朋友", 6) == "Hello"
    # "there." was played. The next letter is a new word, not more of "there".
    assert _word_prefix("Hello there.Friend today", len("Hello there.")) == "Hello there."
    assert _word_prefix("Hello there.Friend today", len("Hello there.Fr")) == "Hello there."
    assert _word_prefix("Hello there!Friend today", len("Hello there!Fr")) == "Hello there!"
    assert _word_prefix("Hello,there friend today", len("Hello,")) == "Hello,"
    assert _word_prefix("Привет,друг сегодня", len("Привет,")) == "Привет,"
    assert _word_prefix("don't go today", 4) == ""
    assert _word_prefix("don't go today", 5) == "don't"
    assert _word_prefix("3.14 today friend", 2) == ""
    assert _word_prefix("3.14 today friend", 3) == ""
    assert _word_prefix("Hello.Friend today", len("Hello.Fr")) == "Hello."
    assert _word_prefix("Привет.друг сегодня", len("Привет.дру")) == "Привет."


def test_spoken_prefix_keeps_cjk_played_after_an_english_word() -> None:
    assert _word_prefix("Hello你好朋友", 8) == "Hello你好朋"
    assert _word_prefix("Hello", 3) == ""
    heard = _spoken_prefix(
        "Hello你好朋友",
        bytes_played=16_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert heard == "Hello你好朋"


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
    assert is_cancel_noise(err)


def test_speak_isolated_reraises_a_sink_that_fails_to_open() -> None:
    tts = IsolatedFishTts(api_key="k", voice_id="v")

    class Sink:
        def start(self) -> None:
            raise PortAudioMissingError("missing")

        def write(self, chunk: bytes) -> None:
            del chunk

        def finish(self, *, kill: bool = False) -> None:
            del kill

        def bytes_played(self) -> int:
            return 0

    with pytest.raises(PortAudioMissingError, match="missing"):
        tts.speak_isolated("Hello there.", Sink())


def test_missing_portaudio_hint() -> None:
    err = missing_portaudio()
    assert isinstance(err, PortAudioMissingError)
    assert "libportaudio2" in str(err)
    assert "nix run" in str(err)


def test_sounddevice_sink_stops_after_the_slice_that_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = threading.Event()
    sink = SounddeviceSink(sample_rate=16_000, cancel=cancel)
    written: list[bytes] = []
    monkeypatch.setattr("fish_audio_suite_voice.playback.tap_playback", lambda *_args: None)

    class Stream:
        def write(self, chunk: bytes) -> None:
            written.append(chunk)
            cancel.set()

    sink._stream = Stream()
    sink.write(b"\x00\x00" * 8_000)
    assert len(written) == 1
    assert sink.bytes_played() == len(written[0])


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
