from __future__ import annotations

import asyncio
import io
import threading
import wave
from pathlib import Path

import pytest
from fishaudio.exceptions import AuthenticationError, RateLimitError
from voice_fakes import make_result

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_voice.cancel import is_cancel_noise
from fish_audio_suite_voice.playback import (
    FileSink,
    MpvSink,
    PlaybackKind,
    PlaybackSink,
    PortAudioMissingError,
    SounddeviceSink,
    StdoutSink,
    audio_format_for,
    duplex_playback_problem,
    make_sink,
    missing_portaudio,
    parse_playback,
    write_mono_wav,
)
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.wire import (
    TtsResult,
    _classify_fish_exc,
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


def test_mpv_kill_does_not_close_stdin_first(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(sink, "proc", proc)
    sink.finish(kill=True)
    assert proc.order == ["kill", "wait"]
    assert proc.stdin.closed is False
    assert sink.proc is None

    proc = Proc()
    monkeypatch.setattr(sink, "proc", proc)
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


def test_speak_reraises_a_sink_that_fails_to_open() -> None:
    tts = FishSpeaker(api_key="k", voice_id="v")

    class Sink:
        output_latency_s = 0.0

        def start(self) -> None:
            raise PortAudioMissingError("missing")

        def write(self, chunk: bytes) -> None:
            del chunk

        def finish(self, *, kill: bool = False) -> None:
            del kill

        def bytes_played(self) -> int:
            return 0

    with pytest.raises(PortAudioMissingError, match="missing"):
        tts.speak("Hello there.", Sink())


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


def test_speak_works_inside_asyncio_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_speak(
        self: FishSpeaker,
        text: str,
        sink: FileSink,
        cancel: threading.Event,
        on_first_audio: object = None,
    ) -> TtsResult:
        sink.start()
        sink.write(b"\x00\x00" * 64)
        sink.finish()
        return make_result(
            "ok", bytes_played=128, got_audio=True, tts_first_audio_ms=1.0, tts_first_text_ms=1.0
        )

    monkeypatch.setattr(FishSpeaker, "_speak_here", fake_speak)
    tts = FishSpeaker(api_key="k", voice_id="v")
    sink = FileSink(tmp_path / "t.wav", sample_rate=44100, wav=True)

    async def outer() -> TtsResult:
        return tts.speak("[clear] hi", sink)

    result = asyncio.run(outer())
    assert result.got_audio
    assert result.bytes_played == 128


def test_parse_playback_names_every_sink_and_rejects_the_rest() -> None:
    assert parse_playback(" MPV ") is PlaybackKind.MPV
    assert parse_playback("Speakers") is PlaybackKind.SPEAKERS
    assert parse_playback("nope") is None
    assert parse_playback("") is None
    assert {kind.value for kind in PlaybackKind} == {
        "sounddevice",
        "speakers",
        "pcm",
        "stdout",
        "mpv",
        "file",
    }


def test_only_the_sound_card_sinks_are_speakers() -> None:
    speakers = {kind for kind in PlaybackKind if kind.is_speaker}
    assert speakers == {PlaybackKind.SOUNDDEVICE, PlaybackKind.SPEAKERS, PlaybackKind.PCM}


@pytest.mark.parametrize("name", ["sounddevice", "SPEAKERS", " pcm "])
def test_make_sink_builds_the_sound_card_sink_for_each_speaker_name(name: str) -> None:
    assert isinstance(make_sink(name), SounddeviceSink)


def test_make_sink_and_the_duplex_check_agree_on_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown playback sink"):
        make_sink("nope")
    assert duplex_playback_problem("nope") == "unknown playback sink 'nope'"


def test_every_sink_reports_an_output_latency() -> None:
    for sink in (StdoutSink(), FileSink(Path("x.wav")), MpvSink(), SounddeviceSink()):
        assert isinstance(sink, PlaybackSink)
        assert sink.output_latency_s == 0.0


def test_write_mono_wav_accepts_a_path_a_string_and_a_file_object(tmp_path: Path) -> None:
    pcm = b"\x01\x00" * 160
    target = tmp_path / "clip.wav"
    write_mono_wav(target, pcm, 16_000)
    write_mono_wav(str(tmp_path / "text.wav"), pcm, 16_000)
    buf = io.BytesIO()
    write_mono_wav(buf, pcm, 16_000)
    for source in (target, tmp_path / "text.wav", io.BytesIO(buf.getvalue())):
        with wave.open(str(source) if isinstance(source, Path) else source, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16_000
            assert wf.getnframes() == 160
