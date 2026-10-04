"""The voice evaluation script, tested without the network: a fake speaker stands in."""

from __future__ import annotations

import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import eval_voices
import pytest
from voice_fakes import make_result

from fish_audio_suite_voice import FileSink
from fish_audio_suite_voice.wire import TtsResult


class _FakeSpeaker:
    def __init__(self, **kwargs: Any) -> None:
        self.voice = str(kwargs["voice_id"])
        self.model = str(kwargs["model"])

    def speak(
        self,
        text: str,
        sink: FileSink,
        *,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        del cancel
        assert text.startswith("[happy]")
        if self.voice == "badvoice":
            return make_result("", error_status=404, error_message="voice not found")
        sink.start()
        sink.write(b"\x01\x00" * 44100)  # one second at 44.1 kHz
        sink.finish()
        if on_first_audio is not None:
            on_first_audio()
        return make_result(text, bytes_played=88200, got_audio=True)


@pytest.fixture
def fake_speaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_voices, "FishSpeaker", _FakeSpeaker)


def test_the_file_name_holds_the_timestamp_voice_and_model() -> None:
    path = eval_voices.clip_path(Path("tmp"), "98655a12fa944e26", "s2.1-pro", "20261004-123456")
    assert path == Path("tmp/20261004-123456_98655a12fa944e26_s2.1-pro.wav")
    unsafe = eval_voices.clip_path(Path("tmp"), "a/b c", "x:y", "T")
    assert unsafe == Path("tmp/T_a-b-c_x-y.wav")


def test_voice_ids_ignore_labels_blanks_and_repeats() -> None:
    assert eval_voices.voice_ids(["aaa  # warm", "bbb", "", "  ", "aaa", "# just a note"]) == [
        "aaa",
        "bbb",
    ]


@pytest.mark.usefixtures("fake_speaker")
def test_every_voice_and_model_is_saved_and_a_failure_does_not_stop_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setattr(eval_voices, "VOICES", ("voice1  # warm", "badvoice", "voice2"))
    monkeypatch.setattr(eval_voices, "TTS_MODELS", ("s2.1-pro", "s1"))
    monkeypatch.setattr(eval_voices, "TEXT", "[happy] Hello there!")
    out = tmp_path / "clips"
    code = eval_voices.main(["--out-dir", str(out), "--env-file", str(tmp_path / "none.env")])
    assert code == 0
    files = sorted(p.name for p in out.glob("*.wav"))
    assert len(files) == 4  # two good voices x two models; the bad voice wrote nothing
    assert any("_voice1_s2.1-pro.wav" in name for name in files)
    assert any("_voice2_s1.wav" in name for name in files)
    with wave.open(str(out / files[0])) as clip:
        assert clip.getframerate() == 44100
        assert clip.getnframes() == 44100
    table = capsys.readouterr().out
    assert "FAILED: voice not found" in table
    assert "voice1" in table
    assert "1.0" in table  # seconds


@pytest.mark.usefixtures("fake_speaker")
def test_the_voice_from_env_is_used_when_the_list_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_VOICE_ID", "envvoice")
    monkeypatch.setattr(eval_voices, "VOICES", ())
    monkeypatch.setattr(eval_voices, "TTS_MODELS", ("s2.1-pro",))
    monkeypatch.setattr(eval_voices, "TEXT", "[happy] Hi.")
    out = tmp_path / "clips"
    assert eval_voices.main(["--out-dir", str(out), "--env-file", str(tmp_path / "none.env")]) == 0
    assert [p.name.split("_", 1)[1] for p in out.glob("*.wav")] == ["envvoice_s2.1-pro.wav"]


def test_it_needs_a_key_and_something_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("FISH_API_KEY", "FISH_VOICE_ID"):
        monkeypatch.delenv(name, raising=False)
    none = ["--env-file", str(tmp_path / "none.env"), "--out-dir", str(tmp_path / "clips")]
    monkeypatch.setattr(eval_voices, "VOICES", ())
    assert eval_voices.main(none) == 2  # no voice at all
    monkeypatch.setattr(eval_voices, "VOICES", ("v",))
    assert eval_voices.main(none) == 2  # a voice, but no key
    assert eval_voices.main([*none, "--dry-run"]) == 0
    assert not (tmp_path / "clips").exists()  # a dry run writes nothing
    assert eval_voices.main([*none, "--find", "narrator"]) == 2  # --find needs a key too
