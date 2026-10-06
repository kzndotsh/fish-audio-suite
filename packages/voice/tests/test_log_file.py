"""``FISH_VOICE_LOG_DIR``: a complete log of the run in a file, whatever the screen shows."""

from __future__ import annotations

import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from fish_audio_suite_voice import debug as debug_module
from fish_audio_suite_voice.debug import (
    configure_voice_logging,
    debug,
    log_file_path,
    log_to_file,
    trace,
    warn,
)
from fish_audio_suite_voice.events import EVENTS, Heard, ReplyEnd, forward_logs, mirror_conversation


@pytest.fixture(autouse=True)
def _fresh_log_file(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(debug_module, "_FILE", debug_module._FileLog())  # pyright: ignore[reportPrivateUsage]
    yield
    debug_module._FILE.close()  # pyright: ignore[reportPrivateUsage]
    configure_voice_logging(debug=False)


def _log(folder: Path) -> str:
    (path,) = folder.glob("fish-voice-*.log")
    return path.read_text(encoding="utf-8")


def test_everything_goes_to_the_file_while_the_screen_shows_only_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert configure_voice_logging(debug=False, log_dir=str(tmp_path)) is not None
    debug("llm.request model={}", "m")
    trace("listen.mic frames={}", 20)
    warn("[tts] retry")
    err = capsys.readouterr().err
    assert "retry" in err
    assert "llm" not in err
    assert "listen" not in err
    text = _log(tmp_path)
    assert "llm" in text
    assert "request model=m" in text
    assert "listen" in text
    assert "frames=20" in text
    assert "retry" in text


def test_a_line_the_screen_shows_is_in_the_file_once_and_a_trace_line_is_file_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_voice_logging(debug=1, log_dir=str(tmp_path))
    debug("llm.request shown")
    trace("listen.mic frames=1 hidden")
    err = capsys.readouterr().err
    assert "shown" in err
    assert "hidden" not in err
    text = _log(tmp_path)
    assert text.count("shown") == 1
    assert "hidden" in text


def test_the_file_is_private_and_named_for_the_run(tmp_path: Path) -> None:
    path = configure_voice_logging(debug=False, log_dir=str(tmp_path / "logs" / "deeper"))
    assert path is not None
    assert path.name.startswith("fish-voice-")
    assert path.suffix == ".log"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # it holds what was said
    assert log_file_path() == path


def test_calling_configure_again_keeps_the_same_file(tmp_path: Path) -> None:
    first = configure_voice_logging(debug=False, log_dir=str(tmp_path))
    second = configure_voice_logging(debug=False, to_stderr=False)  # as the full-screen app does
    assert first == second
    debug("llm.after the second call")
    assert "after the second call" in _log(tmp_path)


def test_what_was_said_and_answered_is_in_the_file(tmp_path: Path) -> None:
    configure_voice_logging(debug=False, log_dir=str(tmp_path))
    stop = mirror_conversation()
    try:
        EVENTS.emit(Heard("hello there", 120.0))
        EVENTS.emit(ReplyEnd("[calm] Hi back."))
    finally:
        stop()
    text = _log(tmp_path)
    assert "you" in text
    assert "hello there" in text
    assert "Hi back." in text


def test_nothing_is_written_when_no_folder_is_set(tmp_path: Path) -> None:
    assert configure_voice_logging(debug=False) is None
    debug("llm.request")
    log_to_file("you", "ignored")
    assert log_file_path() is None
    assert list(tmp_path.iterdir()) == []


def test_a_folder_that_cannot_be_made_is_reported_and_the_run_goes_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("x")
    assert configure_voice_logging(debug=False, log_dir=str(blocked / "logs")) is None
    assert "cannot write the log file" in capsys.readouterr().err
    debug("llm.request")  # does not raise


def test_the_full_screen_log_pane_does_not_get_lines_only_the_file_wants(tmp_path: Path) -> None:
    configure_voice_logging(debug=False, to_stderr=False, log_dir=str(tmp_path))
    from fish_audio_suite_voice.events import LogLine

    shown: list[str] = []
    stop_forwarding = forward_logs()
    stop = EVENTS.subscribe(lambda e: shown.append(e.text) if isinstance(e, LogLine) else None)
    try:
        debug("llm.request quiet")
        warn("[tts] loud")
    finally:
        stop()
        stop_forwarding()
    assert shown == ["loud"]
    assert "quiet" in _log(tmp_path)
