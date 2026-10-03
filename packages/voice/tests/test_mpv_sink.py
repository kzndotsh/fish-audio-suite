from __future__ import annotations

from typing import Any

import pytest

from fish_audio_suite_voice.playback import MpvSink


def test_start_without_mpv_on_path_fails_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_audio_suite_voice.playback.shutil.which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="mpv is not on PATH"):
        MpvSink().start()


def test_start_runs_the_resolved_binary_without_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    class Proc:
        stdin = None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def popen(args: list[str], **kwargs: Any) -> Proc:
        calls.append((args, kwargs))
        return Proc()

    monkeypatch.setattr("fish_audio_suite_voice.playback.shutil.which", lambda _n: "/opt/bin/mpv")
    monkeypatch.setattr("fish_audio_suite_voice.playback.subprocess.Popen", popen)
    sink = MpvSink()
    sink.start()
    args, kwargs = calls[0]
    assert args[0] == "/opt/bin/mpv"
    assert args[-1] == "-"
    assert "shell" not in kwargs
    sink.finish(kill=True)
