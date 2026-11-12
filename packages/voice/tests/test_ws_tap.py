from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest
from fishaudio.resources import realtime

from fish_audio_suite_voice import ws_tap
from fish_audio_suite_voice.log_setup import configure_voice_logging
from fish_audio_suite_voice.ws_tap import install_fish_ws_tap

# The two SDK functions the tap wraps (the SDK has no public hook for its events).
_STOP = "_should_stop"
_PROCESS = "_process_audio_event"


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """The Fish SDK's realtime module, with the tap not yet installed.

    The tap installs once per process, so ``ws_tap`` is reloaded before and after the
    test, and the SDK functions are put back afterwards, so no wrapper outlives the test.
    """
    for name in (_STOP, _PROCESS):
        monkeypatch.setattr(realtime, name, _sdk_function(realtime, name))
    importlib.reload(ws_tap)
    yield realtime
    importlib.reload(ws_tap)


def _sdk_function(module: ModuleType, name: str) -> Any:
    return vars(module).get(name)


def test_the_installed_sdk_still_has_the_functions_the_tap_wraps(
    sdk: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # The debug tap wraps private SDK functions. If an SDK release renames or
    # re-shapes one, this fails in CI instead of the tap failing in a session.
    before = {name: _sdk_function(sdk, name) for name in (_STOP, _PROCESS)}
    configure_voice_logging(debug=1)
    install_fish_ws_tap()
    for name, original in before.items():
        assert callable(original), f"fishaudio.resources.realtime.{name} is gone"
        assert _sdk_function(sdk, name) is not original, f"{name} was not wrapped"
    assert "skipped" not in capsys.readouterr().err


def test_the_tap_is_skipped_with_one_debug_line_when_a_function_is_missing(
    sdk: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def only_stop(data: dict[str, object]) -> bool:
        del data
        return False

    monkeypatch.setattr(sdk, _STOP, only_stop)
    monkeypatch.delattr(sdk, _PROCESS)
    configure_voice_logging(debug=1)
    install_fish_ws_tap()
    install_fish_ws_tap()
    assert _sdk_function(sdk, _STOP) is only_stop  # the SDK is left untouched
    assert _sdk_function(sdk, _PROCESS) is None
    err = capsys.readouterr().err
    assert err.count("skipped") == 1
    assert _PROCESS in err


def test_the_tap_is_skipped_when_a_function_changes_its_signature(
    sdk: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def two_args(data: dict[str, object], extra: object) -> bool:
        del data, extra
        return False

    def process(data: dict[str, object]) -> None:
        del data

    monkeypatch.setattr(sdk, _STOP, two_args)
    monkeypatch.setattr(sdk, _PROCESS, process)
    configure_voice_logging(debug=1)
    install_fish_ws_tap()
    assert _sdk_function(sdk, _STOP) is two_args
    assert _sdk_function(sdk, _PROCESS) is process
    err = capsys.readouterr().err
    assert "skipped" in err
    assert _STOP in err


def test_the_tap_logs_each_event_and_passes_results_through(
    sdk: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[dict[str, Any]] = []

    def should_stop(data: dict[str, Any]) -> bool:
        seen.append(data)
        return data.get("event") == "finish"

    def process(data: dict[str, Any]) -> bytes | None:
        return data.get("audio")

    monkeypatch.setattr(sdk, _STOP, should_stop)
    monkeypatch.setattr(sdk, _PROCESS, process)
    configure_voice_logging(debug=2)
    install_fish_ws_tap()
    stop = _sdk_function(sdk, _STOP)
    proc = _sdk_function(sdk, _PROCESS)
    assert stop is not should_stop
    audio = {"event": "audio", "audio": b"secret-audio"}
    assert stop({"event": "finish", "reason": "stop"}) is True
    assert stop(audio) is False
    assert stop({"event": "log", "message": "hi"}) is False
    assert proc(audio) == b"secret-audio"
    assert len(seen) == 3
    err = capsys.readouterr().err
    assert "finish reason=stop" in err
    assert "audio 12 bytes" in err
    assert "{'event': 'log', 'message': 'hi'}" in err
    # Audio is logged as its byte length only.
    assert "secret-audio" not in err
