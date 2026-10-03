from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from fishaudio.resources import realtime

from fish_audio_suite_voice import debug as voice_debug
from fish_audio_suite_voice.debug import configure_voice_logging, install_fish_ws_tap

TAPPED = voice_debug._TAPPED_FUNCTIONS


def test_the_installed_sdk_still_has_the_functions_the_tap_wraps() -> None:
    # The debug tap wraps private SDK functions. If an SDK release renames or
    # re-shapes one, this fails in CI instead of the tap failing in a session.
    for name in TAPPED:
        function = getattr(realtime, name, None)
        assert callable(function), f"fishaudio.resources.realtime.{name} is gone"
        assert len(inspect.signature(function).parameters) == 1, f"{name} changed its signature"


def _fake_realtime(**present: Any) -> SimpleNamespace:
    return SimpleNamespace(**present)


def test_the_tap_is_skipped_with_one_debug_line_when_a_function_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(voice_debug._WS_TAP, "on", False)
    monkeypatch.setattr(voice_debug._WS_TAP, "skipped", False)

    def only_stop(data: dict[str, object]) -> bool:
        del data
        return False

    fake = _fake_realtime(_should_stop=only_stop)
    monkeypatch.setattr(voice_debug, "_fish_realtime", fake)
    # configure_voice_logging installs the tap itself, so the skip is logged there.
    configure_voice_logging(debug=1)
    install_fish_ws_tap()
    install_fish_ws_tap()
    assert voice_debug._WS_TAP.on is False
    assert voice_debug._WS_TAP.skipped is True
    assert fake._should_stop is only_stop
    assert not hasattr(fake, "_process_audio_event")
    err = capsys.readouterr().err
    assert err.count("skipped") == 1
    assert "_process_audio_event" in err


def test_the_tap_is_skipped_when_a_function_changes_its_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_debug._WS_TAP, "on", False)
    monkeypatch.setattr(voice_debug._WS_TAP, "skipped", False)

    def two_args(data: dict[str, object], extra: object) -> bool:
        del data, extra
        return False

    fake = _fake_realtime(_should_stop=two_args, _process_audio_event=lambda data: None)
    monkeypatch.setattr(voice_debug, "_fish_realtime", fake)
    install_fish_ws_tap()
    assert voice_debug._WS_TAP.on is False
    assert fake._should_stop is two_args


def test_the_tap_wraps_both_functions_and_passes_results_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_debug._WS_TAP, "on", False)
    monkeypatch.setattr(voice_debug._WS_TAP, "skipped", False)
    seen: list[dict[str, Any]] = []

    def should_stop(data: dict[str, Any]) -> bool:
        seen.append(data)
        return data.get("event") == "finish"

    def process(data: dict[str, Any]) -> bytes | None:
        return data.get("audio")

    fake = _fake_realtime(_should_stop=should_stop, _process_audio_event=process)
    monkeypatch.setattr(voice_debug, "_fish_realtime", fake)
    install_fish_ws_tap()
    assert voice_debug._WS_TAP.on is True
    assert fake._should_stop({"event": "finish", "reason": "stop"}) is True
    assert fake._should_stop({"event": "audio", "audio": b"xy"}) is False
    assert fake._process_audio_event({"event": "audio", "audio": b"xy"}) == b"xy"
    assert len(seen) == 2
