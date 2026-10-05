"""``fish-voice --tui``: the flag, the missing extra, and how the session is handed to the app."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import pytest

from fish_audio_suite_voice import cli
from fish_audio_suite_voice.cli import EXIT_FATAL, load_config, main, run_tui
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.debug import DebugLevel
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.signals import DuplexSession


def _configured(monkeypatch: pytest.MonkeyPatch) -> VoiceCliConfig:
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_VOICE_ID", "voice1234abcd")
    monkeypatch.setenv("FISH_LLM_API_KEY", "k")
    monkeypatch.setenv("FISH_LLM_MODEL", "org/model")
    return replace(load_config(), playback="stdout")


def test_the_tui_flag_without_the_extra_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert main(["--tui"]) == EXIT_FATAL
    err = capsys.readouterr().err
    assert "needs the tui extra" in err
    assert "fish-audio-suite-voice[tui]" in err


def test_the_tui_flag_runs_the_app_and_not_the_plain_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_tui(c: object, *, debug: DebugLevel) -> int:
        seen["tui"] = (c, debug)
        return 7

    async def fake_loop(_c: object) -> int:
        raise AssertionError("the plain loop must not run")

    monkeypatch.setattr("fish_audio_suite_voice.cli.apply_cli_env_files", lambda *_a, **_k: [])
    monkeypatch.setattr("fish_audio_suite_voice.cli.configure_voice_logging", lambda **_k: None)
    monkeypatch.setattr("fish_audio_suite_voice.cli.load_config", object)
    monkeypatch.setattr("fish_audio_suite_voice.cli.warn_if_insecure_base", lambda _c: False)
    monkeypatch.setattr("fish_audio_suite_voice.cli.run_tui", fake_tui)
    monkeypatch.setattr("fish_audio_suite_voice.cli.run_loop", fake_loop)
    assert main(["--tui", "--debug"]) == 7
    assert seen["tui"][1] >= DebugLevel.EVENTS


def test_the_plain_loop_is_still_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_loop(_c: object) -> int:
        return 3

    async def fake_tui(_c: object, *, debug: DebugLevel) -> int:
        raise AssertionError("the app must not start without --tui")

    monkeypatch.setattr("fish_audio_suite_voice.cli.apply_cli_env_files", lambda *_a, **_k: [])
    monkeypatch.setattr("fish_audio_suite_voice.cli.configure_voice_logging", lambda **_k: None)
    monkeypatch.setattr("fish_audio_suite_voice.cli.load_config", object)
    monkeypatch.setattr("fish_audio_suite_voice.cli.warn_if_insecure_base", lambda _c: False)
    monkeypatch.setattr("fish_audio_suite_voice.cli.run_loop", fake_loop)
    monkeypatch.setattr("fish_audio_suite_voice.cli.run_tui", fake_tui)
    assert main([]) == 3


def test_run_tui_stops_on_the_same_blockers_as_the_plain_loop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    c = replace(_configured(monkeypatch), playback="nope")
    assert asyncio.run(run_tui(c, debug=DebugLevel.OFF)) == EXIT_FATAL
    assert "unknown playback sink" in capsys.readouterr().err


class _FakeApp:
    """Stands in for ``VoiceApp``: runs the session once, like the real worker does."""

    instances: list[_FakeApp] = []  # noqa: RUF012 - reset by each test

    def __init__(
        self,
        runner: Callable[[], Awaitable[int]],
        *,
        live: LiveInput,
        session: DuplexSession,
        info: str = "",
    ) -> None:
        self.runner = runner
        self.live = live
        self.session = session
        self.info = info
        self.code: int | None = 0
        _FakeApp.instances.append(self)

    async def run_async(self) -> int | None:
        await self.runner()
        return self.code


class _Backend:
    """An open chat backend, as ``open_chat_backend`` hands one out."""

    def __init__(self, *_a: object, **_k: object) -> None:
        pass

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _patch_session(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    _FakeApp.instances = []
    monkeypatch.setattr("fish_audio_suite_voice.tui.VoiceApp", _FakeApp)
    monkeypatch.setattr(cli, "open_chat_backend", _Backend)
    monkeypatch.setattr(cli, "_fish_tts", lambda *_a: object())

    async def fake_duplex(*args: Any, **kwargs: Any) -> int:
        calls["duplex"] = (args, kwargs)
        return 0

    monkeypatch.setattr(cli, "duplex_turns", fake_duplex)

    def fake_logging(**kwargs: Any) -> None:
        calls["logging"] = kwargs

    monkeypatch.setattr(cli, "configure_voice_logging", fake_logging)
    monkeypatch.setattr(cli, "forward_logs", lambda: lambda: calls.setdefault("stopped", True))


def test_run_tui_gives_the_session_to_the_app_with_the_console_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _configured(monkeypatch)
    calls: dict[str, Any] = {}
    _patch_session(monkeypatch, calls)
    assert asyncio.run(run_tui(c, debug=DebugLevel.EVENTS)) == 0
    app = _FakeApp.instances[0]
    assert "voice123" in app.info
    assert "model" in app.info
    args, kwargs = calls["duplex"]
    assert args[0] is c
    assert args[4] is app.session
    assert kwargs["source"] is app.live
    assert kwargs["console"] is False
    # Logging moves off stderr, which the screen owns, and the forwarding is undone.
    assert calls["logging"] == {"debug": DebugLevel.EVENTS, "to_stderr": False}
    assert calls["stopped"] is True


def test_run_tui_returns_the_code_the_app_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _configured(monkeypatch)
    _patch_session(monkeypatch, {})

    class _Failing(_FakeApp):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.code = 2

    monkeypatch.setattr("fish_audio_suite_voice.tui.VoiceApp", _Failing)
    assert asyncio.run(run_tui(c, debug=DebugLevel.OFF)) == 2
