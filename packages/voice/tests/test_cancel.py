from __future__ import annotations

import asyncio

import pytest

from fish_audio_suite_voice.cancel import is_cancel_noise, quiet_shutdown
from fish_audio_suite_voice.log_setup import configure_voice_logging


def test_quiet_shutdown_cancels_leftover_tasks() -> None:
    async def scene() -> bool:
        hung = asyncio.create_task(asyncio.sleep(30))
        await quiet_shutdown(asyncio.get_running_loop())
        return hung.cancelled()

    assert asyncio.run(scene()) is True


def test_cancel_noise_checks_types_and_the_cancel_flag_before_message_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert is_cancel_noise(asyncio.CancelledError())
    assert is_cancel_noise(GeneratorExit())
    odd = RuntimeError("some brand new anyio wording")
    assert not is_cancel_noise(odd)
    assert is_cancel_noise(odd, cancelled=True)
    assert is_cancel_noise(ExceptionGroup("g", [odd]), cancelled=True)
    assert not is_cancel_noise(ExceptionGroup("g", [odd]))
    assert not is_cancel_noise(KeyError("auth"))


def test_cancel_noise_falls_back_to_the_message_and_logs_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_voice_logging(debug=1)
    try:
        assert is_cancel_noise(RuntimeError("Attempted to exit cancel scope in a different task"))
        # A match on the wording is logged, so a reworded SDK message shows up.
        assert "matched by message" in capsys.readouterr().err
    finally:
        configure_voice_logging(debug=False)
