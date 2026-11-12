from __future__ import annotations

from collections.abc import Iterator

import pytest

from fish_audio_suite_voice import debug as voice_debug
from fish_audio_suite_voice import log_format, speaker


def _no_display() -> None:
    return None


def _reset() -> None:
    voice_debug.reset_debug_level()  # so ``warn`` writes to stderr again
    # A display a test left open: take the hook over, then let it go back to nothing.
    voice_debug.on_terminal_line(_no_display)()
    log_format.stop_turn_clock()
    speaker._warn_coerced.cache_clear()


@pytest.fixture(autouse=True)
def _reset_voice_globals() -> Iterator[None]:
    """Clear the few process-wide switches so test order cannot matter.

    Duplex cancel state lives on ``DuplexSession`` and the echo tap on
    ``EchoCanceller``, so only logging state and the warn-once cache are global.
    """
    _reset()
    yield
    _reset()
