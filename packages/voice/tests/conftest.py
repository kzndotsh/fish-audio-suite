from __future__ import annotations

from collections.abc import Iterator

import pytest

from fish_audio_suite_voice import debug as voice_debug
from fish_audio_suite_voice import live


@pytest.fixture(autouse=True)
def _reset_voice_globals() -> Iterator[None]:
    """Clear the few process-wide switches so test order cannot matter.

    Duplex cancel state lives on ``DuplexSession`` and the echo tap on
    ``EchoCanceller``, so only logging state and the warn-once cache are global.
    """
    voice_debug._DEBUG.level = 0
    voice_debug._DEBUG.frozen = None
    voice_debug._TURN.t0 = None
    voice_debug._REPLY.open = False
    live._warn_coerced.cache_clear()
    yield
    voice_debug._DEBUG.level = 0
    voice_debug._DEBUG.frozen = None
    voice_debug._TURN.t0 = None
    voice_debug._REPLY.open = False
    live._warn_coerced.cache_clear()
