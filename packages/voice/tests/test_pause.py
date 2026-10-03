from __future__ import annotations

import asyncio
import time

import pytest

from fish_audio_suite_voice.pause import seconds_value, sleep_unless


def test_seconds_value_ignores_an_integer_too_large_for_a_float() -> None:
    assert seconds_value(10**400) is None
    assert seconds_value("1" + "0" * 400) is None
    assert seconds_value(2) == 2.0
    assert seconds_value("1.5") == 1.5
    assert seconds_value(True) is None


@pytest.mark.perf
def test_sleep_unless_with_a_bad_poll_still_finishes() -> None:
    async def run(poll: float) -> bool:
        return await asyncio.wait_for(sleep_unless(0.05, lambda: False, poll_s=poll), timeout=2.0)

    for poll in (0.0, -1.0, float("nan"), float("inf")):
        started = time.perf_counter()
        assert asyncio.run(run(poll)) is False
        assert time.perf_counter() - started < 1.0


def test_sleep_unless_stops_when_cancelled() -> None:
    assert asyncio.run(sleep_unless(30.0, lambda: True, poll_s=0.0)) is True
