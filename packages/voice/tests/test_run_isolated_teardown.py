"""``run_isolated`` must tear its loop down without closing a live websocket iterator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import anyio
import pytest
from voice_fakes import make_result

from fish_audio_suite_voice.live import TtsResult
from fish_audio_suite_voice.tts_turn import run_isolated


async def _ws_like() -> AsyncIterator[int]:
    # Same shape as the Fish SDK's iterator: an async generator that owns an anyio
    # task group. Closing it from a task other than the one that entered it raises.
    async with anyio.create_task_group() as group:

        async def idle() -> None:
            await asyncio.sleep(3600)

        group.start_soon(idle)
        for i in range(10):
            yield i
            await asyncio.sleep(0)


def test_an_abandoned_task_group_iterator_is_not_closed_at_teardown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = make_result()

    async def turn() -> TtsResult:
        events = _ws_like()
        await anext(events)  # a cancelled turn stops iterating mid-stream
        return result

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        assert run_isolated(turn()) is result
    # asyncio.Runner (or asyncio.run) would aclose() the iterator at shutdown and log
    # "Attempted to exit cancel scope in a different task".
    assert caplog.records == []
