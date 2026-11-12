from __future__ import annotations

import asyncio
import time

import pytest

from fish_audio_suite_kit import FISH_RETRY_ATTEMPTS
from fish_audio_suite_voice import pause
from fish_audio_suite_voice.pause import Retry, fish_retries, sleep_unless


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


def test_fish_retries_gives_up_with_what_the_last_attempt_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The caller never counts attempts: an attempt that always asks for another still ends,
    # with what its Retry says the failure comes to.
    monkeypatch.setattr(pause, "fish_backoff_s", lambda number, retry_after=None: 0.0)
    tried: list[int] = []
    announced: list[int] = []

    async def attempt(number: int) -> str | Retry[str]:
        tried.append(number)
        return Retry(
            lambda: f"gave up after {number + 1}", on_retry=lambda: announced.append(number)
        )

    assert (
        asyncio.run(fish_retries(attempt, lambda: False)) == f"gave up after {FISH_RETRY_ATTEMPTS}"
    )
    assert tried == list(range(FISH_RETRY_ATTEMPTS))
    assert announced == list(range(FISH_RETRY_ATTEMPTS - 1))  # the last one is not retried


def test_fish_retries_stops_without_giving_up_when_cancelled_mid_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backoff_s = 30.0
    monkeypatch.setattr(pause, "fish_backoff_s", lambda number, retry_after=None: backoff_s)
    stopped = [False]
    tried: list[int] = []

    async def attempt(number: int) -> str | Retry[str]:
        tried.append(number)
        return Retry(lambda: "gave up")

    async def run() -> str | None:
        # The quit arrives once the backoff has begun, not before it.
        asyncio.get_running_loop().call_later(0.05, lambda: stopped.__setitem__(0, True))
        return await fish_retries(attempt, lambda: stopped[0])

    started = time.perf_counter()
    assert asyncio.run(run()) is None
    assert time.perf_counter() - started < backoff_s / 10
    assert tried == [0]


def test_fish_retries_stops_without_giving_up_when_cancelled_during_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pause, "fish_backoff_s", lambda number, retry_after=None: 0.0)
    stopped = [False]

    def give_up() -> str:
        raise RuntimeError("reported a failure after a quit")

    async def attempt(number: int) -> str | Retry[str]:
        if number == FISH_RETRY_ATTEMPTS - 1:
            stopped[0] = True  # quit arrives while the last attempt runs
        return Retry(give_up)

    assert asyncio.run(fish_retries(attempt, lambda: stopped[0])) is None
