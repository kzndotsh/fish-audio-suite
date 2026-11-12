"""Cancellable waits, and the Fish retry loop that the TTS and ASR requests share."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from fish_audio_suite_kit import fish_attempt_exhausted, fish_backoff_s

__all__ = [
    "Retry",
    "fish_retries",
    "sleep_unless",
]

# A cancel ends the wait within 50 ms.
POLL_S: Final = 0.05


async def sleep_unless(
    seconds: float,
    cancelled: Callable[[], bool],
    *,
    poll_s: float = POLL_S,
) -> bool:
    """Sleep for ``seconds``, waking early when ``cancelled`` turns true.

    Parameters
    ----------
    seconds : float
        How long to wait.
    cancelled : Callable
        Returns True once the wait should stop, for example ``event.is_set``.
    poll_s : float, optional
        How often ``cancelled`` is checked.

    Returns
    -------
    bool
        True when the wait was cancelled, before or during the sleep.

    Notes
    -----
    One long ``asyncio.sleep`` ignores barge-in and Ctrl+C. The next attempt
    would then run after the user already interrupted.
    """
    # A zero or negative poll would never shrink ``left``.
    poll = poll_s if math.isfinite(poll_s) and poll_s > 0 else POLL_S
    left = seconds
    while left > 0:
        if cancelled():
            return True
        step = min(poll, left)
        await asyncio.sleep(step)
        left -= step
    return cancelled()


@dataclass(frozen=True, slots=True)
class Retry[T]:
    """What a failed Fish attempt returns: try again after the backoff, or give up.

    Attributes
    ----------
    give_up : Callable
        What the failure comes to when no attempt is left. ``fish_retries`` calls it after
        the last attempt and returns its result, so it raises, or returns what the caller
        reports.
    retry_after : float or None
        The wait the reply asked for in ``Retry-After``, which the backoff honours. None
        when there was no header: a TTS failure never has one, because the SDK's
        ``APIError`` carries no headers.
    on_retry : Callable or None
        Called when another attempt will follow, before the wait, such as to log it.
    """

    give_up: Callable[[], T]
    retry_after: float | None = None
    on_retry: Callable[[], None] | None = None


async def fish_retries[T](
    attempt: Callable[[int], Awaitable[T | Retry[T]]],
    stop: Callable[[], bool],
) -> T | None:
    """Run ``attempt`` until it returns a result, waiting out the Fish backoff between tries.

    Parameters
    ----------
    attempt : Callable
        Called with the zero-based attempt number. It returns its result, or ``Retry`` to
        be called again after the backoff.
    stop : Callable
        Says whether the caller was cancelled, for example ``event.is_set``. It is read
        before each attempt and during each wait, so a quit never starts another try.

    Returns
    -------
    T or None
        What the attempt returned, or what ``Retry.give_up`` returned for the last attempt.
        None when ``stop`` turned true first, including during the last attempt.

    Notes
    -----
    At most ``FISH_RETRY_ATTEMPTS`` attempts: the last one's ``Retry`` gives up instead of
    waiting. The wait is ``fish_backoff_s``, with jitter.
    """
    number = 0
    while True:
        if stop():
            return None
        outcome = await attempt(number)
        if not isinstance(outcome, Retry):
            return outcome
        if fish_attempt_exhausted(number):
            # A quit during the last attempt is a stop, not a failure to report.
            return None if stop() else outcome.give_up()
        if outcome.on_retry is not None:
            outcome.on_retry()
        # A plain sleep would ignore barge-in or quit, and the next try would run anyway.
        if await sleep_unless(fish_backoff_s(number, retry_after=outcome.retry_after), stop):
            return None
        number += 1
