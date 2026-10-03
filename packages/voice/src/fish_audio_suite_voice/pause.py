"""Cancellable waits shared by the Fish and LLM retry loops."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from typing import Final

__all__ = [
    "sleep_unless",
]

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
