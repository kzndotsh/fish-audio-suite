"""Cancellable waits and ``Retry-After`` parsing shared by the Fish and LLM retry loops."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping

POLL_S = 0.05


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


def seconds_value(raw: object) -> float | None:
    """Parse a non-negative finite seconds value.

    Parameters
    ----------
    raw : object
        A number or numeric string. Booleans are rejected.

    Returns
    -------
    float or None
        The value, or None when ``raw`` is not usable.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        try:
            value = float(raw)
        except OverflowError:
            return None
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None
    else:
        return None
    if math.isfinite(value) and value >= 0:
        return value
    return None


def header_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Read ``Retry-After`` in seconds from response headers.

    Parameters
    ----------
    headers : Mapping or None
        Response headers. Lookup is case-insensitive for ``httpx.Headers``.

    Returns
    -------
    float or None
        Seconds, or None when the header is missing or is an HTTP date.
    """
    if headers is None:
        return None
    return seconds_value(headers.get("retry-after"))
