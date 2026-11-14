"""The clock helper that times a step in milliseconds."""

from __future__ import annotations

import time
from typing import Final

__all__ = [
    "MS_PER_S",
    "elapsed_ms",
]


MS_PER_S: Final = 1000


def elapsed_ms(started: float) -> float:
    """Milliseconds since a ``time.perf_counter`` reading.

    Parameters
    ----------
    started : float
        Value previously returned by ``time.perf_counter``.

    Returns
    -------
    float
        Elapsed milliseconds. Not rounded.
    """
    return (time.perf_counter() - started) * MS_PER_S
