"""Timing guard for text functions that must stay linear on hostile input."""

from __future__ import annotations

import time
from collections.abc import Callable

FAST_ENOUGH_S = 0.5
MAX_RATIO = 10.0
# A quarter of the text. Linear work takes about 4x as long on the whole text,
# quadratic work about 16x, so 10x separates them with room for noise.
SMALLER_BY = 4


def _best_of_two(run: Callable[[str], object], text: str) -> float:
    best = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        run(text)
        best = min(best, time.perf_counter() - started)
    return best


def assert_linear_time(run: Callable[[str], object], text: str) -> None:
    """Fail when ``run(text)`` is slow and grows faster than the text does.

    Parameters
    ----------
    run : Callable
        The function under test, called with one string.
    text : str
        A hostile input built from a repeated unit, so its prefix is hostile too.

    Notes
    -----
    A fast run passes outright, so a slow CI runner cannot fail a linear
    function. A run over ``FAST_ENOUGH_S`` seconds is compared with the run on
    a quarter of the text, and it fails only if it took ``MAX_RATIO`` times as
    long, which is the signature of quadratic work.
    """
    full = _best_of_two(run, text)
    if full < FAST_ENOUGH_S:
        return
    quarter = _best_of_two(run, text[: len(text) // SMALLER_BY])
    ratio = full / max(quarter, 1e-3)
    assert ratio < MAX_RATIO, (
        f"{len(text)} characters took {full:.2f}s, {ratio:.0f}x the time of a quarter of "
        f"them ({quarter:.3f}s). Linear work would be about {SMALLER_BY}x."
    )
