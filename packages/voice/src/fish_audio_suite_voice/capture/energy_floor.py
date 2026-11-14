"""Quiet-percentile RMS floor that every mic gate scores frames against."""

from __future__ import annotations

import collections
from typing import Final

import numpy as np

__all__ = [
    "AdaptiveFloor",
]

# Quiet 30 ms frames remembered (about 2.4 s), and how many (about 0.75 s) before the
# estimate replaces the seed.
_FLOOR_WINDOW: Final = 80
_FLOOR_FILL: Final = 25
# The floor is this percentile of the quiet frames times the gain, so speech has to be
# well above the room's hiss. The estimate is then held between the low and high bounds.
_FLOOR_PERCENTILE: Final = 20.0
_FLOOR_GAIN: Final = 2.5
_FLOOR_LO: Final = 80.0
_FLOOR_HI: Final = 450.0


class AdaptiveFloor:
    """Quiet-percentile RMS floor that follows the room.

    Parameters
    ----------
    default : float
        The seed, used until enough quiet frames are seen. The floor never drops below it.
    window : int, optional
        Quiet frames remembered.

    Notes
    -----
    The estimate is the 20th percentile of the quiet frames times 2.5, held between 80
    and 450. The seed still wins when it is higher.
    """

    def __init__(self, default: float, *, window: int = _FLOOR_WINDOW) -> None:
        self.default = default
        # A window smaller than the warm-up count could never fill.
        self.window: collections.deque[float] = collections.deque(maxlen=max(1, window))

    def observe(self, rms: float, *, quiet: bool) -> None:
        """Record RMS while the room is quiet so the floor can track hiss."""
        if quiet:
            self.window.append(rms)

    def value(self) -> float:
        """Return the raised noise floor, or the seed until the window fills."""
        if len(self.window) < min(_FLOOR_FILL, self.window.maxlen or _FLOOR_FILL):
            return self.default
        quiet = np.fromiter(self.window, dtype=np.float64)
        est = float(np.percentile(quiet, _FLOOR_PERCENTILE) * _FLOOR_GAIN)
        # The high cap limits the estimate. It must not undercut the seed, or a seed
        # set above it (FISH_VOICE_MIN_RMS, FISH_VOICE_BARGE_RMS) would accept quieter frames.
        capped = min(_FLOOR_HI, max(_FLOOR_LO, est))
        return float(max(self.default, capped))
