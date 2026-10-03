"""Quiet-percentile RMS floor shared by the listen and barge gates."""

from __future__ import annotations

import collections
from typing import Final

import numpy as np

__all__ = [
    "AdaptiveFloor",
]

_FLOOR_WINDOW: Final = 80
_FLOOR_FILL: Final = 25
_FLOOR_PERCENTILE: Final = 20.0
_FLOOR_GAIN: Final = 2.5
_FLOOR_LO: Final = 80.0
_FLOOR_HI: Final = 450.0


class AdaptiveFloor:
    """Quiet-percentile RMS gate. ``default`` is the seed until the window fills."""

    def __init__(
        self,
        default: float,
        *,
        window: int = _FLOOR_WINDOW,
        percentile: float = _FLOOR_PERCENTILE,
        gain: float = _FLOOR_GAIN,
        lo: float = _FLOOR_LO,
        hi: float = _FLOOR_HI,
    ) -> None:
        self.default = default
        self.percentile = percentile
        self.gain = gain
        self.lo = lo
        self.hi = hi
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
        est = float(np.percentile(quiet, self.percentile) * self.gain)
        # The high cap limits the estimate. It must not undercut the seed,
        # or a loud FISH_VOICE_MIN_RMS starts accepting quieter frames.
        capped = min(self.hi, max(self.lo, est))
        return float(max(self.default, capped))
