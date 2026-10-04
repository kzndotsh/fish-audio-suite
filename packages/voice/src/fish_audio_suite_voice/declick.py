"""Fade the edges of each sentence that meets silence, so it does not click.

Fish speaks a streamed reply as separate pieces. A piece can begin on a loud sample
straight after digital silence, and it can stop while still loud, straight into
digital silence. Either step from sound to nothing, or nothing to sound, is heard as a
faint click, and a question that ends on a held, rising note is the most likely to be
cut hard. A few milliseconds of fade at each such edge removes the step and leaves the
rest of the waveform alone.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EdgeFade",
]

# Samples this small count as silence (about -62 dBFS for 16-bit audio).
_MS_PER_S: Final = 1000
_QUIET: Final = 24
# Silence this long around a sound makes its edge a sentence edge. Sentence gaps are
# far longer; the closure before a plosive inside a word is far shorter, so those stay
# sharp.
_MIN_QUIET_MS: Final = 5.0
# How much silence has to follow before the end of a sound counts as an edge. It is
# also how long audio is held back so the fade can still reach the sound before it.
_CONFIRM_MS: Final = _MIN_QUIET_MS


class EdgeFade:
    """Fade in a sound that follows silence, and fade out a sound that ends in it.

    Parameters
    ----------
    sample_rate : int
        Samples per second of the audio.
    fade_ms : float
        Length of each fade. 0 turns the effect off.

    Notes
    -----
    One instance handles one stream of 16-bit mono PCM, chunk by chunk, and remembers
    its state between chunks. To fade a sound out it has to see that silence follows, so
    it holds back ``fade_ms`` plus 5 ms of audio: the output runs that much behind
    the input, and chunks come out a different size from how they went in. Call
    ``finish`` at the end of the stream to get the held audio. The total length does not
    change. The start of the stream counts as following silence, and its end as
    followed by it.
    """

    def __init__(self, sample_rate: int, fade_ms: float) -> None:
        self._fade = max(0, round(sample_rate * fade_ms / _MS_PER_S))
        self._min_quiet = max(1, round(sample_rate * _MIN_QUIET_MS / _MS_PER_S))
        self._confirm = max(1, round(sample_rate * _CONFIRM_MS / _MS_PER_S))
        self._hold = self._fade + self._confirm
        steps = (np.arange(self._fade, dtype=np.float64) + 1.0) / (self._fade + 1.0)
        self._up = steps
        self._down = steps[::-1].copy()
        self._raw = np.zeros(0, dtype=np.int16)
        self._gain = np.zeros(0, dtype=np.float64)
        self._base = 0  # absolute index of the first held sample
        # Quiet samples just before the first held sample. The stream start counts as quiet.
        self._quiet_before = self._min_quiet
        self._last_onset = -1
        self._last_offset = -1
        self._pending_up = 0  # samples of a fade-in that run past the end of the buffer
        self._odd = b""

    def process(self, pcm: bytes) -> bytes:
        """Fade the edges in a chunk of little-endian 16-bit mono PCM.

        Parameters
        ----------
        pcm : bytes
            One chunk of audio.

        Returns
        -------
        bytes
            Audio that is ready to play. It lags the input by the held-back tail, so it
            can be shorter than ``pcm``, or empty.
        """
        if self._fade == 0:
            return pcm
        data = self._odd + pcm
        usable = len(data) - len(data) % 2
        self._odd = data[usable:]
        if usable == 0:
            return b""
        new = np.frombuffer(data[:usable], dtype="<i2").astype(np.int16)
        start = self._raw.size
        self._raw = np.concatenate([self._raw, new])
        self._gain = np.concatenate([self._gain, np.ones(new.size, dtype=np.float64)])
        self._continue_fade_in(start)
        self._find_edges()
        return self._emit(keep=self._hold)

    def finish(self) -> bytes:
        """Return the audio still held back, with the end of the stream faded out.

        Returns
        -------
        bytes
            The last few milliseconds. Empty when nothing is held.
        """
        if self._fade == 0 or self._raw.size == 0:
            return b""
        # The stream ends here, so whatever sound is left ramps down to nothing.
        tail = min(self._fade, self._raw.size)
        self._gain[-tail:] = np.minimum(self._gain[-tail:], self._down[self._fade - tail :])
        return self._emit(keep=0)

    def _continue_fade_in(self, start: int) -> None:
        if self._pending_up <= 0:
            return
        done = self._fade - self._pending_up
        take = min(self._pending_up, self._raw.size - start)
        self._gain[start : start + take] = self._up[done : done + take]
        self._pending_up -= take

    def _find_edges(self) -> None:
        raw = self._raw
        loud = np.abs(raw.astype(np.int32)) >= _QUIET
        index = np.arange(raw.size)
        last_loud = np.maximum.accumulate(np.where(loud, index, -1))
        earlier = np.concatenate(([-1], last_loud[:-1]))
        before = np.where(earlier >= 0, index - earlier - 1, self._quiet_before + index)
        for onset in np.flatnonzero(loud & (before >= self._min_quiet)):
            absolute = self._base + int(onset)
            if absolute > self._last_onset:
                self._last_onset = absolute
                self._fade_in(int(onset))
        quiet = ~loud
        for edge in np.flatnonzero(quiet[1:] & loud[:-1]) + 1:
            absolute = self._base + int(edge)
            end = int(edge) + self._confirm
            if absolute > self._last_offset and end <= raw.size and quiet[int(edge) : end].all():
                self._last_offset = absolute
                self._fade_out(int(edge))

    def _fade_in(self, onset: int) -> None:
        take = min(self._fade, self._raw.size - onset)
        self._gain[onset : onset + take] = np.minimum(
            self._gain[onset : onset + take], self._up[:take]
        )
        self._pending_up = self._fade - take

    def _fade_out(self, edge: int) -> None:
        first = max(0, edge - self._fade)
        length = edge - first
        self._gain[first:edge] = np.minimum(
            self._gain[first:edge], self._down[self._fade - length :]
        )

    def _emit(self, *, keep: int) -> bytes:
        emit = max(0, self._raw.size - keep)
        if emit == 0:
            return b""
        shaped: NDArray[np.float64] = self._raw[:emit] * self._gain[:emit]
        out = np.rint(shaped).astype("<i2").tobytes()
        quiet = np.abs(self._raw[:emit].astype(np.int32)) < _QUIET
        if quiet.all():
            self._quiet_before += emit
        else:
            self._quiet_before = int(emit - 1 - np.flatnonzero(~quiet)[-1])
        self._raw = self._raw[emit:]
        self._gain = self._gain[emit:]
        self._base += emit
        return out
