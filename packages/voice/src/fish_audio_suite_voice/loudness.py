"""Bring a reply to a steady loudness while it streams, so voices are comparable.

Fish voices differ by up to 19 dB in level, and the realtime websocket cannot ask Fish to
level them. This measures the speech in the reply as it arrives and applies one smoothed
gain toward a target, then a soft limiter keeps a boosted peak from clipping.

The gain is decided from the first chunk or two, because the first words are already
playing by the time later chunks exist, and it moves slowly after that, so a loud word
does not make the next quiet one pump. Silence is not counted when measuring and the gain
is bounded, so a pause is never turned into hiss.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from fish_audio_suite_voice.aec import SAMPLE_BYTES

__all__ = [
    "LoudnessMatch",
]

_BLOCK_MS: Final = 20
_MS_PER_S: Final = 1000
_FULL_SCALE: Final = 32768.0
# Blocks quieter than this are pauses and breath, not speech, so they do not set the level.
_GATE_DB: Final = -50.0
# Speech needed before the level is trusted. Until then the reply plays untouched.
_MIN_SPEECH_S: Final = 0.1
# How fast the gain may move once it is set, in dB per second.
_SLEW_DB_PER_S: Final = 6.0
# The limiter leaves everything below the knee alone and bends the rest toward the ceiling.
_KNEE: Final = 0.8
_CEILING: Final = 0.98


def _soft_limit(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bend peaks above the knee toward the ceiling, leaving smaller samples unchanged."""
    magnitude = np.abs(samples)
    room = _CEILING - _KNEE
    bent = _KNEE + room * np.tanh((magnitude - _KNEE) / room)
    return np.where(magnitude > _KNEE, np.copysign(bent, samples), samples)


class LoudnessMatch:
    """Apply a steady gain to a stream of 16-bit mono PCM so its speech reaches a target level.

    Parameters
    ----------
    sample_rate : int
        Samples per second of the audio.
    target_db : float
        The level to bring speech to, in dB below full scale (RMS), such as ``-18``.
    max_boost_db : float, optional
        The most the audio may be made louder, in dB.
    max_cut_db : float, optional
        The most the audio may be made quieter, in dB.

    Notes
    -----
    One instance handles one reply, chunk by chunk, and keeps its state between chunks. It
    adds no delay: every chunk comes back the same length, so it can sit in front of the
    far-end tap and the speaker. The level is the running average over all the speech in
    the reply so far, so it settles as more arrives instead of following each word.
    """

    def __init__(
        self,
        sample_rate: int,
        target_db: float,
        *,
        max_boost_db: float = 12.0,
        max_cut_db: float = 12.0,
    ) -> None:
        self._block = max(1, sample_rate * _BLOCK_MS // _MS_PER_S)
        self._target_db = target_db
        self._max_boost_db = max_boost_db
        self._max_cut_db = max_cut_db
        self._step_db = _SLEW_DB_PER_S * _BLOCK_MS / _MS_PER_S
        self._min_speech = int(sample_rate * _MIN_SPEECH_S)
        self._energy = 0.0
        self._speech = 0
        self._gain_db = 0.0
        self._decided = False
        self._unity = 0
        self._odd = b""

    @property
    def gain_db(self) -> float:
        """The gain in dB applied at the end of the audio processed so far."""
        return self._gain_db

    def process(self, pcm: bytes) -> bytes:
        """Level a chunk of little-endian 16-bit mono PCM.

        Parameters
        ----------
        pcm : bytes
            One chunk of audio.

        Returns
        -------
        bytes
            The chunk with the gain applied and peaks limited. The same length, apart
            from a trailing odd byte, which is held for the next chunk.
        """
        data = self._odd + pcm
        usable = len(data) - len(data) % SAMPLE_BYTES
        self._odd = data[usable:]
        if usable == 0:
            return b""
        samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float64) / _FULL_SCALE
        self._measure(samples)
        wanted = self._wanted_db()
        if wanted is None:
            self._unity += usable
            return data[:usable]
        # The first decision takes effect at once: from the first sample when nothing has
        # played yet, otherwise over one block. After that the gain only drifts.
        first = not self._decided
        self._decided = True
        out = np.empty_like(samples)
        for start in range(0, samples.size, self._block):
            block = samples[start : start + self._block]
            before = self._gain_db
            if first:
                self._gain_db = wanted
                before = wanted if self._unity == 0 else before
                first = False
            else:
                self._slew_toward(wanted)
            ramp = np.linspace(before, self._gain_db, block.size + 1)[1:]
            out[start : start + block.size] = block * np.power(10.0, ramp / 20.0)
        shaped = _soft_limit(out)
        return (
            np.rint(shaped * _FULL_SCALE)
            .clip(-_FULL_SCALE, _FULL_SCALE - 1)
            .astype("<i2")
            .tobytes()
        )

    def _measure(self, samples: NDArray[np.float64]) -> None:
        for start in range(0, samples.size, self._block):
            block = samples[start : start + self._block]
            energy = float(np.dot(block, block))
            mean_square = energy / block.size
            if mean_square > 0 and 10.0 * math.log10(mean_square) >= _GATE_DB:
                self._energy += energy
                self._speech += block.size

    def _wanted_db(self) -> float | None:
        if self._speech < self._min_speech:
            return None
        level_db = 10.0 * math.log10(self._energy / self._speech)
        return min(self._max_boost_db, max(-self._max_cut_db, self._target_db - level_db))

    def _slew_toward(self, wanted: float) -> None:
        delta = wanted - self._gain_db
        self._gain_db += max(-self._step_db, min(self._step_db, delta))
