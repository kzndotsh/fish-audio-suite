"""What is done to a reply's PCM between Fish and the sink: fade the edges, then level it."""

from __future__ import annotations

from fish_audio_suite_voice.declick import EdgeFade
from fish_audio_suite_voice.loudness import LoudnessMatch

__all__ = [
    "AudioChain",
]


class AudioChain:
    """Fade the edges of each sentence, then bring the speech to the target level.

    Fish can start or stop a sentence on a loud sample next to silence, which clicks. The
    level goes last and before the sink, so the far-end tap sees what the speaker plays.

    Parameters
    ----------
    pcm : bool
        Whether the audio is raw 16-bit PCM. Encoded audio passes through unchanged.
    sample_rate : int
        Samples per second of the PCM.
    fade_ms : float
        Fade length at each edge. 0 turns the fade off.
    loudness_db : float or None
        Target level in dB below full scale, or None to leave the level alone.
    """

    def __init__(
        self, *, pcm: bool, sample_rate: int, fade_ms: float, loudness_db: float | None
    ) -> None:
        self._fade = EdgeFade(sample_rate, fade_ms) if pcm and fade_ms > 0 else None
        self._level = (
            LoudnessMatch(sample_rate, loudness_db) if pcm and loudness_db is not None else None
        )

    def process(self, chunk: bytes) -> bytes:
        """Return the part of ``chunk`` that is ready to play, faded and levelled."""
        shaped = chunk if self._fade is None else self._fade.process(chunk)
        return shaped if self._level is None or not shaped else self._level.process(shaped)

    def finish(self) -> bytes:
        """Return what the fade still held at the end of the reply, levelled."""
        tail = b"" if self._fade is None else self._fade.finish()
        return tail if self._level is None or not tail else self._level.process(tail)
