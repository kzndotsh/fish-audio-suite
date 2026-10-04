"""Per-turn latency numbers and the clock helper that makes them."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

__all__ = [
    "MS_PER_S",
    "LatencySnapshot",
    "elapsed_ms",
]


@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """One cascade turn. Times are milliseconds. Never store utterance text.

    Attributes
    ----------
    asr_ms : float or None
        From sending the finished utterance to Fish ASR until its transcript is back.
    llm_first_token_ms : float or None
        From the LLM request until its first token.
    tts_first_text_ms : float or None
        From the start of the TTS turn until the first text is sent to Fish. When
        the reply streams, this includes waiting for the model's first piece.
    tts_first_audio_ms : float or None
        From the start of the TTS turn until the first audio chunk arrives from Fish.
    voice_to_voice_ms : float or None
        From sending the utterance to ASR until the TTS turn ends, after playback
        finishes or is cut off.
    trace_id : str or None
        W3C trace id of the turn, when one was made.
    first_audio_ms : float or None
        From sending the utterance to ASR until the first Fish audio chunk goes to
        the speaker. This is the wait the user hears after they stop talking.

    Notes
    -----
    The fields keep their order, so a snapshot built by position keeps its
    meaning; a new field goes last and is not keyword-only.
    """

    asr_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_text_ms: float | None = None
    tts_first_audio_ms: float | None = None
    voice_to_voice_ms: float | None = None
    trace_id: str | None = None
    # Last, and deliberately not keyword-only: a caller that builds a snapshot by
    # position keeps its meaning, and a new field must never shift the old ones.
    first_audio_ms: float | None = None

    def log_line(self) -> str:
        """One stdout timing line. Missing times are omitted. No utterance text.

        Returns
        -------
        str
            ``[timing asr=…ms … trace=…]``. Each key is its field name without
            ``_ms``. ``trace`` appears only when set.

        Examples
        --------
        >>> LatencySnapshot(asr_ms=40, tts_first_audio_ms=12.4).log_line()
        '[timing asr=40ms tts_first_audio=12ms]'
        """
        parts = [
            _timing_field("asr", self.asr_ms),
            _timing_field("llm_first_token", self.llm_first_token_ms),
            _timing_field("tts_first_text", self.tts_first_text_ms),
            _timing_field("tts_first_audio", self.tts_first_audio_ms),
            _timing_field("first_audio", self.first_audio_ms),
            _timing_field("voice_to_voice", self.voice_to_voice_ms),
        ]
        if self.trace_id:
            parts.append(f"trace={self.trace_id}")
        shown = [part for part in parts if part]
        return "[timing " + " ".join(shown) + "]"


def _timing_field(name: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"{name}={value:.0f}ms"


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
