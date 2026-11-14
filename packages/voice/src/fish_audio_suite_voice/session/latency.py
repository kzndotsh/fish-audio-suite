"""One turn's timings, and the lines a display prints about one turn or a whole session."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "LatencySnapshot",
    "percentile",
    "session_summary",
    "turn_summary",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class LatencySnapshot:
    """One cascade turn. Times are milliseconds. Never store utterance text.

    Attributes
    ----------
    asr_ms : float or None
        From the start of recognition until the final transcript is back. With
        Fish ASR that starts when the finished utterance is sent. With a
        streaming recogniser (Deepgram Flux) it starts at the last voiced audio.
    llm_first_token_ms : float or None
        From the LLM request until its first token.
    tts_first_text_ms : float or None
        From the start of the TTS turn until the first text is sent to Fish. When
        the reply streams, this includes waiting for the model's first piece.
    tts_first_audio_ms : float or None
        From the start of the TTS turn until the first audio chunk arrives from Fish.
    voice_to_voice_ms : float or None
        From the start of ``asr_ms`` until the TTS turn ends, after playback
        finishes or is cut off.
    trace_id : str or None
        W3C trace id of the turn, when one was made.
    first_audio_ms : float or None
        From the start of ``asr_ms`` until the first Fish audio chunk goes to
        the speaker. It does not count the silence the listener waited out first.
    end_wait_ms : float or None
        The silence the listener waited out after the last word before it sent the
        utterance. None when the recogniser decided the end itself (Deepgram Flux).
    """

    asr_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_text_ms: float | None = None
    tts_first_audio_ms: float | None = None
    voice_to_voice_ms: float | None = None
    trace_id: str | None = None
    first_audio_ms: float | None = None
    end_wait_ms: float | None = None

    @property
    def since_stop_ms(self) -> float | None:
        """The wait the user hears: from their last word until the first sound.

        Returns
        -------
        float or None
            ``end_wait_ms + first_audio_ms``, or None when either is unknown.

        Examples
        --------
        >>> LatencySnapshot(first_audio_ms=900, end_wait_ms=600).since_stop_ms
        1500
        >>> LatencySnapshot(first_audio_ms=900).since_stop_ms is None
        True
        """
        if self.end_wait_ms is None or self.first_audio_ms is None:
            return None
        return self.end_wait_ms + self.first_audio_ms

    def log_line(self) -> str:
        """One timing line for the log. Missing times are omitted. No utterance text.

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
            _timing_field("end_wait", self.end_wait_ms),
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


def percentile(values: Sequence[float], fraction: float, *, nearest_rank: bool = False) -> float:
    """Return the value that ``fraction`` of ``values`` fall at or below.

    Parameters
    ----------
    values : Sequence[float]
        Any numbers. Must not be empty.
    fraction : float
        From 0 (the smallest) to 1 (the largest).
    nearest_rank : bool, optional
        True takes the smallest value with at least ``fraction`` of the values at or below
        it, the usual reading of "p90 of the turns". False takes the value at the rounded
        position ``fraction * (len - 1)``, which is what a level meter uses.

    Returns
    -------
    float
        A value from ``values``.

    Examples
    --------
    >>> percentile([4.0, 1.0, 3.0, 2.0], 0.5)
    3.0
    >>> percentile([4.0, 1.0, 3.0, 2.0], 0.5, nearest_rank=True)
    2.0
    """
    ordered = sorted(values)
    if nearest_rank:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def session_summary(snapshots: Sequence[LatencySnapshot]) -> str:
    """One human line for a whole session: the typical and the slow turn.

    Parameters
    ----------
    snapshots : Sequence of LatencySnapshot
        One per answered turn, in any order.

    Returns
    -------
    str
        ``N turns · since you stopped p50 … p90 …`` and the same for first audio, the
        LLM's first token and the total, for each one any turn recorded. Empty when there
        were no turns. A display adds its own label.
    """
    if not snapshots:
        return ""
    fields = [
        ("since you stopped", [s.since_stop_ms for s in snapshots]),
        ("first audio", [s.first_audio_ms for s in snapshots]),
        ("llm first token", [s.llm_first_token_ms for s in snapshots]),
        ("total", [s.voice_to_voice_ms for s in snapshots]),
    ]
    parts = [f"{len(snapshots)} turns"]
    for name, values in fields:
        known = [value for value in values if value is not None]
        if known:
            parts.append(
                f"{name} p50 {_seconds(percentile(known, 0.5, nearest_rank=True))}"
                f" p90 {_seconds(percentile(known, 0.9, nearest_rank=True))}"
            )
    return " · ".join(parts)


def turn_summary(snapshot: LatencySnapshot) -> str:
    """One human line for a finished turn. The first number is the wait that matters.

    Parameters
    ----------
    snapshot : LatencySnapshot
        The finished turn.

    Returns
    -------
    str
        ``since you stopped … · first audio … · …`` for each time the turn recorded, or
        empty when it recorded none. A display adds its own indent or marker.
    """
    fields = [
        ("since you stopped", snapshot.since_stop_ms),
        ("first audio", snapshot.first_audio_ms),
        ("asr", snapshot.asr_ms),
        ("llm first token", snapshot.llm_first_token_ms),
        ("tts first audio", snapshot.tts_first_audio_ms),
        ("total", snapshot.voice_to_voice_ms),
    ]
    shown = [f"{name} {_seconds(value)}" for name, value in fields if value is not None]
    return " · ".join(shown)
