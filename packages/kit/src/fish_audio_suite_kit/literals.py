"""Closed value sets the three packages share. Types only, no behavior.

A helper that reads one of these from a string (``known_latency``) narrows it with a
table lookup built from the type, so no caller needs a ``cast`` and each set is
written once. ``normalize_tts_model`` stays ``str`` because another model id passes
through.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "AsrFormat",
    "AudioFormat",
    "ChatRole",
    "FishLatency",
    "TtsModel",
]

# Fish realtime latency modes, from fastest and roughest to slowest and best.
FishLatency = Literal["low", "balanced", "normal"]
# Audio formats Fish returns for text to speech.
AudioFormat = Literal["wav", "pcm", "mp3", "opus"]
# Response formats a transcription request can ask for.
AsrFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]
# Catalog TTS models, in the order ``FISH_TTS_MODEL_IDS`` lists them. A caller may
# still pass another id through, so the model helpers return ``str``.
TtsModel = Literal["s2.1-pro", "s2.1-pro-free", "s2-pro", "s1", "drama-3"]
ChatRole = Literal["system", "user", "assistant"]
