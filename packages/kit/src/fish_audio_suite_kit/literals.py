"""Closed value sets the three packages share. Types only, no behavior."""

from __future__ import annotations

from typing import Literal, TypedDict

__all__ = [
    "AsrFormat",
    "AudioFormat",
    "ChatMessage",
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
# Catalog TTS models. A caller may still pass another id through, so the models
# helpers return ``str`` and ``catalog_tts_model`` narrows to this set.
TtsModel = Literal["s1", "s2-pro", "s2.1-pro", "s2.1-pro-free"]
ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """One chat message in the shape OpenAI-style APIs take."""

    role: ChatRole
    content: str
