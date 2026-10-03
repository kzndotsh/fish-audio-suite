"""TTS text filters. Re-exports the scrub, dialogue, and stream-hold modules."""

from __future__ import annotations

from fish_audio_suite_kit._charsets import utf8_text
from fish_audio_suite_kit.dialogue import extract_quoted_speech, is_tts_junk
from fish_audio_suite_kit.scrub_markdown import scrub_tts
from fish_audio_suite_kit.stream_holds import (
    hold_tts,
    is_empty_delta,
    sentence_closer_hold_at,
    skip_empty_delta,
    tts_hold_at,
)

__all__ = [
    "extract_quoted_speech",
    "hold_tts",
    "is_empty_delta",
    "is_tts_junk",
    "scrub_tts",
    "sentence_closer_hold_at",
    "skip_empty_delta",
    "tts_hold_at",
    "utf8_text",
]
