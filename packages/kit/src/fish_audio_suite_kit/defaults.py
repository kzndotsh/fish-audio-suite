"""The default value of every shared Fish TTS and ASR knob."""

from __future__ import annotations

from dataclasses import dataclass

from fish_audio_suite_kit.catalog import DEFAULT_TTS_MODEL, OPUS_AUTO
from fish_audio_suite_kit.cuts import PARTIAL_CHARS
from fish_audio_suite_kit.literals import AudioFormat, FishLatency

__all__ = [
    "SuiteDefaults",
]


@dataclass(frozen=True, slots=True)
class SuiteDefaults:
    """Shared Fish TTS and ASR knobs. Callers still clamp before a request.

    Notes
    -----
    ``chunk_length`` starts at 200. Cloud accepts 100-300; a self-hosted
    base accepts up to 1000 (see ``chunk_length_hi``). ``opus_bitrate``
    of -1000 asks Fish to pick the rate. ``tts_partial_chars`` is how far
    ``next_tts_cut`` will flush without a sentence end. ``asr_language``
    is empty unless the caller or env sets a hint.
    """

    tts_model: str = DEFAULT_TTS_MODEL
    asr_model: str = "transcribe-1-pro"
    asr_language: str = ""
    latency: FishLatency = "normal"
    chunk_length: int = 200
    min_chunk_length: int = 50
    audio_format: AudioFormat = "mp3"
    mp3_bitrate: int = 128
    opus_bitrate: int = OPUS_AUTO
    opus_sample_rate: int = 48000
    speed: float = 1.0
    volume: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    max_new_tokens: int = 1024
    early_stop_threshold: float = 1.0
    normalize: bool = True
    normalize_loudness: bool = True
    condition_on_previous_chunks: bool = True
    tts_partial_chars: int = PARTIAL_CHARS
    sample_rate: int = 44100
    fish_base: str = "https://api.fish.audio"
