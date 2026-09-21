"""Shared Fish TTS/ASR knobs. Env names are comments; values live on the dataclass."""

from __future__ import annotations

from dataclasses import dataclass

# FISH_MODEL / FISH_TTS_MODEL
DEFAULT_TTS_MODEL = "s2.1-pro"
# FISH_ASR_MODEL
DEFAULT_ASR_MODEL = "transcribe-1"
# FISH_ASR_LANGUAGE
DEFAULT_ASR_LANGUAGE = "en"
# FISH_LATENCY
DEFAULT_LATENCY = "normal"
# FISH_CHUNK_LENGTH
DEFAULT_CHUNK_LENGTH = 200
# FISH_MIN_CHUNK_LENGTH
DEFAULT_MIN_CHUNK_LENGTH = 50
# FISH_FORMAT (proxy HTTP; voice live defaults to pcm)
DEFAULT_FORMAT = "mp3"
# FISH_MP3_BITRATE
DEFAULT_MP3_BITRATE = 128
# FISH_SPEED / FISH_SPEED_SCALE
DEFAULT_SPEED = 1.05
# FISH_TEMPERATURE
DEFAULT_TEMPERATURE = 0.70
# FISH_TOP_P
DEFAULT_TOP_P = 0.7
# FISH_REPETITION_PENALTY
DEFAULT_REPETITION_PENALTY = 1.15
# FISH_TTS_PARTIAL_CHARS
DEFAULT_TTS_PARTIAL_CHARS = 40
# FISH_SAMPLE_RATE
DEFAULT_SAMPLE_RATE = 44100
# FISH_BASE
DEFAULT_FISH_BASE = "https://api.fish.audio"

DEFAULT_SYSTEM_PROMPT = (
    "You are a spoken assistant using Fish Audio TTS. "
    "English by default. No markdown, bullets, or URLs. Keep replies speakable and bounded. "
    "The first token of each sentence is a square-bracket [cue] tag "
    "(for example [clear], [happy], [calm], [laughing], [sighing], [whispering], [break]). "
    "Lowercase tags only. Those brackets are synthesis instructions and are never spoken. "
    "Do not start a sentence with a bare mood word. Speak the user's language if they switch."
)


@dataclass(frozen=True)
class SuiteDefaults:
    tts_model: str = DEFAULT_TTS_MODEL
    asr_model: str = DEFAULT_ASR_MODEL
    asr_language: str = DEFAULT_ASR_LANGUAGE
    latency: str = DEFAULT_LATENCY
    chunk_length: int = DEFAULT_CHUNK_LENGTH
    min_chunk_length: int = DEFAULT_MIN_CHUNK_LENGTH
    audio_format: str = DEFAULT_FORMAT
    mp3_bitrate: int = DEFAULT_MP3_BITRATE
    speed: float = DEFAULT_SPEED
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    tts_partial_chars: int = DEFAULT_TTS_PARTIAL_CHARS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    fish_base: str = DEFAULT_FISH_BASE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class LatencySnapshot:
    """One cascade turn. Times are milliseconds. Never store utterance text."""

    srt: float | None = None
    llm_ttft: float | None = None
    llm_ttfs: float | None = None
    ttfa: float | None = None
    voice_to_voice: float | None = None

    def log_line(self) -> str:
        def fmt(name: str, value: float | None) -> str:
            if value is None:
                return f"{name}=-1"
            return f"{name}={value:.0f}ms"

        return (
            "[timing "
            f"{fmt('srt', self.srt)} "
            f"{fmt('llm_ttft', self.llm_ttft)} "
            f"{fmt('llm_ttfs', self.llm_ttfs)} "
            f"{fmt('ttfa', self.ttfa)} "
            f"{fmt('voice_to_voice', self.voice_to_voice)}]"
        )
