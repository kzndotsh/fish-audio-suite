"""Shared Fish TTS/ASR knobs. Literals live on SuiteDefaults."""

from __future__ import annotations

from dataclasses import dataclass

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
    tts_model: str = "s2.1-pro"
    asr_model: str = "transcribe-1"
    asr_language: str = "en"
    latency: str = "normal"
    chunk_length: int = 200
    min_chunk_length: int = 50
    audio_format: str = "mp3"
    mp3_bitrate: int = 128
    speed: float = 1.05
    temperature: float = 0.70
    top_p: float = 0.7
    repetition_penalty: float = 1.15
    tts_partial_chars: int = 40
    sample_rate: int = 44100
    fish_base: str = "https://api.fish.audio"
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
