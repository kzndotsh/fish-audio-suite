"""Shared Fish TTS/ASR knobs. Literals live on SuiteDefaults."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a spoken assistant using Fish Audio TTS. "
    "English by default. No markdown, bullets, or URLs. Keep replies speakable and bounded. "
    "Use square-bracket [cue] tags for S2 (for example [happy], [whispering], [break], [long-break]). "
    "Tags may stack ([sad][whispering] …) and may be free-form ([whisper in small voice]). "
    "Lowercase. One primary emotion per sentence. Cues are synthesis instructions and are never spoken. "
    "Do not start a sentence with a bare mood word. Speak the user's language if they switch."
)


@dataclass(frozen=True)
class SuiteDefaults:
    tts_model: str = "s2.1-pro"
    asr_model: str = "transcribe-1"
    asr_language: str = ""
    latency: str = "normal"
    chunk_length: int = 200
    min_chunk_length: int = 50
    audio_format: str = "mp3"
    mp3_bitrate: int = 128
    speed: float = 1.05
    temperature: float = 0.70
    top_p: float = 0.7
    repetition_penalty: float = 1.2
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
    trace_id: str | None = None

    def log_line(self) -> str:
        def fmt(name: str, value: float | None) -> str:
            if value is None:
                return f"{name}=-1"
            return f"{name}={value:.0f}ms"

        parts = [
            fmt("srt", self.srt),
            fmt("llm_ttft", self.llm_ttft),
            fmt("llm_ttfs", self.llm_ttfs),
            fmt("ttfa", self.ttfa),
            fmt("voice_to_voice", self.voice_to_voice),
        ]
        if self.trace_id:
            parts.append(f"trace={self.trace_id}")
        return "[timing " + " ".join(parts) + "]"
