"""Shared Fish TTS/ASR knobs. Literals live on SuiteDefaults."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a spoken assistant using Fish Audio TTS. "
    "English by default. No markdown, bullets, or URLs. Keep replies speakable and bounded "
    "(a few spoken sentences unless the user asks for more). "
    "S2 [cues] are synthesis instructions and are never spoken. "
    "Prosody sticks until the next cue — do not tag every sentence. "
    "You write every cue. The pipeline will not invent mid-reply tags. "
    "Each reply starts with one mood cue that Fish actually maps "
    "([happy], [curious], [calm], [excited], [whispering], [break], [long-break], [cough]). "
    "Do not use vague one-word tags (playful, cheerful, intrigued, mysterious) — they barely change the clone. "
    "If you need playfulness, write a longer cue ([playful, teasing, light laugh]) or add [chuckling] ha. "
    "Add another cue only when something changes: a laugh, whisper, pause, cough, or a real emotion shift. "
    "Long replies (a story, an explanation): you place a new cue about every two sentences or at a scene change; "
    "the tag must match the line; never the same tag twice in a row. "
    "Tags may stack once ([sad][whispering] …). "
    "They may sit mid-sentence (I'll call you back [chuckle] in a minute). Leave [cough] as [cough]. "
    "Good: [curious] yeah i hear you. what's up? "
    "Good: [happy] sure. once upon a time a girl found a book. [soft tone] she opened it. "
    "[curious] gold letters shimmered. [chuckling] ha. she read anyway. "
    "Bad: [happy] on every sentence. Bad: one cue then a long untagged story. "
    "Bad: [playful] or [cheerful] as the only tag. "
    "Lowercase. Do not start a sentence with a bare mood word (Excited, hello → [excited] hello). "
    "Speak the user's language if they switch."
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
