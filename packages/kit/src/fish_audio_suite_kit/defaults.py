"""Shared Fish TTS/ASR knobs. Literals live on SuiteDefaults."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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


_OPUS_AUTO = -1000


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
    opus_bitrate: int = _OPUS_AUTO
    opus_sample_rate: int = 48000
    speed: float = 1.05
    volume: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    max_new_tokens: int = 1024
    early_stop_threshold: float = 1.0
    normalize: bool = True
    normalize_loudness: bool = True
    condition_on_previous_chunks: bool = True
    tts_partial_chars: int = 40
    sample_rate: int = 44100
    fish_base: str = "https://api.fish.audio"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class LatencySnapshot:
    """One cascade turn. Times are milliseconds. Never store utterance text."""

    asr_ms: float | None = None
    llm_ttft: float | None = None
    llm_ttfs: float | None = None
    ttfa: float | None = None
    voice_to_voice: float | None = None
    trace_id: str | None = None

    def log_line(self) -> str:
        parts = [
            _timing_field("asr", self.asr_ms),
            _timing_field("llm_ttft", self.llm_ttft),
            _timing_field("llm_ttfs", self.llm_ttfs),
            _timing_field("ttfa", self.ttfa),
            _timing_field("voice_to_voice", self.voice_to_voice),
        ]
        if self.trace_id:
            parts.append(f"trace={self.trace_id}")
        shown = [part for part in parts if part]
        return "[timing " + " ".join(shown) + "]"


def _timing_field(name: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"{name}={value:.0f}ms"


MS_PER_S = 1000


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * MS_PER_S


def strip_base(url: str) -> str:
    """Drop surrounding space and trailing slashes so a joined path is not `//`."""
    return url.strip().rstrip("/")


def env_base(name: str, default: str) -> str:
    """Process env URL. A missing key keeps the default. Space and trailing slashes are removed."""
    return strip_base(os.environ.get(name, default))


def _env_word(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower()


_OFF_WORDS = frozenset({"0", "false", "no", "off"})
_ON_WORDS = frozenset({"1", "true", "yes", "on"})


def env_off(name: str) -> bool:
    """True only for 0/false/no/off. Blank and any other value are not off."""
    return _env_word(name) in _OFF_WORDS


def env_bool(name: str, default: bool = False) -> bool:
    """Process env flag. Blank keeps the default. Only 1/true/yes/on are true."""
    word = _env_word(name)
    if not word:
        return default
    return word in _ON_WORDS


def env_token(name: str, default: str) -> str:
    """Process env token. Missing or blank keeps the default. The value is stripped."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    return text or default


def env_text(name: str, default: str = "") -> str:
    """Process env text. A missing key keeps the default. The value is stripped."""
    return os.environ.get(name, default).strip()


def number_or[T: int | float](value: Any, default: T, parse: Callable[[Any], T]) -> T:
    """Parse a number. Junk keeps the default."""
    try:
        parsed = parse(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return default
    return parsed


def _env_num[T: int | float](name: str, default: T, parse: Callable[[str], T]) -> T:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return number_or(raw, default, parse)


def env_int(name: str, default: int) -> int:
    """Process env int. Blank or non-numeric values keep the default."""
    return _env_num(name, default, int)


def env_float(name: str, default: float) -> float:
    """Process env float. Blank or non-numeric values keep the default."""
    return _env_num(name, default, float)


CHUNK_LENGTH_LO = 100
CLOUD_CHUNK_HI = 300
SELF_HOST_CHUNK_HI = 1000
MIN_CHUNK_LO = 0
MIN_CHUNK_HI = 100
TTS_SPEED_LO = 0.5
TTS_SPEED_HI = 2.0
UNIT_LO = 0.0
UNIT_HI = 1.0


def chunk_length_hi(fish_base: str) -> int:
    """Cloud OpenAPI max is 300; self-hosted fish-speech allows 1000."""
    if "api.fish.audio" in fish_base.lower():
        return CLOUD_CHUNK_HI
    return SELF_HOST_CHUNK_HI


def clamp_num[T: int | float](
    value: Any,
    lo: T,
    hi: T,
    default: T,
    parse: Callable[[Any], T],
) -> T:
    """Parse a Fish numeric knob and keep it inside the documented range."""
    try:
        n = parse(value)
    except (TypeError, ValueError, OverflowError):
        n = default
    if isinstance(n, float) and not math.isfinite(n):
        return default
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


FISH_TTS_MODEL_IDS = (
    "s2.1-pro",
    "s2.1-pro-free",
    "s2-pro",
    "s1",
    "drama-3-preview",
)
FISH_LATENCIES = frozenset({"low", "balanced", "normal"})


def known_tts_model(name: str) -> str:
    """Catalog ids are lowercase. Any other id is returned stripped."""
    text = name.strip()
    key = text.lower()
    if key in FISH_TTS_MODEL_IDS:
        return key
    return text


def known_latency(name: str, default: str) -> str:
    key = name.strip().lower()
    if key in FISH_LATENCIES:
        return key
    return default


_MP3_BITRATES: dict[int, Literal[64, 128, 192]] = {64: 64, 192: 192}


def known_mp3_bitrate(rate: int) -> Literal[64, 128, 192]:
    return _MP3_BITRATES.get(rate, 128)


_OPUS_BITRATES = frozenset({_OPUS_AUTO, 24000, 32000, 48000, 64000})


def known_opus_bitrate(rate: int) -> int:
    if rate in _OPUS_BITRATES:
        return rate
    return _OPUS_AUTO
