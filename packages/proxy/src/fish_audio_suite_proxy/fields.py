"""Shared OpenAI audio knobs: models, format, trace, and runtime defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    FISH_TTS_MODEL_IDS,
    MIN_CHUNK_HI,
    MIN_CHUNK_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    SuiteDefaults,
    chunk_length_hi,
    clamp_num,
    ensure_trace_headers,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_text,
    env_token,
    extract_quoted_speech,
    known_latency,
    known_mp3_bitrate,
    known_tts_model,
    normalize_cues,
    number_or,
    scrub_tts,
)

_MEDIA = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "wav": "audio/wav",
}

_PCM16_RATE = 24_000
_FORMAT_ALIAS = {fmt: fmt for fmt in _MEDIA} | {"aac": "mp3", "flac": "mp3"}

_TTS_MODEL_ALIASES = {
    "tts-1": "s2.1-pro",
    "tts-1-hd": "s2.1-pro",
    "gpt-4o-mini-tts": "s2.1-pro",
    "playai-tts": "s2.1-pro",
}

_ASR_MODELS = [
    "transcribe-1",
    "transcribe-1-pro",
    "whisper-1",
]

_MODELS = [*FISH_TTS_MODEL_IDS, *_ASR_MODELS]

SILENT_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 64


def runtime_defaults() -> SuiteDefaults:
    stock = SuiteDefaults()
    fish_base = env_base("FISH_BASE", stock.fish_base)
    return SuiteDefaults(
        tts_model=known_tts_model(env_token("FISH_MODEL", stock.tts_model)),
        asr_model=resolve_asr_model(None, env_token("FISH_ASR_MODEL", stock.asr_model)),
        asr_language=env_text("FISH_ASR_LANGUAGE", stock.asr_language),
        latency=known_latency(env_token("FISH_LATENCY", stock.latency), stock.latency),
        chunk_length=clamp_num(
            env_int("FISH_CHUNK_LENGTH", stock.chunk_length),
            CHUNK_LENGTH_LO,
            chunk_length_hi(fish_base),
            stock.chunk_length,
            int,
        ),
        min_chunk_length=clamp_num(
            env_int("FISH_MIN_CHUNK_LENGTH", stock.min_chunk_length),
            MIN_CHUNK_LO,
            MIN_CHUNK_HI,
            stock.min_chunk_length,
            int,
        ),
        audio_format=pick_format(
            {"format": env_token("FISH_FORMAT", stock.audio_format)},
            stock.audio_format,
        ),
        mp3_bitrate=known_mp3_bitrate(env_int("FISH_MP3_BITRATE", stock.mp3_bitrate)),
        speed=clamp_num(
            env_float("FISH_SPEED_SCALE", stock.speed),
            TTS_SPEED_LO,
            TTS_SPEED_HI,
            stock.speed,
            float,
        ),
        fish_base=fish_base,
    )


def quality_guard_env() -> bool:
    return env_bool("FISH_QUALITY_GUARD")


def strip_speakers_env() -> bool:
    return env_bool("FISH_ASR_STRIP_SPEAKERS")


def dialogue_only_env() -> bool:
    return env_bool("FISH_TTS_DIALOGUE_ONLY")


def pick_reference_id(body: dict[str, Any]) -> str | list[str] | None:
    for key in ("reference_id", "voice"):
        value = body.get(key)
        if isinstance(value, list):
            ids = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            return ids or None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def present_value(body: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in body:
            return body[key]
    return None


def explicit_bool(body: dict[str, Any], *keys: str, default: bool) -> bool:
    """A key that is present wins, including false. Missing keys keep the default."""
    flag = present_value(body, *keys)
    if flag is None:
        return default
    return bool(flag)


def first_choice(body: dict[str, Any], *keys: str, default: str) -> str:
    chosen: Any = default
    for key in keys:
        value = body.get(key)
        if value:
            chosen = value
            break
    return str(chosen).lower().strip()


def pick_format(body: dict[str, Any], default: str) -> str:
    raw = first_choice(body, "format", "response_format", "fish_format", default=default)
    mapped = _FORMAT_ALIAS.get(raw)
    if mapped is not None:
        return mapped
    return default if default in _MEDIA else "mp3"


def pcm_sample_rate(fmt: str, body: dict[str, Any], default: int) -> int:
    fallback = _PCM16_RATE if fmt == "pcm16" else default
    raw = body.get("sample_rate")
    if raw is None:
        return fallback
    rate = number_or(raw, fallback, int)
    return rate if rate > 0 else fallback


def media_type(fmt: str) -> str:
    return _MEDIA.get(fmt, "audio/mpeg")


def _native_model_id(raw: str) -> str:
    name = raw.strip()
    if name.lower().startswith("fish-audio/"):
        return name.split("/", 1)[1].strip()
    return name


def _model_name(model: object, default: str) -> str:
    if isinstance(model, str) and model.strip():
        return model
    return default


def resolve_tts_model(model: object, default: str) -> str:
    raw = _native_model_id(_model_name(model, default))
    key = raw.lower()
    aliased = _TTS_MODEL_ALIASES.get(key)
    if aliased is not None:
        return aliased
    return known_tts_model(raw)


_ASR_NATIVE = frozenset({"transcribe-1", "transcribe-1-pro"})


def resolve_asr_model(model: object, default: str) -> str:
    chosen = _native_model_id(_model_name(model, default)).lower()
    if chosen in _ASR_NATIVE:
        return chosen
    fallback = _native_model_id(default).lower()
    if fallback in _ASR_NATIVE:
        return fallback
    return default


def catalog_ids() -> list[str]:
    prefixed = [f"fish-audio/{name}" for name in (*FISH_TTS_MODEL_IDS, *_ASR_NATIVE)]
    return [*_MODELS, *prefixed]


def prepare_tts_text(raw_input: str, *, dialogue_only: bool) -> str:
    cleaned = scrub_tts(raw_input)
    if dialogue_only:
        cleaned = extract_quoted_speech(cleaned)
    return normalize_cues(cleaned)


def upstream_trace_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return ensure_trace_headers(headers)


def traced_model_headers(model: str, incoming: Mapping[str, str]) -> dict[str, str]:
    return {"model": model, **upstream_trace_headers(incoming)}
