"""Voice CLI settings from the process environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    MIN_CHUNK_HI,
    MIN_CHUNK_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_HI,
    UNIT_LO,
    SuiteDefaults,
    chunk_length_hi,
    clamp_num,
    env_base,
    env_float,
    env_int,
    env_text,
    env_token,
    known_latency,
    known_tts_model,
    strip_base,
)
from fish_audio_suite_voice.playback import DEFAULT_PLAYBACK, playback_key

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class VoiceCliConfig:
    """Duplex settings read from the process environment. No YAML.

    Notes
    -----
    ``fish_api_key`` and ``fish_voice_id`` are empty unless the env sets them.
    ``llm_backend`` is only the label printed on the ready line. The transport
    is chosen from ``llm_base``. Process env wins over ``--env-file``.
    """

    fish_api_key: str
    fish_base: str
    fish_voice_id: str
    fish_asr_language: str
    tts_model: str
    latency: str
    speed: float
    temperature: float
    top_p: float
    repetition_penalty: float
    chunk_length: int
    min_chunk_length: int
    volume: float
    sample_rate: int
    playback: str
    system_prompt: str
    device: str | None
    llm_backend: str
    llm_base: str
    llm_key: str
    llm_model: str


def _existing(default: str, *names: str) -> str:
    """First key that exists wins, including a blank value. Surrounding space is removed."""
    for name in names:
        if name in os.environ:
            return os.environ[name].strip()
    return default


def _base_url(default: str, *names: str) -> str:
    """First non-blank URL. A blank value is not a host, so the next key is used."""
    for name in names:
        if name not in os.environ:
            continue
        text = strip_base(os.environ[name])
        if text:
            return text
    return strip_base(default)


def _model_name(*names: str) -> str:
    for name in names:
        raw = env_token(name, "")
        if raw:
            return raw
    return ""


def cfg() -> VoiceCliConfig:
    """Read ``VoiceCliConfig`` from the current process environment.

    Returns
    -------
    VoiceCliConfig
        Clamped numeric knobs. A missing key keeps the ``SuiteDefaults`` value.
        Call this after dotenv loading; an earlier call will not see those keys.

    Notes
    -----
    ``FISH_API_KEY`` uses ``env_text``, so a blank value stays blank instead
    of falling back to a default key. There is no default voice id.
    """
    d = SuiteDefaults()
    fish_base = env_base("FISH_BASE", d.fish_base)
    sample_rate = env_int("FISH_SAMPLE_RATE", d.sample_rate)
    return VoiceCliConfig(
        fish_api_key=env_text("FISH_API_KEY"),
        fish_base=fish_base,
        fish_voice_id=env_text("FISH_VOICE_ID"),
        fish_asr_language=env_text("FISH_ASR_LANGUAGE", d.asr_language),
        tts_model=known_tts_model(env_token("FISH_TTS_MODEL", d.tts_model)),
        latency=known_latency(env_token("FISH_LATENCY", d.latency), d.latency),
        speed=clamp_num(
            env_float("FISH_SPEED", d.speed),
            TTS_SPEED_LO,
            TTS_SPEED_HI,
            d.speed,
            float,
        ),
        temperature=clamp_num(
            env_float("FISH_TEMPERATURE", d.temperature),
            UNIT_LO,
            UNIT_HI,
            d.temperature,
            float,
        ),
        top_p=clamp_num(env_float("FISH_TOP_P", d.top_p), UNIT_LO, UNIT_HI, d.top_p, float),
        repetition_penalty=env_float("FISH_REPETITION_PENALTY", d.repetition_penalty),
        chunk_length=clamp_num(
            env_int("FISH_CHUNK_LENGTH", d.chunk_length),
            CHUNK_LENGTH_LO,
            chunk_length_hi(fish_base),
            d.chunk_length,
            int,
        ),
        min_chunk_length=clamp_num(
            env_int("FISH_MIN_CHUNK_LENGTH", d.min_chunk_length),
            MIN_CHUNK_LO,
            MIN_CHUNK_HI,
            d.min_chunk_length,
            int,
        ),
        volume=env_float("FISH_VOLUME", d.volume),
        sample_rate=sample_rate if sample_rate > 0 else d.sample_rate,
        playback=playback_key(env_token("FISH_PLAYBACK", DEFAULT_PLAYBACK)),
        system_prompt=os.environ.get("FISH_SYSTEM_PROMPT", d.system_prompt),
        device=os.environ.get("FISH_VOICE_DEVICE"),
        llm_backend=env_token("FISH_LLM_BACKEND", "openrouter"),
        llm_base=_base_url(
            OPENROUTER_API_BASE,
            "FISH_LLM_BASE",
            "OPENROUTER_BASE_URL",
        ),
        llm_key=_existing("", "FISH_LLM_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        llm_model=_model_name("FISH_LLM_MODEL", "OPENROUTER_MODEL"),
    )
