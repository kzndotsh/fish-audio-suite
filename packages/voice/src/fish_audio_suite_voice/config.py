"""Voice CLI settings from the process environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlsplit

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    MIN_CHUNK_HI,
    MIN_CHUNK_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_HI,
    UNIT_LO,
    FishLatency,
    SuiteDefaults,
    chunk_length_hi,
    clamp_num,
    env_base,
    env_float,
    env_int,
    env_text,
    env_token,
    is_insecure_fish_base,
    known_latency,
    known_tts_model,
)
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.playback import DEFAULT_PLAYBACK, playback_key
from fish_audio_suite_voice.tune import (
    DEFAULT_HISTORY_TURNS,
    OPENROUTER_API_BASE,
    AecTune,
    BargeTune,
    ListenTune,
    LlmTune,
    read_flag,
    read_float,
    read_int,
)

# A fresh utterance cannot end sooner: the turn needs about 1.2 s of silence plus
# the minimum voiced time. A faster clip is mostly pre-roll or leftover audio.
DEFAULT_REPEAT_WINDOW_S: Final = 1.5

__all__ = [
    "OPENROUTER_API_BASE",
    "VoiceCliConfig",
    "cfg",
    "warn_if_insecure_base",
]


@dataclass(frozen=True, slots=True)
class VoiceCliConfig:
    """Duplex settings read from the process environment. No YAML.

    Notes
    -----
    ``fish_api_key`` and ``fish_voice_id`` are empty unless the env sets them.
    The four tune objects are read once here and passed down, so nothing below
    this layer reads the environment. Process env wins over ``--env-file``.

    Attributes
    ----------
    llm : LlmTune
        Chat backend and request settings.
    listen : ListenTune
        Microphone capture limits.
    barge : BargeTune
        Barge-in gate and the pauses around a reply.
    aec : AecTune
        Echo cancellation.
    history_turns : int
        User and assistant pairs kept in the chat history.
    repeat_window_s : float
        A transcript equal to the previous line is dropped only when it ends
        within this many seconds of the mic opening. 0 never drops a repeat.
    mood_lead : bool
        Rewrite a sentence-leading mood word into a ``[cue]``.
    stream_tts : bool
        Speak the reply while the model is still writing it. Off by default.
    drop_narration : bool
        Drop stage-direction lines such as ``She smiles.`` before TTS.
    """

    fish_api_key: str = field(repr=False)
    fish_base: str
    fish_voice_id: str
    fish_asr_language: str
    tts_model: str
    latency: FishLatency
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
    asr_model: str = "transcribe-1"
    llm: LlmTune = field(default_factory=LlmTune)
    listen: ListenTune = field(default_factory=ListenTune)
    barge: BargeTune = field(default_factory=BargeTune)
    aec: AecTune = field(default_factory=AecTune)
    history_turns: int = DEFAULT_HISTORY_TURNS
    repeat_window_s: float = DEFAULT_REPEAT_WINDOW_S
    mood_lead: bool = False
    drop_narration: bool = False
    stream_tts: bool = False


def _asr_model(default: str) -> str:
    # Voice only runs the two native ids. Anything else would 4xx on every turn.
    chosen = env_token("FISH_ASR_MODEL", default).lower()
    return chosen if chosen in {"transcribe-1", "transcribe-1-pro"} else default


def _base_host(base: str) -> str:
    # Only the hostname is safe to print. A base can carry user:password@ or a
    # query string with a credential.
    try:
        return urlsplit(base).hostname or "unknown"
    except ValueError:
        return "unknown"


def warn_if_insecure_base(c: VoiceCliConfig) -> bool:
    """Warn when an API key would travel over plain http to a non-loopback host.

    Parameters
    ----------
    c : VoiceCliConfig
        The settings built by ``cfg``.

    Returns
    -------
    bool
        True when at least one warning was printed. Self-hosting over http on a
        LAN is legitimate, so this only warns and never stops the run.
    """
    warned = False
    if c.fish_api_key and is_insecure_fish_base(c.fish_base):
        warn(
            f"[fish] FISH_BASE host {_base_host(c.fish_base)} is plain http, so FISH_API_KEY is sent "
            "unencrypted. Use https unless this host is on a network you trust."
        )
        warned = True
    if c.llm.key and is_insecure_fish_base(c.llm.base):
        warn(
            f"[llm] FISH_LLM_BASE host {_base_host(c.llm.base)} is plain http, so the LLM key is sent "
            "unencrypted. Use https unless this host is on a network you trust."
        )
        warned = True
    return warned


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
        asr_model=_asr_model(d.asr_model),
        llm=LlmTune.from_env(),
        listen=ListenTune.from_env(),
        barge=BargeTune.from_env(),
        aec=AecTune.from_env(),
        history_turns=read_int("FISH_HISTORY_TURNS", DEFAULT_HISTORY_TURNS, lo=1),
        repeat_window_s=read_float("FISH_VOICE_REPEAT_WINDOW_S", DEFAULT_REPEAT_WINDOW_S, lo=0.0),
        mood_lead=read_flag("FISH_MOOD_LEAD", default=False),
        drop_narration=read_flag("FISH_DROP_NARRATION", default=False),
        stream_tts=read_flag("FISH_STREAM_TTS", default=False),
    )
