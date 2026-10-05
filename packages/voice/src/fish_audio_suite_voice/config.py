"""Voice CLI settings from the process environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    MIN_CHUNK_LENGTH_HI,
    MIN_CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_INTERVAL_HI,
    UNIT_INTERVAL_LO,
    FishLatency,
    SuiteDefaults,
    chunk_length_hi,
    clamp_number,
    env_base,
    env_float,
    env_int,
    env_text,
    env_token,
    is_insecure_fish_base,
    known_latency,
    normalize_tts_model,
)
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.llm_tune import DEFAULT_HISTORY_TURNS, OPENROUTER_API_BASE, LlmTune
from fish_audio_suite_voice.playback import DEFAULT_PLAYBACK, playback_key
from fish_audio_suite_voice.tune import (
    DEFAULT_FADE_MS,
    AecTune,
    BargeTune,
    ListenTune,
    SttTune,
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
    "load_config",
    "system_prompt_from_file",
    "warn_if_insecure_base",
]


@dataclass(frozen=True, slots=True)
class VoiceCliConfig:
    """Duplex settings read from the process environment. No YAML.

    Notes
    -----
    ``fish_api_key`` and ``fish_voice_id`` are empty unless the env sets them.
    The five tune objects are read once here and passed down, so nothing below
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
    stt : SttTune
        Which speech recognition hears the user.
    history_turns : int
        User and assistant pairs kept in the chat history.
    repeat_window_s : float
        A transcript equal to the previous line is dropped only when it ends
        within this many seconds of the mic opening. 0 never drops a repeat.
    mood_lead : bool
        Rewrite a sentence-leading mood word into a ``[cue]``.
    stream_tts : bool
        Speak the reply while the model is still writing it. Off by default.
    fade_ms : float
        Fade-in at each sound that starts out of silence, so a sentence does not
        click when it begins. 0 turns it off.
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
    asr_model: str = "transcribe-1-pro"
    pin_seed: bool = True
    llm: LlmTune = field(default_factory=LlmTune)
    listen: ListenTune = field(default_factory=ListenTune)
    barge: BargeTune = field(default_factory=BargeTune)
    aec: AecTune = field(default_factory=AecTune)
    stt: SttTune = field(default_factory=SttTune)
    history_turns: int = DEFAULT_HISTORY_TURNS
    repeat_window_s: float = DEFAULT_REPEAT_WINDOW_S
    mood_lead: bool = False
    drop_narration: bool = False
    stream_tts: bool = False
    fade_ms: float = DEFAULT_FADE_MS


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
        The settings built by ``load_config``.

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
    if c.llm.api_key and is_insecure_fish_base(c.llm.base):
        warn(
            f"[llm] FISH_LLM_BASE host {_base_host(c.llm.base)} is plain http, so the LLM key is sent "
            "unencrypted. Use https unless this host is on a network you trust."
        )
        warned = True
    return warned


def load_config() -> VoiceCliConfig:
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
    prompt = _system_prompt(d.system_prompt)
    return VoiceCliConfig(
        fish_api_key=env_text("FISH_API_KEY"),
        fish_base=fish_base,
        fish_voice_id=env_text("FISH_VOICE_ID"),
        fish_asr_language=env_text("FISH_ASR_LANGUAGE", d.asr_language),
        tts_model=normalize_tts_model(env_token("FISH_TTS_MODEL", d.tts_model)),
        latency=known_latency(env_token("FISH_LATENCY", d.latency), d.latency),
        speed=clamp_number(
            env_float("FISH_SPEED", d.speed),
            TTS_SPEED_LO,
            TTS_SPEED_HI,
            d.speed,
            float,
        ),
        temperature=clamp_number(
            env_float("FISH_TEMPERATURE", d.temperature),
            UNIT_INTERVAL_LO,
            UNIT_INTERVAL_HI,
            d.temperature,
            float,
        ),
        top_p=clamp_number(
            env_float("FISH_TOP_P", d.top_p), UNIT_INTERVAL_LO, UNIT_INTERVAL_HI, d.top_p, float
        ),
        repetition_penalty=env_float("FISH_REPETITION_PENALTY", d.repetition_penalty),
        chunk_length=clamp_number(
            env_int("FISH_CHUNK_LENGTH", d.chunk_length),
            CHUNK_LENGTH_LO,
            chunk_length_hi(fish_base),
            d.chunk_length,
            int,
        ),
        min_chunk_length=clamp_number(
            env_int("FISH_MIN_CHUNK_LENGTH", d.min_chunk_length),
            MIN_CHUNK_LENGTH_LO,
            MIN_CHUNK_LENGTH_HI,
            d.min_chunk_length,
            int,
        ),
        volume=env_float("FISH_VOLUME", d.volume),
        sample_rate=sample_rate if sample_rate > 0 else d.sample_rate,
        playback=playback_key(
            env_token(
                "FISH_VOICE_PLAYBACK",
                DEFAULT_PLAYBACK,
            )
        ),
        system_prompt=prompt.text,
        pin_seed=prompt.pin_seed,
        device=os.environ.get("FISH_VOICE_DEVICE"),
        asr_model=_asr_model(d.asr_model),
        llm=LlmTune.from_env(),
        listen=ListenTune.from_env(),
        barge=BargeTune.from_env(),
        aec=AecTune.from_env(),
        stt=SttTune.from_env(),
        history_turns=read_int(
            "FISH_VOICE_HISTORY_TURNS",
            DEFAULT_HISTORY_TURNS,
            lo=1,
        ),
        repeat_window_s=read_float(
            "FISH_VOICE_REPEAT_WINDOW",
            DEFAULT_REPEAT_WINDOW_S,
            lo=0.0,
        ),
        fade_ms=read_float("FISH_VOICE_FADE_MS", DEFAULT_FADE_MS, lo=0.0, hi=_FADE_MAX_MS),
        mood_lead=read_flag("FISH_TTS_MOOD_LEAD", default=False),
        drop_narration=read_flag(
            "FISH_TTS_DROP_NARRATION",
            default=False,
        ),
        stream_tts=read_flag(
            "FISH_VOICE_STREAM_TTS",
            default=False,
        ),
    )


@dataclass(frozen=True, slots=True)
class _Prompt:
    text: str
    # The opening exchange that shows several cues is pinned for the default prompt
    # and for a character file (which keeps the voice rules). A prompt written out in
    # full is left alone.
    pin_seed: bool


# A character card is a few KB. This stops a wrong path (a log, a binary) from
# becoming the system prompt.
_PROMPT_FILE_MAX_BYTES: Final = 64 * 1024
# A fade-in longer than this would blur the start of every word.
_FADE_MAX_MS: Final = 50.0
# Where a character file asks for the default voice rules. ``{{default_prompt}}``, with
# optional spaces inside the braces.
_DEFAULT_SLOT: Final = re.compile(r"\{\{\s*default_prompt\s*\}\}")


def system_prompt_from_file(path: str, default: str) -> str | None:
    """Read a character file and combine it with the voice rules.

    Parameters
    ----------
    path : str
        File with the character or scene, in UTF-8. ``~`` is expanded and a
        relative path is read from the working directory.
    default : str
        The voice rules (cue tags, short spoken replies).

    Returns
    -------
    str or None
        When the file holds ``{{default_prompt}}``, the file text with every
        such slot replaced by ``default``, so the file decides where the voice
        rules go. Otherwise the file text, a blank line, then ``default``. None,
        after a warning, when the file is missing, too large, not UTF-8 or
        empty, so the caller can fall back to the default prompt.
    """
    file = Path(path).expanduser()
    if file.exists() and not file.is_file():
        # A directory, a FIFO or a device would block or never end when read.
        warn(f"fish-voice: prompt file {file} is not a regular file, ignoring it")
        return None
    try:
        # One byte past the limit is enough to tell a file is too big, and a
        # huge file is never read into memory.
        with file.open("rb") as handle:
            raw = handle.read(_PROMPT_FILE_MAX_BYTES + 1)
        if len(raw) > _PROMPT_FILE_MAX_BYTES:
            warn(
                f"fish-voice: prompt file {file} is over {_PROMPT_FILE_MAX_BYTES // 1024} KiB, ignoring it"
            )
            return None
        card = raw.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        warn(f"fish-voice: cannot read prompt file {file}: {exc}")
        return None
    if not card:
        warn(f"fish-voice: prompt file {file} is empty, ignoring it")
        return None
    if _DEFAULT_SLOT.search(card):
        # A function, not a string: a backslash in ``default`` is not an escape.
        return _DEFAULT_SLOT.sub(lambda _: default, card)
    return f"{card}\n\n{default}"


def _system_prompt(default: str) -> _Prompt:
    file = os.environ.get("FISH_VOICE_SYSTEM_PROMPT_FILE", "").strip()
    inline = os.environ.get("FISH_VOICE_SYSTEM_PROMPT")
    if file:
        if inline is not None:
            warn(
                "fish-voice: FISH_VOICE_SYSTEM_PROMPT_FILE is set, so FISH_VOICE_SYSTEM_PROMPT is ignored"
            )
        composed = system_prompt_from_file(file, default)
        if composed is not None:
            return _Prompt(composed, pin_seed=True)
        return _Prompt(default, pin_seed=True)
    # A prompt set but blank means "no system prompt", so unlike the other keys
    # a blank value still counts here.
    if inline is None:
        return _Prompt(default, pin_seed=True)
    return _Prompt(inline, pin_seed=inline == default)
