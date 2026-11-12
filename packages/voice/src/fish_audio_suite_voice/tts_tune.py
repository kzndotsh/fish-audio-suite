"""The Fish TTS settings a speaker sends with every turn, read once from the environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

from fish_audio_suite_kit import (
    CHUNK_LENGTH_CLOUD_HI,
    CHUNK_LENGTH_LO,
    MIN_CHUNK_LENGTH_HI,
    MIN_CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_INTERVAL_HI,
    UNIT_INTERVAL_LO,
    FishLatency,
    SuiteDefaults,
    clamp_number,
    env_float,
    env_int,
    env_token,
    known_latency,
)
from fish_audio_suite_voice.pcm import wav_rate
from fish_audio_suite_voice.tune import read_flag, read_float

__all__ = [
    "DEFAULT_FADE_MS",
    "DEFAULT_LOUDNESS_TARGET_DB",
    "TtsTune",
]

_STOCK: Final = SuiteDefaults()
# Fade, in milliseconds, at each edge between a sound and silence, so a sentence does not
# click when it starts or stops. 0 turns it off.
DEFAULT_FADE_MS: Final = 4.0
# The level, in dB below full scale, that a reply is brought to when loudness is normalised.
DEFAULT_LOUDNESS_TARGET_DB: Final = -18.0
# A fade longer than this would blur the start and end of every word.
_FADE_MAX_MS: Final = 50.0
# Quieter than this leaves nothing to hear, and louder leaves no room for a peak.
_LOUDNESS_TARGET_LO_DB: Final = -40.0
_LOUDNESS_TARGET_HI_DB: Final = -6.0


@dataclass(frozen=True, slots=True)
class TtsTune:
    """How Fish speaks: the generation settings and what is done to the audio before the sink.

    Attributes
    ----------
    latency : {"normal", "balanced", "low"}
        Fish's latency mode. ``low`` is sent as ``balanced``, the fastest the SDK accepts.
    speed : float
        Speaking rate, kept in 0.5 to 2.
    sample_rate : int
        PCM rate Fish is asked for. A rate a WAV header cannot hold becomes the default.
    temperature, top_p : float
        Sampling, each kept in 0 to 1.
    repetition_penalty : float
        Penalty for repeated tokens.
    chunk_length : int
        Characters Fish generates at a time, kept in 100 to 300. A self-hosted server's
        HTTP API allows up to 1000, but this websocket goes through the SDK's model, which
        refuses more than 300 for every base, so 300 is the most either base is sent.
    min_chunk_length : int
        The shortest chunk Fish generates.
    volume : float
        Loudness change in dB. Fish is sent at most 20 either way.
    partial_chars : int
        How far a streamed reply is cut at a comma when no sentence ends sooner.
    fade_ms : float
        Fade at each edge between a sound and silence, so a sentence does not click when it
        starts or stops. 0 turns it off. PCM only.
    loudness_db : float or None
        The level, in dB below full scale, each reply is brought to while it plays, or None
        to play it as Fish sends it. PCM only.

    Notes
    -----
    Speed, temperature, top_p, chunk_length and sample_rate are brought into range when the
    value is built, so a speaker built in code is held to the same ranges as the environment.
    """

    latency: FishLatency = _STOCK.latency
    speed: float = _STOCK.speed
    sample_rate: int = _STOCK.sample_rate
    temperature: float = _STOCK.temperature
    top_p: float = _STOCK.top_p
    repetition_penalty: float = _STOCK.repetition_penalty
    chunk_length: int = _STOCK.chunk_length
    min_chunk_length: int = _STOCK.min_chunk_length
    volume: float = _STOCK.volume
    partial_chars: int = _STOCK.tts_partial_chars
    fade_ms: float = DEFAULT_FADE_MS
    loudness_db: float | None = None

    def __post_init__(self) -> None:
        """Bring the ranged settings into the ranges Fish accepts."""
        fit = {
            "speed": clamp_number(self.speed, TTS_SPEED_LO, TTS_SPEED_HI, _STOCK.speed, float),
            "temperature": clamp_number(
                self.temperature, UNIT_INTERVAL_LO, UNIT_INTERVAL_HI, _STOCK.temperature, float
            ),
            "top_p": clamp_number(
                self.top_p, UNIT_INTERVAL_LO, UNIT_INTERVAL_HI, _STOCK.top_p, float
            ),
            "chunk_length": clamp_number(
                self.chunk_length, CHUNK_LENGTH_LO, CHUNK_LENGTH_CLOUD_HI, _STOCK.chunk_length, int
            ),
            "sample_rate": wav_rate(self.sample_rate),
        }
        for name, value in fit.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_*`` TTS keys and ``FISH_VOICE_FADE_MS`` and the loudness keys.

        Returns
        -------
        TtsTune
            Unset keys keep the ``SuiteDefaults`` values. Loudness stays off unless
            ``FISH_VOICE_NORMALIZE_LOUDNESS`` is on.
        """
        return cls(
            latency=known_latency(env_token("FISH_LATENCY", _STOCK.latency), _STOCK.latency),
            speed=env_float("FISH_SPEED", _STOCK.speed),
            temperature=env_float("FISH_TEMPERATURE", _STOCK.temperature),
            top_p=env_float("FISH_TOP_P", _STOCK.top_p),
            repetition_penalty=env_float("FISH_REPETITION_PENALTY", _STOCK.repetition_penalty),
            chunk_length=env_int("FISH_CHUNK_LENGTH", _STOCK.chunk_length),
            min_chunk_length=clamp_number(
                env_int("FISH_MIN_CHUNK_LENGTH", _STOCK.min_chunk_length),
                MIN_CHUNK_LENGTH_LO,
                MIN_CHUNK_LENGTH_HI,
                _STOCK.min_chunk_length,
                int,
            ),
            volume=env_float("FISH_VOLUME", _STOCK.volume),
            sample_rate=env_int("FISH_SAMPLE_RATE", _STOCK.sample_rate),
            fade_ms=read_float("FISH_VOICE_FADE_MS", DEFAULT_FADE_MS, lo=0.0, hi=_FADE_MAX_MS),
            loudness_db=_loudness_db(),
        )


def _loudness_db() -> float | None:
    """Read ``FISH_VOICE_NORMALIZE_LOUDNESS`` and, when it is on, ``FISH_VOICE_LOUDNESS_TARGET_DB``."""
    if not read_flag("FISH_VOICE_NORMALIZE_LOUDNESS", default=False):
        return None
    return read_float(
        "FISH_VOICE_LOUDNESS_TARGET_DB",
        DEFAULT_LOUDNESS_TARGET_DB,
        lo=_LOUDNESS_TARGET_LO_DB,
        hi=_LOUDNESS_TARGET_HI_DB,
    )
