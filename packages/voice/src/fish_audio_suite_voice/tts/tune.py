"""The Fish TTS settings a speaker sends with every turn, read once from the environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

from fish_audio_suite_kit import (
    TTS_RANGES,
    FishLatency,
    SuiteDefaults,
    clamp_number,
    known_latency,
)
from fish_audio_suite_voice.capture.pcm import wav_rate
from fish_audio_suite_voice.config.env import (
    FloatSetting,
    read_flag,
    read_float,
    read_int,
    read_word,
)

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
# 0 turns the fade off, and a longer one than the most would blur the start and end of
# every word.
_FADE_MIN_MS: Final = 0.0
_FADE_MAX_MS: Final = 50.0
# Quieter than this leaves nothing to hear, and louder leaves no room for a peak.
_LOUDNESS_TARGET_LO_DB: Final = -40.0
_LOUDNESS_TARGET_HI_DB: Final = -6.0
# Read by ``from_env`` and checked by ``TtsTune`` with the same range.
_FADE = FloatSetting("FISH_VOICE_FADE_MS", DEFAULT_FADE_MS, lo=_FADE_MIN_MS, hi=_FADE_MAX_MS)
_LOUDNESS_TARGET = FloatSetting(
    "FISH_VOICE_LOUDNESS_TARGET_DB",
    DEFAULT_LOUDNESS_TARGET_DB,
    lo=_LOUDNESS_TARGET_LO_DB,
    hi=_LOUDNESS_TARGET_HI_DB,
)


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
        The shortest chunk Fish generates, kept in 0 to 100.
    volume : float
        Loudness change in dB. Fish is sent at most 20 either way.
    partial_chars : int
        How far a streamed reply is cut at a comma when no sentence ends sooner.
    fade_ms : float
        Fade at each edge between a sound and silence, so a sentence does not click when it
        starts or stops. 0 turns it off, and 50 is the most. PCM only.
    loudness_db : float or None
        The level, in dB below full scale, each reply is brought to while it plays, from -40
        to -6, or None to play it as Fish sends it. PCM only.

    Raises
    ------
    ValueError
        When ``fade_ms`` or ``loudness_db`` is out of the range ``from_env`` accepts.

    Notes
    -----
    Speed, temperature, top_p, chunk_length, min_chunk_length and sample_rate are brought
    into range when the value is built, so a speaker built in code is held to the same ranges
    as the environment.
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
        """Bring the Fish settings into the ranges Fish accepts, and check the audio ones.

        Raises
        ------
        ValueError
            When ``fade_ms`` or ``loudness_db`` is out of the range ``from_env`` accepts.
        """
        fit = {
            "speed": clamp_number(self.speed, *TTS_RANGES.speed, _STOCK.speed, float),
            "temperature": clamp_number(
                self.temperature, *TTS_RANGES.temperature, _STOCK.temperature, float
            ),
            "top_p": clamp_number(self.top_p, *TTS_RANGES.top_p, _STOCK.top_p, float),
            "chunk_length": clamp_number(
                self.chunk_length, *TTS_RANGES.chunk_length, _STOCK.chunk_length, int
            ),
            "min_chunk_length": clamp_number(
                self.min_chunk_length, *TTS_RANGES.min_chunk_length, _STOCK.min_chunk_length, int
            ),
            "sample_rate": wav_rate(self.sample_rate),
        }
        for name, value in fit.items():
            object.__setattr__(self, name, value)
        _FADE.check("TtsTune.fade_ms", self.fade_ms)
        if self.loudness_db is not None:
            _LOUDNESS_TARGET.check("TtsTune.loudness_db", self.loudness_db)

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_*`` TTS keys and ``FISH_VOICE_FADE_MS`` and the loudness keys.

        Returns
        -------
        TtsTune
            Unset keys keep the ``SuiteDefaults`` values. Loudness stays off unless
            ``FISH_VOICE_NORMALIZE_LOUDNESS`` is on.
        """
        # The Fish knobs keep the policy they have in the proxy: no warning, a number that
        # does not parse is the default and one out of range is clamped.
        return cls(
            latency=known_latency(read_word("FISH_LATENCY", _STOCK.latency), _STOCK.latency),
            speed=_fish_float("FISH_SPEED", _STOCK.speed, TTS_RANGES.speed),
            temperature=_fish_float("FISH_TEMPERATURE", _STOCK.temperature, TTS_RANGES.temperature),
            top_p=_fish_float("FISH_TOP_P", _STOCK.top_p, TTS_RANGES.top_p),
            repetition_penalty=read_float(
                "FISH_REPETITION_PENALTY", _STOCK.repetition_penalty, quiet=True
            ),
            chunk_length=_fish_int(
                "FISH_CHUNK_LENGTH", _STOCK.chunk_length, TTS_RANGES.chunk_length
            ),
            min_chunk_length=_fish_int(
                "FISH_MIN_CHUNK_LENGTH", _STOCK.min_chunk_length, TTS_RANGES.min_chunk_length
            ),
            volume=read_float("FISH_VOLUME", _STOCK.volume, quiet=True),
            sample_rate=read_int("FISH_SAMPLE_RATE", _STOCK.sample_rate, quiet=True),
            fade_ms=_FADE.read(),
            loudness_db=_loudness_db(),
        )


def _fish_float(name: str, default: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return read_float(name, default, lo=lo, hi=hi, quiet=True)


def _fish_int(name: str, default: int, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return read_int(name, default, lo=lo, hi=hi, quiet=True)


def _loudness_db() -> float | None:
    """Read ``FISH_VOICE_NORMALIZE_LOUDNESS`` and, when it is on, ``FISH_VOICE_LOUDNESS_TARGET_DB``."""
    if not read_flag("FISH_VOICE_NORMALIZE_LOUDNESS", default=False):
        return None
    return _LOUDNESS_TARGET.read()
