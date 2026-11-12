"""The Fish TTS models and modes, what each model can do, and the ranges Fish accepts.

Each lookup table is built from its ``literals`` type, so a model or a mode is listed
once. What a model can do beyond the common cues lives in one row of ``_TRAITS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

from fish_audio_suite_kit._charsets import is_header_safe
from fish_audio_suite_kit.literals import FishLatency, TtsModel
from fish_audio_suite_kit.urls import is_cloud_base

__all__ = [
    "CHUNK_LENGTH_CLOUD_HI",
    "CHUNK_LENGTH_LO",
    "FISH_TTS_MODEL_IDS",
    "MIN_CHUNK_LENGTH_HI",
    "MIN_CHUNK_LENGTH_LO",
    "TTS_SPEED_HI",
    "TTS_SPEED_LO",
    "UNIT_INTERVAL_HI",
    "UNIT_INTERVAL_LO",
    "chunk_length_hi",
    "known_latency",
    "known_mp3_bitrate",
    "known_opus_bitrate",
    "normalize_tts_model",
    "takes_directions",
]

CHUNK_LENGTH_LO: Final = 100
CHUNK_LENGTH_CLOUD_HI: Final = 300
_CHUNK_LENGTH_SELF_HOSTED_HI: Final = 1000
MIN_CHUNK_LENGTH_LO: Final = 0
MIN_CHUNK_LENGTH_HI: Final = 100
TTS_SPEED_LO: Final = 0.5
TTS_SPEED_HI: Final = 2.0
UNIT_INTERVAL_LO: Final = 0.0
UNIT_INTERVAL_HI: Final = 1.0
# An Opus bitrate of -1000 asks Fish to pick the rate.
OPUS_AUTO: Final = -1000

FISH_TTS_MODEL_IDS: Final[tuple[TtsModel, ...]] = get_args(TtsModel)
DEFAULT_TTS_MODEL: Final[TtsModel] = "s2.1-pro"


@dataclass(frozen=True, slots=True)
class _ModelTraits:
    # Reads a long [direction] as written instead of a cue from a known list.
    takes_directions: bool = False
    # Tags it reads around the words they change, such as <emphasis>...</emphasis>.
    paired_tags: tuple[str, ...] = ()


# One row per catalog model. A model with no entry of its own takes the plain S-model traits.
_TRAITS: Final[dict[str, _ModelTraits]] = {
    **{model: _ModelTraits() for model in FISH_TTS_MODEL_IDS},
    "drama-3": _ModelTraits(
        takes_directions=True,
        paired_tags=("whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"),
    ),
}
# Every tag a directed model reads in pairs, which the scrubber keeps for it.
DIRECTED_PAIRED_TAGS: Final = tuple(
    dict.fromkeys(tag for traits in _TRAITS.values() for tag in traits.paired_tags)
)

# Maps a lowercase name to the Literal it stands for, so a lookup narrows the type
# without a cast.
_LATENCY_BY_NAME: Final[dict[str, FishLatency]] = {name: name for name in get_args(FishLatency)}


def chunk_length_hi(fish_base: str) -> int:
    """Return the highest ``chunk_length`` the Fish base accepts.

    Parameters
    ----------
    fish_base : str
        Fish origin. The cloud host is matched on the parsed hostname, so a
        path, a port, or a look-alike domain does not count as cloud.

    Returns
    -------
    int
        300 for the cloud API, 1000 for self-hosted fish-speech.
    """
    return CHUNK_LENGTH_CLOUD_HI if is_cloud_base(fish_base) else _CHUNK_LENGTH_SELF_HOSTED_HI


def normalize_tts_model(name: str) -> str:
    """Clean a TTS model id so it is safe to send as a header.

    Parameters
    ----------
    name : str
        A model id from a caller or the environment.

    Returns
    -------
    str
        The lowercase catalog id, the stripped id when it is a single printable
        token that Fish may know (a model newer than the catalog), or the
        default model when ``name`` is blank or could split a header. The result is
        ``str`` and not ``TtsModel`` because other ids pass through.
    """
    text = name.strip()
    # The id is a request header. A newline would split that header, and a
    # non-ASCII character makes the client refuse to send it.
    if not is_header_safe(text, token=True):
        return DEFAULT_TTS_MODEL
    key = text.lower()
    return key if key in _TRAITS else text


def takes_directions(model: str) -> bool:
    """Say whether a TTS model is directed in plain language, with paired tags around words.

    Parameters
    ----------
    model : str
        A model id. Compared after strip and lowercase.

    Returns
    -------
    bool
        True for ``drama-3``, which reads a long ``[direction]`` as written and tags such as
        ``<emphasis>``. False for the S models, whose cues are short and from a known list,
        and for a model the catalog does not list.

    Examples
    --------
    >>> takes_directions(" Drama-3 "), takes_directions("s2.1-pro")
    (True, False)
    """
    traits = _TRAITS.get(model.strip().lower())
    return traits is not None and traits.takes_directions


def known_latency(name: str, default: FishLatency) -> FishLatency:
    """Accept ``low``, ``balanced``, or ``normal``. Anything else keeps ``default``.

    Parameters
    ----------
    name : str
        Caller or env latency. Compared after strip and lowercase.
    default : FishLatency
        Value returned when ``name`` is not one of the three Fish modes.

    Returns
    -------
    FishLatency
        A known latency, or ``default``.

    Examples
    --------
    >>> known_latency(" Balanced ", "normal")
    'balanced'
    >>> known_latency("turbo", "normal")
    'normal'
    """
    return _LATENCY_BY_NAME.get(name.strip().lower(), default)


_MP3_BITRATES: Final[dict[int, Literal[64, 128, 192]]] = {64: 64, 192: 192}


def known_mp3_bitrate(rate: int) -> Literal[64, 128, 192]:
    """Keep 64 and 192. Every other rate snaps to 128.

    Parameters
    ----------
    rate : int
        Requested MP3 bitrate.

    Returns
    -------
    Literal[64, 128, 192]
        A Fish-accepted MP3 bitrate.
    """
    return _MP3_BITRATES.get(rate, 128)


_OPUS_BITRATES: Final = frozenset({OPUS_AUTO, 24000, 32000, 48000, 64000})


def known_opus_bitrate(rate: int) -> int:
    """Keep -1000, 24000, 32000, 48000, or 64000. Anything else snaps to -1000.

    Parameters
    ----------
    rate : int
        Requested Opus bitrate. ``-1000`` means Fish chooses.

    Returns
    -------
    int
        A documented Opus bitrate, or ``-1000``.
    """
    if rate in _OPUS_BITRATES:
        return rate
    return OPUS_AUTO
