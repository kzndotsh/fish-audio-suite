"""The names ``FISH_VOICE_PLAYBACK`` and ``--playback`` accept, read with the other settings.

The sinks themselves are in ``tts.playback``. Their names live here so the settings can be read
and checked without importing the audio code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "DEFAULT_PLAYBACK",
    "PlaybackKind",
    "parse_playback",
    "playback_key",
]


class PlaybackKind(StrEnum):
    """Sink names accepted by ``FISH_VOICE_PLAYBACK`` and ``--playback``."""

    SOUNDDEVICE = "sounddevice"
    SPEAKERS = "speakers"
    PCM = "pcm"
    STDOUT = "stdout"
    MPV = "mpv"
    FILE = "file"

    @property
    def is_speaker(self) -> bool:
        """Whether the sink plays through the sound card (``sounddevice``, ``speakers``, ``pcm``)."""
        return self in {PlaybackKind.SOUNDDEVICE, PlaybackKind.SPEAKERS, PlaybackKind.PCM}


DEFAULT_PLAYBACK: Final = PlaybackKind.SOUNDDEVICE.value


def playback_key(name: str) -> str:
    """Normalize a playback name.

    Parameters
    ----------
    name : str
        ``FISH_VOICE_PLAYBACK`` or ``--playback``.

    Returns
    -------
    str
        Stripped, lowercased name. Unknown names are returned as written so
        ``duplex_playback_problem`` can reject them.
    """
    return name.strip().lower()


def parse_playback(name: str) -> PlaybackKind | None:
    """Look up a sink by name.

    Parameters
    ----------
    name : str
        ``FISH_VOICE_PLAYBACK`` or ``--playback``, in any case and with surrounding space.

    Returns
    -------
    PlaybackKind or None
        The sink, or None when the name is not one of them.
    """
    try:
        return PlaybackKind(playback_key(name))
    except ValueError:
        return None
