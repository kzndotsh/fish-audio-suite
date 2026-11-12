"""Every streaming recogniser, by its ``FISH_VOICE_STT`` name.

Adding one is an adapter module that declares its ``Recogniser``, an entry here, its package
extra, and its name in ``tune.STREAMING_STT_PROVIDERS`` (a test keeps the two in step).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fish_audio_suite_voice.deepgram import DEEPGRAM
from fish_audio_suite_voice.fish_realtime import FISH_REALTIME
from fish_audio_suite_voice.turn_stream import Recogniser

__all__ = [
    "RECOGNISERS",
    "recogniser",
]

RECOGNISERS: Final[Mapping[str, Recogniser]] = MappingProxyType(
    {spec.name: spec for spec in (DEEPGRAM, FISH_REALTIME)}
)


def recogniser(provider: str) -> Recogniser | None:
    """Look up the streaming recogniser a ``FISH_VOICE_STT`` value names.

    Parameters
    ----------
    provider : str
        ``SttTune.provider``.

    Returns
    -------
    Recogniser or None
        Its spec, or None for ``fish``, which hears a clip after the user stops.
    """
    return RECOGNISERS.get(provider)
