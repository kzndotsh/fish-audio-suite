"""Importable Fish live TTS toolkit. Duplex CLI is one recipe."""

from typing import Final

from fish_audio_suite_voice._aliases import resolve_alias
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.live import IsolatedFishTts, TtsResult
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.playback import (
    FileSink,
    MpvSink,
    PlaybackKind,
    PlaybackSink,
    PortAudioMissingError,
    SounddeviceSink,
    StdoutSink,
    make_sink,
)
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.tune import AecTune, BargeTune, ListenTune, LlmSettings

__all__ = [
    "AecTune",
    "BargeGate",
    "BargeTune",
    "ChatBackend",
    "DuplexSession",
    "EchoCanceller",
    "FileSink",
    "IsolatedFishTts",
    "ListenTune",
    "LlmSettings",
    "MpvSink",
    "PlaybackKind",
    "PlaybackSink",
    "PortAudioMissingError",
    "SounddeviceSink",
    "StdoutSink",
    "TtsResult",
    "make_sink",
]

# Renamed in 0.2.0. Each old name still resolves through ``__getattr__`` with a
# DeprecationWarning and is kept out of ``__all__``.
_DEPRECATED_ALIASES: Final[dict[str, tuple[str, str]]] = {
    "IsolatedResult": ("TtsResult", "0.2.0"),
    "LlmTune": ("LlmSettings", "0.2.0"),
}


def __getattr__(name: str) -> object:
    """Resolve a renamed class by its old name, with a ``DeprecationWarning``."""
    return resolve_alias(__name__, name, _DEPRECATED_ALIASES, globals())
