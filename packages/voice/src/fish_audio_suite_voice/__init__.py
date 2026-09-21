"""Importable Fish live TTS toolkit. Duplex CLI is one recipe."""

from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.live import IsolatedFishTts, IsolatedResult
from fish_audio_suite_voice.playback import (
    FileSink,
    MpvSink,
    PlaybackSink,
    SounddeviceSink,
    StdoutSink,
    make_sink,
)

__all__ = [
    "BargeGate",
    "FileSink",
    "IsolatedFishTts",
    "IsolatedResult",
    "MpvSink",
    "PlaybackSink",
    "SounddeviceSink",
    "StdoutSink",
    "make_sink",
]
