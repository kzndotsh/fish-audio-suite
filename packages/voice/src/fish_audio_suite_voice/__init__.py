"""Importable Fish live TTS toolkit. Duplex CLI is one recipe."""

from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.live import FishSpeaker, TtsResult
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
from fish_audio_suite_voice.tune import AecTune, BargeTune, ListenTune, LlmTune

__all__ = [
    "AecTune",
    "BargeGate",
    "BargeTune",
    "ChatBackend",
    "DuplexSession",
    "EchoCanceller",
    "FileSink",
    "FishSpeaker",
    "ListenTune",
    "LlmTune",
    "MpvSink",
    "PlaybackKind",
    "PlaybackSink",
    "PortAudioMissingError",
    "SounddeviceSink",
    "StdoutSink",
    "TtsResult",
    "make_sink",
]
