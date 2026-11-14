"""Fish live TTS, playback, mic cleanup and barge-in as a library. The CLI is one recipe."""

from fish_audio_suite_voice.capture.aec import EchoCanceller
from fish_audio_suite_voice.capture.barge_in import BargeGate
from fish_audio_suite_voice.capture.mic import Mic
from fish_audio_suite_voice.capture.portaudio import PortAudioMissingError
from fish_audio_suite_voice.config.dotenv import load_dotenv
from fish_audio_suite_voice.config.playback_kind import PlaybackKind
from fish_audio_suite_voice.config.tune import AecTune, BargeTune, ListenTune
from fish_audio_suite_voice.llm.client import ChatBackend
from fish_audio_suite_voice.llm.tune import LlmTune
from fish_audio_suite_voice.session.signals import DuplexSession, TurnSignals
from fish_audio_suite_voice.tts.playback import (
    FileSink,
    MpvSink,
    PlaybackSink,
    SounddeviceSink,
    StdoutSink,
    make_sink,
)
from fish_audio_suite_voice.tts.speaker import FishSpeaker
from fish_audio_suite_voice.tts.tune import TtsTune
from fish_audio_suite_voice.tts.websocket import TtsResult

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
    "Mic",
    "MpvSink",
    "PlaybackKind",
    "PlaybackSink",
    "PortAudioMissingError",
    "SounddeviceSink",
    "StdoutSink",
    "TtsResult",
    "TtsTune",
    "TurnSignals",
    "load_dotenv",
    "make_sink",
]
