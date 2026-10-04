"""State shared by the duplex loop and its listen and reply steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import httpx

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.history import KEEP_SYSTEM
from fish_audio_suite_voice.live import FishSpeaker
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.signals import DuplexSession

__all__ = [
    "EXIT_FATAL",
    "EXIT_OK",
    "DuplexContext",
]

EXIT_OK: Final = 0
EXIT_FATAL: Final = 2


@dataclass(slots=True)
class DuplexContext:
    """Everything one duplex session shares between the listen, reply and history steps."""

    config: VoiceCliConfig
    tts: FishSpeaker
    device: str | int | None
    backend: ChatBackend
    session: DuplexSession
    asr_http: httpx.AsyncClient
    history: list[ChatMessage]
    barge_prefix: bytes = b""
    # What a barge-in cut off, kept until the next line shows the interrupt was
    # real speech. If it was only noise, this is spoken again.
    resume_text: str = ""
    pinned: int = KEEP_SYSTEM
