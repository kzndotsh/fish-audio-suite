"""State shared by the duplex loop and its listen and reply steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import httpx

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.cancel import never
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.history import KEEP_SYSTEM
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.mic import Mic
from fish_audio_suite_voice.recognised import BargeIn
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.speculate import Speculation

__all__ = [
    "EXIT_CRASH",
    "EXIT_FATAL",
    "EXIT_OK",
    "CarryOver",
    "DuplexContext",
]

EXIT_OK: Final = 0
EXIT_FATAL: Final = 2
# What Python itself exits with after an uncaught exception.
EXIT_CRASH: Final = 1


@dataclass(slots=True)
class CarryOver:
    """What a barge-in leaves for the next turn.

    Notes
    -----
    ``clip`` is the speech that tripped the barge-in. The next listen starts from it
    instead of waiting out the cooldown, and takes it. ``resume`` is the part of the cut
    reply that was not heard. It is spoken again if the interrupt turns out to be noise,
    and dropped once real speech is heard.
    """

    clip: BargeIn | None = None
    resume: str = ""

    @property
    def barged(self) -> bool:
        """Whether a barge-in left speech for the next listen."""
        return self.clip is not None

    def keep(self, clip: BargeIn, resume: str) -> None:
        """Hold the clip that cut a reply off, and what of the reply was not heard.

        Parameters
        ----------
        clip : BargeIn
            The speech that tripped the barge-in, where the mic was after it, and the
            level the gate admitted it at.
        resume : str
            The unheard rest of the reply. Empty when none of it played.
        """
        self.clip = clip
        self.resume = resume

    def take_clip(self) -> BargeIn | None:
        """Return the barge-in clip for the next listen and forget it."""
        clip, self.clip = self.clip, None
        return clip

    def take_resume(self) -> str:
        """Return the unheard rest of the cut reply and forget it."""
        resume, self.resume = self.resume, ""
        return resume


@dataclass(slots=True)
class DuplexContext:
    """Everything one duplex session shares between the listen, reply and history steps."""

    config: VoiceCliConfig
    tts: FishSpeaker
    device: str | int | None
    mic: Mic
    backend: ChatBackend
    session: DuplexSession
    asr_http: httpx.AsyncClient
    history: list[ChatMessage]
    carry: CarryOver = field(default_factory=CarryOver)
    # Says whether the user has muted the mic. A source with a mute switch sets it.
    muted: Callable[[], bool] = never
    # A reply being written for a turn Flux thinks is probably over. Used if it is confirmed.
    speculation: Speculation | None = None
    pinned: int = KEEP_SYSTEM
