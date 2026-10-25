"""State shared by the duplex loop and its listen and reply steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, NamedTuple

import httpx

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.history import KEEP_SYSTEM
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.mic import Mic
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.speculate import Speculation

__all__ = [
    "EXIT_CRASH",
    "EXIT_FATAL",
    "EXIT_OK",
    "CarriedClip",
    "CarryOver",
    "DuplexContext",
]

EXIT_OK: Final = 0
EXIT_FATAL: Final = 2
# What Python itself exits with after an uncaught exception.
EXIT_CRASH: Final = 1


def _never_muted() -> bool:
    return False


class CarriedClip(NamedTuple):
    """The clip a barge-in kept, and the mic position the next listen reads on from."""

    pcm: bytes
    since: int | None


@dataclass(slots=True)
class CarryOver:
    """What a barge-in leaves for the next turn.

    Notes
    -----
    ``prefix`` is the speech that tripped the barge-in. The next listen starts from
    it instead of waiting out the cooldown, and takes it. ``since`` is the mic
    position just after it, so that listen reads on from there and loses none of
    the words said while the reply stopped. ``resume`` is the part of
    the cut reply that was not heard. It is spoken again if the interrupt turns out
    to be noise, and dropped once real speech is heard.
    """

    prefix: bytes = b""
    resume: str = ""
    since: int | None = None

    @property
    def barged(self) -> bool:
        """Whether a barge-in left speech for the next listen."""
        return bool(self.prefix)

    def keep(self, prefix: bytes, resume: str, since: int | None = None) -> None:
        """Hold the clip that cut a reply off, and what of the reply was not heard.

        Parameters
        ----------
        prefix : bytes
            The PCM that tripped the barge-in.
        resume : str
            The unheard rest of the reply. Empty when none of it played.
        since : int or None, optional
            The mic position just after ``prefix``. None when it is not known.
        """
        self.prefix = prefix
        self.resume = resume
        self.since = since

    def take_prefix(self) -> CarriedClip:
        """Return the barge-in clip for the next listen and forget it."""
        clip = CarriedClip(self.prefix, self.since)
        self.prefix, self.since = b"", None
        return clip

    def hand_back(self, prefix: bytes) -> None:
        """Keep ``prefix`` for the next listen, such as speech a failed stream already caught."""
        self.prefix = prefix
        self.since = None

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
    muted: Callable[[], bool] = _never_muted
    # A reply being written for a turn Flux thinks is probably over. Used if it is confirmed.
    speculation: Speculation | None = None
    pinned: int = KEEP_SYSTEM
