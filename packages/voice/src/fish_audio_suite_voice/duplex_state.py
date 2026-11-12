"""State shared by the duplex loop and its listen and reply steps, grouped by who uses it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from fish_audio_suite_voice.cancel import never
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.history import History
from fish_audio_suite_voice.llm import ChatBackend
from fish_audio_suite_voice.mic import Mic
from fish_audio_suite_voice.recognised import BargeIn
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.speculate import Speculation

__all__ = [
    "CarryOver",
    "Chat",
    "DuplexContext",
    "TurnInput",
]


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


@dataclass(frozen=True, slots=True)
class TurnInput:
    """What a turn source works with: the mic side of the session.

    Attributes
    ----------
    session : DuplexSession
        Quit, and the handles of the reply in flight.
    mic : Mic
        The session's one input stream.
    carry : CarryOver
        What the last barge-in left for the next listen.
    muted : Callable
        Says whether the user has muted the mic. A source with a mute switch supplies it.
    """

    session: DuplexSession
    mic: Mic
    carry: CarryOver = field(default_factory=CarryOver)
    muted: Callable[[], bool] = never


@dataclass(slots=True)
class Chat:
    """The conversation with the model: the backend, the history, and a reply written ahead.

    Attributes
    ----------
    backend : ChatBackend
        Streams the model's replies.
    history : History
        What was said so far.
    speculation : Speculation or None
        A reply being written for a turn the recogniser thinks is probably over. Hearing
        starts it, and the reply uses it if the same words are confirmed.
    """

    backend: ChatBackend
    history: History
    speculation: Speculation | None = None


@dataclass(frozen=True, slots=True)
class DuplexContext:
    """One duplex session, as hearing and the reply see it.

    Attributes
    ----------
    config : VoiceCliConfig
        The session's settings.
    device : str or int or None
        The PortAudio device for the mic and the speaker.
    turn_input : TurnInput
        The mic side: quit, the mic, the carried clip and the mute switch.
    chat : Chat
        The model, the history and a reply written ahead.
    tts : FishSpeaker
        Speaks each reply. Only the reply uses it.
    asr_http : httpx.AsyncClient
        The client Fish ASR requests share. Only hearing uses it.
    """

    config: VoiceCliConfig
    device: str | int | None
    turn_input: TurnInput
    chat: Chat
    tts: FishSpeaker
    asr_http: httpx.AsyncClient
