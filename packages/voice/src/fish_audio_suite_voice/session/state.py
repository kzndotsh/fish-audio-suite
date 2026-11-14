"""State shared by the duplex loop and its listen and reply steps, grouped by who uses it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from fish_audio_suite_voice.asr.result import BargeIn
from fish_audio_suite_voice.capture.aec import EchoCanceller
from fish_audio_suite_voice.capture.mic import Mic
from fish_audio_suite_voice.config.settings import SinkLatency, VoiceCliConfig, sink_latency
from fish_audio_suite_voice.llm.client import ChatBackend
from fish_audio_suite_voice.llm.speculate import Speculation
from fish_audio_suite_voice.session.cancel import never
from fish_audio_suite_voice.session.history import History
from fish_audio_suite_voice.session.signals import DuplexSession
from fish_audio_suite_voice.tts.speaker import FishSpeaker

__all__ = [
    "CarryOver",
    "Chat",
    "DuplexContext",
    "TurnInput",
    "session_context",
    "session_speaker",
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
        The PortAudio device for the speaker. The mic is the one ``open_mic`` opened.
    turn_input : TurnInput
        The mic side: quit, the mic, the carried clip and the mute switch.
    chat : Chat
        The model, the history and a reply written ahead.
    tts : FishSpeaker
        Speaks each reply. Only the reply uses it.
    asr_http : httpx.AsyncClient
        The client Fish ASR requests share. Only hearing uses it.
    sink_latency : {"low", "high"}
        The playback buffer each reply's sink asks for, ``config.output_latency`` with
        ``auto`` decided for this run by ``config.sink_latency``.
    """

    config: VoiceCliConfig
    device: str | int | None
    turn_input: TurnInput
    chat: Chat
    tts: FishSpeaker
    asr_http: httpx.AsyncClient
    sink_latency: SinkLatency


def session_speaker(config: VoiceCliConfig) -> FishSpeaker:
    """Build the speaker a session replies with, from its settings.

    Parameters
    ----------
    config : VoiceCliConfig
        The Fish key, voice, model, base URL, speech settings and cue style.

    Returns
    -------
    FishSpeaker
        The speaker ``fish-voice`` uses for its session and its smoke test.
    """
    return FishSpeaker(
        api_key=config.fish_api_key,
        voice_id=config.fish_voice_id,
        model=config.tts_model,
        tune=config.tts,
        cue_style=config.cue_style,
        base_url=config.fish_base,
    )


def session_context(
    config: VoiceCliConfig,
    *,
    session: DuplexSession,
    open_mic: Callable[[EchoCanceller], Mic],
    backend: ChatBackend,
    tts: FishSpeaker,
    asr_http: httpx.AsyncClient,
    device: str | int | None = None,
    muted: Callable[[], bool] = never,
    full_screen: bool = False,
) -> DuplexContext:
    """Assemble one session's context from its settings and what it holds open.

    Parameters
    ----------
    config : VoiceCliConfig
        The session's settings.
    session : DuplexSession
        Quit, and the handles of the reply in flight.
    open_mic : Callable
        Opens the session's one input stream around the echo canceller built here from
        ``config.aec``.
    backend : ChatBackend
        The open chat backend.
    tts : FishSpeaker
        Speaks each reply.
    asr_http : httpx.AsyncClient
        The client Fish ASR requests share.
    device : str or int or None, optional
        The PortAudio device for the speaker. The host default when None. The mic is
        chosen by ``open_mic``.
    muted : Callable, optional
        The mute switch of the session's turn source. Never muted when omitted.
    full_screen : bool, optional
        Whether the session runs in the full-screen app, which decides an ``auto``
        ``output_latency``.

    Returns
    -------
    DuplexContext
        The context with the opening history: the system prompt, and the pinned seed
        exchange when the prompt asks for one.
    """
    # One input for the whole session, so nothing is lost between listens. The echo
    # canceller lives with it: the mic cleans every frame and the speaker feeds it.
    mic = open_mic(EchoCanceller(config.aec))
    return DuplexContext(
        config=config,
        device=device,
        turn_input=TurnInput(session=session, mic=mic, muted=muted),
        chat=Chat(
            backend,
            History.opening(
                config.prompt.text, seed=config.prompt.pin_seed, turns=config.history_turns
            ),
        ),
        tts=tts,
        asr_http=asr_http,
        sink_latency=sink_latency(config.output_latency, full_screen=full_screen),
    )
