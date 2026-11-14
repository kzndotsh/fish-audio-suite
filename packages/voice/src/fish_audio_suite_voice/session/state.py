"""State shared by the duplex loop and its listen and reply steps, grouped by who uses it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.asr.result import BargeIn
from fish_audio_suite_voice.capture.aec import EchoCanceller
from fish_audio_suite_voice.capture.mic import Mic
from fish_audio_suite_voice.config.settings import SinkLatency, VoiceCliConfig, sink_latency
from fish_audio_suite_voice.llm.client import ChatBackend
from fish_audio_suite_voice.llm.speculate import Speculation
from fish_audio_suite_voice.logutil.debug import warn
from fish_audio_suite_voice.session.cancel import never
from fish_audio_suite_voice.session.chat_store import ChatFile, SavedChat
from fish_audio_suite_voice.session.conversation import Conversation, HistoryView
from fish_audio_suite_voice.session.signals import DuplexSession
from fish_audio_suite_voice.tts.played_text import Unheard
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
    resume: Unheard | None = None

    @property
    def barged(self) -> bool:
        """Whether a barge-in left speech for the next listen."""
        return self.clip is not None

    def keep(self, clip: BargeIn, resume: Unheard | None) -> None:
        """Hold the clip that cut a reply off, and what of the reply was not heard.

        Parameters
        ----------
        clip : BargeIn
            The speech that tripped the barge-in, where the mic was after it, and the
            level the gate admitted it at.
        resume : Unheard or None
            The unheard rest of the reply. None when none of it played, or too little is
            left to speak again.
        """
        self.clip = clip
        self.resume = resume

    def take_clip(self) -> BargeIn | None:
        """Return the barge-in clip for the next listen and forget it."""
        clip, self.clip = self.clip, None
        return clip

    def take_resume(self) -> Unheard | None:
        """Return the unheard rest of the cut reply and forget it."""
        resume, self.resume = self.resume, None
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
    """The conversation with the model: the backend, the record, and a reply written ahead.

    Attributes
    ----------
    backend : ChatBackend
        Streams the model's replies.
    history : Conversation
        Every turn so far, as the user heard it.
    view : HistoryView
        Which of those turns a request sends, and how much of their cues.
    speculation : Speculation or None
        A reply being written for a turn the recogniser thinks is probably over. Hearing
        starts it, and the reply uses it if the same words are confirmed.
    file : ChatFile or None
        Where ``save`` writes the record. None when chats are not saved.
    """

    backend: ChatBackend
    history: Conversation
    view: HistoryView
    speculation: Speculation | None = None
    file: ChatFile | None = None
    _save_failed: bool = field(default=False, repr=False, compare=False)

    async def save(self) -> None:
        """Write the record to its file, off the event loop. A failure only warns, once.

        Notes
        -----
        Call it after each turn and after each edit, from the session loop only, one save
        at a time, so there is one writer. A chat with no turns is not written unless its
        file is already there, as it is for a resumed chat whose turns were all deleted.
        """
        file = self.file
        if file is None or (not self.history.turns and not file.path.exists()):
            return
        try:
            # A copy, so the loop can go on changing the record while the file is written.
            await asyncio.to_thread(file.save, self.history.copy())
        except OSError as exc:
            if not self._save_failed:
                self._save_failed = True
                warn(f"fish-voice: cannot save the chat to {file.path}: {exc}")

    def messages(self, tone: tuple[str, ...] = ()) -> list[ChatMessage]:
        """Return the messages the next reply is asked for with.

        Parameters
        ----------
        tone : tuple of str, optional
            How the last line sounded. It goes into this request only, never into the record.

        Returns
        -------
        list of ChatMessage
            ``view`` applied to ``history``.
        """
        return self.view.messages(self.history, tone)


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
        ``auto`` decided for this run by ``config.settings.sink_latency``.
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
    resumed: SavedChat | None = None,
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
    resumed : SavedChat or None, optional
        A saved chat to go on with. Its turns and id are kept, and so is its file, which
        the session keeps writing to. The prompt and seed are the ones in ``config``.

    Returns
    -------
    DuplexContext
        The context with the opening conversation: the system prompt, the pinned seed
        exchange when the prompt asks for one, and the resumed turns. The chat has a file
        when ``config`` saves chats.
    """
    # One input for the whole session, so nothing is lost between listens. The echo
    # canceller lives with it: the mic cleans every frame and the speaker feeds it.
    mic = open_mic(EchoCanceller(config.aec))
    prompt = config.prompt
    if resumed is None:
        history = Conversation.opening(prompt.text, seed=prompt.pin_seed)
    else:
        history = resumed.conversation.reopened(prompt.text, seed=prompt.pin_seed)
    file = None
    if config.save_chats and config.chat_dir is not None:
        file = resumed.file if resumed is not None else ChatFile.new(config.chat_dir, history)
    return DuplexContext(
        config=config,
        device=device,
        turn_input=TurnInput(session=session, mic=mic, muted=muted),
        chat=Chat(
            backend,
            history,
            HistoryView(config.history_turns, config.history_cue_turns),
            file=file,
        ),
        tts=tts,
        asr_http=asr_http,
        sink_latency=sink_latency(config.output_latency, full_screen=full_screen),
    )
