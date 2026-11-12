"""What a recogniser hears in one turn, and the speech a listen carries in from before it.

Every recogniser, Fish ASR on a clip or a streaming one, returns a ``Recognised``: the words
(``Transcript``), a reason there is no turn (``NoTurn``), or, from a streaming recogniser that
could not be reached once speech had started, the speech so far (``StreamFallback``) for the
batch recogniser to carry on from. ``heard.heard_line`` turns any of them into the line the
loop acts on.

A listen can start from speech it did not record itself: a ``BargeIn`` clip that cut the last
reply off, or a ``StreamFallback``. The two differ in what they mean, so each says where it
came from and the listen branches on that, never on whether there is audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.pcm import take_frames
from fish_audio_suite_voice.portaudio import PortAudioMissingError

__all__ = [
    "BargeIn",
    "CarriedClip",
    "NoTurn",
    "Recognised",
    "StreamFallback",
    "Transcript",
    "mic_failure",
    "said_over_reply",
]


@dataclass(frozen=True, slots=True)
class BargeIn:
    """Speech that cut the last reply off, kept for the next listen.

    Attributes
    ----------
    pcm : bytes
        The frames the barge-in gate kept when it tripped.
    since : int or None
        The mic position just after them, so the next listen reads on from there and loses
        none of the words said while the reply stopped. None when it is not known.
    gate_rms : float
        The loudness floor the gate that admitted them started from (``BargeTune.min_rms``,
        ``FISH_VOICE_BARGE_RMS``). The listen scores the clip against it, not its own floor.
    """

    pcm: bytes
    since: int | None
    gate_rms: float

    def frames(self) -> list[bytes]:
        """Return the clip cut into whole frames, the way every listen reads it.

        Returns
        -------
        list of bytes
            One ``pcm.FRAME_BYTES`` frame each. A partial frame at the end is left out.
        """
        return take_frames(bytearray(), self.pcm)


@dataclass(frozen=True, slots=True)
class StreamFallback:
    """The speech a streaming recogniser could not take, for Fish ASR to carry on from.

    The recogniser could not be reached once speech had started. This is ordinary speech that
    began a turn, not an interruption, unless the turn itself began with a barge-in.

    Attributes
    ----------
    frames : tuple of (bytes, bool)
        Every frame captured so far, the pre-roll included, each with whether the listen gate
        called it speech.
    since : int or None
        The mic position just after the last frame, so the batch listen reads on from there
        and no audio is lost between the two. None when it is not known.
    barge : BargeIn or None
        The barge-in clip the turn started from, whose frames come first. None for a turn
        that started on its own.
    """

    frames: tuple[tuple[bytes, bool], ...]
    since: int | None
    barge: BargeIn | None = None

    @property
    def pcm(self) -> bytes:
        """Every frame, joined."""
        return b"".join(frame for frame, _speech in self.frames)


type CarriedClip = BargeIn | StreamFallback


def said_over_reply(clip: CarriedClip | None) -> bool:
    """Say whether a listen that starts from ``clip`` is speech said over the assistant's reply.

    Parameters
    ----------
    clip : BargeIn or StreamFallback or None
        What the listen started from.

    Returns
    -------
    bool
        True for a barge-in clip, and for a fallback from a turn that began with one.
    """
    match clip:
        case BargeIn():
            return True
        case StreamFallback(barge=barge):
            return barge is not None
        case None:
            return False


@dataclass(frozen=True, slots=True)
class Transcript:
    """The words of one turn, with the timings the reply counts from.

    Attributes
    ----------
    text : str
        What was recognised.
    tone : tuple of str
        How the voice sounded (``laughter``, ``tired``), from Fish ASR with
        ``FISH_VOICE_HEAR_TONE``. Empty otherwise.
    asr_ms : float
        How long recognition took after the speech ended.
    started : float
        ``time.perf_counter()`` when the speech ended, which the reply's timings count from.
    trace_id : str or None
        The trace id shared with the reply's TTS turn.
    end_wait_ms : float or None
        The silence the listener waited out after the last word. None when the recogniser
        ended the turn itself.
    clips : int
        How many clips the end-of-turn check joined into ``text``. 1 for a single clip or a
        streamed turn.
    """

    text: str
    tone: tuple[str, ...] = ()
    asr_ms: float = 0.0
    started: float = 0.0
    trace_id: str | None = None
    end_wait_ms: float | None = None
    clips: int = 1


@dataclass(frozen=True, slots=True)
class NoTurn:
    """Why a listen produced no words to answer.

    Attributes
    ----------
    reason : str
        ``"noise"`` (nothing recognisable was said, or it was too short to be speech),
        ``"stopped"`` (quit, mute or a typed line ended it), ``"again"`` (the recogniser failed
        in a way worth another listen) or ``"fatal"`` (the key is refused, or there is no mic).
    """

    reason: Literal["noise", "stopped", "again", "fatal"]


type Recognised = Transcript | NoTurn | StreamFallback


def mic_failure(error: PortAudioMissingError) -> NoTurn:
    """Report that the mic cannot open, and end the session.

    Parameters
    ----------
    error : PortAudioMissingError
        What the device loader raised, with its install hints.

    Returns
    -------
    NoTurn
        ``fatal``: without PortAudio no listen can work.
    """
    warn(str(error))
    return NoTurn("fatal")
