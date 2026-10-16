"""The end-of-turn check on the Fish path: a short silence, then the words decide.

Fish ASR hears a clip only after the speaker stops, so the silence that ends the clip is a
wait in front of every reply. With the check on, that silence is short. While the clip is
transcribed the mic is already open again, and when the words stop mid-thought ("and",
"because", a comma) the turn continues into that second recording instead of being
answered half-finished. The pieces are joined and judged like one line.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

from fish_audio_suite_kit import TurnJudgement, judge_turn
from fish_audio_suite_voice.barge import StopFlag
from fish_audio_suite_voice.tune import ListenTune, TurnTune

__all__ = [
    "OpenTurn",
    "Reopened",
    "continuation_tune",
    "first_clip_tune",
    "wait_frames",
]

# A continuation is by definition right after speech, so it needs less voice than a fresh
# turn to count. A word like "and" is only a few frames.
_CONTINUATION_VOICED_FRAMES = 6
_TONE_MAX = 3


@dataclass(frozen=True, slots=True)
class OpenTurn:
    """The words of one turn so far, across the clips it was recorded in.

    Attributes
    ----------
    text : str
        Every clip's transcript, joined in order.
    tone : tuple of str
        The tone Fish heard across the clips, without repeats.
    clips : int
        How many clips are in ``text``. A turn that has only its first clip has 1.
    """

    text: str = ""
    tone: tuple[str, ...] = ()
    clips: int = 0

    def add(self, text: str, tone: tuple[str, ...]) -> OpenTurn:
        """Return the turn with one more clip's transcript joined on.

        Parameters
        ----------
        text : str
            What the new clip said.
        tone : tuple of str
            The tone Fish heard in it.

        Returns
        -------
        OpenTurn
            A new turn. The first clip's words are never replaced.
        """
        return OpenTurn(
            text=f"{self.text.rstrip()} {text.lstrip()}".strip(),
            tone=tuple(dict.fromkeys((*self.tone, *tone)))[:_TONE_MAX],
            clips=self.clips + 1,
        )

    def room_for_more(self, tune: TurnTune) -> bool:
        """Return whether another clip may follow the one about to be transcribed.

        Parameters
        ----------
        tune : TurnTune
            Holds the cap on continuations.

        Returns
        -------
        bool
            False once the turn has used every continuation it is allowed.
        """
        return self.clips + 1 <= tune.max_extensions

    def judgement(self) -> TurnJudgement:
        """Return whether the words so far look like a finished turn."""
        return judge_turn(self.text)


def first_clip_tune(listen: ListenTune, turn: TurnTune) -> ListenTune:
    """Return the listen limits for the first clip: the short silence while the check is on.

    Parameters
    ----------
    listen : ListenTune
        The ordinary limits.
    turn : TurnTune
        The check's settings.

    Returns
    -------
    ListenTune
        ``listen`` unchanged when the check is off. Otherwise the silence is the shorter of
        the check's and the ordinary one, so the check never makes a turn wait longer.
    """
    if not turn.enabled:
        return listen
    return replace(listen, end_silence_frames=min(turn.silence_frames, listen.end_silence_frames))


def continuation_tune(listen: ListenTune, turn: TurnTune) -> ListenTune:
    """Return the listen limits for a continuation clip.

    Parameters
    ----------
    listen : ListenTune
        The ordinary limits.
    turn : TurnTune
        The check's settings.

    Returns
    -------
    ListenTune
        The longer silence ends the clip, and fewer voiced frames are needed to keep it.
    """
    return replace(
        listen,
        end_silence_frames=turn.extend_frames,
        min_voiced_frames=min(listen.min_voiced_frames, _CONTINUATION_VOICED_FRAMES),
    )


def wait_frames(turn: TurnTune) -> int:
    """Return how many frames to wait for a continuation to begin.

    Parameters
    ----------
    turn : TurnTune
        The check's settings.

    Returns
    -------
    int
        The longer silence less the short one already spent, so a turn that was not over
        waits about as long in all as it would have without the check.
    """
    return max(1, turn.extend_frames - turn.silence_frames)


class _Either:
    def __init__(self, first: StopFlag, second: StopFlag) -> None:
        self._first = first
        self._second = second

    def is_set(self) -> bool:
        return self._first.is_set() or self._second.is_set()


class Reopened:
    """The mic, listening for a continuation while the clip so far is transcribed.

    Parameters
    ----------
    record : Callable
        Records one clip until the stop flag it is given is set, and returns the WAV or None.
        It runs on a worker thread.
    stop : StopFlag
        The session's own stop (quit, mute, a typed line). It ends the recording too.

    Notes
    -----
    Recording while the first clip is in flight is what keeps the words said right after
    the pause. If the turn turns out to be over, ``drop`` closes the mic again before the
    reply opens it.
    """

    def __init__(self, record: Callable[[StopFlag], bytes | None], stop: StopFlag) -> None:
        self._dropped = threading.Event()
        self._task = asyncio.ensure_future(asyncio.to_thread(record, _Either(stop, self._dropped)))

    async def clip(self) -> bytes | None:
        """Wait for the continuation.

        Returns
        -------
        bytes or None
            The WAV of what was said, or None when nothing more came, the clip was too
            short, or the stop flag ended it.
        """
        return await self._task

    async def drop(self) -> None:
        """Close the mic and discard whatever it heard."""
        self._dropped.set()
        with contextlib.suppress(OSError):
            await self._task
