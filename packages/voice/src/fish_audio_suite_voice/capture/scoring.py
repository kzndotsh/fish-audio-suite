"""One score per mic frame, shared by every gate: loudness, the voice detector, the floor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fish_audio_suite_voice.capture.energy_floor import AdaptiveFloor
from fish_audio_suite_voice.capture.pcm import MIC_LEVEL_EVERY_FRAMES, SAMPLE_RATE, pcm_rms
from fish_audio_suite_voice.capture.vad import Vad
from fish_audio_suite_voice.session.events import EVENTS, MicLevel

__all__ = [
    "FrameScore",
    "FrameScorer",
]


@dataclass(frozen=True, slots=True)
class FrameScore:
    """What one cleaned mic frame measured.

    Attributes
    ----------
    rms : float
        The frame's loudness.
    voiced : bool
        Whether the voice detector called the frame speech. False, without asking it,
        for a frame quieter than the ``detect_from`` it was scored with.
    floor : float
        The adaptive floor as it stood before this frame was heard.
    """

    rms: float
    voiced: bool
    floor: float


class FrameScorer:
    """Scores each frame of one read of the mic once, in order, as it arrives.

    Parameters
    ----------
    vad : Vad
        A fresh detector for this read. It hears every frame it is asked about, in
        order, once.
    seed : float
        The floor's seed and lowest value (``AdaptiveFloor``).
    source : {"listen", "barge"} or None, optional
        Names the gate in the ``MicLevel`` events ``meter`` sends. None sends none.

    Notes
    -----
    Each gate keeps its own rule for what counts as speech on top of the score. Listen
    and the cooldown log ask the detector about every frame. Barge-in asks only about
    frames at its level (``detect_from``), so a detector that keeps state between frames
    (both engines do) hears just the loud parts. Asked about every frame, both engines
    interrupted sooner on recorded speech and webrtc interrupted on typing, so the
    gaps stay until that is weighed on real hardware.
    """

    def __init__(
        self, vad: Vad, seed: float, source: Literal["listen", "barge"] | None = None
    ) -> None:
        self._vad = vad
        self._floor = AdaptiveFloor(seed)
        self._now = self._floor.value()
        self._source: Literal["listen", "barge"] | None = source
        self._frames = 0

    @property
    def floor(self) -> float:
        """The adaptive floor the next frame is judged by."""
        return self._now

    def score(self, frame: bytes, *, learn: bool = True, detect_from: float = 0.0) -> FrameScore:
        """Measure one frame and, unless it is too quiet, ask the detector about it.

        Parameters
        ----------
        frame : bytes
            The next 30 ms of cleaned 16 kHz mic audio.
        learn : bool, optional
            Let a frame under the floor teach the floor the room's hiss. False once
            the gate is inside an utterance.
        detect_from : float, optional
            A frame quieter than this is unvoiced and the detector never hears it. The
            default asks about every frame.

        Returns
        -------
        FrameScore
            The frame's loudness, the detector's verdict and the floor it is judged by.
        """
        self._frames += 1
        rms = pcm_rms(frame)
        voiced = rms >= detect_from and bool(self._vad.is_speech(frame, SAMPLE_RATE))
        floor = self._now
        if learn:
            # Only a frame under the floor is hiss. Counting louder ones raises the floor
            # until the same voice can never pass it. A gate that raises its own level
            # (barge-in while the speaker plays) still learns against this floor, or
            # speech between the two would count as hiss.
            self._floor.observe(rms, quiet=rms < floor)
            self._now = self._floor.value()
        return FrameScore(rms, voiced, floor)

    def meter(self, rms: float, need: float) -> None:
        """Send a ``MicLevel`` for the level meter every few frames.

        Parameters
        ----------
        rms : float
            The latest frame's loudness.
        need : float
            The level that gate's frame had to reach.
        """
        if self._source is not None and self._frames % MIC_LEVEL_EVERY_FRAMES == 0:
            EVENTS.emit(MicLevel(rms, need, self._source))
