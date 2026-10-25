"""Mic barge-in gate: lookback, energy, min-speech, speaker-bleed delay."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Final

from fish_audio_suite_voice.aec import EchoCanceller, pcm_rms
from fish_audio_suite_voice.debug import debug, heartbeat_due, trace, warn
from fish_audio_suite_voice.events import EVENTS, BargedIn, MicLevel
from fish_audio_suite_voice.floor import AdaptiveFloor
from fish_audio_suite_voice.mic import SAMPLE_RATE, Mic, SounddeviceMic
from fish_audio_suite_voice.tune import (
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_PLAYING_GAIN,
    DEFAULT_BARGE_RMS,
    DEFAULT_BLEED_DELAY_S,
    DEFAULT_POST_SPEAK_COOLDOWN_S,
    BargeTune,
)

__all__ = [
    "BARGE_MISS_DECAY_FRAMES",
    "DEFAULT_BARGE_HIT_FRAMES",
    "DEFAULT_BARGE_PLAYING_GAIN",
    "DEFAULT_BARGE_RMS",
    "DEFAULT_BLEED_DELAY_S",
    "DEFAULT_POST_SPEAK_COOLDOWN_S",
    "LISTEN_HEARTBEAT_FRAMES",
    "MIC_LEVEL_EVERY_FRAMES",
    "BargeGate",
    "barge_rms_need",
    "frame_is_speech",
]

LISTEN_HEARTBEAT_FRAMES: Final = 20
# One MicLevel event every this many frames (about 90 ms), so a display can draw a meter.
MIC_LEVEL_EVERY_FRAMES: Final = 3
BARGE_MISS_DECAY_FRAMES: Final = 3
BARGE_LOOKBACK_FRAMES: Final = 20
# Mode 3 dropped frames listen already accepted (RMS above the floor, hit stayed 0).
_BARGE_VAD: Final = 1


def _barge_step(
    *,
    hit: int,
    miss: int,
    voiced: bool,
    rms: float,
    need: float,
    far: bool,
    want: int,
) -> tuple[int, int, bool]:
    if not voiced:
        miss += 1
        if miss >= BARGE_MISS_DECAY_FRAMES and hit:
            trace(
                "barge.decay hit={hit} rms={rms:.0f} need={need:.0f} far={far}",
                hit=hit,
                rms=rms,
                need=need,
                far=far,
            )
            return max(0, hit - 1), 0, False
        return hit, miss, False
    miss = 0
    hit += 1
    debug("barge.hit {hit}/{want} rms={rms:.0f}", hit=hit, want=want, rms=rms)
    if hit >= want:
        debug("barge.interrupt rms={rms:.0f} hits={hit}", rms=rms, hit=hit)
        return hit, miss, True
    return hit, miss, False


def barge_rms_need(
    min_rms: float,
    *,
    far_playing: bool,
    playing_gain: float,
    aec_on: bool = False,
) -> float:
    """Raise the barge floor only when speaker bleed is still in the mic.

    AEC3 already subtracts far-end. Stacking over-gain on cleaned RMS blocks real speech.
    """
    if far_playing and not aec_on:
        # Below 1 the floor drops while the speaker is on, so bleed trips barge-in.
        gain = playing_gain if playing_gain >= 1 else DEFAULT_BARGE_PLAYING_GAIN
        return min_rms * gain
    return min_rms


def _barge_heartbeat(
    *,
    idle_frames: int,
    rms: float,
    need: float,
    far: bool,
    hit: int,
    aec_on: bool,
) -> None:
    if not heartbeat_due(idle_frames, LISTEN_HEARTBEAT_FRAMES):
        return
    trace(
        "barge.mic rms={rms:.0f} need={need:.0f} far={far} hit={hit} aec={aec}",
        rms=rms,
        need=need,
        far=far,
        hit=hit,
        aec=aec_on,
    )


class BargeGate:
    """Interrupt TTS when the mic stays above the floor for enough frames.

    Parameters
    ----------
    mic : Mic or None, optional
        The session's mic. Its echo canceller is shared with the playback sink so
        the far-end reference is the audio that sink played. None opens the host
        default input without AEC.
    tune : BargeTune or None, optional
        Gate settings. None uses the defaults.
    bleed_delay_s, hit_frames, min_rms : optional
        Per-gate overrides of the matching ``tune`` fields.
    muted : Callable or None, optional
        Says whether the user has muted the mic. While it is true the mic stays closed: a
        reply that starts muted is not watched, and muting during a reply ends the watch
        without interrupting it. Unmuting does not restart a watch that ended.

    Notes
    -----
    The bleed delay is short when AEC3 is loaded and ``BargeTune.bleed_delay_s``
    otherwise, so the speaker's own voice is not treated as the user. The floor
    follows the room: it is the quiet-percentile of recent frames, never below
    ``min_rms``. It is multiplied by ``playing_gain`` while audio is playing only when
    AEC is off. A trip keeps the last 20 frames; the next listen starts from
    that clip and skips the post-speak cooldown. Speech is scored with the same
    VAD mode as listen. Hits decay after 3 missed frames so a short gap does
    not reset the phrase.
    """

    def __init__(
        self,
        *,
        mic: Mic | None = None,
        tune: BargeTune | None = None,
        bleed_delay_s: float | None = None,
        hit_frames: int | None = None,
        min_rms: float | None = None,
        muted: Callable[[], bool] | None = None,
    ) -> None:
        base = tune or BargeTune()
        # No mic means no AEC. A default EchoCanceller would load AEC3 and
        # shorten the bleed delay without ever receiving the sink's far-end audio.
        self.mic: Mic = mic if mic is not None else SounddeviceMic()
        self.tune: BargeTune = replace(
            base,
            bleed_delay_s=base.bleed_delay_s
            if bleed_delay_s is None or bleed_delay_s < 0
            else bleed_delay_s,
            hit_frames=base.hit_frames if hit_frames is None or hit_frames < 1 else hit_frames,
            min_rms=base.min_rms if min_rms is None or min_rms <= 0 else min_rms,
        )
        self.aec: EchoCanceller = self.mic.aec
        # An explicit delay is the caller's choice. It must win over the shorter
        # AEC default, or a longer window could never be forced.
        self._bleed_override = (
            bleed_delay_s if bleed_delay_s is not None and bleed_delay_s >= 0 else None
        )
        self.bleed_delay_s: float = self.tune.bleed_delay_s
        self.hit_frames: int = self.tune.hit_frames
        self.min_rms: float = self.tune.min_rms
        self._heard: deque[bytes] = deque(maxlen=BARGE_LOOKBACK_FRAMES)
        self.captured: bytes = b""
        self.failure: Exception | None = None
        self._muted: Callable[[], bool] = muted or _never

    def _bleed_wait(self) -> float:
        if self._bleed_override is not None:
            return self._bleed_override
        return self.aec.effective_bleed_s(self.tune.bleed_delay_s)

    def watch(self, cancel: threading.Event) -> None:
        """Listen on the mic and set ``cancel`` after enough speech frames.

        Parameters
        ----------
        cancel : threading.Event
            Set by the caller to stop watching. Set here on a trip.
        """
        import webrtcvad

        vad = webrtcvad.Vad(_BARGE_VAD)
        hit = 0
        miss = 0
        floor = AdaptiveFloor(self.min_rms)
        aec_on = self.aec.available()
        debug(
            "barge.arm after {delay}s bleed, {hits} loud frames of rms>={min_rms:.0f} interrupt, "
            "aec={aec}",
            delay=self.bleed_delay_s,
            hits=self.hit_frames,
            min_rms=self.min_rms,
            aec="on" if aec_on else "off",
        )
        # The ring still holds audio from before the bleed delay. Keep only
        # what is about to reach the mic.
        self.aec.align()

        try:
            for idle_frames, frame in enumerate(
                self.mic.frames(_CancelOrMuted(cancel, self._muted)),
                start=1,
            ):
                rms = pcm_rms(frame)
                far = self.aec.far_end_playing()
                base_need = floor.value()
                need = barge_rms_need(
                    base_need, far_playing=far, playing_gain=self.tune.playing_gain, aec_on=aec_on
                )
                if idle_frames % MIC_LEVEL_EVERY_FRAMES == 0:
                    EVENTS.emit(MicLevel(rms, need, "barge"))
                _barge_heartbeat(
                    idle_frames=idle_frames,
                    rms=rms,
                    need=need,
                    far=far,
                    hit=hit,
                    aec_on=aec_on,
                )
                self._heard.append(frame)
                voiced = rms >= need and frame_is_speech(vad, frame)
                # A frame under the floor is room hiss. Loud ones must not
                # raise it, or the same voice could never trip the gate. The test
                # uses the floor itself: ``need`` is multiplied while the speaker
                # plays, and speech between the two would otherwise count as hiss.
                floor.observe(rms, quiet=rms < base_need)
                hit, miss, tripped = _barge_step(
                    hit=hit,
                    miss=miss,
                    voiced=voiced,
                    rms=rms,
                    need=need,
                    far=far,
                    want=self.hit_frames,
                )
                if tripped:
                    self.captured = b"".join(self._heard)
                    self.aec.clear()
                    debug("barge.keep frames={} bytes={}", len(self._heard), len(self.captured))
                    EVENTS.emit(BargedIn())
                    cancel.set()
                    return
        except Exception as e:  # noqa: BLE001 - daemon-thread boundary, kept in self.failure and reported
            # A missing PortAudio or a dead device ends this watcher thread. The
            # reply goes on without barge-in, so the user is told and the
            # failure is kept for the thread that started the gate.
            self.failure = e
            warn(f"[barge-in] off for this reply: {e}")

    def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
        """Sleep out speaker bleed, then start ``watch`` unless already cancelled.

        Parameters
        ----------
        cancel : threading.Event
            Ends the sleep and the watch.

        Returns
        -------
        threading.Thread
            The started daemon thread.
        """

        def _run() -> None:
            try:
                if self._muted():  # no need to wait out the bleed to find out
                    debug("barge.muted the mic is muted, so there is no barge-in watch")
                    return
                delay = self._bleed_wait()
                self.bleed_delay_s = delay
                trace("barge.bleed sleep_s={}", delay)
                if self._muted():
                    debug("barge.muted the mic is muted, so there is no barge-in watch")
                    return
                # sleep() ignores cancel. A finished turn would wait out the rest
                # of the bleed, or the next listen would open the mic twice.
                if cancel.wait(timeout=delay):
                    debug("barge.bleed skipped (already cancelled)")
                    return
                self.watch(cancel)
            except Exception as e:  # noqa: BLE001 - daemon-thread boundary, kept in self.failure and reported
                # Setup can fail before watch() has its own handler, for
                # example loading AEC or the VAD. Without this the thread
                # would die silently and the reply would play with no barge-in.
                self.failure = e
                warn(f"[barge-in] off for this reply: {e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


class _CancelOrMuted:
    """A stop flag for the mic stream: set when the watch is cancelled or the user mutes."""

    def __init__(self, cancel: threading.Event, muted: Callable[[], bool]) -> None:
        self._cancel = cancel
        self._muted = muted

    def is_set(self) -> bool:
        return self._cancel.is_set() or self._muted()


def _never() -> bool:
    return False


def frame_is_speech(vad: Any, frame: bytes) -> bool:
    """Return whether WebRTC VAD scores this 16 kHz frame as speech."""
    return bool(vad.is_speech(frame, SAMPLE_RATE))
