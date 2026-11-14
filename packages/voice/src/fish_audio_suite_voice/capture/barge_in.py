"""Barge-in gate: stop a reply when the user talks over it.

After the bleed delay the gate reads the session mic and counts frames that are both
over an adaptive floor and voiced by the voice detector. Enough of them in a row stop
the reply, and the frames that did it are kept for the next listen.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Final

from fish_audio_suite_voice.asr.result import BargeIn
from fish_audio_suite_voice.capture.aec import EchoCanceller
from fish_audio_suite_voice.capture.mic import Mic
from fish_audio_suite_voice.capture.pcm import LISTEN_HEARTBEAT_FRAMES
from fish_audio_suite_voice.capture.scoring import FrameScorer
from fish_audio_suite_voice.capture.vad import open_vad
from fish_audio_suite_voice.config.tune import BargeTune, ListenTune
from fish_audio_suite_voice.logutil.debug import debug, heartbeat_due, trace, warn
from fish_audio_suite_voice.session.cancel import any_of, never
from fish_audio_suite_voice.session.events import EVENTS, BargedIn

__all__ = [
    "BARGE_MISS_DECAY_FRAMES",
    "BargeGate",
    "barge_rms_need",
]

# Unvoiced frames in a row that take one hit back, so a short pause between words does
# not reset the count.
BARGE_MISS_DECAY_FRAMES: Final = 3
# Frames (about 600 ms) kept for the next listen when the gate trips, so the words
# that interrupted the reply reach speech recognition.
BARGE_LOOKBACK_FRAMES: Final = 20


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

    With echo cancellation on, the far-end is already subtracted, so raising the floor
    on the cleaned mic would also block real speech. ``playing_gain`` is at least 1
    (``BargeTune`` checks it).
    """
    if far_playing and not aec_on:
        return min_rms * playing_gain
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
    """Interrupt TTS when the mic hears speech over the floor for enough frames.

    Parameters
    ----------
    mic : Mic
        The session's mic. Its echo canceller is shared with the playback sink so
        the far-end reference is the audio that sink played.
    tune : BargeTune or None, optional
        Gate settings. None uses the defaults. For one gate's own values, pass
        ``dataclasses.replace(tune, ...)``.
    listen : ListenTune or None, optional
        Supplies the voice detector and its strictness. Barge-in uses listen's, so it
        never rejects speech that listen would accept. None uses the defaults.
    muted : Callable or None, optional
        Says whether the user has muted the mic. A reply that starts muted is not
        watched, and muting during a reply ends the watch without interrupting it.
        Unmuting does not restart a watch that ended.

    Notes
    -----
    The bleed delay is ``BargeTune.aec_bleed_delay_s`` when echo cancellation is loaded and
    ``BargeTune.bleed_delay_s`` otherwise, so the speaker's own voice is not taken for
    the user. The floor follows the room: it is the quiet-percentile of recent frames,
    never below ``min_rms``. It is multiplied by ``playing_gain`` while audio plays,
    only when AEC3 is not running. A frame counts only when it is over the floor and the
    voice detector calls it speech; the detector is asked only about frames over the
    floor (see ``scorer.FrameScorer``). A trip keeps the last 20 frames and the mic position after them; the next
    listen starts from that clip, reads on from that position, and skips the post-reply
    cooldown. Three missed frames in a row take one hit back, so a short gap does not
    reset the phrase.
    """

    def __init__(
        self,
        *,
        mic: Mic,
        tune: BargeTune | None = None,
        listen: ListenTune | None = None,
        muted: Callable[[], bool] | None = None,
    ) -> None:
        self.mic: Mic = mic
        self.tune: BargeTune = tune or BargeTune()
        self.aec: EchoCanceller = self.mic.aec
        self._heard: deque[bytes] = deque(maxlen=BARGE_LOOKBACK_FRAMES)
        self.captured: bytes = b""
        # The mic position just after the frame that tripped the gate.
        self.resume_at: int | None = None
        self.failure: Exception | None = None
        self._muted: Callable[[], bool] = muted or never
        self._listen: ListenTune = listen or ListenTune()

    def clip(self) -> BargeIn | None:
        """Return the speech that tripped the gate, for the next listen to start from.

        Returns
        -------
        BargeIn or None
            The kept frames, the mic position after them and the loudness floor this gate
            started from (``BargeTune.min_rms``). None when the gate did not trip.
        """
        if not self.captured:
            return None
        return BargeIn(self.captured, self.resume_at, self.tune.min_rms)

    def _bleed_wait(self) -> float:
        # AEC3 takes the speaker out of the mic, so the watch can start sooner with it.
        if self.aec.available():
            return self.tune.aec_bleed_delay_s
        return self.tune.bleed_delay_s

    def _off_for_this_reply(self, step: Callable[[], None]) -> None:
        try:
            step()
        except Exception as e:  # noqa: BLE001 - daemon-thread boundary, kept in self.failure and reported
            # A missing PortAudio, a dead device, or a processor or voice detector that
            # fails to load ends the watch. The reply goes on without barge-in, so the
            # user is told and the failure is kept for the thread that started the gate.
            self.failure = e
            warn(f"[barge-in] off for this reply: {e}")

    def watch(self, cancel: threading.Event) -> None:
        """Listen on the mic and set ``cancel`` after enough speech frames.

        Parameters
        ----------
        cancel : threading.Event
            Set by the caller to stop watching. Set here on a trip.
        """
        self._off_for_this_reply(lambda: self._watch(cancel))

    def _watch(self, cancel: threading.Event) -> None:
        scorer = FrameScorer(
            open_vad(self._listen.vad_engine, self._listen.vad_aggressiveness),
            self.tune.min_rms,
            "barge",
        )
        hit = 0
        miss = 0
        aec_on = self.aec.available()
        debug(
            "barge.arm after {delay}s bleed, {hits} loud frames of rms>={min_rms:.0f} interrupt, "
            "aec={aec}",
            delay=self._bleed_wait(),
            hits=self.tune.hit_frames,
            min_rms=self.tune.min_rms,
            aec="on" if aec_on else "off",
        )
        # The ring still holds audio from before the bleed delay. Keep only
        # what is about to reach the mic.
        self.aec.align()
        start = self.mic.position
        for idle_frames, frame in enumerate(
            self.mic.frames(any_of(cancel, self._muted), since=start),
            start=1,
        ):
            far = self.aec.far_end_playing()
            need = barge_rms_need(
                scorer.floor, far_playing=far, playing_gain=self.tune.playing_gain, aec_on=aec_on
            )
            heard = scorer.score(frame, detect_from=need)
            scorer.meter(heard.rms, need)
            _barge_heartbeat(
                idle_frames=idle_frames,
                rms=heard.rms,
                need=need,
                far=far,
                hit=hit,
                aec_on=aec_on,
            )
            self._heard.append(frame)
            hit, miss, tripped = _barge_step(
                hit=hit,
                miss=miss,
                voiced=heard.voiced,
                rms=heard.rms,
                need=need,
                far=far,
                want=self.tune.hit_frames,
            )
            if tripped:
                self.captured = b"".join(self._heard)
                self.resume_at = start + idle_frames
                self.aec.clear()
                debug("barge.keep frames={} bytes={}", len(self._heard), len(self.captured))
                EVENTS.emit(BargedIn())
                cancel.set()
                return

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

        def _bleed_then_watch() -> None:
            if self._muted():  # no need to wait out the bleed to find out
                debug("barge.muted the mic is muted, so there is no barge-in watch")
                return
            delay = self._bleed_wait()
            trace("barge.bleed sleep_s={}", delay)
            if self._muted():
                debug("barge.muted the mic is muted, so there is no barge-in watch")
                return
            # Wait on cancel, not sleep(), so a reply that ends during the bleed
            # starts no watch and the thread ends at once.
            if cancel.wait(timeout=delay):
                debug("barge.bleed skipped (already cancelled)")
                return
            self.watch(cancel)

        thread = threading.Thread(
            target=self._off_for_this_reply, args=(_bleed_then_watch,), daemon=True
        )
        thread.start()
        return thread
