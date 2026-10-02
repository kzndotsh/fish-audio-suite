"""Mic barge-in gate: lookback, energy, min-speech, speaker-bleed delay."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from fish_audio_suite_kit import MS_PER_S
from fish_audio_suite_voice.aec import (
    AEC_RATE,
    SAMPLE_BYTES,
    EchoCanceller,
    pcm_rms,
)
from fish_audio_suite_voice.debug import debug, heartbeat_due, warn
from fish_audio_suite_voice.floor import AdaptiveFloor
from fish_audio_suite_voice.playback import load_sounddevice, pcm_stream_kwargs
from fish_audio_suite_voice.tune import (
    DEFAULT_BARGE_HIT_FRAMES,
    DEFAULT_BARGE_OVER,
    DEFAULT_BARGE_RMS,
    DEFAULT_BLEED_DELAY_S,
    DEFAULT_POST_SPEAK_COOLDOWN_S,
    BargeTune,
)

__all__ = [
    "DEFAULT_BARGE_HIT_FRAMES",
    "DEFAULT_BARGE_OVER",
    "DEFAULT_BARGE_RMS",
    "DEFAULT_BLEED_DELAY_S",
    "DEFAULT_POST_SPEAK_COOLDOWN_S",
    "BargeGate",
    "barge_rms_need",
    "frame_is_speech",
    "mic_frames",
]

SAMPLE_RATE = AEC_RATE
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // MS_PER_S
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_BYTES
LISTEN_HEARTBEAT_FRAMES = 20
BARGE_MISS_DECAY_FRAMES = 3
BARGE_LOOKBACK_FRAMES = 20
_BARGE_POLL_S = 0.2
# Mode 3 dropped frames listen already accepted (RMS above the floor, hit stayed 0).
_BARGE_VAD = 1


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
            debug(
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
    over: float,
    aec_on: bool = False,
) -> float:
    """Raise the barge floor only when speaker bleed is still in the mic.

    AEC3 already subtracts far-end. Stacking over-gain on cleaned RMS blocks real speech.
    """
    if far_playing and not aec_on:
        # Below 1 the floor drops while the speaker is on, so bleed trips barge-in.
        gain = over if over >= 1 else DEFAULT_BARGE_OVER
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
    debug(
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
    device : str or int or None, optional
        PortAudio input. None uses the host default.
    tune : BargeTune or None, optional
        Gate settings. None uses the defaults.
    aec : EchoCanceller or None, optional
        The session's echo canceller, shared with the playback sink so the
        far-end reference is the audio that sink played. None runs without AEC.
    bleed_delay_s, hit_frames, min_rms : optional
        Per-gate overrides of the matching ``tune`` fields.

    Notes
    -----
    The bleed delay is short when AEC3 is loaded and ``BargeTune.bleed_delay_s``
    otherwise, so the speaker's own voice is not treated as the user. The floor
    follows the room: it is the quiet-percentile of recent frames, never below
    ``min_rms``. It is multiplied by ``over`` while audio is playing only when
    AEC is off. A trip keeps the last 20 frames; the next listen starts from
    that clip and skips the post-speak cooldown. Speech is scored with the same
    VAD mode as listen. Hits decay after 3 missed frames so a short gap does
    not reset the phrase.
    """

    def __init__(
        self,
        *,
        device: str | int | None = None,
        tune: BargeTune | None = None,
        aec: EchoCanceller | None = None,
        bleed_delay_s: float | None = None,
        hit_frames: int | None = None,
        min_rms: float | None = None,
    ) -> None:
        base = tune or BargeTune()
        self.device = device
        self.tune = replace(
            base,
            bleed_delay_s=base.bleed_delay_s
            if bleed_delay_s is None or bleed_delay_s < 0
            else bleed_delay_s,
            hit_frames=base.hit_frames if hit_frames is None or hit_frames < 1 else hit_frames,
            min_rms=base.min_rms if min_rms is None or min_rms <= 0 else min_rms,
        )
        self.aec = aec or EchoCanceller()
        self.bleed_delay_s = self.tune.bleed_delay_s
        self.hit_frames = self.tune.hit_frames
        self.min_rms = self.tune.min_rms
        self._heard: deque[bytes] = deque(maxlen=BARGE_LOOKBACK_FRAMES)
        self.captured = b""

    def _bleed_wait(self) -> float:
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
            "barge.arm delay_s={delay} hit_frames={hits} min_rms={min_rms} over={over} "
            "aec={aec} vad={vad}",
            delay=self.bleed_delay_s,
            hits=self.hit_frames,
            min_rms=self.min_rms,
            over=self.tune.over,
            aec=aec_on,
            vad=_BARGE_VAD,
        )
        # The ring still holds audio from before the bleed delay. Keep only
        # what is about to reach the mic.
        self.aec.align()

        try:
            for idle_frames, frame in enumerate(
                mic_frames(self.device, cancel, timeout=_BARGE_POLL_S, aec=self.aec), start=1
            ):
                rms = pcm_rms(frame)
                far = self.aec.far_end_playing()
                base_need = floor.value()
                need = barge_rms_need(
                    base_need, far_playing=far, over=self.tune.over, aec_on=aec_on
                )
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
                # raise it, or the same voice could never trip the gate.
                floor.observe(rms, quiet=rms < need)
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
                    cancel.set()
                    return
        except Exception as e:
            warn(f"[barge-in] {e}")

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
            delay = self._bleed_wait()
            self.bleed_delay_s = delay
            debug("barge.bleed sleep_s={}", delay)
            # sleep() ignores cancel. A finished turn would wait out the rest
            # of the bleed, or the next listen would open the mic twice.
            if cancel.wait(timeout=delay):
                debug("barge.bleed skipped (already cancelled)")
                return
            self.watch(cancel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


def frame_is_speech(vad: Any, frame: bytes) -> bool:
    """Return whether WebRTC VAD scores this 16 kHz frame as speech."""
    return bool(vad.is_speech(frame, SAMPLE_RATE))


def _take_full_frames(pending: bytearray, chunk: bytes, frame_bytes: int) -> list[bytes]:
    # PortAudio may deliver a short buffer or more than one block. A short
    # buffer has to wait for the next callback, and the tail of a long buffer
    # has to be kept. Dropping either cuts a hole in the utterance.
    if frame_bytes < 1:
        return []
    pending.extend(chunk)
    frames: list[bytes] = []
    while len(pending) >= frame_bytes:
        frames.append(bytes(pending[:frame_bytes]))
        del pending[:frame_bytes]
    return frames


def mic_frames(
    device: str | int | None,
    stop: threading.Event | None,
    *,
    timeout: float,
    aec: EchoCanceller | None = None,
) -> Iterator[bytes]:
    """Yield 16 kHz mic frames, AEC-cleaned when ``aec`` is given, until ``stop`` is set.

    Parameters
    ----------
    device : str or int or None
        PortAudio input.
    stop : threading.Event or None
        Ends the stream when set.
    timeout : float
        Seconds to wait for audio before checking ``stop`` again.
    aec : EchoCanceller or None, optional
        Cleans each frame against the far-end tap.

    Yields
    ------
    bytes
        One 30 ms int16 frame.
    """
    sd = load_sounddevice()
    audio: queue.Queue[bytes] = queue.Queue()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info, status
        audio.put(bytes(indata))

    with sd.RawInputStream(
        **pcm_stream_kwargs(SAMPLE_RATE, device),
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        pending = bytearray()
        while stop is None or not stop.is_set():
            try:
                chunk = audio.get(timeout=timeout)
            except queue.Empty:
                continue
            for frame in _take_full_frames(pending, chunk, FRAME_BYTES):
                yield aec.clean(frame) if aec is not None else frame
