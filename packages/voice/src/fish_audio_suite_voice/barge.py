"""Mic barge-in gate: lookback, energy, min-speech, speaker-bleed delay."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from fish_audio_suite_kit import MS_PER_S, env_float, env_int
from fish_audio_suite_voice.aec import (
    AEC_RATE,
    SAMPLE_BYTES,
    aec_available,
    clean_mic_frame,
    effective_bleed_s,
    far_end_playing,
    pcm_rms,
)
from fish_audio_suite_voice.debug import debug, heartbeat_due
from fish_audio_suite_voice.playback import load_sounddevice, pcm_stream_kwargs

SAMPLE_RATE = AEC_RATE
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // MS_PER_S
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_BYTES
LISTEN_HEARTBEAT_FRAMES = 20
DEFAULT_BARGE_HIT_FRAMES = 10
DEFAULT_BARGE_RMS = 220.0
DEFAULT_BARGE_OVER = 2.2
BARGE_MISS_DECAY_FRAMES = 3
DEFAULT_BLEED_DELAY_S = 0.9
DEFAULT_POST_SPEAK_COOLDOWN_S = 0.8
_BARGE_POLL_S = 0.2
_BARGE_VAD = 3


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
        return min_rms * over
    return min_rms


def post_speak_cooldown_s() -> float:
    return env_float("FISH_VOICE_COOLDOWN", DEFAULT_POST_SPEAK_COOLDOWN_S)


def _or_env[T: int | float](
    value: T | None,
    name: str,
    default: T,
    read: Callable[[str, T], T],
) -> T:
    if value is not None:
        return value
    return read(name, default)


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
    """VAD-energy + min-speech. Apps that own the mic can import this."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        bleed_delay_s: float | None = None,
        hit_frames: int | None = None,
        min_rms: float | None = None,
    ) -> None:
        self.device = device
        self._bleed_override = bleed_delay_s
        delay = _or_env(bleed_delay_s, "FISH_VOICE_BLEED_DELAY", DEFAULT_BLEED_DELAY_S, env_float)
        self.bleed_delay_s = delay if delay >= 0 else DEFAULT_BLEED_DELAY_S
        hits = _or_env(hit_frames, "FISH_VOICE_BARGE_FRAMES", DEFAULT_BARGE_HIT_FRAMES, env_int)
        self.hit_frames = hits if hits > 0 else DEFAULT_BARGE_HIT_FRAMES
        self.min_rms = _or_env(min_rms, "FISH_VOICE_BARGE_RMS", DEFAULT_BARGE_RMS, env_float)

    def _bleed_wait(self) -> float:
        if self._bleed_override is not None:
            delay = self._bleed_override
            return delay if delay >= 0 else DEFAULT_BLEED_DELAY_S
        delay = env_float("FISH_VOICE_BLEED_DELAY", DEFAULT_BLEED_DELAY_S)
        if delay < 0:
            delay = DEFAULT_BLEED_DELAY_S
        return effective_bleed_s(delay)

    def watch(self, cancel: threading.Event) -> None:
        import webrtcvad

        vad = webrtcvad.Vad(_BARGE_VAD)
        hit = 0
        miss = 0
        over = env_float("FISH_VOICE_BARGE_OVER", DEFAULT_BARGE_OVER)
        aec_on = aec_available()
        debug(
            "barge.arm delay_s={delay} hit_frames={hits} min_rms={min_rms} over={over} "
            "aec={aec} vad={vad}",
            delay=self.bleed_delay_s,
            hits=self.hit_frames,
            min_rms=self.min_rms,
            over=over,
            aec=aec_on,
            vad=_BARGE_VAD,
        )

        try:
            for idle_frames, frame in enumerate(
                mic_frames(self.device, cancel, timeout=_BARGE_POLL_S), start=1
            ):
                rms = pcm_rms(frame)
                far = far_end_playing()
                need = barge_rms_need(self.min_rms, far_playing=far, over=over, aec_on=aec_on)
                _barge_heartbeat(
                    idle_frames=idle_frames,
                    rms=rms,
                    need=need,
                    far=far,
                    hit=hit,
                    aec_on=aec_on,
                )
                voiced = rms >= need and frame_is_speech(vad, frame)
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
                    cancel.set()
                    return
        except Exception as e:
            print(f"[barge-in] {e}", file=sys.stderr)

    def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
        def _run() -> None:
            delay = self._bleed_wait()
            self.bleed_delay_s = delay
            debug("barge.bleed sleep_s={}", delay)
            time.sleep(delay)
            if cancel.is_set():
                debug("barge.bleed skipped (already cancelled)")
                return
            self.watch(cancel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


def frame_is_speech(vad: Any, frame: bytes) -> bool:
    return bool(vad.is_speech(frame, SAMPLE_RATE))


def mic_frames(
    device: str | int | None,
    stop: threading.Event | None,
    *,
    timeout: float,
) -> Iterator[bytes]:
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
        while stop is None or not stop.is_set():
            try:
                frame = audio.get(timeout=timeout)
            except queue.Empty:
                continue
            if len(frame) < FRAME_BYTES:
                continue
            yield clean_mic_frame(frame[:FRAME_BYTES])
