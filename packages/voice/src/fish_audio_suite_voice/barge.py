"""Mic barge-in gate: lookback, energy, min-speech, speaker-bleed delay."""

from __future__ import annotations

import collections
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
VAD_AGGRESSIVENESS = int(os.environ.get("FISH_VOICE_VAD", "2"))
SILENCE_FRAMES_END = int(os.environ.get("FISH_VOICE_SILENCE_FRAMES", "22"))
SPEECH_FRAMES_START = int(os.environ.get("FISH_VOICE_SPEECH_FRAMES", "8"))
MAX_UTTERANCE_FRAMES = 500
MIN_SPEECH_RMS = float(os.environ.get("FISH_VOICE_MIN_RMS", "280"))
PRE_PAD_FRAMES = int(os.environ.get("FISH_VOICE_PRE_PAD", "10"))  # ~300 ms
BARGE_HIT_FRAMES = int(os.environ.get("FISH_VOICE_BARGE_FRAMES", "16"))
BARGE_RMS = float(os.environ.get("FISH_VOICE_BARGE_RMS", "400"))
BLEED_DELAY_S = float(os.environ.get("FISH_VOICE_BLEED_DELAY", "0.9"))
POST_SPEAK_COOLDOWN_S = float(os.environ.get("FISH_VOICE_COOLDOWN", "0.8"))


def pcm_rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)) + 1e-9)


class BargeGate:
    """VAD-energy + min-speech. Apps that own the mic can import this."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        bleed_delay_s: float = BLEED_DELAY_S,
        hit_frames: int = BARGE_HIT_FRAMES,
        min_rms: float = BARGE_RMS,
    ) -> None:
        self.device = device
        self.bleed_delay_s = bleed_delay_s
        self.hit_frames = hit_frames
        self.min_rms = min_rms

    def watch(self, cancel: threading.Event) -> None:
        import sounddevice as sd
        import webrtcvad

        vad = webrtcvad.Vad(3)
        hit = 0
        q: queue.Queue[bytes] = queue.Queue()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                device=self.device,
                callback=callback,
            ):
                while not cancel.is_set():
                    try:
                        frame = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if len(frame) < FRAME_BYTES:
                        continue
                    frame = frame[:FRAME_BYTES]
                    if pcm_rms(frame) < self.min_rms:
                        hit = max(0, hit - 1)
                        continue
                    if vad.is_speech(frame, SAMPLE_RATE):
                        hit += 1
                        if hit >= self.hit_frames:
                            cancel.set()
                            return
                    else:
                        hit = max(0, hit - 1)
        except Exception as e:
            print(f"[barge-in] {e}", file=sys.stderr)

    def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
        def _run() -> None:
            time.sleep(self.bleed_delay_s)
            if cancel.is_set():
                return
            self.watch(cancel)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


def record_utterance(
    device: str | int | None = None,
    stop: threading.Event | None = None,
) -> bytes | None:
    """Block until one VAD utterance. Returns WAV bytes (16 kHz mono) or None."""
    import sounddevice as sd
    import webrtcvad

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        q.put(bytes(indata))

    voiced: list[bytes] = []
    ring: collections.deque[bytes] = collections.deque(maxlen=PRE_PAD_FRAMES)
    triggered = False
    silence = 0

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        device=device,
        callback=callback,
    ):
        while True:
            if stop is not None and stop.is_set():
                return None
            try:
                frame = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if len(frame) < FRAME_BYTES:
                continue
            frame = frame[:FRAME_BYTES]
            rms = pcm_rms(frame)
            hold_rms = MIN_SPEECH_RMS * 0.55
            need = MIN_SPEECH_RMS if not triggered else hold_rms
            loud = rms >= need
            is_speech = loud and bool(vad.is_speech(frame, SAMPLE_RATE))

            if not triggered:
                ring.append(frame)
                speechish = sum(
                    1
                    for f in ring
                    if pcm_rms(f) >= MIN_SPEECH_RMS and vad.is_speech(f, SAMPLE_RATE)
                )
                if speechish >= SPEECH_FRAMES_START:
                    triggered = True
                    voiced.extend(ring)
                    ring.clear()
                    silence = 0
            else:
                voiced.append(frame)
                if is_speech:
                    silence = 0
                else:
                    silence += 1
                    if silence >= SILENCE_FRAMES_END:
                        break
                if len(voiced) >= MAX_UTTERANCE_FRAMES:
                    break

    if len(voiced) < SPEECH_FRAMES_START + 3:
        return None

    pcm = b"".join(voiced)
    with tempfile.SpooledTemporaryFile(max_size=2_000_000) as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        buf.seek(0)
        return buf.read()
