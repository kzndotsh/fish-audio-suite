"""One microphone utterance: VAD start, silence end, WAV for Fish ASR."""

from __future__ import annotations

import collections
import io
import threading
from typing import Any, Final

from fish_audio_suite_voice.aec import EchoCanceller, pcm_rms
from fish_audio_suite_voice.barge import (
    DEFAULT_BARGE_RMS,
    FRAME_BYTES,
    FRAME_MS,
    LISTEN_HEARTBEAT_FRAMES,
    MIC_LEVEL_EVERY_FRAMES,
    SAMPLE_RATE,
    frame_is_speech,
    mic_frames,
)
from fish_audio_suite_voice.debug import debug, heartbeat_due, trace
from fish_audio_suite_voice.events import EVENTS, MicLevel
from fish_audio_suite_voice.floor import AdaptiveFloor
from fish_audio_suite_voice.playback import write_mono_wav
from fish_audio_suite_voice.tune import (
    IMPULSE_START_EXTRA,
    MAX_UTTERANCE_FRAMES,
    ListenTune,
)

__all__ = [
    "listen_reject_reason",
    "record_utterance",
    "spike_start_allowed",
    "start_frames_needed",
    "start_hit",
    "trailing_start_hits",
]

HOLD_RMS_RATIO: Final = 0.55
MIN_UTTERANCE_FRAMES: Final = 4
# One quiet frame is 30 ms, a normal dip inside a word. Ending on that dip
# after the cap sends a cut-off sentence to ASR.
_CAP_QUIET_FRAMES: Final = 4
_MIC_POLL_S: Final = 0.25
_WAV_HEADER_BYTES: Final = 44
IMPULSE_PEAK_RATIO: Final = 8.0
IMPULSE_EXTRA_VOICED: Final = 12
IMPULSE_NOW_RATIO: Final = 0.5


def start_hit(rms: float, min_rms: float, *, vad_speech: bool) -> bool:
    """Count a listen-start frame. Score VAD once per capture; do not replay the ring.

    Parameters
    ----------
    rms : float
        Frame loudness.
    min_rms : float
        The loudness a frame must reach.
    vad_speech : bool
        Whether the voice detector scored the frame as speech.

    Returns
    -------
    bool
        True when the detector heard speech and the frame is at least ``min_rms``.

    Notes
    -----
    The hold ratio is only for staying in an utterance after it has already
    started, so it does not apply here.
    """
    if not vad_speech:
        return False
    return rms >= min_rms


def _impulse(peak_rms: float, min_rms: float) -> bool:
    return peak_rms >= min_rms * IMPULSE_PEAK_RATIO


def listen_reject_reason(
    *,
    clip_frames: int,
    speech_hits: int,
    peak_rms: float,
    min_voiced_frames: int,
    min_speech_rms: float,
    credited_hits: int = 0,
) -> str | None:
    """Drop coughs and spikes. None means send the clip to ASR."""
    if clip_frames < MIN_UTTERANCE_FRAMES:
        return "too_short"
    # Barge trips at 10 frames. The listen minimum is 12, so the same
    # interrupt was discarded when the user stopped talking.
    need_voice = min(min_voiced_frames, credited_hits) if credited_hits > 0 else min_voiced_frames
    if speech_hits < need_voice:
        return "too_little_voice"
    # A barge run is already several loud frames. The impulse gate is for a
    # short spike that never passed that gate, so it must not drop the interrupt.
    if (
        credited_hits <= 0
        and _impulse(peak_rms, min_speech_rms)
        and speech_hits <= min_voiced_frames + IMPULSE_EXTRA_VOICED
    ):
        return "impulse"
    return None


def start_frames_needed(peak_rms: float, min_rms: float, start_speech_frames: int) -> int:
    """Require extra VAD hits before an 8x RMS spike counts as speech."""
    if _impulse(peak_rms, min_rms):
        return start_speech_frames + IMPULSE_START_EXTRA
    return start_speech_frames


def trailing_start_hits(ring: collections.deque[tuple[bytes, bool]]) -> int:
    """Count scored frames at the newest end of the pre-pad. Gaps do not count."""
    n = 0
    for _, counted in reversed(ring):
        if not counted:
            break
        n += 1
    return n


def spike_start_allowed(peak_rms: float, now_rms: float, min_rms: float) -> bool:
    """Reject a decaying bang: 8x peak in the ring but this frame already dropped."""
    if not _impulse(peak_rms, min_rms):
        return True
    return now_rms >= peak_rms * IMPULSE_NOW_RATIO


def _encode_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    write_mono_wav(buf, pcm, SAMPLE_RATE)
    return buf.getvalue()


class _Listen:
    def __init__(self, tune: ListenTune, vad: Any) -> None:
        self.tune: ListenTune = tune
        self.vad: Any = vad
        self.clip_frames: list[bytes] = []
        self.ring: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=tune.pre_pad_frames
        )
        self.triggered: bool = False
        self.silence: int = 0
        self.window_peak: float = 0.0
        self.clip_peak: float = 0.0
        self.speech_hits: int = 0
        self.speech_flags: list[bool] = []
        self.primed_hits: int = 0
        self.floor: AdaptiveFloor = AdaptiveFloor(tune.min_speech_rms)
        self.min_speech_now: float = tune.min_speech_rms

    def take(self, frame: bytes, idle_frames: int) -> bool:
        rms = pcm_rms(frame)
        vad_speech = frame_is_speech(self.vad, frame)
        self.min_speech_now = self.floor.value()
        self.window_peak = max(self.window_peak, rms)
        self.clip_peak = max(self.clip_peak, rms)
        if idle_frames % MIC_LEVEL_EVERY_FRAMES == 0:
            EVENTS.emit(MicLevel(rms, self.min_speech_now, "listen"))
        self._heartbeat(idle_frames, rms, vad_speech)
        if not self.triggered:
            # A frame already at the floor is not room hiss. Counting it raises
            # the floor until the same voice can never start the clip.
            self.floor.observe(rms, quiet=rms < self.min_speech_now)
            return self._arm(frame, rms, vad_speech)
        is_speech = rms >= self.min_speech_now * HOLD_RMS_RATIO and vad_speech
        return self._hold(frame, is_speech, vad_speech=vad_speech)

    def remember(self, frame: bytes, hit: bool) -> None:
        """Store one frame and whether it counted as speech.

        Parameters
        ----------
        frame : bytes
            One PCM frame.
        hit : bool
            True when this frame is speech.
        """
        self.clip_frames.append(frame)
        self.speech_flags.append(hit)
        # A talker who never pauses used to keep every frame. The cap is the
        # last 15 seconds, and the hit count has to match those frames.
        extra = len(self.clip_frames) - MAX_UTTERANCE_FRAMES
        if extra > 0:
            del self.clip_frames[:extra]
            del self.speech_flags[:extra]
        self.speech_hits = sum(self.speech_flags)
        # primed_hits is credit for frames still in this clip. After the cap
        # drops the barge prefix, that credit must not keep a short tail.
        self.primed_hits = min(self.primed_hits, self.speech_hits)

    def _hold(self, frame: bytes, is_speech: bool, *, vad_speech: bool) -> bool:
        self.remember(frame, is_speech)
        # A quiet frame that VAD still calls speech is a hesitation, not the end.
        if is_speech or vad_speech:
            self.silence = 0
            return False
        self.silence += 1
        if self.silence >= self.tune.end_silence_frames:
            return True
        return len(self.clip_frames) >= MAX_UTTERANCE_FRAMES and self.silence >= _CAP_QUIET_FRAMES

    def _heartbeat(self, idle_frames: int, rms: float, vad_speech: bool) -> None:
        if not heartbeat_due(idle_frames, LISTEN_HEARTBEAT_FRAMES):
            return
        hits = (
            sum(1 for _, counted in self.ring if counted)
            if not self.triggered
            else self.speech_hits
        )
        debug(
            "listen.mic frames={} peak_rms={:.0f} last_rms={:.0f} floor={:.0f} vad={} triggered={} hits={}",
            idle_frames,
            self.window_peak,
            rms,
            self.min_speech_now,
            vad_speech,
            self.triggered,
            hits,
        )
        self.window_peak = 0.0

    def _arm(self, frame: bytes, rms: float, vad_speech: bool) -> bool:
        self.ring.append((frame, start_hit(rms, self.min_speech_now, vad_speech=vad_speech)))
        peak_ring = max((pcm_rms(pcm) for pcm, _ in self.ring), default=0.0)
        need_start = start_frames_needed(
            peak_ring, self.min_speech_now, self.tune.start_speech_frames
        )
        hits_now = trailing_start_hits(self.ring)
        if hits_now < need_start or not spike_start_allowed(peak_ring, rms, self.min_speech_now):
            return False
        self.triggered = True
        # The trailing run only decides that speech started. Earlier scored
        # frames in the pre-pad are part of the clip; leaving them out makes
        # a short hesitation look like too little voice and drops the utterance.
        for pcm, counted in self.ring:
            self.remember(pcm, counted)
        debug(
            "listen.speech_start rms={rms:.0f} vad={vad} prepad={prepad} "
            "hits={hits} start_need={start_need} peak={peak:.0f}",
            rms=rms,
            vad=vad_speech,
            prepad=len(self.clip_frames),
            hits=self.speech_hits,
            start_need=need_start,
            peak=peak_ring,
        )
        self.ring.clear()
        self.silence = 0
        return False


def _clip_wav(heard: _Listen, tune: ListenTune) -> bytes | None:
    why = listen_reject_reason(
        clip_frames=len(heard.clip_frames),
        speech_hits=heard.speech_hits,
        peak_rms=heard.clip_peak,
        min_voiced_frames=tune.min_voiced_frames,
        min_speech_rms=heard.min_speech_now if heard.triggered else tune.min_speech_rms,
        credited_hits=heard.primed_hits,
    )
    if why is not None:
        debug(
            "listen.reject {} frames={} voiced_hits={} peak_rms={:.0f} min_voiced_frames={}",
            why,
            len(heard.clip_frames),
            heard.speech_hits,
            heard.clip_peak,
            tune.min_voiced_frames,
        )
        return None
    pcm = b"".join(heard.clip_frames)
    debug(
        "listen.end {:.2f}s clip, {} voiced frames, peak rms {}",
        len(heard.clip_frames) * FRAME_MS / 1000,
        heard.speech_hits,
        round(heard.clip_peak),
    )
    return _encode_wav(pcm)


def _prime_listen(heard: _Listen, pcm: bytes) -> None:
    """Start a listen from barge audio. Those frames already passed the interrupt gate."""
    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]
    if not frames:
        return
    heard.triggered = True
    # The interrupt gate already accepted this clip. A higher listen floor
    # used to score every frame as silence, so the utterance never reached ASR.
    floor = min(heard.min_speech_now, DEFAULT_BARGE_RMS)
    for frame in frames:
        heard.remember(frame, pcm_rms(frame) >= floor)
    heard.primed_hits = heard.speech_hits
    heard.clip_peak = max(pcm_rms(frame) for frame in frames)
    heard.silence = 0


def record_utterance(
    device: str | int | None = None,
    quit_requested: threading.Event | None = None,
    *,
    prefix: bytes = b"",
    tune: ListenTune | None = None,
    aec: EchoCanceller | None = None,
) -> bytes | None:
    """Block until one VAD utterance.

    Parameters
    ----------
    device : str or int or None, optional
        PortAudio input. None uses the host default.
    quit_requested : threading.Event or None, optional
        The session's quit flag (Ctrl+C). When set, return None.
    prefix : bytes, optional
        PCM kept from the barge-in that interrupted the previous reply.
        The next listen starts from this clip instead of a cooldown.
    tune : ListenTune or None, optional
        VAD and utterance limits. None uses the defaults.
    aec : EchoCanceller or None, optional
        Cleans each frame against the far-end tap. None skips AEC.

    Returns
    -------
    bytes or None
        16 kHz mono WAV, or None when the clip is too short, an impulse, or
        ``quit_requested`` is set.

    Notes
    -----
    Speech must hold VAD and RMS at the newest pre-pad end. A peak at least
    8 times the floor also needs this frame to be at least half the peak and
    10 trailing hits. An impulse (peak >= 8 times the floor and at most 24
    voiced hits) is dropped. Silence of about 1.2 seconds ends the turn.
    ``webrtcvad`` is imported here so the voice package imports without the
    ``vad`` extra.
    """
    import webrtcvad

    tune = tune or ListenTune()
    heard = _Listen(tune, webrtcvad.Vad(tune.vad_aggressiveness))
    _prime_listen(heard, prefix)
    trace(
        "listen.open vad={} start_speech_frames={} min_rms={} min_voiced_frames={} "
        "pre_pad_frames={} end_silence_frames={} prefix_frames={}",
        tune.vad_aggressiveness,
        tune.start_speech_frames,
        tune.min_speech_rms,
        tune.min_voiced_frames,
        tune.pre_pad_frames,
        tune.end_silence_frames,
        len(heard.clip_frames),
    )

    for idle_frames, frame in enumerate(
        mic_frames(device, quit_requested, timeout=_MIC_POLL_S, aec=aec), start=1
    ):
        if heard.take(frame, idle_frames):
            break
    if quit_requested is not None and quit_requested.is_set():
        return None
    return _clip_wav(heard, tune)
