"""One microphone utterance: VAD start, silence end, then a WAV for Fish ASR or a live stream."""

from __future__ import annotations

import collections
import io
from collections.abc import Callable, Sequence
from typing import Final

from fish_audio_suite_kit import MS_PER_S
from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.debug import debug, heartbeat_due, trace, warn
from fish_audio_suite_voice.mic import Mic
from fish_audio_suite_voice.pcm import (
    FRAME_MS,
    LISTEN_HEARTBEAT_FRAMES,
    SAMPLE_RATE,
    pcm_rms,
    write_mono_wav,
)
from fish_audio_suite_voice.portaudio import PortAudioMissingError
from fish_audio_suite_voice.recognised import BargeIn, CarriedClip, NoTurn, StreamFallback
from fish_audio_suite_voice.scorer import FrameScorer
from fish_audio_suite_voice.tune import (
    IMPULSE_EXTRA_VOICED,
    IMPULSE_NOW_RATIO,
    IMPULSE_PEAK_RATIO,
    IMPULSE_START_EXTRA,
    MAX_UTTERANCE_FRAMES,
    ListenTune,
)
from fish_audio_suite_voice.vad import Vad, open_vad

__all__ = [
    "cooldown_report",
    "listen_reject_reason",
    "mic_failure",
    "record_utterance",
    "spike_start_allowed",
    "start_frames_needed",
    "start_hit",
    "stream_utterance",
    "trailing_start_hits",
    "voice_needed",
]

# Once speech has started, a voiced frame at 55% of the floor still counts as
# speech, so soft word endings add to the voiced total.
HOLD_RMS_RATIO: Final = 0.55
# 120 ms. Anything shorter is a click, not a word.
MIN_UTTERANCE_FRAMES: Final = 4
# One quiet frame is 30 ms, a normal dip inside a word. Ending on that dip
# after the cap sends a cut-off sentence to ASR.
_CAP_QUIET_FRAMES: Final = 4


def start_hit(rms: float, min_rms: float, *, vad_speech: bool) -> bool:
    """Return whether a frame counts toward starting speech.

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
    started, so it does not apply here. ``vad_speech`` comes from the gate's
    ``FrameScorer``, which feeds the detector every frame once, as it arrives.
    """
    if not vad_speech:
        return False
    return rms >= min_rms


def _impulse(peak_rms: float, min_rms: float) -> bool:
    return peak_rms >= min_rms * IMPULSE_PEAK_RATIO


def voice_needed(min_voiced_frames: int, credited_hits: int) -> int:
    """Return how many voiced frames make a clip speech, given the frames a barge-in credited.

    Parameters
    ----------
    min_voiced_frames : int
        What a fresh utterance needs (``ListenTune.min_voiced_frames``).
    credited_hits : int
        The voiced frames of a barge-in clip the listen started from. 0 for none.

    Returns
    -------
    int
        ``min_voiced_frames``, or fewer when a barge-in clip credited fewer: the interrupt
        gate already judged that clip to be speech, so the voice in it is enough.

    Notes
    -----
    The batch path applies this when the clip ends (``listen_reject_reason``), and the
    streaming path before anything is connected, so the two drop the same clips.
    """
    if credited_hits > 0:
        return min(min_voiced_frames, credited_hits)
    return min_voiced_frames


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
    # A barge-in trips on fewer voiced frames than a listen needs (10 and 12 by
    # default). The frames it credited are enough, or the words that stopped the
    # reply would be thrown away when the user stops talking.
    if speech_hits < voice_needed(min_voiced_frames, credited_hits):
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
    """Count the unbroken run of start hits at the newest end of the pre-pad."""
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
    def __init__(self, tune: ListenTune, vad: Vad) -> None:
        self.tune: ListenTune = tune
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
        self.scorer: FrameScorer = FrameScorer(vad, tune.min_speech_rms, "listen")
        self.min_speech_now: float = tune.min_speech_rms

    def take(self, frame: bytes, idle_frames: int) -> bool:
        # Once speech has started the floor stops learning, so the voice cannot raise it.
        heard = self.scorer.score(frame, learn=not self.triggered)
        rms, vad_speech = heard.rms, heard.voiced
        self.min_speech_now = heard.floor
        self.window_peak = max(self.window_peak, rms)
        self.clip_peak = max(self.clip_peak, rms)
        self.scorer.meter(rms, self.min_speech_now)
        self._heartbeat(idle_frames, rms, vad_speech)
        if not self.triggered:
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
        # A talker who never pauses would grow the clip without bound. Keep the
        # last 15 seconds, and count hits over only those frames.
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
        trace(
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


def cooldown_report(frames: Sequence[bytes], tune: ListenTune, vad: Vad) -> str:
    """Describe what the mic heard while the cooldown kept it from listening.

    Parameters
    ----------
    frames : Sequence of bytes
        The 30 ms frames from the end of a reply to the end of its cooldown.
    tune : ListenTune
        Supplies the floor's seed and how many frames start speech.
    vad : Vad
        A fresh voice-activity detector.

    Returns
    -------
    str
        ``frames=… peak_rms=… floor=… voiced=… over_floor=… speech_like=… start_needs=…``.
        Each frame is judged against the adaptive floor a listen opened at the end of
        the reply would use, and ``floor`` is where it ended. ``speech_like`` counts
        frames both over the floor and voiced, the ones that count toward starting a
        turn. Near ``start_needs``, the bot's own echo could
        start a turn if the cooldown were shorter.

    Examples
    --------
    >>> class Quiet:
    ...     def is_speech(self, frame, rate):
    ...         return False
    >>> cooldown_report([], ListenTune(), Quiet())
    'frames=0'
    """
    if not frames:
        return "frames=0"
    # Scored as a listen opened at the end of the reply would score them.
    scorer = FrameScorer(vad, tune.min_speech_rms)
    scores = [scorer.score(frame) for frame in frames]
    over = [heard.rms >= heard.floor for heard in scores]
    voiced = sum(heard.voiced for heard in scores)
    speech_like = sum(loud and heard.voiced for loud, heard in zip(over, scores, strict=True))
    return (
        f"frames={len(frames)} peak_rms={max(heard.rms for heard in scores):.0f} "
        f"floor={scorer.floor:.0f} voiced={voiced} over_floor={sum(over)} "
        f"speech_like={speech_like} start_needs={tune.start_speech_frames}"
    )


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
        len(heard.clip_frames) * FRAME_MS / MS_PER_S,
        heard.speech_hits,
        round(heard.clip_peak),
    )
    return _encode_wav(pcm)


def _prime_listen(heard: _Listen, clip: CarriedClip | None) -> None:
    """Start a listen from speech it did not record itself, in the way its origin calls for."""
    match clip:
        case None:
            return
        case BargeIn():
            frames = clip.frames()
            if not frames:
                return
            # The interrupt gate already accepted this clip. Scored against a listen
            # floor above the gate's own, every frame would be silence and the
            # utterance would never reach ASR.
            floor = min(heard.min_speech_now, clip.gate_rms)
            for frame in frames:
                heard.remember(frame, pcm_rms(frame) >= floor)
            heard.primed_hits = heard.speech_hits
        case StreamFallback():
            frames = [frame for frame, _speech in clip.frames]
            if not frames:
                return
            # The listen gate already judged each frame as it was captured, so the clip
            # goes on as if this listen had recorded it. Only a barge-in it began with
            # earns credit.
            for frame, speech in clip.frames:
                heard.remember(frame, speech)
            credited = len(clip.barge.frames()) if clip.barge is not None else 0
            heard.primed_hits = min(
                heard.speech_hits, sum(speech for _frame, speech in clip.frames[:credited])
            )
    heard.triggered = True
    heard.clip_peak = max(pcm_rms(frame) for frame in frames)
    heard.silence = 0


def record_utterance(
    mic: Mic,
    quit_requested: StopFlag | None = None,
    *,
    carried: CarriedClip | None = None,
    tune: ListenTune | None = None,
    give_up_frames: int | None = None,
) -> bytes | None:
    """Block until one VAD utterance.

    Parameters
    ----------
    mic : Mic
        Where the frames come from, already cleaned of echo.
    quit_requested : StopFlag or None, optional
        Ends the listen when set, such as the session's quit flag (Ctrl+C). Returns None.
    carried : BargeIn or StreamFallback or None, optional
        Speech the listen starts from: the barge-in clip that interrupted the previous
        reply, or what a streaming recogniser caught before it failed. The listen reads on
        from the mic position just after it. None starts fresh and reads from now.
    tune : ListenTune or None, optional
        VAD and utterance limits. None uses the defaults.
    give_up_frames : int or None, optional
        Stop and return None when speech has not started after this many 30 ms frames. None
        waits as long as it takes.

    Returns
    -------
    bytes or None
        16 kHz mono WAV, or None when the clip is too short, an impulse, speech never
        started within ``give_up_frames``, or ``quit_requested`` is set.

    Notes
    -----
    With the defaults, speech starts after 4 frames in a row that the detector
    calls speech and that reach the adaptive floor. When the pre-pad holds a
    peak at least 8 times the floor, the run must be 10 frames and this frame
    at least half the peak. A clip with such a peak and at most 24 voiced
    frames is an impulse and is dropped. About 1.2 seconds of silence ends the
    turn. Each call opens a fresh detector from ``tune.vad_engine``, because a
    detector keeps state between frames.
    """
    tune = tune or ListenTune()
    heard = _Listen(tune, open_vad(tune.vad_engine, tune.vad_aggressiveness))
    _prime_listen(heard, carried)
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

    since = carried.since if carried is not None else None
    for idle_frames, frame in enumerate(mic.frames(quit_requested, since=since), start=1):
        if heard.take(frame, idle_frames):
            break
        if give_up_frames is not None and not heard.triggered and idle_frames >= give_up_frames:
            return None
    if quit_requested is not None and quit_requested.is_set():
        return None
    return _clip_wav(heard, tune)


class _StreamListen(_Listen):
    """A listen that hands each frame of the utterance to ``sink`` as it is kept."""

    def __init__(self, tune: ListenTune, vad: Vad, sink: Callable[[bytes, bool], None]) -> None:
        super().__init__(tune, vad)
        self._sink = sink

    def remember(self, frame: bytes, hit: bool) -> None:
        """Keep the frame as ``_Listen`` does, and pass it on with whether it was speech.

        Parameters
        ----------
        frame : bytes
            One PCM frame.
        hit : bool
            True when this frame is speech.
        """
        super().remember(frame, hit)
        self._sink(frame, hit)


def stream_utterance(
    mic: Mic,
    quit_requested: StopFlag | None,
    sink: Callable[[bytes, bool], None],
    *,
    carried: BargeIn | None = None,
    tune: ListenTune | None = None,
    since: int | None = None,
) -> bool:
    """Block until one utterance ends, passing its frames to ``sink`` as they come.

    Parameters
    ----------
    mic : Mic
        Where the frames come from, already cleaned of echo.
    quit_requested : StopFlag or None
        Ends the listen when set.
    sink : Callable
        Called with ``(frame, voiced)`` for every frame of the utterance, in order: first
        any ``carried`` clip and the pre-pad, then each frame as it is captured. It runs on this
        thread and has to return quickly. Nothing is passed while the mic is waiting for
        speech to start, so only the speech leaves the machine.
    carried : BargeIn or None, optional
        The barge-in clip that interrupted the previous reply. Its frames are passed on
        first, as speech wherever its gate's level says so.
    tune : ListenTune or None, optional
        VAD and utterance limits. None uses the defaults.
    since : int or None, optional
        The mic position to read from. None reads from now.

    Returns
    -------
    bool
        True when the silence at the end of the speech ended it, so the caller should decide
        the turn is over. False when ``quit_requested`` stopped it.

    Notes
    -----
    This is ``record_utterance`` without the WAV: the same start gate (VAD, level, impulse
    rejection) and the same ``MicLevel`` events, but the end of the turn is for the caller to
    judge, with this silence as the limit.
    """
    tune = tune or ListenTune()
    heard = _StreamListen(tune, open_vad(tune.vad_engine, tune.vad_aggressiveness), sink)
    _prime_listen(heard, carried)
    for idle_frames, frame in enumerate(mic.frames(quit_requested, since=since), start=1):
        if heard.take(frame, idle_frames):
            return True
    return False


def mic_failure(error: PortAudioMissingError) -> NoTurn:
    """Report that the mic cannot open, and end the session.

    Parameters
    ----------
    error : PortAudioMissingError
        What the device loader raised when a listen opened the mic, with its install hints.

    Returns
    -------
    NoTurn
        ``fatal``: without PortAudio no listen can work.
    """
    warn(str(error))
    return NoTurn("fatal")
