"""Frozen duplex tuning read once from the environment.

Every knob lives on one of these dataclasses. ``from_env`` validates each key and
warns when a value is unusable, then uses the default. Library classes take a
tune object and never read the environment themselves.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Final, Self

from fish_audio_suite_voice.debug import warn

__all__ = [
    "DEEPGRAM_REGIONS",
    "DEFAULT_AEC_BLEED_DELAY_S",
    "DEFAULT_BARGE_HIT_FRAMES",
    "DEFAULT_BARGE_PLAYING_GAIN",
    "DEFAULT_BARGE_RMS",
    "DEFAULT_BLEED_DELAY_S",
    "DEFAULT_DEEPGRAM_MODEL",
    "DEFAULT_END_SILENCE_FRAMES",
    "DEFAULT_EOT_THRESHOLD",
    "DEFAULT_FADE_MS",
    "DEFAULT_MIN_SPEECH_RMS",
    "DEFAULT_MIN_VOICED_FRAMES",
    "DEFAULT_POST_SPEAK_COOLDOWN_S",
    "DEFAULT_VAD_AGGRESSIVENESS",
    "HTTP_KEEPALIVE_S",
    "IMPULSE_START_EXTRA",
    "MAX_UTTERANCE_FRAMES",
    "STT_PROVIDERS",
    "AecTune",
    "BargeTune",
    "ListenTune",
    "SttTune",
    "read_flag",
    "read_float",
    "read_int",
    "read_raw",
]

# Which speech recognition hears the user: Fish's, on a clip sent when they stop, or Deepgram's
# Flux, streamed while they speak.
STT_PROVIDERS: Final = ("fish", "deepgram")
DEFAULT_STT: Final = "fish"
DEFAULT_DEEPGRAM_MODEL: Final = "flux-general-en"
# Where Deepgram processes the audio. The regional hosts never route outside their region.
DEEPGRAM_REGIONS: Final = ("global", "eu", "au", "in")
DEFAULT_DEEPGRAM_REGION: Final = "global"
DEFAULT_EOT_THRESHOLD: Final = 0.7
EOT_THRESHOLD_LO: Final = 0.5
EOT_THRESHOLD_HI: Final = 1.0

DEFAULT_VAD_AGGRESSIVENESS: Final = 1
VAD_MODE_HI: Final = 3
DEFAULT_END_SILENCE_FRAMES: Final = 40
DEFAULT_START_SPEECH_FRAMES: Final = 4
DEFAULT_MIN_SPEECH_RMS: Final = 200.0
DEFAULT_PRE_PAD_FRAMES: Final = 20
DEFAULT_MIN_VOICED_FRAMES: Final = 12
MAX_UTTERANCE_FRAMES: Final = 500
IMPULSE_START_EXTRA: Final = 6

DEFAULT_BARGE_HIT_FRAMES: Final = 10
DEFAULT_BARGE_RMS: Final = 220.0
DEFAULT_BARGE_PLAYING_GAIN: Final = 2.2
DEFAULT_BLEED_DELAY_S: Final = 0.9
DEFAULT_POST_SPEAK_COOLDOWN_S: Final = 0.8

DEFAULT_AEC_BLEED_DELAY_S: Final = 0.3
DEFAULT_AEC_WET: Final = 0.85
# Fade-in, in milliseconds, at each sound that starts out of silence, so a sentence does
# not click when it begins. 0 turns it off.
DEFAULT_FADE_MS: Final = 4.0
# Idle seconds an ASR or LLM connection stays open. httpx drops it after 5 s,
# which is shorter than the gap between two turns, so every turn paid a new
# TLS handshake.
HTTP_KEEPALIVE_S: Final = 120.0

_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off"})


def read_raw(name: str) -> str | None:
    """Read an environment variable, treating blank as unset.

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    str or None
        The stripped value, or None when it is unset or only whitespace.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def read_float(
    name: str,
    default: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    positive: bool = False,
) -> float:
    """Read a float key, or warn and return ``default`` when it is unusable.

    Parameters
    ----------
    name : str
        Environment key.
    default : float
        Used when the key is unset, blank, not a finite number, or out of range.
    lo : float or None, optional
        Smallest accepted value.
    hi : float or None, optional
        Largest accepted value.
    positive : bool, optional
        Reject zero and below.

    Returns
    -------
    float
        The parsed value or ``default``.
    """
    raw = read_raw(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    bad = (
        not math.isfinite(value)
        or (positive and value <= 0)
        or (lo is not None and value < lo)
        or (hi is not None and value > hi)
    )
    if bad:
        warn(f"fish-voice: ignoring {name}={raw!r}, using {default}")
        return default
    return value


def read_int(
    name: str,
    default: int,
    *,
    lo: int | None = None,
    hi: int | None = None,
) -> int:
    """Read an int key, or warn and return ``default`` when it is unusable.

    Parameters
    ----------
    name : str
        Environment key.
    default : int
        Used when the key is unset, blank, not an integer, or out of range.
    lo : int or None, optional
        Smallest accepted value.
    hi : int or None, optional
        Largest accepted value.

    Returns
    -------
    int
        The parsed value or ``default``.
    """
    raw = read_raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        warn(f"fish-voice: ignoring {name}={raw!r}, using {default}")
        return default
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        warn(f"fish-voice: ignoring {name}={raw!r}, using {default}")
        return default
    return value


def read_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean key. Unknown spellings warn and return ``default``.

    Parameters
    ----------
    name : str
        Environment key.
    default : bool, optional
        Used when the key is unset, blank, or not a known spelling.

    Returns
    -------
    bool
        ``1/true/yes/on`` is True and ``0/false/no/off`` is False.
    """
    raw = read_raw(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    warn(f"fish-voice: ignoring {name}={raw!r}, using {default}")
    return default


def read_text(name: str, default: str = "") -> str:
    """Read a stripped string key. A key that is set but blank returns ``""``.

    Parameters
    ----------
    name : str
        Environment key.
    default : str, optional
        Used only when the key is not set at all.

    Returns
    -------
    str
        The stripped value or ``default``.
    """
    value = os.environ.get(name)
    return default if value is None else value.strip()


@dataclass(frozen=True, slots=True)
class ListenTune:
    """Voice-activity and utterance limits for one microphone capture.

    Attributes
    ----------
    vad_aggressiveness : int
        WebRTC VAD mode, 0 to 3.
    end_silence_frames : int
        Quiet 30 ms frames that end an utterance.
    start_speech_frames : int
        Voiced frames needed to start one.
    min_speech_rms : float
        Seed for the adaptive RMS floor, in int16 units.
    pre_pad_frames : int
        Frames kept before the start. Always at least the start run plus the
        impulse extra, or the mic would never trigger.
    min_voiced_frames : int
        Voiced frames an utterance needs before it is sent to ASR.
    """

    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
    end_silence_frames: int = DEFAULT_END_SILENCE_FRAMES
    start_speech_frames: int = DEFAULT_START_SPEECH_FRAMES
    min_speech_rms: float = DEFAULT_MIN_SPEECH_RMS
    pre_pad_frames: int = DEFAULT_PRE_PAD_FRAMES
    min_voiced_frames: int = DEFAULT_MIN_VOICED_FRAMES

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_VOICE_*`` listen keys.

        Returns
        -------
        ListenTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        start = read_int("FISH_VOICE_SPEECH_FRAMES", DEFAULT_START_SPEECH_FRAMES, lo=1)
        pad = read_int(
            "FISH_VOICE_PRE_PAD_FRAMES",
            DEFAULT_PRE_PAD_FRAMES,
            lo=0,
        )
        return cls(
            vad_aggressiveness=read_int(
                "FISH_VOICE_VAD", DEFAULT_VAD_AGGRESSIVENESS, lo=0, hi=VAD_MODE_HI
            ),
            end_silence_frames=read_int(
                "FISH_VOICE_SILENCE_FRAMES", DEFAULT_END_SILENCE_FRAMES, lo=1
            ),
            start_speech_frames=start,
            min_speech_rms=read_float("FISH_VOICE_MIN_RMS", DEFAULT_MIN_SPEECH_RMS, positive=True),
            pre_pad_frames=max(pad, start + IMPULSE_START_EXTRA),
            min_voiced_frames=read_int(
                "FISH_VOICE_MIN_VOICED_FRAMES",
                DEFAULT_MIN_VOICED_FRAMES,
                lo=1,
                hi=MAX_UTTERANCE_FRAMES,
            ),
        )


@dataclass(frozen=True, slots=True)
class BargeTune:
    """Barge-in gate and the pauses around a spoken reply.

    Attributes
    ----------
    hit_frames : int
        Loud voiced 30 ms frames in a row that interrupt a reply.
    min_rms : float
        Seed for the adaptive barge floor, in int16 units.
    playing_gain : float
        Floor multiplier while the speaker plays and AEC is off. At least 1.
    bleed_delay_s : float
        Seconds to ignore the mic after TTS starts when AEC is off.
    cooldown_s : float
        Seconds to wait after a reply before the mic opens again.
    """

    hit_frames: int = DEFAULT_BARGE_HIT_FRAMES
    min_rms: float = DEFAULT_BARGE_RMS
    playing_gain: float = DEFAULT_BARGE_PLAYING_GAIN
    bleed_delay_s: float = DEFAULT_BLEED_DELAY_S
    cooldown_s: float = DEFAULT_POST_SPEAK_COOLDOWN_S

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_VOICE_BARGE_*`` and delay keys.

        Returns
        -------
        BargeTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        return cls(
            hit_frames=read_int("FISH_VOICE_BARGE_FRAMES", DEFAULT_BARGE_HIT_FRAMES, lo=1),
            min_rms=read_float("FISH_VOICE_BARGE_RMS", DEFAULT_BARGE_RMS, positive=True),
            playing_gain=read_float(
                "FISH_VOICE_BARGE_PLAYING_GAIN",
                DEFAULT_BARGE_PLAYING_GAIN,
                lo=1.0,
            ),
            bleed_delay_s=read_float("FISH_VOICE_BLEED_DELAY", DEFAULT_BLEED_DELAY_S, lo=0.0),
            cooldown_s=read_float("FISH_VOICE_COOLDOWN", DEFAULT_POST_SPEAK_COOLDOWN_S, lo=0.0),
        )


@dataclass(frozen=True, slots=True)
class AecTune:
    """In-process echo cancellation.

    Attributes
    ----------
    enabled : bool
        False passes the mic through untouched.
    wet : float
        Mix of cleaned audio, 0 to 1.
    bleed_delay_s : float
        Seconds to ignore the mic after TTS starts when AEC3 is loaded. Named
        like ``BargeTune.bleed_delay_s``.
    """

    enabled: bool = True
    wet: float = DEFAULT_AEC_WET
    bleed_delay_s: float = DEFAULT_AEC_BLEED_DELAY_S

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_AEC``, ``FISH_VOICE_AEC_WET``, ``FISH_VOICE_AEC_BLEED_DELAY``.

        Returns
        -------
        AecTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        return cls(
            enabled=read_flag("FISH_VOICE_AEC", default=True),
            wet=read_float("FISH_VOICE_AEC_WET", DEFAULT_AEC_WET, lo=0.0, hi=1.0),
            bleed_delay_s=read_float(
                "FISH_VOICE_AEC_BLEED_DELAY",
                DEFAULT_AEC_BLEED_DELAY_S,
                lo=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class SttTune:
    """Which speech recognition hears the user, and how Deepgram's is set up.

    Attributes
    ----------
    provider : str
        ``"fish"`` sends one clip to Fish ASR when the user stops speaking. ``"deepgram"``
        streams the speech to Deepgram's Flux while the user talks, which decides for itself
        when the turn is over.
    deepgram_key : str
        The Deepgram API key. Empty unless the environment sets it.
    deepgram_model : str
        The Flux model: ``flux-general-en``, or ``flux-general-multi`` for ten languages.
    eot_threshold : float
        How sure Flux must be that the user has finished, from 0.5 to 1. Higher waits a little
        longer and cuts in less often.
    deepgram_region : str
        ``"global"``, or ``"eu"``, ``"au"`` or ``"in"`` to have the audio processed within the
        European Union, Australia or India. Deepgram fails a regional request rather than
        send it elsewhere.
    save_dir : str
        A folder to write each streamed turn's audio to, as a 16 kHz WAV, to hear what Deepgram
        was sent. Empty (the default) saves nothing.
    """

    provider: str = DEFAULT_STT
    deepgram_key: str = field(default="", repr=False)
    deepgram_model: str = DEFAULT_DEEPGRAM_MODEL
    eot_threshold: float = DEFAULT_EOT_THRESHOLD
    deepgram_region: str = DEFAULT_DEEPGRAM_REGION
    save_dir: str = ""

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_STT`` and the ``DEEPGRAM_*`` and Deepgram keys.

        Returns
        -------
        SttTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        provider = read_text("FISH_VOICE_STT", DEFAULT_STT).lower() or DEFAULT_STT
        if provider not in STT_PROVIDERS:
            warn(f"fish-voice: ignoring FISH_VOICE_STT={provider!r}, using {DEFAULT_STT}")
            provider = DEFAULT_STT
        model = read_text("FISH_VOICE_DEEPGRAM_MODEL", DEFAULT_DEEPGRAM_MODEL)
        if not model.startswith("flux"):
            warn(
                f"fish-voice: ignoring FISH_VOICE_DEEPGRAM_MODEL={model!r}, "
                f"using {DEFAULT_DEEPGRAM_MODEL} (only Flux models stream turns)"
            )
            model = DEFAULT_DEEPGRAM_MODEL
        region = read_text("FISH_VOICE_DEEPGRAM_REGION", DEFAULT_DEEPGRAM_REGION).lower()
        if region not in DEEPGRAM_REGIONS:
            warn(
                f"fish-voice: ignoring FISH_VOICE_DEEPGRAM_REGION={region!r}, "
                f"using {DEFAULT_DEEPGRAM_REGION} (one of {', '.join(DEEPGRAM_REGIONS)})"
            )
            region = DEFAULT_DEEPGRAM_REGION
        return cls(
            provider=provider,
            deepgram_key=read_text("DEEPGRAM_API_KEY"),
            deepgram_model=model,
            eot_threshold=read_float(
                "FISH_VOICE_EOT_THRESHOLD",
                DEFAULT_EOT_THRESHOLD,
                lo=EOT_THRESHOLD_LO,
                hi=EOT_THRESHOLD_HI,
            ),
            deepgram_region=region,
            save_dir=read_text("FISH_VOICE_STT_SAVE_DIR"),
        )
