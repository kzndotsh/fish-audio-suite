"""The tunes of hearing the user: listening, barge-in, mic cleanup and speech recognition.

Each tune is a frozen dataclass that a library class takes. ``from_env`` reads it once
through ``env_read``, which warns about a value it cannot use and uses the default. Built
in code, a tune holds its values to the same bounds: ``__post_init__`` raises
``ValueError`` for one ``from_env`` would refuse. Library classes never read the
environment themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Self

from fish_audio_suite_voice.config.env import (
    ChoiceSetting,
    FloatSetting,
    IntSetting,
    check_range,
    read_flag,
    read_float,
    read_text,
    read_word,
)
from fish_audio_suite_voice.logutil.debug import warn

__all__ = [
    "DEEPGRAM_HOSTS",
    "DEEPGRAM_REGIONS",
    "DEFAULT_AEC_BLEED_DELAY_S",
    "DEFAULT_BARGE_HIT_FRAMES",
    "DEFAULT_BARGE_PLAYING_GAIN",
    "DEFAULT_BARGE_RMS",
    "DEFAULT_BLEED_DELAY_S",
    "DEFAULT_DEEPGRAM_MODEL",
    "DEFAULT_DEEPGRAM_REGION",
    "DEFAULT_END_SILENCE_FRAMES",
    "DEFAULT_EOT_THRESHOLD",
    "DEFAULT_MIN_SPEECH_RMS",
    "DEFAULT_MIN_VOICED_FRAMES",
    "DEFAULT_NOISE_SUPPRESSION_DB",
    "DEFAULT_POST_SPEAK_COOLDOWN_S",
    "DEFAULT_TURN_EXTENSIONS",
    "DEFAULT_TURN_SILENCE_FRAMES",
    "DEFAULT_VAD_AGGRESSIVENESS",
    "DEFAULT_VAD_ENGINE",
    "IMPULSE_EXTRA_VOICED",
    "IMPULSE_NOW_RATIO",
    "IMPULSE_PEAK_RATIO",
    "IMPULSE_START_EXTRA",
    "MAX_UTTERANCE_FRAMES",
    "STREAMING_STT_PROVIDERS",
    "STT_PROVIDERS",
    "AecTune",
    "BargeTune",
    "DeepgramTune",
    "ListenTune",
    "SttProvider",
    "SttTune",
    "TurnTune",
]

# Which speech recognition hears the user: Fish's, on a clip sent when they stop, or a
# streaming one, while they speak.
SttProvider = Literal["fish", "deepgram", "fish-realtime"]
DEFAULT_STT: Final[SttProvider] = "fish"
# The recognisers that hear a turn as it is spoken and end it themselves. Each is an adapter
# listed in ``recognisers.RECOGNISERS``, which a test keeps this in step with.
STREAMING_STT_PROVIDERS: Final[tuple[SttProvider, ...]] = ("deepgram", "fish-realtime")
STT_PROVIDERS: Final[tuple[SttProvider, ...]] = (DEFAULT_STT, *STREAMING_STT_PROVIDERS)
DEFAULT_DEEPGRAM_MODEL: Final = "flux-general-en"
# Only Flux models end a turn themselves, which the streaming path needs.
_DEEPGRAM_MODEL_FAMILY: Final = "flux"
# Where Deepgram processes the audio, and the host that does it. The regional hosts never
# route outside their region.
DEEPGRAM_HOSTS: Final = {
    "global": "api.deepgram.com",
    "eu": "api.eu.deepgram.com",
    "au": "api.au.deepgram.com",
    "in": "api.in.deepgram.com",
}
DEEPGRAM_REGIONS: Final = tuple(DEEPGRAM_HOSTS)
DEFAULT_DEEPGRAM_REGION: Final = "global"
DEFAULT_EOT_THRESHOLD: Final = 0.7
_EOT_THRESHOLD_LO: Final = 0.5
_EOT_THRESHOLD_HI: Final = 1.0
# The early end of turn: 0 is off, else this range and no higher than the end of turn.
_EAGER_EOT_OFF: Final = 0.0
_EAGER_EOT_LO: Final = 0.3
_EAGER_EOT_HI: Final = 0.9

DEFAULT_VAD_AGGRESSIVENESS: Final = 1
DEFAULT_VAD_ENGINE: Final = "webrtc"
_VAD_MODE_LO: Final = 0
_VAD_MODE_HI: Final = 3
DEFAULT_END_SILENCE_FRAMES: Final = 40
DEFAULT_START_SPEECH_FRAMES: Final = 4
DEFAULT_MIN_SPEECH_RMS: Final = 200.0
DEFAULT_PRE_PAD_FRAMES: Final = 20
DEFAULT_MIN_VOICED_FRAMES: Final = 12
MAX_UTTERANCE_FRAMES: Final = 500
# A peak this many times the floor is a bang (a door, a desk knock) until proven otherwise.
# Such a clip is dropped unless it holds ``IMPULSE_EXTRA_VOICED`` voiced frames beyond the
# minimum, it cannot start while the current frame is under ``IMPULSE_NOW_RATIO`` of the
# peak, and it needs ``IMPULSE_START_EXTRA`` more voiced frames to start. The pre-roll must
# hold that longer start, so ``ListenTune`` keeps ``pre_pad_frames`` at least that long.
IMPULSE_PEAK_RATIO: Final = 8.0
IMPULSE_EXTRA_VOICED: Final = 12
IMPULSE_NOW_RATIO: Final = 0.5
IMPULSE_START_EXTRA: Final = 6
# Frame counts are at least one frame.
_ONE_FRAME: Final = 1

# The end-of-turn check on the Fish path: a short silence ends the clip, and a transcript
# that stops mid-thought reopens the mic.
DEFAULT_TURN_SILENCE_FRAMES: Final = 12
DEFAULT_TURN_EXTENSIONS: Final = 1
_TURN_EXTENSIONS_MAX: Final = 2

DEFAULT_BARGE_HIT_FRAMES: Final = 10
DEFAULT_BARGE_RMS: Final = 220.0
DEFAULT_BARGE_PLAYING_GAIN: Final = 2.2
# Below 1 the floor would drop while the speaker plays, so bleed trips barge-in.
_PLAYING_GAIN_LO: Final = 1.0
DEFAULT_BLEED_DELAY_S: Final = 0.9
DEFAULT_AEC_BLEED_DELAY_S: Final = 0.3
DEFAULT_POST_SPEAK_COOLDOWN_S: Final = 0.8
_NO_WAIT_S: Final = 0.0

DEFAULT_AEC_WET: Final = 0.85
_WET_LO: Final = 0.0
_WET_HI: Final = 1.0
# WebRTC's noise suppressor works in these steps of attenuation. 21 dB made typing louder than
# a quiet voice in a measured recording; 18 dB did not.
NOISE_SUPPRESSION_STEPS_DB: Final = (0, 6, 12, 18, 21)
DEFAULT_NOISE_SUPPRESSION_DB: Final = 18

# Each setting a tune reads and checks, with its variable, default and range, declared once:
# ``from_env`` reads it and ``__post_init__`` holds a value given in code to the same range.
_VAD = IntSetting("FISH_VOICE_VAD", DEFAULT_VAD_AGGRESSIVENESS, lo=_VAD_MODE_LO, hi=_VAD_MODE_HI)
_SILENCE = IntSetting("FISH_VOICE_SILENCE_FRAMES", DEFAULT_END_SILENCE_FRAMES, lo=_ONE_FRAME)
_SPEECH = IntSetting("FISH_VOICE_SPEECH_FRAMES", DEFAULT_START_SPEECH_FRAMES, lo=_ONE_FRAME)
_MIN_RMS = FloatSetting("FISH_VOICE_MIN_RMS", DEFAULT_MIN_SPEECH_RMS, positive=True)
_PRE_PAD = IntSetting("FISH_VOICE_PRE_PAD_FRAMES", DEFAULT_PRE_PAD_FRAMES, lo=0)
_MIN_VOICED = IntSetting(
    "FISH_VOICE_MIN_VOICED_FRAMES",
    DEFAULT_MIN_VOICED_FRAMES,
    lo=_ONE_FRAME,
    hi=MAX_UTTERANCE_FRAMES,
)
_TURN_SILENCE = IntSetting(
    "FISH_VOICE_TURN_SILENCE_FRAMES", DEFAULT_TURN_SILENCE_FRAMES, lo=_ONE_FRAME
)
_TURN_EXTEND = IntSetting(
    "FISH_VOICE_TURN_EXTEND_FRAMES", DEFAULT_END_SILENCE_FRAMES, lo=_ONE_FRAME
)
_TURN_EXTENSIONS = IntSetting(
    "FISH_VOICE_TURN_EXTENSIONS", DEFAULT_TURN_EXTENSIONS, lo=1, hi=_TURN_EXTENSIONS_MAX
)
_BARGE_FRAMES = IntSetting("FISH_VOICE_BARGE_FRAMES", DEFAULT_BARGE_HIT_FRAMES, lo=_ONE_FRAME)
_BARGE_RMS = FloatSetting("FISH_VOICE_BARGE_RMS", DEFAULT_BARGE_RMS, positive=True)
_PLAYING_GAIN = FloatSetting(
    "FISH_VOICE_BARGE_PLAYING_GAIN", DEFAULT_BARGE_PLAYING_GAIN, lo=_PLAYING_GAIN_LO
)
_BLEED_DELAY = FloatSetting("FISH_VOICE_BLEED_DELAY", DEFAULT_BLEED_DELAY_S, lo=_NO_WAIT_S)
_AEC_BLEED_DELAY = FloatSetting(
    "FISH_VOICE_AEC_BLEED_DELAY", DEFAULT_AEC_BLEED_DELAY_S, lo=_NO_WAIT_S
)
_COOLDOWN = FloatSetting("FISH_VOICE_COOLDOWN", DEFAULT_POST_SPEAK_COOLDOWN_S, lo=_NO_WAIT_S)
_AEC_WET = FloatSetting("FISH_VOICE_AEC_WET", DEFAULT_AEC_WET, lo=_WET_LO, hi=_WET_HI)
# Read within the outer steps first; ``AecTune`` then takes only a step itself.
_NOISE = IntSetting(
    "FISH_VOICE_NOISE_SUPPRESSION",
    DEFAULT_NOISE_SUPPRESSION_DB,
    lo=NOISE_SUPPRESSION_STEPS_DB[0],
    hi=NOISE_SUPPRESSION_STEPS_DB[-1],
)
_DEEPGRAM_REGION = ChoiceSetting(
    "FISH_VOICE_DEEPGRAM_REGION",
    DEFAULT_DEEPGRAM_REGION,
    DEEPGRAM_REGIONS,
    warning="fish-voice: ignoring {name}={value!r}, using {default} (one of {choices})",
)
_EOT = FloatSetting(
    "FISH_VOICE_EOT_THRESHOLD", DEFAULT_EOT_THRESHOLD, lo=_EOT_THRESHOLD_LO, hi=_EOT_THRESHOLD_HI
)
_STT: Final[ChoiceSetting[SttProvider]] = ChoiceSetting(
    "FISH_VOICE_STT", DEFAULT_STT, STT_PROVIDERS
)


@dataclass(frozen=True, slots=True)
class ListenTune:
    """Voice detection and utterance limits for hearing the user.

    Attributes
    ----------
    vad_aggressiveness : int
        How strict the voice detector is, 0 to 3. Higher calls fewer sounds speech.
        The barge-in gate uses it too.
    end_silence_frames : int
        Quiet 30 ms frames that end an utterance. At least 1.
    start_speech_frames : int
        Voiced frames needed to start one. At least 1.
    min_speech_rms : float
        Seed for the adaptive RMS floor, in int16 units. Above 0.
    pre_pad_frames : int
        Frames kept before the start, at least 0. Raised to the start run plus
        ``IMPULSE_START_EXTRA`` when it is shorter, or the mic would never trigger on a
        loud start.
    min_voiced_frames : int
        Voiced frames an utterance needs before it is sent to ASR, 1 to
        ``MAX_UTTERANCE_FRAMES``.
    vad_engine : str
        The voice detector: a name in ``vad.VAD_ENGINES`` or ``package.module:factory``.
        The barge-in gate uses it too.

    Raises
    ------
    ValueError
        When a value is out of the range ``from_env`` accepts.
    """

    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
    end_silence_frames: int = DEFAULT_END_SILENCE_FRAMES
    start_speech_frames: int = DEFAULT_START_SPEECH_FRAMES
    min_speech_rms: float = DEFAULT_MIN_SPEECH_RMS
    pre_pad_frames: int = DEFAULT_PRE_PAD_FRAMES
    min_voiced_frames: int = DEFAULT_MIN_VOICED_FRAMES
    vad_engine: str = DEFAULT_VAD_ENGINE

    def __post_init__(self) -> None:
        """Reject values the listener cannot use, and lengthen a pre-roll that is too short.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        _VAD.check("ListenTune.vad_aggressiveness", self.vad_aggressiveness)
        _SILENCE.check("ListenTune.end_silence_frames", self.end_silence_frames)
        _SPEECH.check("ListenTune.start_speech_frames", self.start_speech_frames)
        _MIN_RMS.check("ListenTune.min_speech_rms", self.min_speech_rms)
        _PRE_PAD.check("ListenTune.pre_pad_frames", self.pre_pad_frames)
        _MIN_VOICED.check("ListenTune.min_voiced_frames", self.min_voiced_frames)
        shortest = self.start_speech_frames + IMPULSE_START_EXTRA
        if self.pre_pad_frames < shortest:
            object.__setattr__(self, "pre_pad_frames", shortest)

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_VOICE_*`` listen keys.

        Returns
        -------
        ListenTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        start = _SPEECH.read()
        pad = _PRE_PAD.read()
        return cls(
            vad_aggressiveness=_VAD.read(),
            vad_engine=read_word("FISH_VOICE_VAD_ENGINE", DEFAULT_VAD_ENGINE),
            end_silence_frames=_SILENCE.read(),
            start_speech_frames=start,
            min_speech_rms=_MIN_RMS.read(),
            pre_pad_frames=pad,
            min_voiced_frames=_MIN_VOICED.read(),
        )


@dataclass(frozen=True, slots=True)
class TurnTune:
    """The end-of-turn check for the Fish speech recogniser.

    Attributes
    ----------
    enabled : bool
        Off by default. On, a short silence ends the clip and the words decide whether the
        turn is over.
    silence_frames : int
        Quiet 30 ms frames that end the first clip while the check is on. At least 1.
    extend_frames : int
        Quiet frames that end a continuation, at least 1. The wait for it to begin is this
        minus ``silence_frames``, so a turn that was not over never waits longer than the
        ordinary silence limit.
    max_extensions : int
        Continuations one turn may have, 1 or 2.

    Raises
    ------
    ValueError
        When a value is out of the range ``from_env`` accepts.
    """

    enabled: bool = False
    silence_frames: int = DEFAULT_TURN_SILENCE_FRAMES
    extend_frames: int = DEFAULT_END_SILENCE_FRAMES
    max_extensions: int = DEFAULT_TURN_EXTENSIONS

    def __post_init__(self) -> None:
        """Reject values the end-of-turn check cannot use.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        _TURN_SILENCE.check("TurnTune.silence_frames", self.silence_frames)
        _TURN_EXTEND.check("TurnTune.extend_frames", self.extend_frames)
        _TURN_EXTENSIONS.check("TurnTune.max_extensions", self.max_extensions)

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_VOICE_TURN_*`` keys.

        Returns
        -------
        TurnTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        return cls(
            enabled=read_flag("FISH_VOICE_TURN_CHECK", default=False),
            silence_frames=_TURN_SILENCE.read(),
            extend_frames=_TURN_EXTEND.read(),
            max_extensions=_TURN_EXTENSIONS.read(),
        )


@dataclass(frozen=True, slots=True)
class BargeTune:
    """Barge-in gate and the pauses around a spoken reply.

    Attributes
    ----------
    hit_frames : int
        Loud voiced 30 ms frames in a row that interrupt a reply. At least 1.
    min_rms : float
        Seed for the adaptive barge floor, in int16 units. Above 0.
    playing_gain : float
        Floor multiplier while the speaker plays and AEC3 is not running. At least 1.
    bleed_delay_s : float
        Seconds to ignore the mic after TTS starts when AEC3 is not running: echo
        cancellation is off, the ``aec`` extra is missing, or the processor failed to start.
    aec_bleed_delay_s : float
        Seconds to ignore the mic after TTS starts when AEC3 is running, in place of
        ``bleed_delay_s``. It is shorter, since AEC3 takes the speaker out of the mic.
    cooldown_s : float
        Seconds to wait after a reply before the mic opens again.

    Raises
    ------
    ValueError
        When a value is out of the range ``from_env`` accepts: ``hit_frames`` under 1,
        ``min_rms`` not above 0, ``playing_gain`` under 1, a negative delay or cooldown.
    """

    hit_frames: int = DEFAULT_BARGE_HIT_FRAMES
    min_rms: float = DEFAULT_BARGE_RMS
    playing_gain: float = DEFAULT_BARGE_PLAYING_GAIN
    bleed_delay_s: float = DEFAULT_BLEED_DELAY_S
    aec_bleed_delay_s: float = DEFAULT_AEC_BLEED_DELAY_S
    cooldown_s: float = DEFAULT_POST_SPEAK_COOLDOWN_S

    def __post_init__(self) -> None:
        """Reject values the barge gate cannot use.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        _BARGE_FRAMES.check("BargeTune.hit_frames", self.hit_frames)
        _BARGE_RMS.check("BargeTune.min_rms", self.min_rms)
        _PLAYING_GAIN.check("BargeTune.playing_gain", self.playing_gain)
        _BLEED_DELAY.check("BargeTune.bleed_delay_s", self.bleed_delay_s)
        _AEC_BLEED_DELAY.check("BargeTune.aec_bleed_delay_s", self.aec_bleed_delay_s)
        _COOLDOWN.check("BargeTune.cooldown_s", self.cooldown_s)

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_BARGE_*``, the two bleed delays and ``FISH_VOICE_COOLDOWN``.

        Returns
        -------
        BargeTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        return cls(
            hit_frames=_BARGE_FRAMES.read(),
            min_rms=_BARGE_RMS.read(),
            playing_gain=_PLAYING_GAIN.read(),
            bleed_delay_s=_BLEED_DELAY.read(),
            cooldown_s=_COOLDOWN.read(),
            aec_bleed_delay_s=_AEC_BLEED_DELAY.read(),
        )


@dataclass(frozen=True, slots=True)
class AecTune:
    """In-process mic cleanup: echo cancellation, automatic gain and noise suppression.

    Attributes
    ----------
    enabled : bool
        Echo cancellation (AEC3). False leaves the speaker's echo in the mic.
    wet : float
        Mix of echo-cancelled audio, 0 to 1, while the speaker plays.
    auto_gain : bool
        Raise a quiet voice to a steady level, as a call app does.
    noise_suppression_db : int
        Attenuation of steady background noise: 0 (off), 6, 12, 18 or 21.

    Raises
    ------
    ValueError
        When ``wet`` is outside 0 to 1 or ``noise_suppression_db`` is not one of the steps.
    """

    enabled: bool = True
    wet: float = DEFAULT_AEC_WET
    auto_gain: bool = True
    noise_suppression_db: int = DEFAULT_NOISE_SUPPRESSION_DB

    def __post_init__(self) -> None:
        """Reject a mix or a suppression level the audio processor cannot use.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        _AEC_WET.check("AecTune.wet", self.wet)
        if not _is_noise_step(self.noise_suppression_db):
            msg = (
                f"AecTune.noise_suppression_db must be one of {_steps()}, "
                f"not {self.noise_suppression_db}"
            )
            raise ValueError(msg)

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_AEC``, ``FISH_VOICE_AEC_WET``, ``FISH_VOICE_AGC`` and ``FISH_VOICE_NOISE_SUPPRESSION``.

        Returns
        -------
        AecTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        noise = _NOISE.read()
        if not _is_noise_step(noise):
            warn(f"fish-voice: {_NOISE.name} takes {_steps()}, using {_NOISE.default}")
            noise = _NOISE.default
        return cls(
            enabled=read_flag("FISH_VOICE_AEC", default=True),
            wet=_AEC_WET.read(),
            auto_gain=read_flag("FISH_VOICE_AGC", default=True),
            noise_suppression_db=noise,
        )

    @classmethod
    def off(cls) -> Self:
        """Return a tune with every stage off, so the mic passes through untouched.

        Returns
        -------
        AecTune
            No echo cancellation, gain or noise suppression.
        """
        return cls(enabled=False, auto_gain=False, noise_suppression_db=0)


def _steps() -> str:
    return ", ".join(map(str, NOISE_SUPPRESSION_STEPS_DB))


def _is_noise_step(db: int) -> bool:
    return db in NOISE_SUPPRESSION_STEPS_DB


@dataclass(frozen=True, slots=True)
class DeepgramTune:
    """How Deepgram's Flux is reached and how sure it must be that a turn is over.

    Only the ``deepgram`` recogniser reads it.

    Attributes
    ----------
    key : str
        The Deepgram API key. Empty unless the environment sets it.
    model : str
        The Flux model: ``flux-general-en``, or ``flux-general-multi`` for ten languages.
        Only a Flux model streams turns.
    region : str
        ``"global"``, or ``"eu"``, ``"au"`` or ``"in"`` to have the audio processed within the
        European Union, Australia or India. Deepgram fails a regional request rather than
        send it elsewhere. One of ``DEEPGRAM_HOSTS``.
    eot_threshold : float
        How sure Flux must be that the user has finished, from 0.5 to 1. Higher waits a little
        longer and cuts in less often.

    Raises
    ------
    ValueError
        When the model is not a Flux model, the region is unknown or the threshold is out
        of range.
    """

    key: str = field(default="", repr=False)
    model: str = DEFAULT_DEEPGRAM_MODEL
    region: str = DEFAULT_DEEPGRAM_REGION
    eot_threshold: float = DEFAULT_EOT_THRESHOLD

    def __post_init__(self) -> None:
        """Reject a model that cannot stream turns, an unknown region or threshold.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        if not _is_flux(self.model):
            msg = f"DeepgramTune.model must be a Flux model, not {self.model!r}"
            raise ValueError(msg)
        _DEEPGRAM_REGION.check("DeepgramTune.region", self.region)
        _EOT.check("DeepgramTune.eot_threshold", self.eot_threshold)

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_DEEPGRAM_*``, ``FISH_VOICE_EOT_THRESHOLD`` and ``DEEPGRAM_API_KEY``.

        Returns
        -------
        DeepgramTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        name = "FISH_VOICE_DEEPGRAM_MODEL"
        model = read_word(name, DEFAULT_DEEPGRAM_MODEL)
        if not _is_flux(model):
            warn(
                f"fish-voice: ignoring {name}={model!r}, "
                f"using {DEFAULT_DEEPGRAM_MODEL} (only Flux models stream turns)"
            )
            model = DEFAULT_DEEPGRAM_MODEL
        return cls(
            key=read_text("DEEPGRAM_API_KEY"),
            model=model,
            region=_DEEPGRAM_REGION.read(),
            eot_threshold=_EOT.read(),
        )


def _is_flux(model: str) -> bool:
    return model.startswith(_DEEPGRAM_MODEL_FAMILY)


def _eager_cap(provider: SttProvider, deepgram: DeepgramTune) -> float | None:
    """Return the end-of-turn threshold an early one may not pass, or None for no cap.

    Only Deepgram's Flux has one: Fish's realtime recogniser takes any early threshold
    above 0 as "on".
    """
    return deepgram.eot_threshold if provider == "deepgram" else None


def _eager_threshold(cap: float | None) -> float:
    """Read ``FISH_VOICE_EAGER_EOT_THRESHOLD``: 0 for off, else 0.3 to 0.9 and at most ``cap``."""
    name = "FISH_VOICE_EAGER_EOT_THRESHOLD"
    # Anything from 0 up reads, so a value just under the range is raised to it, not dropped.
    value = read_float(name, _EAGER_EOT_OFF, lo=_EAGER_EOT_OFF, hi=_EAGER_EOT_HI)
    if value == _EAGER_EOT_OFF:
        return _EAGER_EOT_OFF
    if value < _EAGER_EOT_LO:
        warn(f"fish-voice: {name} needs {_EAGER_EOT_LO} to {_EAGER_EOT_HI}, using {_EAGER_EOT_LO}")
        value = _EAGER_EOT_LO
    if cap is not None and value > cap:
        warn(f"fish-voice: {name} is above {_EOT.name}, using {cap}")
        value = cap
    return value


@dataclass(frozen=True, slots=True)
class SttTune:
    """Which speech recognition hears the user, and the settings of the streaming ones.

    Attributes
    ----------
    provider : {"fish", "deepgram", "fish-realtime"}
        ``"fish"`` sends one clip to Fish ASR when the user stops speaking. ``"deepgram"``
        streams the speech to Deepgram's Flux while the user talks, which decides for itself
        when the turn is over. ``"fish-realtime"`` does the same with Fish's realtime speech
        to text (``transcribe-stream-1``, in preview), on the Fish key.
    deepgram : DeepgramTune
        Deepgram's key, model, region and end-of-turn threshold.
    eager_eot_threshold : float
        0 (the default) is off. From 0.3 to 0.9 it asks the streaming recogniser to say when
        a turn is probably over, a little before it is sure, so the reply can start being
        written early. Costs 50 to 70 percent more model calls. With Deepgram it is no higher
        than Deepgram's ``eot_threshold``. Fish's realtime recogniser has no threshold: any
        value above 0 turns its early end of turn on, so it is not capped there.
    save_dir : str
        A folder to write each streamed turn's audio to, as a 16 kHz WAV, to hear what the
        streaming recogniser was sent. Empty (the default) saves nothing.

    Raises
    ------
    ValueError
        When the provider is unknown or the early threshold is neither 0 nor in its range.
    """

    provider: SttProvider = DEFAULT_STT
    deepgram: DeepgramTune = field(default_factory=DeepgramTune)
    eager_eot_threshold: float = _EAGER_EOT_OFF
    save_dir: str = ""

    def __post_init__(self) -> None:
        """Reject an unknown recogniser or an early threshold ``from_env`` would not give.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        _STT.check("SttTune.provider", self.provider)
        if self.eager_eot_threshold != _EAGER_EOT_OFF:
            cap = _eager_cap(self.provider, self.deepgram)
            check_range(
                "SttTune.eager_eot_threshold",
                self.eager_eot_threshold,
                lo=_EAGER_EOT_LO,
                hi=_EAGER_EOT_HI if cap is None else min(_EAGER_EOT_HI, cap),
            )

    @property
    def eager(self) -> bool:
        """Whether the early end of turn is asked for."""
        return self.eager_eot_threshold > 0

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_STT``, the Deepgram and end-of-turn keys, and ``DEEPGRAM_API_KEY``.

        Returns
        -------
        SttTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        provider = _STT.read()
        deepgram = DeepgramTune.from_env()
        return cls(
            provider=provider,
            deepgram=deepgram,
            eager_eot_threshold=_eager_threshold(_eager_cap(provider, deepgram)),
            save_dir=read_text("FISH_VOICE_STT_SAVE_DIR"),
        )
