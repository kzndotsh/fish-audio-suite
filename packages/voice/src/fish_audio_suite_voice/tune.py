"""Frozen duplex tuning read once from the environment.

Every knob lives on one of four dataclasses. ``from_env`` validates each key and
warns when a value is unusable, then uses the default. Library classes take a
tune object and never read the environment themselves.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Final, Literal, Self
from urllib.parse import urlsplit

from fish_audio_suite_kit import env_token, strip_base
from fish_audio_suite_voice.debug import warn

__all__ = [
    "DEFAULT_AEC_BLEED_S",
    "DEFAULT_BARGE_HIT_FRAMES",
    "DEFAULT_BARGE_OVER",
    "DEFAULT_BARGE_RMS",
    "DEFAULT_BLEED_DELAY_S",
    "DEFAULT_HISTORY_TURNS",
    "DEFAULT_MIN_SPEECH_RMS",
    "DEFAULT_MIN_VOICED_FRAMES",
    "DEFAULT_POST_SPEAK_COOLDOWN_S",
    "DEFAULT_SILENCE_FRAMES_END",
    "DEFAULT_VAD_AGGRESSIVENESS",
    "EXPERIENTIAL_API_BASE",
    "IMPULSE_START_EXTRA",
    "LLM_PROVIDERS",
    "MAX_UTTERANCE_FRAMES",
    "OPENROUTER_API_BASE",
    "REASONING_EFFORTS",
    "AecTune",
    "BargeTune",
    "ListenTune",
    "LlmBackendName",
    "LlmProvider",
    "LlmProviderName",
    "LlmTune",
    "openrouter_host",
    "provider_for_base",
    "read_flag",
    "read_float",
    "read_int",
]

DEFAULT_VAD_AGGRESSIVENESS: Final = 1
VAD_MODE_HI: Final = 3
DEFAULT_SILENCE_FRAMES_END: Final = 40
DEFAULT_SPEECH_FRAMES_START: Final = 4
DEFAULT_MIN_SPEECH_RMS: Final = 200.0
DEFAULT_PRE_PAD_FRAMES: Final = 20
DEFAULT_MIN_VOICED_FRAMES: Final = 12
MAX_UTTERANCE_FRAMES: Final = 500
IMPULSE_START_EXTRA: Final = 6

DEFAULT_BARGE_HIT_FRAMES: Final = 10
DEFAULT_BARGE_RMS: Final = 220.0
DEFAULT_BARGE_OVER: Final = 2.2
DEFAULT_BLEED_DELAY_S: Final = 0.9
DEFAULT_POST_SPEAK_COOLDOWN_S: Final = 0.8

DEFAULT_AEC_BLEED_S: Final = 0.3
DEFAULT_AEC_WET: Final = 0.85

OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"
EXPERIENTIAL_API_BASE: Final = "https://api.experientiallabs.ai/v1"
# The values Experiential documents for reasoning_effort. OpenAI's own names are a
# subset, so they pass too. Which of them a route accepts is the provider's call.
REASONING_EFFORTS: Final = frozenset({"none", "minimal", "low", "medium", "high", "max"})
DEFAULT_LLM_MAX_TOKENS: Final = 1200
DEFAULT_LLM_TEMPERATURE: Final = 0.8
DEFAULT_LLM_TIMEOUT_S: Final = 120.0
DEFAULT_LLM_REFERER: Final = "https://github.com/kzndotsh/fish-audio-suite"
DEFAULT_LLM_TITLE: Final = "fish-audio-suite-voice"
DEFAULT_LLM_CATEGORIES: Final = "cli-agent"
DEFAULT_HISTORY_TURNS: Final = 20
_LLM_TEMPERATURE_HI: Final = 2.0
_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off"})


def _raw(name: str) -> str | None:
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
    raw = _raw(name)
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
    raw = _raw(name)
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
    raw = _raw(name)
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


LlmBackendName = Literal["openrouter", "openai"]
LlmProviderName = Literal["openrouter", "experiential", "custom"]


@dataclass(frozen=True, slots=True)
class LlmProvider:
    """A chat provider this project knows by name.

    Attributes
    ----------
    name : {"openrouter", "experiential"}
        Value of ``FISH_LLM_PROVIDER``.
    base : str
        Default API base, used when ``FISH_LLM_BASE`` is not set.
    host : str
        The domain the base lives on. Its subdomains count too. A base on this
        host is this provider, which is what decides whose key it may receive.
    key_env : str
        The only environment variable, besides ``FISH_LLM_KEY``, that may supply
        this provider's key.
    model_env : str
        Optional model for this provider, so two providers can share one ``.env``.
    """

    name: Literal["openrouter", "experiential"]
    base: str
    host: str
    key_env: str
    model_env: str


LLM_PROVIDERS: Final[tuple[LlmProvider, ...]] = (
    LlmProvider(
        "openrouter",
        OPENROUTER_API_BASE,
        "openrouter.ai",
        "OPENROUTER_API_KEY",
        "FISH_LLM_MODEL_OPENROUTER",
    ),
    LlmProvider(
        "experiential",
        EXPERIENTIAL_API_BASE,
        "experientiallabs.ai",
        "EXPLABS_API_KEY",
        "FISH_LLM_MODEL_EXPERIENTIAL",
    ),
)


def provider_for_base(base: str) -> LlmProvider | None:
    """Find the named provider that hosts ``base``.

    Parameters
    ----------
    base : str
        LLM API base URL.

    Returns
    -------
    LlmProvider or None
        The provider whose domain, or a subdomain of it, is the host. None for any
        other host, and for a look-alike path or query.

    Examples
    --------
    >>> provider_for_base("https://api.experientiallabs.ai/v1").name
    'experiential'
    >>> provider_for_base("https://example.test/experientiallabs.ai") is None
    True
    """
    try:
        host = (urlsplit(base.strip()).hostname or "").lower()
    except ValueError:
        return None
    for provider in LLM_PROVIDERS:
        if host == provider.host or host.endswith(f".{provider.host}"):
            return provider
    return None


@dataclass(frozen=True, slots=True)
class ListenTune:
    """Voice-activity and utterance limits for one microphone capture.

    Attributes
    ----------
    vad_aggressiveness : int
        WebRTC VAD mode, 0 to 3.
    silence_frames_end : int
        Quiet 30 ms frames that end an utterance.
    speech_frames_start : int
        Voiced frames needed to start one.
    min_speech_rms : float
        Seed for the adaptive RMS floor, in int16 units.
    pre_pad_frames : int
        Frames kept before the start. Always at least the start run plus the
        impulse extra, or the mic would never trigger.
    min_voiced : int
        Voiced frames an utterance needs before it is sent to ASR.
    """

    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
    silence_frames_end: int = DEFAULT_SILENCE_FRAMES_END
    speech_frames_start: int = DEFAULT_SPEECH_FRAMES_START
    min_speech_rms: float = DEFAULT_MIN_SPEECH_RMS
    pre_pad_frames: int = DEFAULT_PRE_PAD_FRAMES
    min_voiced: int = DEFAULT_MIN_VOICED_FRAMES

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_VOICE_*`` listen keys.

        Returns
        -------
        ListenTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        start = read_int("FISH_VOICE_SPEECH_FRAMES", DEFAULT_SPEECH_FRAMES_START, lo=1)
        pad = read_int("FISH_VOICE_PRE_PAD", DEFAULT_PRE_PAD_FRAMES, lo=0)
        return cls(
            vad_aggressiveness=read_int(
                "FISH_VOICE_VAD", DEFAULT_VAD_AGGRESSIVENESS, lo=0, hi=VAD_MODE_HI
            ),
            silence_frames_end=read_int(
                "FISH_VOICE_SILENCE_FRAMES", DEFAULT_SILENCE_FRAMES_END, lo=1
            ),
            speech_frames_start=start,
            min_speech_rms=read_float("FISH_VOICE_MIN_RMS", DEFAULT_MIN_SPEECH_RMS, positive=True),
            pre_pad_frames=max(pad, start + IMPULSE_START_EXTRA),
            min_voiced=read_int(
                "FISH_VOICE_MIN_VOICED",
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
    over : float
        Floor multiplier while the speaker plays and AEC is off. At least 1.
    bleed_delay_s : float
        Seconds to ignore the mic after TTS starts when AEC is off.
    cooldown_s : float
        Seconds to wait after a reply before the mic opens again.
    """

    hit_frames: int = DEFAULT_BARGE_HIT_FRAMES
    min_rms: float = DEFAULT_BARGE_RMS
    over: float = DEFAULT_BARGE_OVER
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
            over=read_float("FISH_VOICE_BARGE_OVER", DEFAULT_BARGE_OVER, lo=1.0),
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
    bleed_s : float
        Seconds to ignore the mic after TTS starts when AEC3 is loaded.
    """

    enabled: bool = True
    wet: float = DEFAULT_AEC_WET
    bleed_s: float = DEFAULT_AEC_BLEED_S

    @classmethod
    def from_env(cls) -> Self:
        """Build from ``FISH_VOICE_AEC``, ``FISH_VOICE_AEC_WET``, ``FISH_VOICE_AEC_BLEED``.

        Returns
        -------
        AecTune
            Validated tune. A bad value is replaced by its default with a warning.
        """
        return cls(
            enabled=read_flag("FISH_VOICE_AEC", default=True),
            wet=read_float("FISH_VOICE_AEC_WET", DEFAULT_AEC_WET, lo=0.0, hi=1.0),
            bleed_s=read_float("FISH_VOICE_AEC_BLEED", DEFAULT_AEC_BLEED_S, lo=0.0),
        )


def openrouter_host(base: str) -> bool:
    """Return whether ``base`` is hosted on ``openrouter.ai``.

    Parameters
    ----------
    base : str
        LLM API base URL.

    Returns
    -------
    bool
        True for ``openrouter.ai`` and its subdomains. A look-alike path or
        query does not count.
    """
    try:
        host = (urlsplit(base.strip()).hostname or "").lower()
    except ValueError:
        return False
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


@dataclass(frozen=True, slots=True)
class LlmTune:
    """Chat backend choice and request settings.

    Attributes
    ----------
    backend : {"openrouter", "openai"}
        ``openai`` (any chat-completions server over httpx) or ``openrouter``
        (the OpenRouter SDK). Experiential and any other server use ``openai``.
    base : str
        API origin without a trailing slash.
    key : str
        Bearer token. Empty until the environment sets one.
    model : str
        Model id. Empty until the environment sets one.
    temperature : float
        Sampling temperature.
    timeout_s : float
        Per-request timeout.
    max_tokens : int
        Completion cap.
    nitro : bool
        OpenRouter only. Adds ``:nitro`` to the model and sorts providers.
    provider_sort : str
        OpenRouter provider sort sent when ``nitro`` is on.
    referer, title, categories : str
        OpenRouter attribution. Empty disables the field.
    continuation : bool
        Send one more request when a reply stops before a sentence end.
    reasoning_effort : str
        Sent as ``reasoning_effort`` on the ``openai`` backend. Empty omits it, and
        the provider's default applies. A reasoning model that defaults to a high
        effort can take seconds to the first word, so voice use wants ``low``.
    """

    backend: LlmBackendName = "openrouter"
    base: str = OPENROUTER_API_BASE
    key: str = field(default="", repr=False)
    model: str = ""
    temperature: float = DEFAULT_LLM_TEMPERATURE
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    nitro: bool = False
    provider_sort: str = "throughput"
    referer: str = DEFAULT_LLM_REFERER
    title: str = DEFAULT_LLM_TITLE
    categories: str = DEFAULT_LLM_CATEGORIES
    continuation: bool = False
    reasoning_effort: str = ""

    @property
    def provider(self) -> LlmProviderName:
        """Which provider ``base`` belongs to, or ``custom`` for any other host."""
        found = provider_for_base(self.base)
        return found.name if found is not None else "custom"

    @property
    def openrouter(self) -> bool:
        """Whether this tune selects the OpenRouter SDK."""
        return self.backend == "openrouter"

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_LLM_*`` keys and their provider fallbacks.

        Returns
        -------
        LlmTune
            ``FISH_LLM_PROVIDER`` names a provider and supplies its default base,
            and ``FISH_LLM_BASE`` overrides the base. The provider is then read
            from the final base's host, so a key can only come from the variable
            that belongs to that host (``OPENROUTER_API_KEY``, ``EXPLABS_API_KEY``),
            or ``OPENAI_API_KEY`` for any other server. ``FISH_LLM_BACKEND`` picks
            the backend. Unset, it follows the base URL host.
        """
        named = _named_provider()
        if named is not None:
            # A named provider is only overridden by FISH_LLM_BASE. The older
            # OPENROUTER_BASE_URL alias must not point its key at another host.
            base = _first_base("FISH_LLM_BASE") or named.base
        else:
            base = _first_base("FISH_LLM_BASE", "OPENROUTER_BASE_URL") or OPENROUTER_API_BASE
        provider = provider_for_base(base)
        backend = _backend(base)
        if provider is not None:
            key_env = provider.key_env
        else:
            key_env = "OPENROUTER_API_KEY" if backend == "openrouter" else "OPENAI_API_KEY"
        model_names = [provider.model_env] if provider is not None else []
        model_names.append("FISH_LLM_MODEL")
        if provider is None or provider.name == "openrouter":
            model_names.append("OPENROUTER_MODEL")
        return cls(
            backend=backend,
            base=base,
            key=_first_text("FISH_LLM_KEY", key_env),
            model=_first_token(*model_names),
            temperature=read_float(
                "FISH_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, lo=0.0, hi=_LLM_TEMPERATURE_HI
            ),
            timeout_s=read_float("FISH_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT_S, positive=True),
            max_tokens=read_int("FISH_LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, lo=1),
            nitro=read_flag("FISH_LLM_NITRO", default=False),
            provider_sort=read_text("FISH_LLM_PROVIDER_SORT", "throughput"),
            referer=read_text("FISH_LLM_REFERER", DEFAULT_LLM_REFERER),
            title=read_text("FISH_LLM_TITLE", DEFAULT_LLM_TITLE),
            categories=read_text("FISH_LLM_CATEGORIES", DEFAULT_LLM_CATEGORIES),
            continuation=read_flag("FISH_LLM_CONTINUE", default=False),
            reasoning_effort=_reasoning_effort(),
        )


def _named_provider() -> LlmProvider | None:
    raw = _raw("FISH_LLM_PROVIDER")
    if raw is None:
        return None
    low = raw.lower()
    for provider in LLM_PROVIDERS:
        if provider.name == low:
            return provider
    if low != "custom":
        names = ", ".join(provider.name for provider in LLM_PROVIDERS)
        warn(f"fish-voice: unknown FISH_LLM_PROVIDER={raw!r} (use {names}), reading FISH_LLM_BASE")
    return None


def _reasoning_effort() -> str:
    raw = _raw("FISH_LLM_REASONING_EFFORT")
    if raw is None:
        return ""
    low = raw.lower()
    if low in REASONING_EFFORTS:
        return low
    values = ", ".join(sorted(REASONING_EFFORTS))
    warn(f"fish-voice: FISH_LLM_REASONING_EFFORT={raw!r} is not one of {values}, leaving it unset")
    return ""


def _first_base(*names: str) -> str:
    for name in names:
        raw = _raw(name)
        if raw is None:
            continue
        text = strip_base(raw)
        if not text:
            continue
        try:
            urlsplit(text).hostname  # noqa: B018 - raises ValueError for a bad bracketed host
        except ValueError:
            warn(f"fish-voice: {name} is not a valid URL, trying the next base or the default")
            continue
        return text
    return ""


def _first_text(*names: str) -> str:
    # A blank value is skipped, so an empty FISH_LLM_KEY from the copied
    # .env.example does not hide a real provider key.
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _first_token(*names: str) -> str:
    for name in names:
        raw = env_token(name, "")
        if raw:
            return raw
    return ""


def _backend(base: str) -> LlmBackendName:
    raw = _raw("FISH_LLM_BACKEND")
    if raw is not None:
        low = raw.lower()
        if low == "openai":
            return "openai"
        if low == "openrouter":
            host = provider_for_base(base)
            if host is not None and host.name != "openrouter":
                # The SDK backend only fits OpenRouter. A setting left over from
                # before another provider existed must not point it elsewhere.
                warn(
                    f"fish-voice: FISH_LLM_BACKEND=openrouter does not fit {host.name}, "
                    "using the openai backend"
                )
                return "openai"
            return "openrouter"
        warn(f"fish-voice: unknown FISH_LLM_BACKEND={raw!r}, choosing from the base URL")
    return "openrouter" if openrouter_host(base) else "openai"
