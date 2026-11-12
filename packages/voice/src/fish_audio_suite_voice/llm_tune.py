"""The chat model's settings, read once from the environment.

They cover the provider, base URL, key and model, sampling, and OpenRouter's
routing options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Self
from urllib.parse import urlsplit

from fish_audio_suite_kit import strip_base
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.env_read import (
    ChoiceSetting,
    FloatSetting,
    IntSetting,
    read_choice,
    read_first,
    read_flag,
    read_raw,
    read_text,
)

__all__ = [
    "DEFAULT_LLM_PROVIDER_SORT",
    "EXPERIENTIAL_API_BASE",
    "LLM_BASE_ENV",
    "LLM_PROVIDERS",
    "OPENAI_KEY_ENV",
    "OPENROUTER_API_BASE",
    "KnownLlmProvider",
    "LlmBackendName",
    "LlmProvider",
    "LlmProviderName",
    "LlmTune",
    "is_openrouter_host",
    "llm_problem",
    "provider_for_base",
]


OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"


EXPERIENTIAL_API_BASE: Final = "https://api.experientiallabs.ai/v1"


# The variables named in more than one place: a read, a warning, a startup problem.
LLM_BASE_ENV: Final = "FISH_LLM_BASE"
_LLM_KEY_ENV: Final = "FISH_LLM_API_KEY"
_LLM_MODEL_ENV: Final = "FISH_LLM_MODEL"
_LLM_PROVIDER_ENV: Final = "FISH_LLM_PROVIDER"
# What a request leaves out when a choice below is empty: the provider decides.
_UNSET: Final = ""

# The values Experiential documents for reasoning_effort. OpenAI's own names are a
# subset, so they pass too. Which of them a route accepts is the provider's call.
_REASONING_EFFORTS: Final = ("none", "minimal", "low", "medium", "high", "max")
# OpenRouter's provider.sort values. Voice wants the first token soon, so the
# default is latency; OpenRouter's own default weighs price. ``off`` sends none.
_PROVIDER_SORTS: Final = ("latency", "throughput", "price")
_NO_SORT: Final = {"off": _UNSET, "none": _UNSET}
DEFAULT_LLM_PROVIDER_SORT: Final = "latency"
# OpenRouter's service_tier values. Unset leaves the tier to OpenRouter. priority
# costs more for steadier speed, and ultrafast (select OpenAI models only) far
# more, falling back to priority and then default where it is not offered.
_SERVICE_TIERS: Final = ("default", "flex", "priority", "ultrafast")
# The warning for an unknown word in a choice that is left out when unset.
_NOT_ONE_OF: Final = "fish-voice: {name}={raw!r} is not one of {sorted}, leaving it unset"


DEFAULT_LLM_MAX_TOKENS: Final = 1200


DEFAULT_LLM_TEMPERATURE: Final = 0.8


DEFAULT_LLM_TIMEOUT_S: Final = 120.0


DEFAULT_LLM_REFERER: Final = "https://github.com/kzndotsh/fish-audio-suite"


DEFAULT_LLM_TITLE: Final = "fish-audio-suite-voice"


DEFAULT_LLM_CATEGORIES: Final = "cli-agent"


_LLM_TEMPERATURE_LO: Final = 0.0
_LLM_TEMPERATURE_HI: Final = 2.0
_LLM_MAX_TOKENS_LO: Final = 1
# Read by ``from_env`` and checked by ``LlmTune`` with the same bounds.
_TEMPERATURE = FloatSetting(
    "FISH_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, lo=_LLM_TEMPERATURE_LO, hi=_LLM_TEMPERATURE_HI
)
_TIMEOUT = FloatSetting("FISH_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT_S, positive=True)
_MAX_TOKENS = IntSetting("FISH_LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, lo=_LLM_MAX_TOKENS_LO)
_PROVIDER_SORT = ChoiceSetting(
    "FISH_LLM_PROVIDER_SORT",
    DEFAULT_LLM_PROVIDER_SORT,
    _PROVIDER_SORTS,
    aliases=_NO_SORT,
    warning="fish-voice: {name}={raw!r} is not one of {sorted} or off, using {default}",
)
_SERVICE_TIER = ChoiceSetting("FISH_LLM_SERVICE_TIER", _UNSET, _SERVICE_TIERS, warning=_NOT_ONE_OF)
_REASONING_EFFORT = ChoiceSetting(
    "FISH_LLM_REASONING_EFFORT", _UNSET, _REASONING_EFFORTS, warning=_NOT_ONE_OF
)


LlmBackendName = Literal["openrouter", "openai"]


# The providers this project knows by name. ``LLM_PROVIDERS`` holds one entry for each.
KnownLlmProvider = Literal["openrouter", "experiential"]


LlmProviderName = KnownLlmProvider | Literal["custom"]


# The key a server this project does not know by name may take, besides FISH_LLM_API_KEY.
OPENAI_KEY_ENV: Final = "OPENAI_API_KEY"


@dataclass(frozen=True, slots=True)
class LlmProvider:
    """A chat provider this project knows by name.

    Attributes
    ----------
    name : KnownLlmProvider
        Value of ``FISH_LLM_PROVIDER``.
    base : str
        Default API base, used when ``FISH_LLM_BASE`` is not set.
    host : str
        The domain the base lives on. Its subdomains count too. A base on this
        host is this provider, which is what decides whose key it may receive.
    key_env : str
        The only environment variable, besides ``FISH_LLM_API_KEY``, that may supply
        this provider's key.
    model_env : str
        Optional model for this provider, so two providers can share one ``.env``.
    """

    name: KnownLlmProvider
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


def is_openrouter_host(base: str) -> bool:
    """Return whether ``base`` is hosted on ``openrouter.ai``.

    Parameters
    ----------
    base : str
        LLM API base URL.

    Returns
    -------
    bool
        True when ``provider_for_base`` finds the ``openrouter`` provider: its host
        and its subdomains. A look-alike path or query does not count.
    """
    found = provider_for_base(base)
    return found is not None and found.name == "openrouter"


@dataclass(frozen=True, slots=True)
class LlmTune:
    """Chat backend choice and request settings.

    Attributes
    ----------
    base : str
        API base URL without a trailing slash. Its host decides the backend
        (``backend``) and the provider (``provider``).
    api_key : str
        Bearer token. Empty until the environment sets one.
    model : str
        Model id. Empty until the environment sets one.
    temperature : float
        Sampling temperature.
    timeout_s : float
        Per-request timeout in seconds.
    max_tokens : int
        Completion cap.
    nitro : bool
        OpenRouter only. Adds ``:nitro`` to the model.
    provider_sort : str
        OpenRouter only. Sent as ``provider.sort``: ``latency`` (default),
        ``throughput`` or ``price``. Empty sends no sort, so OpenRouter's own
        price-weighted routing applies. It only matters for a model that more
        than one provider serves.
    service_tier : str
        OpenRouter only. Sent as ``service_tier``: ``default``, ``flex``,
        ``priority`` or ``ultrafast``. Empty sends none, and OpenRouter picks.
    referer, title, categories : str
        OpenRouter attribution. Empty disables the field.
    continuation : bool
        Send one more request when a reply stops before a sentence end.
    reasoning_effort : str
        Sent as ``reasoning_effort`` on the ``openai`` backend and as
        ``reasoning.effort`` to OpenRouter. Empty omits it, and the provider's
        default applies. A reasoning model that defaults to a high
        effort can take seconds to the first word, so voice use wants ``low``.

    Raises
    ------
    ValueError
        When a value is out of the range ``from_env`` accepts: a temperature outside 0 to 2,
        a timeout not above 0, ``max_tokens`` under 1, or a sort, tier or effort that is
        not one of the listed words or empty.
    """

    base: str = OPENROUTER_API_BASE
    api_key: str = field(default="", repr=False)
    model: str = ""
    temperature: float = DEFAULT_LLM_TEMPERATURE
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    nitro: bool = False
    provider_sort: str = DEFAULT_LLM_PROVIDER_SORT
    service_tier: str = ""
    referer: str = DEFAULT_LLM_REFERER
    title: str = DEFAULT_LLM_TITLE
    categories: str = DEFAULT_LLM_CATEGORIES
    continuation: bool = False
    reasoning_effort: str = ""

    def __post_init__(self) -> None:
        """Reject request settings the providers would refuse.

        Raises
        ------
        ValueError
            When a value is out of range.
        """
        _TEMPERATURE.check("LlmTune.temperature", self.temperature)
        _TIMEOUT.check("LlmTune.timeout_s", self.timeout_s)
        _MAX_TOKENS.check("LlmTune.max_tokens", self.max_tokens)
        _PROVIDER_SORT.check("LlmTune.provider_sort", self.provider_sort)
        _SERVICE_TIER.check("LlmTune.service_tier", self.service_tier)
        _REASONING_EFFORT.check("LlmTune.reasoning_effort", self.reasoning_effort)

    @property
    def provider(self) -> LlmProviderName:
        """Which provider ``base`` belongs to, or ``custom`` for any other host."""
        found = provider_for_base(self.base)
        return found.name if found is not None else "custom"

    @property
    def backend(self) -> LlmBackendName:
        """Which client talks to ``base``: ``openrouter`` (the OpenRouter SDK) or ``openai``.

        The OpenRouter SDK only talks to OpenRouter, so the host decides: an OpenRouter
        host gets the SDK, and every other server (Experiential, a self-hosted one) the
        OpenAI-compatible httpx client.
        """
        return "openrouter" if is_openrouter_host(self.base) else "openai"

    @property
    def uses_openrouter_sdk(self) -> bool:
        """Whether these settings select the OpenRouter SDK backend."""
        return self.backend == "openrouter"

    @classmethod
    def from_env(cls) -> Self:
        """Build from the ``FISH_LLM_*`` keys and their provider fallbacks.

        Returns
        -------
        LlmTune
            ``FISH_LLM_PROVIDER`` names a provider and supplies its default base,
            and ``FISH_LLM_BASE`` overrides the base. The provider is then read
            from the final base's host. ``FISH_LLM_API_KEY`` comes first. Past it,
            a key can only come from the variable that belongs to that host
            (``OPENROUTER_API_KEY``, ``EXPLABS_API_KEY``), or ``OPENAI_API_KEY``
            for any other server, and a known provider's key is never sent to a
            base that is not https. The backend follows the host. OpenRouter gets
            the OpenRouter SDK, and everything else the OpenAI-compatible client.
        """
        named = _named_provider()
        base = _first_base(LLM_BASE_ENV) or (
            named.base if named is not None else OPENROUTER_API_BASE
        )
        provider = provider_for_base(base)
        key_env = _key_env(provider)
        key_names = [_LLM_KEY_ENV]
        if provider is None or _is_https(base):
            key_names.append(key_env)
        elif read_raw(key_env) is not None:
            warn(
                f"fish-voice: {key_env} is not used because the base is not https, which would "
                f"send it unencrypted. Use an https base, or set {_LLM_KEY_ENV} to send a key anyway."
            )
        model_names = _model_names(provider)
        return cls(
            base=base,
            api_key=read_first(*key_names),
            model=read_first(*model_names),
            temperature=_TEMPERATURE.read(),
            timeout_s=_TIMEOUT.read(),
            max_tokens=_MAX_TOKENS.read(),
            nitro=read_flag("FISH_LLM_NITRO", default=False),
            provider_sort=_PROVIDER_SORT.read(),
            service_tier=_SERVICE_TIER.read(),
            referer=read_text("FISH_LLM_REFERER", DEFAULT_LLM_REFERER),
            title=read_text("FISH_LLM_TITLE", DEFAULT_LLM_TITLE),
            categories=read_text("FISH_LLM_CATEGORIES", DEFAULT_LLM_CATEGORIES),
            continuation=read_flag("FISH_LLM_CONTINUE", default=False),
            reasoning_effort=_REASONING_EFFORT.read(),
        )


def _key_env(provider: LlmProvider | None) -> str:
    """Return the variable, besides ``FISH_LLM_API_KEY``, that may hold the key for a base."""
    # An OpenRouter host is always a known provider, so an unknown host is OpenAI-compatible.
    return provider.key_env if provider is not None else OPENAI_KEY_ENV


def _model_names(provider: LlmProvider | None) -> list[str]:
    """Return the variables the model is read from, first match wins."""
    return [*([provider.model_env] if provider is not None else []), _LLM_MODEL_ENV]


def llm_problem(tune: LlmTune) -> str | None:
    """Say which LLM setting a session cannot start without, naming where it is read from.

    Parameters
    ----------
    tune : LlmTune
        The chat settings.

    Returns
    -------
    str or None
        ``FISH_LLM_API_KEY / <the provider's key variable> missing`` or
        ``FISH_LLM_MODEL / <the model variables> missing``, or None when both are set.
    """
    provider = provider_for_base(tune.base)
    if not tune.api_key:
        return f"{_LLM_KEY_ENV} / {_key_env(provider)} missing"
    if not tune.model:
        return f"{_LLM_MODEL_ENV} / {' / '.join(_model_names(provider))} missing"
    return None


def _named_provider() -> LlmProvider | None:
    # ``custom`` names no provider, the same as leaving it unset: the base decides.
    name = read_choice(
        _LLM_PROVIDER_ENV,
        _UNSET,
        tuple(provider.name for provider in LLM_PROVIDERS),
        aliases={"custom": _UNSET},
        warning="fish-voice: unknown {name}={raw!r} (use {choices}), reading " + LLM_BASE_ENV,
    )
    return next((provider for provider in LLM_PROVIDERS if provider.name == name), None)


def _is_https(base: str) -> bool:
    try:
        return urlsplit(base.strip()).scheme.lower() == "https"
    except ValueError:
        return False


def _first_base(*names: str) -> str:
    for name in names:
        raw = read_raw(name)
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
