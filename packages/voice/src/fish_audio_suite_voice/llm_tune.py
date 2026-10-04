"""The chat model's settings, read once from the environment: provider, base URL, key and sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final, Literal, Self
from urllib.parse import urlsplit

from fish_audio_suite_kit import env_token, strip_base
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.tune import read_flag, read_float, read_int, read_raw, read_text

__all__ = [
    "DEFAULT_HISTORY_TURNS",
    "DEFAULT_LLM_PROVIDER_SORT",
    "EXPERIENTIAL_API_BASE",
    "LLM_PROVIDERS",
    "OPENROUTER_API_BASE",
    "PROVIDER_SORTS",
    "REASONING_EFFORTS",
    "LlmBackendName",
    "LlmProvider",
    "LlmProviderName",
    "LlmTune",
    "is_openrouter_host",
    "provider_for_base",
]


OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"


EXPERIENTIAL_API_BASE: Final = "https://api.experientiallabs.ai/v1"


# The values Experiential documents for reasoning_effort. OpenAI's own names are a
# subset, so they pass too. Which of them a route accepts is the provider's call.
REASONING_EFFORTS: Final = frozenset({"none", "minimal", "low", "medium", "high", "max"})
# OpenRouter's provider.sort values. Voice wants the first token soon, so the
# default is latency; OpenRouter's own default weighs price.
PROVIDER_SORTS: Final = frozenset({"latency", "throughput", "price"})
DEFAULT_LLM_PROVIDER_SORT: Final = "latency"


DEFAULT_LLM_MAX_TOKENS: Final = 1200


DEFAULT_LLM_TEMPERATURE: Final = 0.8


DEFAULT_LLM_TIMEOUT_S: Final = 120.0


DEFAULT_LLM_REFERER: Final = "https://github.com/kzndotsh/fish-audio-suite"


DEFAULT_LLM_TITLE: Final = "fish-audio-suite-voice"


DEFAULT_LLM_CATEGORIES: Final = "cli-agent"


DEFAULT_HISTORY_TURNS: Final = 20


_LLM_TEMPERATURE_HI: Final = 2.0


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
        The only environment variable, besides ``FISH_LLM_API_KEY``, that may supply
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


def is_openrouter_host(base: str) -> bool:
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
    api_key : str
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
        OpenRouter only. Adds ``:nitro`` to the model.
    provider_sort : str
        OpenRouter only. Sent as ``provider.sort``: ``latency`` (default),
        ``throughput`` or ``price``. Empty sends no sort, so OpenRouter's own
        price-weighted routing applies. It only matters for a model that more
        than one provider serves.
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
    api_key: str = field(default="", repr=False)
    model: str = ""
    temperature: float = DEFAULT_LLM_TEMPERATURE
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    nitro: bool = False
    provider_sort: str = DEFAULT_LLM_PROVIDER_SORT
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
        key_names = ["FISH_LLM_API_KEY"]
        if provider is None or _is_https(base):
            key_names.append(key_env)
        elif read_raw(key_env) is not None:
            warn(
                f"fish-voice: {key_env} is not used because the base is not https, which would "
                f"send it unencrypted. Use an https base, or set FISH_LLM_API_KEY to send a key anyway."
            )
        model_names = [provider.model_env] if provider is not None else []
        model_names.append("FISH_LLM_MODEL")
        if provider is None or provider.name == "openrouter":
            model_names.append("OPENROUTER_MODEL")
        return cls(
            backend=backend,
            base=base,
            api_key=_first_text(*key_names),
            model=_first_token(*model_names),
            temperature=read_float(
                "FISH_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE, lo=0.0, hi=_LLM_TEMPERATURE_HI
            ),
            timeout_s=read_float("FISH_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT_S, positive=True),
            max_tokens=read_int("FISH_LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS, lo=1),
            nitro=read_flag("FISH_LLM_NITRO", default=False),
            provider_sort=_provider_sort(),
            referer=read_text("FISH_LLM_REFERER", DEFAULT_LLM_REFERER),
            title=read_text("FISH_LLM_TITLE", DEFAULT_LLM_TITLE),
            categories=read_text("FISH_LLM_CATEGORIES", DEFAULT_LLM_CATEGORIES),
            continuation=read_flag("FISH_LLM_CONTINUE", default=False),
            reasoning_effort=_reasoning_effort(),
        )


def _named_provider() -> LlmProvider | None:
    raw = read_raw("FISH_LLM_PROVIDER")
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


def _is_https(base: str) -> bool:
    try:
        return urlsplit(base.strip()).scheme.lower() == "https"
    except ValueError:
        return False


def _reasoning_effort() -> str:
    raw = read_raw("FISH_LLM_REASONING_EFFORT")
    if raw is None:
        return ""
    low = raw.lower()
    if low in REASONING_EFFORTS:
        return low
    values = ", ".join(sorted(REASONING_EFFORTS))
    warn(f"fish-voice: FISH_LLM_REASONING_EFFORT={raw!r} is not one of {values}, leaving it unset")
    return ""


def _provider_sort() -> str:
    raw = read_raw("FISH_LLM_PROVIDER_SORT")
    if raw is None:
        return DEFAULT_LLM_PROVIDER_SORT
    low = raw.lower()
    if low in PROVIDER_SORTS:
        return low
    if low in {"off", "none"}:
        return ""
    values = ", ".join(sorted(PROVIDER_SORTS))
    warn(
        f"fish-voice: FISH_LLM_PROVIDER_SORT={raw!r} is not one of {values} or off, "
        f"using {DEFAULT_LLM_PROVIDER_SORT}"
    )
    return DEFAULT_LLM_PROVIDER_SORT


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


def _first_text(*names: str) -> str:
    # A blank value is skipped, so an empty FISH_LLM_API_KEY from the copied
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
    raw = read_raw("FISH_LLM_BACKEND")
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
    return "openrouter" if is_openrouter_host(base) else "openai"
