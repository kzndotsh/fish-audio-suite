"""One validated settings object, read from the environment once per process."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    FISH_RETRY_ATTEMPTS,
    MIN_CHUNK_LENGTH_HI,
    MIN_CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    SuiteDefaults,
    chunk_length_hi,
    clamp_number,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_text,
    env_token,
    known_latency,
    known_mp3_bitrate,
    normalize_tts_model,
)
from fish_audio_suite_proxy.fields import ClientFormat, fish_audio_format, known_client_format
from fish_audio_suite_proxy.models import default_tts_aliases, parse_aliases, resolve_asr_model

__all__ = [
    "DEFAULT_CONNECT_S",
    "DEFAULT_GRACEFUL_S",
    "DEFAULT_HOST",
    "DEFAULT_KEEP_ALIVE_S",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_INPUT_CHARS",
    "DEFAULT_POOL_S",
    "DEFAULT_PORT",
    "DEFAULT_READ_S",
    "DEFAULT_RETRY_DEADLINE_S",
    "ProxySettings",
    "SettingsError",
    "load_settings",
    "runtime_defaults",
]

log: Final[logging.Logger] = logging.getLogger("fish-audio-suite-proxy")

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8849
DEFAULT_MAX_BODY_BYTES: Final = 25 * 1024 * 1024
DEFAULT_MAX_INPUT_CHARS: Final = 4096
DEFAULT_CONNECT_S: Final = 10.0
DEFAULT_READ_S: Final = 120.0
DEFAULT_POOL_S: Final = 5.0
DEFAULT_RETRY_DEADLINE_S: Final = 90.0
DEFAULT_KEEP_ALIVE_S: Final = 5
DEFAULT_GRACEFUL_S: Final = 120
_PORT_MAX = 65535
_MAX_RETRY_ATTEMPTS = 10
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def _env_name(*names: str) -> str:
    """Return the first name that is set, preferring the earlier one."""
    for name in names:
        if name in os.environ:
            return name
    return names[0]


def _deprecated(read: Callable[[str, Any], Any], new: str, old: str, default: Any) -> Any:
    name = _env_name(new, old)
    if name == old:
        log.warning("%s is deprecated; use %s", old, new)
    return read(name, default)


def _non_negative(name: str, default: int) -> int:
    value = env_int(name, default)
    return value if value >= 0 else default


def _positive_float(name: str, default: float) -> float:
    value = env_float(name, default)
    return value if value > 0 else default


class SettingsError(ValueError):
    """The environment holds a value the proxy cannot start safely with."""


def _api_keys(raw: str) -> tuple[str, ...]:
    """Parse ``FISH_PROXY_API_KEYS``.

    Parameters
    ----------
    raw : str
        The stripped environment value. Empty means auth is off.

    Returns
    -------
    tuple of str
        The comma-separated keys, stripped.

    Raises
    ------
    SettingsError
        When ``raw`` is not empty but holds no usable key, such as ``" , "``.
        Treating it as "no keys" would switch auth off without saying so.
    """
    keys = tuple(key for key in (part.strip() for part in raw.split(",")) if key)
    if raw and not keys:
        msg = (
            "FISH_PROXY_API_KEYS is set but holds no key. "
            "Unset it to run without client auth, or list at least one key."
        )
        raise SettingsError(msg)
    return keys


@dataclass(frozen=True, slots=True)
class ProxySettings:
    """Everything the proxy reads from the environment, validated.

    Notes
    -----
    Built once in the app lifespan and stored on ``app.state.settings``.
    ``FISH_API_KEY`` and ``FISH_PROXY_API_KEYS`` are read there too, but the
    secret values stay out of ``health``. Request bodies can still override
    most per-call knobs; these are the defaults.
    """

    defaults: SuiteDefaults = field(default_factory=SuiteDefaults)
    tts_aliases: Mapping[str, str] = field(default_factory=lambda: default_tts_aliases("s2.1-pro"))
    quality_guard: bool = False
    asr_strip_speakers: bool = False
    asr_strip_cues: bool = False
    tts_dialogue_only: bool = False
    tts_mood_lead: bool = False
    tts_drop_narration: bool = False
    response_format: ClientFormat = "mp3"
    api_keys: tuple[str, ...] = field(default=(), repr=False)
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    log_text: bool = False
    connect_timeout_s: float = DEFAULT_CONNECT_S
    read_timeout_s: float = DEFAULT_READ_S
    pool_timeout_s: float = DEFAULT_POOL_S
    retry_attempts: int = FISH_RETRY_ATTEMPTS
    retry_deadline_s: float = DEFAULT_RETRY_DEADLINE_S
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    workers: int = 1
    keep_alive_s: int = DEFAULT_KEEP_ALIVE_S
    graceful_shutdown_s: int = DEFAULT_GRACEFUL_S
    limit_concurrency: int = 0

    @property
    def auth_required(self) -> bool:
        """Whether clients must send one of ``FISH_PROXY_API_KEYS``."""
        return bool(self.api_keys)

    @property
    def exposed(self) -> bool:
        """Whether the listen address is reachable beyond this machine."""
        return self.host.strip().lower() not in _LOOPBACK

    def uvicorn_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``uvicorn.run``.

        Returns
        -------
        dict
            Host, port, workers, timeouts, and ``limit_concurrency`` when set.
            ``forwarded-allow-ips`` stays at uvicorn's default.
        """
        kwargs: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "loop": "auto",
            "http": "auto",
            "ws": "none",
            "timeout_keep_alive": self.keep_alive_s,
            "timeout_graceful_shutdown": self.graceful_shutdown_s,
            "proxy_headers": True,
        }
        if self.limit_concurrency > 0:
            kwargs["limit_concurrency"] = self.limit_concurrency
        return kwargs

    def health(self) -> dict[str, Any]:
        """Return the settings safe to show on ``GET /health``. Never includes a key.

        Returns
        -------
        dict
            Defaults, flags, limits, and retry knobs.
        """
        d = self.defaults
        return {
            "model": d.tts_model,
            "asr_model": d.asr_model,
            "asr_language": d.asr_language,
            "latency": d.latency,
            "chunk_length": d.chunk_length,
            "format": self.response_format,
            "speed_scale": d.speed,
            "quality_guard": self.quality_guard,
            "asr_strip_speakers": self.asr_strip_speakers,
            "asr_strip_cues": self.asr_strip_cues,
            "tts_dialogue_only": self.tts_dialogue_only,
            "tts_mood_lead": self.tts_mood_lead,
            "tts_drop_narration": self.tts_drop_narration,
            "auth_required": self.auth_required,
            "max_body_bytes": self.max_body_bytes,
            "max_input_chars": self.max_input_chars,
            "retry_attempts": self.retry_attempts,
            "retry_deadline_s": self.retry_deadline_s,
        }


def _response_format() -> ClientFormat:
    """Read ``FISH_FORMAT``, the format used when a request names none.

    Returns
    -------
    ClientFormat
        A supported format, or ``mp3`` for a missing or unknown value. ``pcm16``
        is allowed here, though Fish itself only knows ``pcm``.
    """
    return known_client_format(env_token("FISH_FORMAT", "mp3"), "mp3")


def _suite_defaults() -> SuiteDefaults:
    stock = SuiteDefaults()
    fish_base = env_base("FISH_BASE", stock.fish_base)
    return SuiteDefaults(
        tts_model=normalize_tts_model(
            _deprecated(env_token, "FISH_TTS_MODEL", "FISH_MODEL", stock.tts_model)
        ),
        asr_model=resolve_asr_model(None, env_token("FISH_ASR_MODEL", stock.asr_model)),
        asr_language=env_text("FISH_ASR_LANGUAGE", stock.asr_language),
        latency=known_latency(env_token("FISH_LATENCY", stock.latency), stock.latency),
        chunk_length=clamp_number(
            env_int("FISH_CHUNK_LENGTH", stock.chunk_length),
            CHUNK_LENGTH_LO,
            chunk_length_hi(fish_base),
            stock.chunk_length,
            int,
        ),
        min_chunk_length=clamp_number(
            env_int("FISH_MIN_CHUNK_LENGTH", stock.min_chunk_length),
            MIN_CHUNK_LENGTH_LO,
            MIN_CHUNK_LENGTH_HI,
            stock.min_chunk_length,
            int,
        ),
        audio_format=fish_audio_format(_response_format()),
        mp3_bitrate=known_mp3_bitrate(env_int("FISH_MP3_BITRATE", stock.mp3_bitrate)),
        speed=clamp_number(
            _deprecated(env_float, "FISH_SPEED", "FISH_SPEED_SCALE", stock.speed),
            TTS_SPEED_LO,
            TTS_SPEED_HI,
            stock.speed,
            float,
        ),
        fish_base=fish_base,
    )


def runtime_defaults() -> SuiteDefaults:
    """Build ``SuiteDefaults`` from the process environment.

    Returns
    -------
    SuiteDefaults
        Clamped chunk length, speed, latency, and format. ``FISH_API_KEY``
        is not read here.

    Notes
    -----
    A ``FISH_BASE`` that is not Fish Cloud allows ``chunk_length`` up to 1000.
    Cloud caps it at 300.
    """
    return _suite_defaults()


def load_settings() -> ProxySettings:
    """Read and validate every proxy setting from the process environment.

    Returns
    -------
    ProxySettings
        A bad value keeps its default. ``FISH_MODEL`` and ``FISH_SPEED_SCALE``
        still work and log a deprecation warning.

    Raises
    ------
    SettingsError
        When ``FISH_PROXY_API_KEYS`` is set but holds no key, so a typo cannot
        switch client auth off.
    """
    defaults = _suite_defaults()
    aliases = {
        **default_tts_aliases(defaults.tts_model),
        **parse_aliases(env_text("FISH_TTS_ALIASES")),
    }
    port = env_int("FISH_PROXY_PORT", DEFAULT_PORT)
    return ProxySettings(
        defaults=defaults,
        tts_aliases=aliases,
        quality_guard=env_bool("FISH_QUALITY_GUARD"),
        asr_strip_speakers=env_bool("FISH_ASR_STRIP_SPEAKERS"),
        asr_strip_cues=env_bool("FISH_ASR_STRIP_CUES"),
        tts_dialogue_only=env_bool("FISH_TTS_DIALOGUE_ONLY"),
        tts_mood_lead=env_bool("FISH_MOOD_LEAD"),
        tts_drop_narration=env_bool("FISH_DROP_NARRATION"),
        response_format=_response_format(),
        api_keys=_api_keys(env_text("FISH_PROXY_API_KEYS")),
        max_body_bytes=_non_negative("FISH_PROXY_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES),
        max_input_chars=_non_negative("FISH_PROXY_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS),
        log_text=env_bool("FISH_PROXY_LOG_TEXT"),
        connect_timeout_s=_positive_float("FISH_PROXY_CONNECT_TIMEOUT", DEFAULT_CONNECT_S),
        read_timeout_s=_positive_float("FISH_PROXY_READ_TIMEOUT", DEFAULT_READ_S),
        pool_timeout_s=_positive_float("FISH_PROXY_POOL_TIMEOUT", DEFAULT_POOL_S),
        retry_attempts=clamp_number(
            env_int("FISH_PROXY_RETRY_ATTEMPTS", FISH_RETRY_ATTEMPTS),
            1,
            _MAX_RETRY_ATTEMPTS,
            FISH_RETRY_ATTEMPTS,
            int,
        ),
        retry_deadline_s=max(env_float("FISH_PROXY_RETRY_DEADLINE", DEFAULT_RETRY_DEADLINE_S), 0.0),
        host=env_token("FISH_PROXY_HOST", DEFAULT_HOST),
        port=port if 1 <= port <= _PORT_MAX else DEFAULT_PORT,
        workers=max(env_int("FISH_PROXY_WORKERS", env_int("WEB_CONCURRENCY", 1)), 1),
        keep_alive_s=_non_negative("FISH_PROXY_KEEP_ALIVE", DEFAULT_KEEP_ALIVE_S),
        graceful_shutdown_s=_non_negative("FISH_PROXY_GRACEFUL_SHUTDOWN", DEFAULT_GRACEFUL_S),
        limit_concurrency=max(env_int("FISH_PROXY_LIMIT_CONCURRENCY", 0), 0),
    )
