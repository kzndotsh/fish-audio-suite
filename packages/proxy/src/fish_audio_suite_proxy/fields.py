"""Shared OpenAI audio knobs: models, format, trace, and runtime defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    FISH_TTS_MODEL_IDS,
    MIN_CHUNK_HI,
    MIN_CHUNK_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    SuiteDefaults,
    chunk_length_hi,
    clamp_num,
    ensure_trace_headers,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_text,
    env_token,
    extract_quoted_speech,
    known_latency,
    known_mp3_bitrate,
    known_tts_model,
    normalize_cues,
    number_or,
    scrub_tts,
)

_MEDIA = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "wav": "audio/wav",
}

_PCM16_RATE = 24_000
_FORMAT_ALIAS = {fmt: fmt for fmt in _MEDIA} | {"aac": "mp3", "flac": "mp3"}

_TTS_MODEL_ALIASES = {
    "tts-1": "s2.1-pro",
    "tts-1-hd": "s2.1-pro",
    "gpt-4o-mini-tts": "s2.1-pro",
    "playai-tts": "s2.1-pro",
}

_ASR_MODELS = [
    "transcribe-1",
    "transcribe-1-pro",
    "whisper-1",
]

_MODELS = [*FISH_TTS_MODEL_IDS, *_ASR_MODELS]

SILENT_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 64


def runtime_defaults() -> SuiteDefaults:
    """Build ``SuiteDefaults`` from the process environment.

    Returns
    -------
    SuiteDefaults
        Clamped chunk length, speed, latency, and format. ``FISH_API_KEY``
        is not read here; the proxy lifespan reads it.

    Notes
    -----
    ``FISH_BASE`` containing ``api.fish.audio`` caps ``chunk_length`` at 300.
    Any other base allows up to 1000.
    """
    stock = SuiteDefaults()
    fish_base = env_base("FISH_BASE", stock.fish_base)
    return SuiteDefaults(
        tts_model=known_tts_model(env_token("FISH_MODEL", stock.tts_model)),
        asr_model=resolve_asr_model(None, env_token("FISH_ASR_MODEL", stock.asr_model)),
        asr_language=env_text("FISH_ASR_LANGUAGE", stock.asr_language),
        latency=known_latency(env_token("FISH_LATENCY", stock.latency), stock.latency),
        chunk_length=clamp_num(
            env_int("FISH_CHUNK_LENGTH", stock.chunk_length),
            CHUNK_LENGTH_LO,
            chunk_length_hi(fish_base),
            stock.chunk_length,
            int,
        ),
        min_chunk_length=clamp_num(
            env_int("FISH_MIN_CHUNK_LENGTH", stock.min_chunk_length),
            MIN_CHUNK_LO,
            MIN_CHUNK_HI,
            stock.min_chunk_length,
            int,
        ),
        audio_format=pick_format(
            {"format": env_token("FISH_FORMAT", stock.audio_format)},
            stock.audio_format,
        ),
        mp3_bitrate=known_mp3_bitrate(env_int("FISH_MP3_BITRATE", stock.mp3_bitrate)),
        speed=clamp_num(
            env_float("FISH_SPEED_SCALE", stock.speed),
            TTS_SPEED_LO,
            TTS_SPEED_HI,
            stock.speed,
            float,
        ),
        fish_base=fish_base,
    )


def quality_guard_env() -> bool:
    """Return whether ``FISH_QUALITY_GUARD`` is on.

    Returns
    -------
    bool
        False when unset. The flag is sent to Fish TTS when a caller does not
        set ``quality_guard`` on the request.
    """
    return env_bool("FISH_QUALITY_GUARD")


def strip_speakers_env() -> bool:
    """Return whether ASR speaker labels are stripped by default.

    Returns
    -------
    bool
        True only when ``FISH_ASR_STRIP_SPEAKERS`` is ``1``, ``true``, ``yes``,
        or ``on``.
    """
    return env_bool("FISH_ASR_STRIP_SPEAKERS")


def dialogue_only_env() -> bool:
    """Return whether TTS should keep quoted speech and drop stage notes.

    Returns
    -------
    bool
        True only when ``FISH_TTS_DIALOGUE_ONLY`` is an on-flag.
    """
    return env_bool("FISH_TTS_DIALOGUE_ONLY")


def pick_reference_id(body: dict[str, Any]) -> str | list[str] | None:
    """Read a Fish voice id from ``reference_id`` or OpenAI ``voice``.

    Parameters
    ----------
    body : dict
        Speech request JSON.

    Returns
    -------
    str or list of str or None
        A single id, or a list for S2 multi-speaker. ``reference_id`` wins
        when both keys are set because it is checked first. Blank values
        are ignored.
    """
    for key in ("reference_id", "voice"):
        value = body.get(key)
        if isinstance(value, list):
            ids = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            return ids or None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def present_value(body: dict[str, Any], *keys: str) -> Any:
    """Return the first key that exists, even when the value is false or empty.

    Parameters
    ----------
    body : dict
        Request object.
    *keys : str
        Field names in preference order.

    Returns
    -------
    Any
        The stored value, including ``False`` or ``""``. None when every key
        is absent or when the first present value is None. Those two Nones
        look the same; use ``key in body`` to tell them apart.
    """
    for key in keys:
        if key in body:
            return body[key]
    return None


def explicit_bool(body: dict[str, Any], *keys: str, default: bool) -> bool:
    """Prefer a present request flag, including false, over the default."""
    flag = present_value(body, *keys)
    if flag is None:
        return default
    return bool(flag)


def first_choice(body: dict[str, Any], *keys: str, default: str) -> str:
    """Return the first non-empty string field, lowercased.

    Parameters
    ----------
    body : dict
        Request object.
    *keys : str
        Field names. An empty string does not count.
    default : str
        Used when every key is missing or empty.

    Returns
    -------
    str
        Stripped, lowercased choice.

    Notes
    -----
    Unlike ``present_value``, a present-but-empty string falls through.
    Use ``present_value`` when false is a real answer.
    """
    chosen: Any = default
    for key in keys:
        value = body.get(key)
        if value:
            chosen = value
            break
    return str(chosen).lower().strip()


def pick_format(body: dict[str, Any], default: str) -> str:
    """Map an OpenAI ``response_format`` onto a Fish audio format.

    Parameters
    ----------
    body : dict
        Request object. ``format``, ``response_format``, then ``fish_format``.
    default : str
        Used when the name is unknown. An unknown default becomes ``mp3``.

    Returns
    -------
    str
        ``mp3``, ``opus``, ``pcm``, ``pcm16``, or ``wav``. ``aac`` and ``flac``
        become ``mp3``.
    """
    raw = first_choice(body, "format", "response_format", "fish_format", default=default)
    mapped = _FORMAT_ALIAS.get(raw)
    if mapped is not None:
        return mapped
    return default if default in _MEDIA else "mp3"


def pcm_sample_rate(fmt: str, body: dict[str, Any], default: int) -> int:
    """Choose the PCM rate. ``pcm16`` is 24 kHz unless the client sets one.

    Parameters
    ----------
    fmt : str
        Format from ``pick_format``.
    body : dict
        May contain ``sample_rate``.
    default : int
        Rate when the format is not ``pcm16`` and the body omits one.

    Returns
    -------
    int
        A positive rate. Non-positive or junk input keeps the fallback.
    """
    fallback = _PCM16_RATE if fmt == "pcm16" else default
    raw = body.get("sample_rate")
    if raw is None:
        return fallback
    rate = number_or(raw, fallback, int)
    return rate if rate > 0 else fallback


def media_type(fmt: str) -> str:
    """Content-Type for a Fish audio format.

    Parameters
    ----------
    fmt : str
        ``mp3``, ``opus``, ``pcm``, ``pcm16``, or ``wav``.

    Returns
    -------
    str
        A MIME type. Unknown formats are ``audio/mpeg``. ``pcm16`` is
        ``audio/pcm``.
    """
    return _MEDIA.get(fmt, "audio/mpeg")


def _native_model_id(raw: str) -> str:
    name = raw.strip()
    if name.lower().startswith("fish-audio/"):
        return name.split("/", 1)[1].strip()
    return name


def _model_name(model: object, default: str) -> str:
    if isinstance(model, str) and model.strip():
        return model
    return default


def resolve_tts_model(model: object, default: str) -> str:
    """Map an OpenAI or prefixed model id onto a Fish TTS model.

    Parameters
    ----------
    model : object
        Client ``model``. Non-strings use ``default``.
    default : str
        Used when ``model`` is missing.

    Returns
    -------
    str
        ``tts-1``, ``tts-1-hd``, ``gpt-4o-mini-tts``, and ``playai-tts`` become
        ``s2.1-pro``. A ``fish-audio/`` prefix is stripped. Catalog ids are
        lowercased. ``s2.1-pro-free`` and ``drama-3-preview`` are not remapped.
        Any other id is returned as written.
    """
    raw = _native_model_id(_model_name(model, default))
    key = raw.lower()
    aliased = _TTS_MODEL_ALIASES.get(key)
    if aliased is not None:
        return aliased
    return known_tts_model(raw)


_ASR_NATIVE = frozenset({"transcribe-1", "transcribe-1-pro"})


def resolve_asr_model(model: object, default: str) -> str:
    """Map a client ASR model onto a native Fish id.

    Parameters
    ----------
    model : object
        Client ``model``. ``whisper-1`` and other aliases are not native.
    default : str
        Used when ``model`` is not ``transcribe-1`` or ``transcribe-1-pro``.

    Returns
    -------
    str
        A native id when the client or the default names one. Otherwise
        ``default`` unchanged, so an unknown alias still reaches Fish as the
        configured default rather than the alias string.
    """
    chosen = _native_model_id(_model_name(model, default)).lower()
    if chosen in _ASR_NATIVE:
        return chosen
    fallback = _native_model_id(default).lower()
    if fallback in _ASR_NATIVE:
        return fallback
    return default


def catalog_ids() -> list[str]:
    """Model ids advertised on ``GET /v1/models``.

    Returns
    -------
    list of str
        Native Fish ids plus the same TTS and native ASR ids prefixed with
        ``fish-audio/``. OpenAI aliases such as ``tts-1`` are not listed.
    """
    prefixed = [f"fish-audio/{name}" for name in (*FISH_TTS_MODEL_IDS, *_ASR_NATIVE)]
    return [*_MODELS, *prefixed]


def prepare_tts_text(raw_input: str, *, dialogue_only: bool) -> str:
    """Scrub model text into what Fish should speak.

    Parameters
    ----------
    raw_input : str
        OpenAI ``input``.
    dialogue_only : bool
        When True, keep quoted speech and drop stage notes after scrubbing.

    Returns
    -------
    str
        ``normalize_cues(scrub_tts(...))``. Cue-only junk is detected later
        by ``is_tts_junk``.
    """
    cleaned = scrub_tts(raw_input)
    if dialogue_only:
        cleaned = extract_quoted_speech(cleaned)
    return normalize_cues(cleaned)


def upstream_trace_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Forward a valid W3C trace header or mint one.

    Parameters
    ----------
    headers : Mapping
        Incoming request headers. Names are matched case-insensitively.

    Returns
    -------
    dict
        ``traceparent`` and, when valid, ``tracestate``. Never empty.
    """
    return ensure_trace_headers(headers)


def traced_model_headers(model: str, incoming: Mapping[str, str]) -> dict[str, str]:
    """Fish ``model`` header plus the trace headers for this request.

    Parameters
    ----------
    model : str
        Already resolved Fish model id.
    incoming : Mapping
        Client headers.

    Returns
    -------
    dict
        Sent on TTS and ASR upstream calls.
    """
    return {"model": model, **upstream_trace_headers(incoming)}
