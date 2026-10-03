"""Model ids: OpenAI aliases onto Fish TTS ids, and the ASR id rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from fish_audio_suite_kit import FISH_TTS_MODEL_IDS, normalize_tts_model

__all__ = [
    "OPENAI_TTS_NAMES",
    "catalog_ids",
    "default_tts_aliases",
    "parse_aliases",
    "resolve_asr_model",
    "resolve_tts_model",
]

# OpenAI names clients send by default. Each maps to the configured TTS model,
# and FISH_TTS_ALIASES adds or replaces entries.
OPENAI_TTS_NAMES: Final = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")

_ASR_NATIVE = ("transcribe-1", "transcribe-1-pro")
# Accepted on the transcription route. It is not a Fish id, so it takes the default.
_ASR_ALIASES = ("whisper-1",)
_PREFIX = "fish-audio/"


def default_tts_aliases(default_model: str) -> dict[str, str]:
    """Map the OpenAI TTS names onto the configured Fish model.

    Parameters
    ----------
    default_model : str
        The proxy's default TTS model.

    Returns
    -------
    dict
        Lowercase alias to Fish model id.
    """
    return dict.fromkeys(OPENAI_TTS_NAMES, default_model)


def parse_aliases(raw: str) -> dict[str, str]:
    """Parse ``a=b,c=d`` pairs from ``FISH_TTS_ALIASES``.

    Parameters
    ----------
    raw : str
        Comma-separated ``alias=model`` pairs.

    Returns
    -------
    dict
        Lowercase alias to model id. A pair without ``=`` or with a blank side
        is skipped.
    """
    pairs: dict[str, str] = {}
    for item in raw.split(","):
        alias, sep, target = item.partition("=")
        alias = alias.strip().lower()
        target = target.strip()
        if sep and alias and target:
            pairs[alias] = target
    return pairs


def _native_model_id(raw: str) -> str:
    name = raw.strip()
    if name.lower().startswith(_PREFIX):
        return name.split("/", 1)[1].strip()
    return name


def _model_name(model: object, default: str) -> str:
    if isinstance(model, str) and model.strip():
        return model
    return default


def resolve_tts_model(
    model: object,
    default: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Map an OpenAI or prefixed model id onto a Fish TTS model.

    Parameters
    ----------
    model : object
        Client ``model``. Non-strings use ``default``.
    default : str
        Used when ``model`` is missing.
    aliases : Mapping or None, optional
        Alias table. ``None`` uses the OpenAI names mapped to ``default``.

    Returns
    -------
    str
        An alias becomes its target. A ``fish-audio/`` prefix is stripped.
        Catalog ids are lowercased and are not remapped. Any other
        single-token id is returned as written. An id with whitespace or a
        control character uses ``default``.
    """
    table = default_tts_aliases(default) if aliases is None else aliases
    raw = _native_model_id(_model_name(model, default))
    target = table.get(raw.lower())
    if target is not None:
        # A target written as fish-audio/<id> is the same model as <id>.
        return normalize_tts_model(_native_model_id(target))
    if any(ch.isspace() or ord(ch) < 32 for ch in raw):
        return normalize_tts_model(default)
    return normalize_tts_model(raw)


def _header_model(text: str) -> str:
    # A newline in the model header is illegal. h11 raises and the
    # transcription request never leaves.
    token = text.strip()
    if token and all(32 < ord(ch) < 127 and not ch.isspace() for ch in token):
        return token
    return ""


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
        A native id when the client or the default names one. Otherwise the
        default when it is a single header token, so an unknown alias still
        reaches Fish as that id. A blank or illegal default is ``transcribe-1``.
    """
    chosen = _header_model(_native_model_id(_model_name(model, default))).lower()
    if chosen in _ASR_NATIVE:
        return chosen
    fallback = _header_model(_native_model_id(default)).lower()
    if fallback in _ASR_NATIVE:
        return fallback
    custom = _header_model(default)
    if custom:
        return custom
    return "transcribe-1"


def catalog_ids(aliases: Mapping[str, str] | None = None) -> list[str]:
    """Model ids advertised on ``GET /v1/models``.

    Parameters
    ----------
    aliases : Mapping or None, optional
        TTS alias table. ``None`` lists the OpenAI names.

    Returns
    -------
    list of str
        Native Fish ids, the accepted aliases, and the native ids prefixed
        with ``fish-audio/``. Every id listed here is accepted on a route.
    """
    alias_names = OPENAI_TTS_NAMES if aliases is None else tuple(aliases)
    native = [*FISH_TTS_MODEL_IDS, *_ASR_NATIVE]
    prefixed = [f"{_PREFIX}{name}" for name in native]
    # An alias can share a name with a native id. Each id is listed once.
    return list(dict.fromkeys([*native, *alias_names, *_ASR_ALIASES, *prefixed]))
