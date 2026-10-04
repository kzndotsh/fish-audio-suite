"""Readers for the optional fields of an OpenAI request body, and the text and headers built from them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fish_audio_suite_kit import (
    ensure_trace_headers,
    extract_quoted_speech,
    normalize_cues,
    scrub_tts,
)
from fish_audio_suite_proxy.audio import CLIENT_FORMATS, SUPPORTED_FORMATS, ClientFormat
from fish_audio_suite_proxy.errors import ProxyError

__all__ = [
    "prepare_tts_text",
    "read_choice",
    "read_flag",
    "read_format",
    "read_present",
    "read_reference_id",
    "traced_model_headers",
]


def read_reference_id(body: dict[str, Any]) -> str | list[str] | None:
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
            # An empty list is blank, same as "". Keep looking at voice.
            if ids:
                return ids
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def read_present(body: dict[str, Any], *keys: str) -> Any:
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


_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})


_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


def read_flag(body: dict[str, Any], *keys: str, default: bool) -> bool:
    """Prefer a present request flag, including false, over the default."""
    flag = read_present(body, *keys)
    if flag is None:
        return default
    if isinstance(flag, str):
        word = flag.strip().lower()
        # bool("false") is True, so a string flag would turn the feature on.
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
        return default
    return bool(flag)


def read_choice(body: dict[str, Any], *keys: str, default: str) -> str:
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
    Unlike ``read_present``, a present-but-empty string falls through.
    Use ``read_present`` when false is a real answer.
    """
    chosen: Any = default
    for key in keys:
        value = body.get(key)
        if value:
            chosen = value
            break
    return str(chosen).lower().strip()


def read_format(body: dict[str, Any], default: ClientFormat) -> ClientFormat:
    """Read the audio format a request asks for.

    Parameters
    ----------
    body : dict
        Request object. ``format``, then ``response_format``.
    default : ClientFormat
        Used when the request names no format.

    Returns
    -------
    ClientFormat
        ``mp3``, ``opus``, ``pcm``, ``pcm16``, or ``wav``.

    Raises
    ------
    ProxyError
        With status 400 when the request names a format Fish cannot produce,
        such as ``aac`` or ``flac``. Returning other bytes than the client
        asked for would break its decoder.
    """
    raw = read_choice(body, "format", "response_format", default="")
    if not raw:
        return default
    known = CLIENT_FORMATS.get(raw)
    if known is not None:
        return known
    supported = ", ".join(SUPPORTED_FORMATS)
    raise ProxyError(400, f"unsupported response_format {raw[:32]!r}; use one of: {supported}")


def prepare_tts_text(
    raw_input: str,
    *,
    dialogue_only: bool,
    mood_lead: bool = False,
) -> str:
    """Scrub model text into what Fish should speak.

    Parameters
    ----------
    raw_input : str
        OpenAI ``input``.
    dialogue_only : bool
        When True, keep quoted speech and drop the text around it. This suits
        fiction or roleplay output where narration sits between the lines.
    mood_lead : bool, optional
        When True, a sentence that opens with a mood word becomes a cue.
        Default False, so ordinary sentences are spoken as written.

    Returns
    -------
    str
        ``normalize_cues(scrub_tts(...))``. Cue-only junk is detected later
        by ``is_tts_junk``.
    """
    cleaned = scrub_tts(raw_input)
    if dialogue_only:
        cleaned = extract_quoted_speech(cleaned)
    return normalize_cues(cleaned, lead=mood_lead)


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
    return {"model": model, **ensure_trace_headers(incoming)}
