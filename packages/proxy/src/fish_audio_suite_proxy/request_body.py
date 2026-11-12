"""Read an OpenAI request body: the JSON object, its loose fields, and base64 audio."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from fastapi import Request

from fish_audio_suite_kit import parse_flag
from fish_audio_suite_proxy.errors import ProxyError

__all__ = ["decode_audio_b64", "read_choice", "read_flag", "read_json_object"]


async def read_json_object(request: Request) -> dict[str, Any]:
    """Parse a JSON object body.

    Parameters
    ----------
    request : Request
        Incoming FastAPI request.

    Returns
    -------
    dict
        The object.

    Raises
    ------
    ProxyError
        400 when the body is not JSON, or is a JSON array or scalar.
    """
    try:
        # utf-8-sig drops a leading BOM. json.loads rejects that byte and the
        # request would 400 before Fish saw the text.
        raw = await request.body()
        parsed = json.loads(raw.decode("utf-8-sig"))
    # ValueError covers bad JSON, bad UTF-8, and an integer past Python's digit
    # limit. A client can nest arrays deep enough to raise RecursionError.
    except (ValueError, RecursionError) as exc:
        raise ProxyError(400, "invalid JSON body") from exc
    if not isinstance(parsed, dict):
        raise ProxyError(400, "JSON body must be an object")
    return parsed


def read_flag(body: dict[str, Any], key: str, *, default: bool) -> bool:
    """Read an on/off request field, keeping an explicit false.

    Parameters
    ----------
    body : dict
        Request object.
    key : str
        Field name.
    default : bool
        Used when the field is missing, null, or a word that is neither on nor off.

    Returns
    -------
    bool
        A string is read with the kit's lenient words (``yes``, ``off``), because
        ``bool("false")`` is True. Any other value is truth-tested.
    """
    flag = body.get(key)
    if flag is None:
        return default
    if isinstance(flag, str):
        word = parse_flag(flag)
        return default if word is None else word
    return bool(flag)


def read_choice(body: dict[str, Any], *keys: str, default: str) -> str:
    """Return the first non-empty field, as a stripped lowercase string.

    Parameters
    ----------
    body : dict
        Request object.
    *keys : str
        Field names in preference order. An empty value does not count.
    default : str
        Used when every key is missing or empty.

    Returns
    -------
    str
        Stripped, lowercased choice.
    """
    chosen: Any = default
    for key in keys:
        value = body.get(key)
        if value:
            chosen = value
            break
    return str(chosen).lower().strip()


_DATA_URI = "data:"


def _b64_audio(value: str, field: str) -> bytes:
    raw = "".join(value.strip().split())
    # The data-URI scheme is case-insensitive, so "DATA:" is a prefix too.
    head, sep, tail = raw.partition(",")
    if sep and head.lower().startswith(_DATA_URI):
        raw = tail
    # URL-safe alphabets use - and _. The strict decoder rejects those, so
    # real audio would 400.
    raw = raw.replace("-", "+").replace("_", "/")
    padded = raw + ("=" * ((-len(raw)) % 4))
    try:
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProxyError(400, f"{field} is not valid base64") from exc


def decode_audio_b64(value: Any, *, field: str) -> bytes:
    """Decode audio from raw bytes or base64, including a data URI.

    Parameters
    ----------
    value : Any
        ``bytes``, or a base64 string. Whitespace inside the string is removed.
    field : str
        What the client calls this audio, used to start the error message:
        ``"input_audio"`` for a transcription, ``"reference audio"`` for a
        voice clip.

    Returns
    -------
    bytes
        Decoded audio. Never empty.

    Raises
    ------
    ProxyError
        400 when the value is empty or not valid base64, with a message such as
        ``"input_audio is empty"`` or ``"input_audio is not valid base64"``.
    """
    if isinstance(value, (bytes, bytearray)):
        audio = bytes(value)
    elif isinstance(value, str) and value.strip():
        audio = _b64_audio(value, field)
    else:
        audio = b""
    if not audio:
        raise ProxyError(400, f"{field} is empty")
    return audio
