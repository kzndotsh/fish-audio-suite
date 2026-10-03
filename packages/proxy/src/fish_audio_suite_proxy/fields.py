"""Shared OpenAI audio knobs: format, silence, request-field readers, and trace."""

from __future__ import annotations

import base64
import binascii
import io
import wave
from collections.abc import Mapping
from typing import Any, Final, Literal, get_args

from fish_audio_suite_kit import (
    AudioFormat,
    ensure_trace_headers,
    extract_quoted_speech,
    normalize_cues,
    parse_number,
    scrub_tts,
)
from fish_audio_suite_proxy.errors import ProxyError

__all__ = [
    "SILENT_MP3",
    "SUPPORTED_FORMATS",
    "AudioDecodeError",
    "ClientFormat",
    "decode_audio_b64",
    "fish_audio_format",
    "known_client_format",
    "media_type",
    "pcm_sample_rate",
    "prepare_tts_text",
    "read_choice",
    "read_flag",
    "read_format",
    "read_present",
    "read_reference_id",
    "silent_speech",
    "traced_model_headers",
]

# Formats a client may ask for. ``pcm16`` is the OpenAI name for 24 kHz PCM, which
# Fish knows as ``pcm``, so it is a request format only (see ``fish_audio_format``).
ClientFormat = Literal["mp3", "opus", "pcm", "pcm16", "wav"]

_MEDIA: dict[ClientFormat, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "wav": "audio/wav",
}

_PCM16_RATE = 24_000
# wave stores the rate as an unsigned 32-bit field. A larger junk-WAV rate
# raises before the silence response is sent.
_WAV_RATE_HI = 2**32 - 1
SUPPORTED_FORMATS: tuple[ClientFormat, ...] = tuple(_MEDIA)
_CLIENT_FORMATS: dict[str, ClientFormat] = {name: name for name in get_args(ClientFormat)}


def fish_audio_format(fmt: ClientFormat) -> AudioFormat:
    """Map a client format to the format Fish produces.

    Parameters
    ----------
    fmt : ClientFormat
        A format from ``read_format``.

    Returns
    -------
    AudioFormat
        The same format, except ``pcm16``, which Fish produces as ``pcm``.
    """
    return "pcm" if fmt == "pcm16" else fmt


# One valid MPEG-1 Layer III frame: 32 kbps, 44.1 kHz, mono, 104 bytes
# (144 * 32000 // 44100). Zeroed side info and main data decode as silence.
SILENT_MP3: Final = b"\xff\xfb\x10\xc0" + b"\x00" * 100
_SILENT_PCM = b"\x00\x00"
# One 20 ms mono Opus page (peak sample 1 after decode). An MP3 frame with
# an audio/opus type is not silence; an Opus decoder plays it as noise.
_SILENT_OPUS = bytes.fromhex(
    "4f6767530002000000000000000032e6b27d00000000b1d60b1101134f707573486561"
    "640101380180bb00000000004f6767530000000000000000000032e6b27d0100000002"
    "f38868013c4f707573546167730c0000004c61766636332e312e313031010000001c00"
    "0000656e636f6465723d4c61766336332e312e313031206c69626f7075734f67675300"
    "04f80400000000000032e6b27d02000000f4decf11020706080be63b23ab600808acb3"
    "0ec6"
)


def silent_speech(fmt: ClientFormat, sample_rate: int) -> tuple[bytes, str]:
    """Return one silent buffer in the format the client asked to play."""
    # An MP3 frame played as PCM is loud garbage, and it is not a WAV or Opus file.
    if fmt in {"pcm", "pcm16"}:
        return _SILENT_PCM, media_type(fmt)
    if fmt == "opus":
        return _SILENT_OPUS, media_type("opus")
    if fmt == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate if 0 < sample_rate <= _WAV_RATE_HI else _PCM16_RATE)
            wf.writeframes(_SILENT_PCM)
        return buf.getvalue(), media_type("wav")
    return SILENT_MP3, media_type("mp3")


_DATA_URI = "data:"


class AudioDecodeError(ProxyError):
    """Audio in a request body could not be decoded. The route returns 400.

    Attributes
    ----------
    status : int
        Always 400.
    message : str
        Shown in the OpenAI error envelope.
    """

    def __init__(self, message: str) -> None:
        """Store ``message`` with status 400.

        Parameters
        ----------
        message : str
            Client-facing reason.
        """
        super().__init__(400, message)


def _b64_audio(value: str, field: str) -> bytes:
    raw = "".join(value.strip().split())
    # The data-URI scheme is case-insensitive. "DATA:" failed the base64
    # check, so the request was rejected before Fish heard the audio.
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
        raise AudioDecodeError(f"{field} is not valid base64") from exc


def decode_audio_b64(value: Any, *, field: str = "audio") -> bytes:
    """Decode audio from raw bytes or base64, including a data URI.

    Parameters
    ----------
    value : Any
        ``bytes``, or a base64 string. Whitespace inside the string is removed.
    field : str, optional
        What the client calls this audio, used to start the error message:
        ``"input_audio"`` for a transcription, ``"reference audio"`` for a
        voice clip. Default ``"audio"``.

    Returns
    -------
    bytes
        Decoded audio. Never empty.

    Raises
    ------
    AudioDecodeError
        When the value is empty or not valid base64, with a message such as
        ``"audio is empty"`` or ``"audio is not valid base64"``.
    """
    if isinstance(value, (bytes, bytearray)):
        audio = bytes(value)
    elif isinstance(value, str) and value.strip():
        audio = _b64_audio(value, field)
    else:
        audio = b""
    if not audio:
        raise AudioDecodeError(f"{field} is empty")
    return audio


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
        Request object. ``format``, ``response_format``, then ``fish_format``.
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
    raw = read_choice(body, "format", "response_format", "fish_format", default="")
    if not raw:
        return default
    known = _CLIENT_FORMATS.get(raw)
    if known is not None:
        return known
    supported = ", ".join(SUPPORTED_FORMATS)
    raise ProxyError(400, f"unsupported response_format {raw[:32]!r}; use one of: {supported}")


def known_client_format(name: str, default: ClientFormat) -> ClientFormat:
    """Read a format from config, keeping ``default`` when the name is unsupported.

    Parameters
    ----------
    name : str
        Env value such as ``FISH_TTS_FORMAT``. Compared after strip and lowercase.
    default : ClientFormat
        Returned when ``name`` is blank or not a supported format.

    Returns
    -------
    ClientFormat
        A supported format.
    """
    return _CLIENT_FORMATS.get(name.strip().lower(), default)


def pcm_sample_rate(fmt: ClientFormat, body: dict[str, Any], default: int) -> int:
    """Choose the PCM rate. ``pcm16`` is 24 kHz unless the client sets one.

    Parameters
    ----------
    fmt : ClientFormat
        Format from ``read_format``.
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
    rate = parse_number(raw, fallback, int)
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
    known = _CLIENT_FORMATS.get(fmt)
    return _MEDIA[known] if known is not None else "audio/mpeg"


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
