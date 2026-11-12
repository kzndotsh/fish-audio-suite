"""Audio formats a client may ask for, and the silent clip sent for junk text."""

from __future__ import annotations

import io
import wave
from typing import Any, Final, Literal, get_args

from fish_audio_suite_kit import AudioFormat, parse_number

__all__ = [
    "SUPPORTED_FORMATS",
    "ClientFormat",
    "client_format",
    "fish_audio_format",
    "media_type",
    "pcm_sample_rate",
    "silent_speech",
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
# wave stores the byte rate (rate x 2 bytes for mono 16-bit) as an unsigned 32-bit
# field and raises on a larger one, so the silent WAV falls back to 24 kHz above this.
_WAV_RATE_HI = (2**32 - 1) // 2
SUPPORTED_FORMATS: tuple[ClientFormat, ...] = get_args(ClientFormat)
_BY_NAME: dict[str, ClientFormat] = {name: name for name in SUPPORTED_FORMATS}


def client_format(name: str) -> ClientFormat | None:
    """Look up a format name from a request or the environment.

    Parameters
    ----------
    name : str
        Compared after strip and lowercase.

    Returns
    -------
    ClientFormat or None
        The format, or None when the proxy cannot produce it (``aac``, ``flac``).
    """
    return _BY_NAME.get(name.strip().lower())


def fish_audio_format(fmt: ClientFormat) -> AudioFormat:
    """Map a client format to the format Fish produces.

    Parameters
    ----------
    fmt : ClientFormat
        A format the client asked for.

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
    """Return one silent buffer in the format the client asked to play.

    Parameters
    ----------
    fmt : ClientFormat
        The requested format.
    sample_rate : int
        Rate written into the WAV header. A rate outside 1 to (2**32 - 1) // 2
        becomes 24 kHz. Other formats ignore it.

    Returns
    -------
    tuple of bytes and str
        The silent audio and its media type.
    """
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


def pcm_sample_rate(
    fmt: ClientFormat, body: dict[str, Any], default: int, *, hi: int | None = None
) -> int:
    """Choose the PCM rate. ``pcm16`` is 24 kHz unless the client sets one.

    Parameters
    ----------
    fmt : ClientFormat
        The format the client asked for.
    body : dict
        May contain ``sample_rate``.
    default : int
        Rate when the format is not ``pcm16`` and the body omits one.
    hi : int or None, optional
        Largest rate the caller can send on. A larger client rate keeps the
        fallback, as junk does. None sets no bound.

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
    return rate if rate > 0 and (hi is None or rate <= hi) else fallback


def media_type(fmt: ClientFormat) -> str:
    """Content-Type for a format the client asked for.

    Parameters
    ----------
    fmt : ClientFormat
        ``mp3``, ``opus``, ``pcm``, ``pcm16``, or ``wav``.

    Returns
    -------
    str
        A MIME type. ``pcm16`` is ``audio/pcm``.
    """
    return _MEDIA[fmt]
