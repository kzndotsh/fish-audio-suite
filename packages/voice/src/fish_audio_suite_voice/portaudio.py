"""PortAudio through ``sounddevice``: loaded when a stream opens, for the mic and the speaker alike.

``sounddevice`` is imported only when a device is opened, so the package imports on a machine
without the PortAudio C library, and a missing library becomes one error with install hints.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = [
    "PORTAUDIO_HINT",
    "PortAudioMissingError",
    "load_sounddevice",
    "missing_portaudio",
    "pcm_stream_kwargs",
]

PORTAUDIO_HINT: Final = """sounddevice needs the PortAudio C library (the Python wheel does not ship it).
  Debian/Ubuntu: sudo apt install libportaudio2
  Fedora: sudo dnf install portaudio
  macOS: brew install portaudio
  NixOS: nix run .#fish-audio-suite-voice   # flake prefixes LD_LIBRARY_PATH
          or ./packages/voice/dev.sh"""

_MONO: Final = 1


class PortAudioMissingError(OSError):
    """sounddevice imported, but libportaudio is not on the loader path."""


def missing_portaudio() -> PortAudioMissingError:
    """Build the error raised when libportaudio is not loadable.

    Returns
    -------
    PortAudioMissingError
        Carrying ``PORTAUDIO_HINT``.
    """
    return PortAudioMissingError(PORTAUDIO_HINT)


def pcm_stream_kwargs(
    sample_rate: int, device: str | int | None, *, latency: str | None = None
) -> dict[str, Any]:
    """Return RawStream kwargs for mono int16 PCM, with a buffer size when ``latency`` is given.

    Parameters
    ----------
    sample_rate : int
        Samples per second.
    device : str or int or None
        The PortAudio device. None is the host default.
    latency : str or None, optional
        ``"low"`` or ``"high"``. None leaves PortAudio's default.

    Returns
    -------
    dict
        Keyword arguments for ``sounddevice.RawInputStream`` or ``RawOutputStream``.
    """
    kwargs: dict[str, Any] = {
        "samplerate": sample_rate,
        "channels": _MONO,
        "dtype": "int16",
        "device": device,
    }
    if latency is not None:
        kwargs["latency"] = latency
    return kwargs


def load_sounddevice() -> Any:
    """Import sounddevice, mapping a missing library to ``PortAudioMissingError``.

    Returns
    -------
    module
        The ``sounddevice`` module.

    Raises
    ------
    PortAudioMissingError
        When the PortAudio C library cannot be loaded.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415 - loaded only when a device opens
    except OSError as exc:
        raise missing_portaudio() from exc
    return sd
