"""Voice-activity detectors behind one interface, chosen by name or loaded as a plugin."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Final, Protocol

from fish_audio_suite_voice.mic import SAMPLE_RATE

__all__ = [
    "SILERO_THRESHOLDS",
    "VAD_ENGINES",
    "Vad",
    "VadFactory",
    "frame_is_speech",
    "open_vad",
    "vad_problem",
]


class Vad(Protocol):
    """Scores 30 ms frames of 16 kHz mono int16 PCM, in the order they were heard.

    ``webrtcvad.Vad`` already has this shape. A detector may keep state between
    frames, so each listen and each barge-in watch opens its own.
    """

    def is_speech(self, buf: bytes, sample_rate: int, /) -> bool:
        """Return whether this frame holds speech."""
        ...


# Takes the strictness, 0 to 3 (FISH_VOICE_VAD), and returns a fresh detector.
type VadFactory = Callable[[int], Vad]

# Speech probability Silero must reach at each strictness. 0.5 at the default of 1
# is Silero's own default; the others are starting points to tune.
SILERO_THRESHOLDS: Final = (0.35, 0.5, 0.65, 0.8)


def _webrtc(strictness: int) -> Vad:
    import webrtcvad

    return webrtcvad.Vad(strictness)


class _Silero:
    """Silero VAD through pysilero-vad. It takes 32 ms chunks, so frames are rebuffered."""

    def __init__(self, threshold: float) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self._model = SileroVoiceActivityDetector()
        self._chunk: int = self._model.chunk_bytes()
        self._pending = b""
        self._threshold = threshold
        self._probability = 0.0

    def is_speech(self, buf: bytes, sample_rate: int) -> bool:
        del sample_rate  # the mic always runs at 16 kHz
        self._pending += buf
        while len(self._pending) >= self._chunk:
            self._probability = self._model(self._pending[: self._chunk])
            self._pending = self._pending[self._chunk :]
        return self._probability >= self._threshold


def _silero(strictness: int) -> Vad:
    return _Silero(SILERO_THRESHOLDS[strictness])


VAD_ENGINES: Final[dict[str, VadFactory]] = {"webrtc": _webrtc, "silero": _silero}
_EXTRAS: Final = {"webrtc": "vad", "silero": "silero"}


def _factory(engine: str) -> VadFactory:
    known = VAD_ENGINES.get(engine)
    if known is not None:
        return known
    module, _, name = engine.partition(":")
    if not name:
        known_names = ", ".join(VAD_ENGINES)
        msg = f"unknown voice detector {engine!r}: use {known_names} or package.module:factory"
        raise ValueError(msg)
    return getattr(importlib.import_module(module), name)


def open_vad(engine: str, strictness: int) -> Vad:
    """Open a fresh detector.

    Parameters
    ----------
    engine : str
        A name in ``VAD_ENGINES``, or ``package.module:factory`` for a plugin. The
        factory takes the strictness and returns an object with ``is_speech``.
    strictness : int
        0 to 3. Higher calls fewer sounds speech.

    Returns
    -------
    Vad
        A detector with no history.
    """
    return _factory(engine)(strictness)


def vad_problem(engine: str, strictness: int) -> str | None:
    """Return why ``engine`` cannot open, or None when it can.

    Parameters
    ----------
    engine : str
        As for ``open_vad``.
    strictness : int
        As for ``open_vad``.

    Returns
    -------
    str or None
        A message that names the missing extra or the failing plugin.
    """
    try:
        open_vad(engine, strictness)
    except ImportError as e:
        extra = _EXTRAS.get(engine)
        if extra is None:
            return f"FISH_VOICE_VAD_ENGINE={engine} cannot be imported: {e}"
        return (
            f"FISH_VOICE_VAD_ENGINE={engine} needs the {extra} extra: "
            f"pip install 'fish-audio-suite-voice[{extra}]' (or uv sync --extra {extra})"
        )
    except Exception as e:  # noqa: BLE001 - a plugin can fail any way; the user is told why
        return f"FISH_VOICE_VAD_ENGINE={engine} failed to open: {e}"
    return None


def frame_is_speech(vad: Vad, frame: bytes) -> bool:
    """Return whether ``vad`` scores this 16 kHz frame as speech."""
    return bool(vad.is_speech(frame, SAMPLE_RATE))
