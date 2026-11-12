"""Typed stand-ins shared by the voice tests.

One mic, one barge gate, one ``TtsResult`` factory and the helpers that install them,
so a test patches objects instead of repeating small fake classes.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from types import ModuleType
from typing import Any

import pytest

from fish_audio_suite_voice import reply
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.cancel import StopFlag
from fish_audio_suite_voice.pcm import FRAME_BYTES
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tune import DEFAULT_BARGE_RMS, AecTune, BargeTune
from fish_audio_suite_voice.wire import TtsResult


class ScriptedMic:
    """A ``Mic`` that plays a script of frames, then goes quiet.

    Parameters
    ----------
    script : Iterable of bytes or Callable, optional
        The frames each read yields, or a function of the stop flag that yields them.
    aec : EchoCanceller or None, optional
        What ``aec`` reports. AEC off when omitted.
    """

    def __init__(
        self,
        script: Iterable[bytes] | Callable[[StopFlag | None], Iterator[bytes]] = (),
        *,
        aec: EchoCanceller | None = None,
    ) -> None:
        self._script = script
        self.aec = aec or EchoCanceller(AecTune.off())
        self.opened = 0
        self.since: int | None = None
        self.released = 0

    def __enter__(self) -> ScriptedMic:
        """Return the mic, as the session's exit stack expects."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Nothing to close."""

    @property
    def position(self) -> int:
        """Always 0: every read plays the script from its start."""
        return 0

    def release(self) -> None:
        """Count the release, as mute does."""
        self.released += 1

    def recent(self, since: int) -> list[bytes]:
        """The script, as if all of it had arrived since ``since``."""
        del since
        script = self._script
        return [] if callable(script) else list(script)

    def frames(self, stop: StopFlag | None, since: int | None = None) -> Iterator[bytes]:
        """Yield the script, ending early once ``stop`` is set. ``since`` is recorded."""
        self.since = since
        self.opened += 1
        script = self._script(stop) if callable(self._script) else self._script
        for frame in script:
            if stop is not None and stop.is_set():
                return
            yield frame


class FakeGate(BargeGate):
    """A ``BargeGate`` that never watches the mic: it counts how often it was armed.

    The gate's state is set as a trip would leave it, and ``clip`` is the real method.

    Parameters
    ----------
    captured : bytes, optional
        The speech a trip kept. Empty when the gate did not trip.
    resume_at : int or None, optional
        The mic position after that speech.
    min_rms : float, optional
        The level the gate admits speech at, as ``BargeTune.min_rms``.
    hold : bool, optional
        When True the thread waits for ``cancel`` like the real watcher, so a
        test can check that the caller releases it.
    """

    def __init__(
        self,
        captured: bytes = b"",
        *,
        resume_at: int | None = None,
        min_rms: float = DEFAULT_BARGE_RMS,
        hold: bool = False,
    ) -> None:
        super().__init__(mic=ScriptedMic(), tune=BargeTune(min_rms=min_rms))
        self.captured = captured
        self.resume_at = resume_at
        self.hold = hold
        self.armed = 0
        self.thread: threading.Thread | None = None

    def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
        """Count the arming and start a thread that holds until ``cancel`` when asked to."""
        self.armed += 1

        def run() -> None:
            if self.hold:
                cancel.wait(timeout=30)

        thread = threading.Thread(target=run)
        self.thread = thread
        thread.start()
        return thread


def install_gate(monkeypatch: pytest.MonkeyPatch, gate: FakeGate | None = None) -> FakeGate:
    """Make every reply watch for barge-in with ``gate``, a fresh ``FakeGate`` when omitted.

    The reply's sink needs no stand-in: the test configs play to ``stdout``, and the fake
    speakers never write to it.
    """
    use_gate = gate or FakeGate()
    monkeypatch.setattr(reply, "BargeGate", lambda **_kwargs: use_gate)
    return use_gate


def set_tts(
    monkeypatch: pytest.MonkeyPatch,
    tts: FishSpeaker,
    *,
    speak: Callable[..., TtsResult] | None = None,
    speak_stream: Callable[..., TtsResult] | None = None,
) -> None:
    """Replace the speak methods of an ``FishSpeaker`` for one test.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The test's patcher.
    tts : FishSpeaker
        The object to patch.
    speak : Callable or None, optional
        Replaces ``speak_shaped``, the call the loop makes for a whole reply.
    speak_stream : Callable or None, optional
        Replaces ``speak_stream``.
    """
    if speak is not None:
        monkeypatch.setattr(tts, "speak_shaped", speak)
    if speak_stream is not None:
        monkeypatch.setattr(tts, "speak_stream", speak_stream)


def make_result(
    spoken: str = "",
    *,
    bytes_played: int = 0,
    got_audio: bool = False,
    cancelled: bool = False,
    tts_first_audio_ms: float | None = None,
    tts_first_text_ms: float | None = None,
    error_status: int | None = None,
    error_message: str | None = None,
) -> TtsResult:
    """Build a ``TtsResult`` by keyword, so reordering its fields cannot change a test.

    ``TtsResult`` derives its ``error`` from ``error_status`` itself.
    """
    return TtsResult(
        spoken_so_far=spoken,
        bytes_played=bytes_played,
        got_audio=got_audio,
        cancelled=cancelled,
        tts_first_audio_ms=tts_first_audio_ms,
        tts_first_text_ms=tts_first_text_ms,
        error_status=error_status,
        error_message=error_message,
    )


def level_frame(level: int) -> bytes:
    """Return one mic frame whose every sample is ``level``, so its RMS is ``level``."""
    return level.to_bytes(2, "little", signed=True) * (FRAME_BYTES // 2)


def install_detector(monkeypatch: pytest.MonkeyPatch, is_speech: Callable[[bytes], bool]) -> None:
    """Make the default voice detector call a frame speech when ``is_speech`` says so."""

    class Vad:
        def __init__(self, mode: int) -> None:
            del mode

        def is_speech(self, frame: bytes, rate: int) -> bool:
            del rate
            return is_speech(frame)

    install_vad(monkeypatch, Vad)


def install_vad(monkeypatch: pytest.MonkeyPatch, vad_class: type[Any]) -> None:
    """Make ``import webrtcvad`` return a module whose ``Vad`` is ``vad_class``."""
    module = ModuleType("webrtcvad")
    vars(module)["Vad"] = vad_class
    monkeypatch.setitem(sys.modules, "webrtcvad", module)
