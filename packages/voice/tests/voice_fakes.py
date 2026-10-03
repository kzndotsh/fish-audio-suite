"""Typed stand-ins shared by the voice tests.

One barge gate, one sink, one ``TtsResult`` factory and the helpers that
install them, so a test patches objects instead of repeating small fake classes.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from fish_audio_suite_kit import FishHttpError
from fish_audio_suite_voice import reply
from fish_audio_suite_voice.live import IsolatedFishTts, TtsResult


class FakeGate:
    """``BargeGate`` stand-in. Counts how often it was armed and can hold the mic.

    Parameters
    ----------
    captured : bytes, optional
        What ``BargeGate.captured`` would hold after a barge-in.
    hold : bool, optional
        When True the thread waits for ``cancel`` like the real watcher, so a
        test can check that the caller releases it.
    """

    def __init__(self, captured: bytes = b"", *, hold: bool = False) -> None:
        self.captured = captured
        self.hold = hold
        self.armed = 0
        self.thread: threading.Thread | None = None
        self.failure: Exception | None = None

    def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
        self.armed += 1

        def run() -> None:
            if self.hold:
                cancel.wait(timeout=30)

        thread = threading.Thread(target=run)
        self.thread = thread
        thread.start()
        return thread


class FakeSink:
    """``PlaybackSink`` stand-in that discards audio and reports ``played`` bytes."""

    def __init__(self, played: int = 0) -> None:
        self.played = played
        self.output_latency_s = 0.0

    def start(self) -> None:
        return None

    def write(self, chunk: bytes) -> None:
        del chunk

    def finish(self, *, kill: bool = False) -> None:
        del kill

    def bytes_played(self) -> int:
        return self.played


def install_audio(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate: FakeGate | None = None,
    sink: FakeSink | None = None,
) -> tuple[FakeGate, FakeSink]:
    """Replace the barge gate and the sink factory used by ``reply``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The test's patcher, so the change is undone afterwards.
    gate : FakeGate or None, optional
        Used for every ``BargeGate(...)``. A fresh one when omitted.
    sink : FakeSink or None, optional
        Returned by every ``make_sink(...)``. A fresh one when omitted.

    Returns
    -------
    tuple of FakeGate and FakeSink
        The objects now in use.
    """
    use_gate = gate or FakeGate()
    use_sink = sink or FakeSink()
    monkeypatch.setattr(reply, "BargeGate", lambda **_kwargs: use_gate)
    monkeypatch.setattr(reply, "make_sink", lambda *_args, **_kwargs: use_sink)
    return use_gate, use_sink


def set_tts(
    monkeypatch: pytest.MonkeyPatch,
    tts: IsolatedFishTts,
    *,
    speak: Callable[..., TtsResult] | None = None,
    speak_stream: Callable[..., TtsResult] | None = None,
) -> None:
    """Replace the speak methods of an ``IsolatedFishTts`` for one test.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The test's patcher.
    tts : IsolatedFishTts
        The object to patch.
    speak : Callable or None, optional
        Replaces ``speak_isolated``.
    speak_stream : Callable or None, optional
        Replaces ``speak_deltas_isolated``.
    """
    if speak is not None:
        monkeypatch.setattr(tts, "speak_isolated", speak)
    if speak_stream is not None:
        monkeypatch.setattr(tts, "speak_deltas_isolated", speak_stream)


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
    """Build an ``TtsResult`` by keyword, so reordering its fields cannot change a test."""
    return TtsResult(
        spoken_so_far=spoken,
        bytes_played=bytes_played,
        got_audio=got_audio,
        cancelled=cancelled,
        tts_first_audio_ms=tts_first_audio_ms,
        tts_first_text_ms=tts_first_text_ms,
        error_status=error_status,
        error_message=error_message,
        error=(
            None
            if error_status is None
            else FishHttpError.from_status(error_status, error_message or "")
        ),
    )


def install_vad(monkeypatch: pytest.MonkeyPatch, vad_class: type[Any]) -> None:
    """Make ``import webrtcvad`` return a module whose ``Vad`` is ``vad_class``."""
    module = ModuleType("webrtcvad")
    vars(module)["Vad"] = vad_class
    monkeypatch.setitem(sys.modules, "webrtcvad", module)
