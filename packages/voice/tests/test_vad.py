"""Choosing, opening and plugging in voice-activity detectors."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from voice_fakes import ScriptedMic, install_vad

from fish_audio_suite_voice.listen import record_utterance, stream_utterance
from fish_audio_suite_voice.mic import FRAME_BYTES
from fish_audio_suite_voice.tune import DEFAULT_VAD_ENGINE, ListenTune
from fish_audio_suite_voice.vad import SILERO_THRESHOLDS, VAD_ENGINES, open_vad, vad_problem

_SILERO_CHUNK = 1024


class _ModeVad:
    def __init__(self, mode: int) -> None:
        self.mode = mode

    def is_speech(self, frame: bytes, rate: int) -> bool:
        del frame, rate
        return False


def test_the_default_engine_is_webrtc_at_the_given_strictness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_vad(monkeypatch, _ModeVad)
    vad = open_vad(DEFAULT_VAD_ENGINE, 2)
    assert isinstance(vad, _ModeVad)
    assert vad.mode == 2


def test_the_engine_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_VAD_ENGINE", "silero")
    assert ListenTune.from_env().vad_engine == "silero"
    monkeypatch.setenv("FISH_VOICE_VAD_ENGINE", "  ")
    assert ListenTune.from_env().vad_engine == DEFAULT_VAD_ENGINE


def test_a_plugin_factory_is_loaded_from_its_module_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = ModuleType("my_vad_plugin")
    vars(plugin)["make"] = _ModeVad
    monkeypatch.setitem(sys.modules, "my_vad_plugin", plugin)
    vad = open_vad("my_vad_plugin:make", 3)
    assert isinstance(vad, _ModeVad)
    assert vad.mode == 3
    assert vad_problem("my_vad_plugin:make", 3) is None


def test_an_unknown_engine_name_lists_the_known_ones() -> None:
    problem = vad_problem("silro", 1)
    assert problem is not None
    assert "'silro'" in problem
    assert all(name in problem for name in VAD_ENGINES)


def test_a_missing_extra_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pysilero_vad", None)
    problem = vad_problem("silero", 1)
    assert problem is not None
    assert "fish-audio-suite-voice[silero]" in problem


def test_a_plugin_that_cannot_import_is_reported() -> None:
    problem = vad_problem("no_such_vad_module:make", 1)
    assert problem is not None
    assert "cannot be imported" in problem


class _FakeSilero:
    """Records the chunks it is given and answers with scripted probabilities."""

    def __init__(self, probabilities: list[float]) -> None:
        self.chunks: list[int] = []
        self._probabilities = iter(probabilities)

    def chunk_bytes(self) -> int:
        return _SILERO_CHUNK

    def __call__(self, audio: bytes) -> float:
        self.chunks.append(len(audio))
        return next(self._probabilities)


def _install_silero(monkeypatch: pytest.MonkeyPatch, fake: _FakeSilero) -> None:
    module = ModuleType("pysilero_vad")
    vars(module)["SileroVoiceActivityDetector"] = lambda: fake
    monkeypatch.setitem(sys.modules, "pysilero_vad", module)


def test_silero_rebuffers_30_ms_frames_into_its_32_ms_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeSilero([0.9, 0.1])
    _install_silero(monkeypatch, fake)
    vad = open_vad("silero", 1)
    frame = b"\x00" * FRAME_BYTES
    # 960-byte frames against 1024-byte chunks: nothing is scored until the second frame.
    assert [vad.is_speech(frame, 16000) for _ in range(3)] == [False, True, False]
    assert fake.chunks == [_SILERO_CHUNK, _SILERO_CHUNK]


def test_silero_strictness_sets_the_probability_it_must_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    between = (SILERO_THRESHOLDS[1] + SILERO_THRESHOLDS[2]) / 2
    frames = [b"\x00" * _SILERO_CHUNK]
    _install_silero(monkeypatch, _FakeSilero([between]))
    assert open_vad("silero", 1).is_speech(frames[0], 16000) is True
    _install_silero(monkeypatch, _FakeSilero([between]))
    assert open_vad("silero", 2).is_speech(frames[0], 16000) is False


def test_the_real_silero_model_hears_no_speech_in_silence() -> None:
    pytest.importorskip("pysilero_vad")
    vad = open_vad("silero", 1)
    assert not any(vad.is_speech(b"\x00" * FRAME_BYTES, 16000) for _ in range(20))


@pytest.mark.parametrize("streamed", [False, True])
def test_listening_opens_the_engine_and_strictness_the_tune_names(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    opened: list[int] = []

    def make(strictness: int) -> _ModeVad:
        opened.append(strictness)
        return _ModeVad(strictness)

    plugin = ModuleType("listen_vad_plugin")
    vars(plugin)["make"] = make
    monkeypatch.setitem(sys.modules, "listen_vad_plugin", plugin)
    tune = ListenTune(vad_aggressiveness=2, vad_engine="listen_vad_plugin:make")
    mic = ScriptedMic([b"\x00" * FRAME_BYTES])
    if streamed:
        stream_utterance(mic, None, lambda _frame, _voiced: None, tune=tune)
    else:
        record_utterance(mic, tune=tune, give_up_frames=1)
    assert opened == [2]
