"""Names renamed in 0.2.0 keep working for one minor release, each with a DeprecationWarning."""

from __future__ import annotations

import importlib
import threading
from typing import Any, cast

import pytest

import fish_audio_suite_voice as voice
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.tune import AecTune, BargeTune, ListenTune, LlmSettings
from fish_audio_suite_voice.wire import TtsResult


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("fish_audio_suite_voice", "IsolatedResult", "TtsResult"),
        ("fish_audio_suite_voice", "LlmTune", "LlmSettings"),
        ("fish_audio_suite_voice.wire", "IsolatedResult", "TtsResult"),
        ("fish_audio_suite_voice.live", "IsolatedResult", "TtsResult"),
        ("fish_audio_suite_voice.tune", "LlmTune", "LlmSettings"),
    ],
)
def test_an_old_class_name_still_resolves_and_warns(module: str, old: str, new: str) -> None:
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated since 0.2.0; use {new}"):
        found = getattr(mod, old)
    assert found is getattr(mod, new)
    assert old not in cast(list[str], mod.__all__)


def test_an_old_name_imports_with_from_and_warns() -> None:
    with pytest.warns(DeprecationWarning, match="use TtsResult"):
        from fish_audio_suite_voice import IsolatedResult
    assert IsolatedResult is voice.TtsResult


def test_an_unknown_name_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = cast(Any, voice).Nope


def _result(**timings: float | None) -> TtsResult:
    return cast(Any, TtsResult)(
        spoken_so_far="hi", bytes_played=2, got_audio=True, cancelled=False, **timings
    )


def test_tts_result_old_timing_keywords_map_to_the_new_fields() -> None:
    with pytest.warns(DeprecationWarning, match=r"TtsResult\(ttfa_ms=\.\.\.\)"):
        result = _result(ttfa_ms=120.0, tts_first_text_ms=40.0)
    assert result.tts_first_audio_ms == 120.0
    with pytest.warns(DeprecationWarning, match=r"TtsResult\(llm_ttfs_ms=\.\.\.\)"):
        result = _result(tts_first_audio_ms=120.0, llm_ttfs_ms=40.0)
    assert result.tts_first_text_ms == 40.0


def test_tts_result_old_timing_attributes_read_the_new_fields() -> None:
    result = TtsResult("hi", 2, True, False, 120.0, 40.0)
    with pytest.warns(DeprecationWarning, match=r"TtsResult\.ttfa_ms .*use tts_first_audio_ms"):
        assert cast(Any, result).ttfa_ms == 120.0
    with pytest.warns(DeprecationWarning, match=r"TtsResult\.llm_ttfs_ms .*use tts_first_text_ms"):
        assert cast(Any, result).llm_ttfs_ms == 40.0


def test_speak_stream_isolated_warns_and_runs_speak_deltas_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = IsolatedFishTts(api_key="k", voice_id="v")
    seen: list[tuple[object, ...]] = []
    sentinel = _result(tts_first_audio_ms=None, tts_first_text_ms=None)

    def fake(deltas: object, sink: object, **kwargs: object) -> TtsResult:
        seen.append((deltas, sink, kwargs))
        return sentinel

    monkeypatch.setattr(tts, "speak_deltas_isolated", fake)
    cancel = threading.Event()
    with pytest.warns(DeprecationWarning, match="use IsolatedFishTts.speak_deltas_isolated"):
        out = tts.speak_stream_isolated(["a"], cast(Any, "sink"), cancel=cancel)
    assert out is sentinel
    assert seen == [(["a"], "sink", {"cancel": cancel, "on_first_audio": None})]


def test_duplex_session_stop_is_a_deprecated_view_of_quit_requested() -> None:
    session = DuplexSession()
    session.request_quit()
    with pytest.warns(DeprecationWarning, match=r"DuplexSession\.stop .*use quit_requested"):
        assert cast(Any, session).stop is session.quit_requested
    event = threading.Event()
    with pytest.warns(DeprecationWarning, match=r"DuplexSession\(stop=\.\.\.\)"):
        built = cast(Any, DuplexSession)(stop=event)
    assert built.quit_requested is event


def test_llm_settings_old_key_and_openrouter_names_still_work() -> None:
    with pytest.warns(DeprecationWarning, match=r"LlmSettings\(key=\.\.\.\)"):
        settings = cast(Any, LlmSettings)(backend="openrouter", key="sk")
    assert settings.api_key == "sk"
    with pytest.warns(DeprecationWarning, match=r"LlmSettings\.key .*use api_key"):
        assert settings.key == "sk"
    with pytest.warns(DeprecationWarning, match="use LlmSettings.uses_openrouter_sdk"):
        assert settings.openrouter is True
    assert "sk" not in repr(settings)


@pytest.mark.parametrize(
    ("cls", "old", "new", "value"),
    [
        (ListenTune, "silence_frames_end", "end_silence_frames", 33),
        (ListenTune, "speech_frames_start", "start_speech_frames", 5),
        (ListenTune, "min_voiced", "min_voiced_frames", 9),
        (BargeTune, "over", "playing_gain", 3.0),
        (AecTune, "bleed_s", "bleed_delay_s", 0.4),
    ],
)
def test_a_renamed_tune_field_keeps_its_old_keyword_and_attribute(
    cls: type, old: str, new: str, value: float
) -> None:
    with pytest.warns(DeprecationWarning, match=rf"{cls.__name__}\({old}=\.\.\.\) .*use {new}"):
        tune = cls(**{old: value})
    assert getattr(tune, new) == value
    with pytest.warns(DeprecationWarning, match=rf"{cls.__name__}\.{old} .*use {new}"):
        assert getattr(tune, old) == value
    with pytest.raises(TypeError, match="got both"):
        cls(**{old: value, new: value})
