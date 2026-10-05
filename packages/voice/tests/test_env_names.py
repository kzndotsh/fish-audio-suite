"""Each voice environment name reaches its setting. A blank system prompt means none."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, NamedTuple

import pytest

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_voice.config import VoiceCliConfig, load_config
from fish_audio_suite_voice.llm_tune import DEFAULT_HISTORY_TURNS


class EnvCase(NamedTuple):
    name: str
    raw: str
    value: Any
    read: Callable[[VoiceCliConfig], Any]


_CASES: list[EnvCase] = [
    EnvCase("FISH_VOICE_PLAYBACK", "file", "file", attrgetter("playback")),
    EnvCase("FISH_VOICE_OUTPUT_LATENCY", " HIGH ", "high", attrgetter("output_latency")),
    EnvCase("FISH_VOICE_HISTORY_TURNS", "7", 7, attrgetter("history_turns")),
    EnvCase("FISH_VOICE_SYSTEM_PROMPT", "a prompt", "a prompt", attrgetter("system_prompt")),
    EnvCase("FISH_VOICE_STREAM_TTS", "1", True, attrgetter("stream_tts")),
    EnvCase("FISH_VOICE_REPEAT_WINDOW", "2.5", 2.5, attrgetter("repeat_window_s")),
    EnvCase("FISH_VOICE_FADE_MS", "6", 6.0, attrgetter("fade_ms")),
    EnvCase("FISH_VOICE_PRE_PAD_FRAMES", "40", 40, attrgetter("listen.pre_pad_frames")),
    EnvCase("FISH_VOICE_MIN_VOICED_FRAMES", "9", 9, attrgetter("listen.min_voiced_frames")),
    EnvCase("FISH_VOICE_BARGE_PLAYING_GAIN", "3.5", 3.5, attrgetter("barge.playing_gain")),
    EnvCase("FISH_VOICE_AEC_BLEED_DELAY", "0.5", 0.5, attrgetter("aec.bleed_delay_s")),
    EnvCase("FISH_TTS_MOOD_LEAD", "1", True, attrgetter("mood_lead")),
    EnvCase("FISH_TTS_DROP_NARRATION", "1", True, attrgetter("drop_narration")),
    EnvCase("FISH_LLM_API_KEY", "sk-new", "sk-new", attrgetter("llm.api_key")),
    EnvCase("FISH_TTS_MODEL", "s1", "s1", attrgetter("tts_model")),
    EnvCase("FISH_SPEED", "1.2", 1.2, attrgetter("speed")),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for case in _CASES:
        monkeypatch.delenv(case.name, raising=False)


@pytest.mark.parametrize("case", _CASES, ids=[case.name for case in _CASES])
def test_the_env_name_reaches_its_setting(monkeypatch: pytest.MonkeyPatch, case: EnvCase) -> None:
    assert case.read(load_config()) != case.value
    monkeypatch.setenv(case.name, case.raw)
    assert case.read(load_config()) == case.value


def test_a_blank_system_prompt_means_no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "")
    assert load_config().system_prompt == ""


def test_unset_keeps_the_defaults() -> None:
    c = load_config()
    assert c.history_turns == DEFAULT_HISTORY_TURNS
    assert c.system_prompt == SuiteDefaults().system_prompt


def test_an_unknown_output_latency_is_ignored_and_auto_means_low_outside_the_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert load_config().sink_latency == "low"  # auto
    monkeypatch.setenv("FISH_VOICE_OUTPUT_LATENCY", "enormous")
    config = load_config()
    assert config.output_latency == "auto"
    assert config.sink_latency == "low"
