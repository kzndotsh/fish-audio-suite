"""Env keys renamed in 0.2.0: the old name still works with a printed warning; the new one wins."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, NamedTuple

import pytest

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_voice.config import VoiceCliConfig, load_config
from fish_audio_suite_voice.tune import DEFAULT_HISTORY_TURNS


class Renamed(NamedTuple):
    new: str
    old: str
    old_raw: str
    old_value: Any
    new_raw: str
    new_value: Any
    read: Callable[[VoiceCliConfig], Any]


_RENAMED: list[Renamed] = [
    Renamed(
        "FISH_VOICE_PLAYBACK",
        "FISH_PLAYBACK",
        "file",
        "file",
        "stdout",
        "stdout",
        attrgetter("playback"),
    ),
    Renamed(
        "FISH_VOICE_HISTORY_TURNS",
        "FISH_HISTORY_TURNS",
        "7",
        7,
        "9",
        9,
        attrgetter("history_turns"),
    ),
    Renamed(
        "FISH_VOICE_SYSTEM_PROMPT",
        "FISH_SYSTEM_PROMPT",
        "old prompt",
        "old prompt",
        "new prompt",
        "new prompt",
        attrgetter("system_prompt"),
    ),
    Renamed(
        "FISH_VOICE_STREAM_TTS", "FISH_STREAM_TTS", "1", True, "0", False, attrgetter("stream_tts")
    ),
    Renamed(
        "FISH_VOICE_REPEAT_WINDOW",
        "FISH_VOICE_REPEAT_WINDOW_S",
        "2.5",
        2.5,
        "0.5",
        0.5,
        attrgetter("repeat_window_s"),
    ),
    Renamed(
        "FISH_VOICE_PRE_PAD_FRAMES",
        "FISH_VOICE_PRE_PAD",
        "40",
        40,
        "50",
        50,
        attrgetter("listen.pre_pad_frames"),
    ),
    Renamed(
        "FISH_VOICE_MIN_VOICED_FRAMES",
        "FISH_VOICE_MIN_VOICED",
        "9",
        9,
        "15",
        15,
        attrgetter("listen.min_voiced_frames"),
    ),
    Renamed(
        "FISH_VOICE_BARGE_PLAYING_GAIN",
        "FISH_VOICE_BARGE_OVER",
        "3.5",
        3.5,
        "1.5",
        1.5,
        attrgetter("barge.playing_gain"),
    ),
    Renamed(
        "FISH_VOICE_AEC_BLEED_DELAY",
        "FISH_VOICE_AEC_BLEED",
        "0.5",
        0.5,
        "0.7",
        0.7,
        attrgetter("aec.bleed_delay_s"),
    ),
    Renamed("FISH_TTS_MOOD_LEAD", "FISH_MOOD_LEAD", "1", True, "0", False, attrgetter("mood_lead")),
    Renamed(
        "FISH_TTS_DROP_NARRATION",
        "FISH_DROP_NARRATION",
        "1",
        True,
        "0",
        False,
        attrgetter("drop_narration"),
    ),
    Renamed(
        "FISH_LLM_API_KEY",
        "FISH_LLM_KEY",
        "sk-old",
        "sk-old",
        "sk-new",
        "sk-new",
        attrgetter("llm.api_key"),
    ),
    Renamed(
        "FISH_TTS_MODEL", "FISH_MODEL", "s1", "s1", "s2-pro", "s2-pro", attrgetter("tts_model")
    ),
    Renamed("FISH_SPEED", "FISH_SPEED_SCALE", "1.2", 1.2, "0.8", 0.8, attrgetter("speed")),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for row in _RENAMED:
        monkeypatch.delenv(row.new, raising=False)
        monkeypatch.delenv(row.old, raising=False)


@pytest.mark.parametrize("row", _RENAMED, ids=[row.old for row in _RENAMED])
def test_the_old_name_still_works_and_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], row: Renamed
) -> None:
    monkeypatch.setenv(row.old, row.old_raw)
    assert row.read(load_config()) == row.old_value
    assert f"{row.old} is deprecated; use {row.new}" in capsys.readouterr().err


@pytest.mark.parametrize("row", _RENAMED, ids=[row.old for row in _RENAMED])
def test_the_new_name_wins_and_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], row: Renamed
) -> None:
    monkeypatch.setenv(row.old, row.old_raw)
    monkeypatch.setenv(row.new, row.new_raw)
    assert row.read(load_config()) == row.new_value
    assert row.old not in capsys.readouterr().err


def test_a_blank_old_system_prompt_still_means_no_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_SYSTEM_PROMPT", "")
    assert load_config().system_prompt == ""
    assert "FISH_SYSTEM_PROMPT is deprecated" in capsys.readouterr().err


def test_neither_name_set_keeps_the_default_and_does_not_warn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    c = load_config()
    assert c.history_turns == DEFAULT_HISTORY_TURNS
    assert c.system_prompt == SuiteDefaults().system_prompt
    assert "deprecated" not in capsys.readouterr().err
