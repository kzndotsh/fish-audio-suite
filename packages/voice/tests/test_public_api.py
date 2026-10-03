"""The public surface: every module says what it exports, and the root exports match."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, cast

import pytest

import fish_audio_suite_voice as voice
from fish_audio_suite_voice.debug import DebugLevel, configure_voice_logging, debug_level
from fish_audio_suite_voice.listen import start_hit
from fish_audio_suite_voice.tune import LlmTune, read_flag


def _modules() -> list[str]:
    return sorted(info.name for info in pkgutil.iter_modules(voice.__path__))


@pytest.mark.parametrize("name", _modules())
def test_every_module_lists_what_it_exports(name: str) -> None:
    module = importlib.import_module(f"fish_audio_suite_voice.{name}")
    names = cast(list[str] | None, getattr(module, "__all__", None))
    assert names is not None, f"{name} has no __all__"
    assert len(set(names)) == len(names)
    for exported in names:
        assert not exported.startswith("_"), f"{name} exports a private name {exported}"
        assert hasattr(module, exported), f"{name}.__all__ lists {exported}, which is missing"


def test_the_root_exports_the_types_public_signatures_use() -> None:
    for name in (
        "EchoCanceller",
        "AecTune",
        "BargeTune",
        "ListenTune",
        "LlmTune",
        "PortAudioMissingError",
        "IsolatedResult",
        "DuplexSession",
        "ChatBackend",
        "PlaybackKind",
    ):
        assert name in voice.__all__, name
        assert getattr(voice, name) is not None


def test_the_debug_level_is_a_named_ordered_value() -> None:
    assert DebugLevel.OFF < DebugLevel.EVENTS < DebugLevel.TRACE
    assert DebugLevel.TRACE == 2
    configure_voice_logging(debug=9)
    assert debug_level() is DebugLevel.TRACE
    configure_voice_logging(debug=False)


def test_the_llm_backend_is_one_of_the_two_names() -> None:
    assert LlmTune().backend in {"openrouter", "openai"}


def test_boolean_flags_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        cast(Any, read_flag)("FISH_X", True)
    with pytest.raises(TypeError):
        cast(Any, start_hit)(200.0, 200.0, True)
