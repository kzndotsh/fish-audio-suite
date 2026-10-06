"""``SttTune``: which recogniser hears the user, read from the environment."""

from __future__ import annotations

import pytest

from fish_audio_suite_voice.tune import (
    DEFAULT_DEEPGRAM_MODEL,
    DEFAULT_EOT_THRESHOLD,
    STT_PROVIDERS,
    SttTune,
)

_KEYS = (
    "FISH_VOICE_STT",
    "DEEPGRAM_API_KEY",
    "FISH_VOICE_DEEPGRAM_MODEL",
    "FISH_VOICE_DEEPGRAM_REGION",
    "FISH_VOICE_EOT_THRESHOLD",
    "FISH_VOICE_STT_SAVE_DIR",
    "FISH_VOICE_EAGER_EOT_THRESHOLD",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_the_defaults_keep_fish_asr_and_need_no_deepgram_setting() -> None:
    tune = SttTune.from_env()
    assert tune == SttTune()
    assert tune.provider == "fish"
    assert tune.deepgram_key == ""
    assert tune.deepgram_model == DEFAULT_DEEPGRAM_MODEL == "flux-general-en"
    assert tune.eot_threshold == DEFAULT_EOT_THRESHOLD == 0.7
    assert tune.deepgram_region == "global"
    assert STT_PROVIDERS == ("fish", "deepgram")


def test_deepgram_is_chosen_by_name_in_any_case_with_its_key_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_STT", " Deepgram ")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "  dg-secret  ")
    monkeypatch.setenv("FISH_VOICE_DEEPGRAM_MODEL", "flux-general-multi")
    monkeypatch.setenv("FISH_VOICE_EOT_THRESHOLD", "0.85")
    monkeypatch.setenv("FISH_VOICE_DEEPGRAM_REGION", " EU ")
    monkeypatch.setenv("FISH_VOICE_STT_SAVE_DIR", " tmp/stt ")
    tune = SttTune.from_env()
    assert tune.provider == "deepgram"
    assert tune.deepgram_key == "dg-secret"
    assert tune.deepgram_model == "flux-general-multi"
    assert tune.eot_threshold == 0.85
    assert tune.deepgram_region == "eu"
    assert tune.save_dir == "tmp/stt"


def test_the_key_is_never_shown_when_the_settings_are_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    assert "dg-secret" not in repr(SttTune.from_env())


@pytest.mark.parametrize(
    ("name", "value", "field", "fallback"),
    [
        ("FISH_VOICE_STT", "whisper", "provider", "fish"),
        ("FISH_VOICE_DEEPGRAM_MODEL", "nova-3", "deepgram_model", DEFAULT_DEEPGRAM_MODEL),
        ("FISH_VOICE_EOT_THRESHOLD", "0.2", "eot_threshold", DEFAULT_EOT_THRESHOLD),
        ("FISH_VOICE_EOT_THRESHOLD", "1.5", "eot_threshold", DEFAULT_EOT_THRESHOLD),
        ("FISH_VOICE_EOT_THRESHOLD", "high", "eot_threshold", DEFAULT_EOT_THRESHOLD),
        ("FISH_VOICE_DEEPGRAM_REGION", "mars", "deepgram_region", "global"),
        ("FISH_VOICE_EAGER_EOT_THRESHOLD", "1.4", "eager_eot_threshold", 0.0),
        ("FISH_VOICE_EAGER_EOT_THRESHOLD", "soon", "eager_eot_threshold", 0.0),
    ],
)
def test_an_unusable_value_is_ignored_with_a_warning_and_the_default_is_used(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    name: str,
    value: str,
    field: str,
    fallback: object,
) -> None:
    monkeypatch.setenv(name, value)
    assert getattr(SttTune.from_env(), field) == fallback
    assert name in capsys.readouterr().err  # the user is told which setting was ignored


def test_a_blank_provider_means_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_STT", "   ")
    assert SttTune.from_env().provider == "fish"


def test_the_early_end_of_turn_is_off_unless_set_and_is_kept_within_flux_limits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert SttTune.from_env().eager_eot_threshold == 0.0  # off: it costs extra model calls
    monkeypatch.setenv("FISH_VOICE_EAGER_EOT_THRESHOLD", "0.5")
    assert SttTune.from_env().eager_eot_threshold == 0.5
    monkeypatch.setenv("FISH_VOICE_EAGER_EOT_THRESHOLD", "0.1")  # below what Flux accepts
    assert SttTune.from_env().eager_eot_threshold == 0.3
    monkeypatch.setenv("FISH_VOICE_EAGER_EOT_THRESHOLD", "0.85")  # above the final threshold
    assert SttTune.from_env().eager_eot_threshold == DEFAULT_EOT_THRESHOLD
    monkeypatch.setenv("FISH_VOICE_EOT_THRESHOLD", "0.9")
    assert SttTune.from_env().eager_eot_threshold == 0.85
    assert "FISH_VOICE_EAGER_EOT_THRESHOLD" in capsys.readouterr().err
