"""Short names for what the session prints."""

from __future__ import annotations

from fish_audio_suite_voice.wording import short_model


def test_short_model_drops_only_the_vendor_prefix() -> None:
    assert short_model("cognitivecomputations/dolphin-mistral-24b-venice-edition") == (
        "dolphin-mistral-24b-venice-edition"
    )
    assert short_model("openai/gpt-4o:nitro") == "gpt-4o:nitro"
    assert short_model("local-model") == "local-model"
    assert short_model(None) == ""
