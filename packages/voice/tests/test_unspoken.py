from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fish_audio_suite_voice.spoken import unspoken_text

SENT = "[curious] Oh, really? [surprised] I'd love to help you with that. [whispering] Tell me more about it."


@pytest.mark.parametrize(
    ("heard", "rest"),
    [
        ("", SENT),
        (
            "Oh, really?",
            "[surprised] I'd love to help you with that. [whispering] Tell me more about it.",
        ),
        (
            "Oh, really? I'd love to help",
            "[surprised] you with that. [whispering] Tell me more about it.",
        ),
        ("Oh, really? I'd love to help you with that.", "[whispering] Tell me more about it."),
    ],
)
def test_returns_the_rest_led_by_the_cue_in_force(heard: str, rest: str) -> None:
    assert unspoken_text(SENT, heard) == rest


@pytest.mark.parametrize(
    "heard",
    ["wrong words entirely", SENT, "Oh, really? I'd love to help you with that. Tell me more"],
)
def test_gives_nothing_back_when_heard_does_not_fit_or_little_is_left(heard: str) -> None:
    assert unspoken_text(SENT, heard) == ""


def test_works_without_spaces_between_words() -> None:
    assert (
        unspoken_text("你好。我们今天见面吧。很高兴认识你。", "你好。我们")
        == "今天见面吧。很高兴认识你。"
    )


@given(st.text(max_size=80), st.integers(min_value=0, max_value=80))
def test_the_rest_is_always_a_suffix_of_the_reply(text: str, cut: int) -> None:
    rest = unspoken_text(text, text[:cut])
    assert rest == "" or text.endswith(rest) or rest.startswith("[")
