from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    hold_tts,
)


def test_hold_tts_keeps_an_unfinished_span() -> None:
    assert hold_tts("<think", line_start=True, sentence_start=True) == 0
    partial = "See https://exa"
    assert hold_tts(partial, line_start=True, sentence_start=False) < len(partial)
    fence = "Open the door.\n```"
    assert hold_tts(fence, line_start=True, sentence_start=False) <= fence.index("`")


@pytest.mark.parametrize(
    ("text", "held_at"),
    [
        ("Hello <thinking>still going", 6),
        ("Hello <thinking>done</thinking> there", None),
        ("Hello <whisper>quiet", 6),
        ("Hello <script>x</style>", 6),
        ("Hello <script>x</script> there", None),
        ("code ```print(1)", 5),
        ("code ~~~print(1)", 5),
        ("see https://exam", 4),
        ("see <http", 4),
        ("Hello (she smiles", 6),
        ("Hello (she smiles) there", None),
        ("Hello <thi", 6),
    ],
)
def test_unfinished_spans_hold_at_their_opener(text: str, held_at: int | None) -> None:
    result = hold_tts(text, line_start=True, sentence_start=False)
    assert result == (held_at if held_at is not None else len(text))
