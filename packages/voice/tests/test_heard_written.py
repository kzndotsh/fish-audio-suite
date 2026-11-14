"""The conversation records what the model wrote of the heard part, not the cues added for Fish."""

from __future__ import annotations

import asyncio
import threading

from fishaudio import TextEvent

from fish_audio_suite_kit import CueStyle
from fish_audio_suite_voice.tts.played_text import heard_written, spoken_prefix, unheard
from fish_audio_suite_voice.tts.text import delta_events

STYLE = CueStyle.for_model("s2.1-pro", official=True, lead=False)
WRITTEN = "[calm] The first thing. The second thing. The third thing."
RATE = 16_000


def _sent(tokens: list[str]) -> str:
    """Return the text Fish is sent for a streamed reply, piece by piece, joined."""

    async def run() -> list[str]:
        events = delta_events(tokens, threading.Event(), partial_chars=20, style=STYLE)
        return [ev.text async for ev in events if isinstance(ev, TextEvent)]

    return "".join(asyncio.run(run()))


def _heard(sent: str, chars: int | None) -> str:
    """Return ``spoken_prefix`` of ``sent``: all of it, or about ``chars`` spoken characters."""
    if chars is None:
        return spoken_prefix(
            sent,
            bytes_played=1,
            sample_rate=RATE,
            audio_format="pcm",
            got_audio=True,
            cancelled=False,
        )
    return spoken_prefix(
        sent,
        bytes_played=chars * RATE * 2 // 16,  # 16 characters a second at speed 1
        sample_rate=RATE,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )


SENT = _sent(["[calm] The first thing. ", "The second thing. ", "The third thing."])


def test_fish_still_gets_the_mood_carried_into_each_sentence() -> None:
    assert SENT.count("[calm]") == 3


def test_a_reply_heard_whole_is_recorded_as_written() -> None:
    assert heard_written(WRITTEN, _heard(SENT, None)) == WRITTEN


def test_a_reply_cut_mid_sentence_keeps_the_written_cue_and_none_carried() -> None:
    cut = len("The first thing. The second")
    assert heard_written(WRITTEN, _heard(SENT, cut)) == "[calm] The first thing. The second"


def test_the_default_lead_cue_is_not_recorded() -> None:
    written = "Hello there friend."
    for heard in (_heard(f"[calm] {written}", None), _heard(f"[calm] {written}", 11)):
        assert "[calm]" not in heard_written(written, heard)
    assert heard_written(written, _heard(f"[calm] {written}", None)) == written


def test_a_cue_the_model_wrote_mid_reply_is_kept() -> None:
    written = "[calm] The first thing. [sad] The second thing."
    sent = _sent(["[calm] The first thing. ", "[sad] The second thing."])
    assert heard_written(written, _heard(sent, None)) == written
    cut = len("The first thing. The second")
    assert heard_written(written, _heard(sent, cut)) == "[calm] The first thing. [sad] The second"


def test_text_that_does_not_match_keeps_no_cue_at_all() -> None:
    # The stream shaped a piece differently from the whole reply: the heard words are kept,
    # and no cue, since none can be told apart from one added for Fish.
    assert heard_written("Something else entirely.", "[calm] Hello. [calm] There.") == (
        "Hello. There."
    )


def test_nothing_heard_records_nothing() -> None:
    assert heard_written(WRITTEN, "") == ""


def test_the_rest_of_a_cut_reply_is_sent_with_the_mood_and_recorded_without() -> None:
    rest = unheard(SENT, WRITTEN, "The first thing.")
    assert rest is not None
    assert rest.sent.startswith("[calm] The second thing.")
    assert rest.written == "The second thing. The third thing."


def test_nothing_is_left_to_resume_once_all_was_heard() -> None:
    assert unheard(SENT, WRITTEN, _heard(SENT, None)) is None
