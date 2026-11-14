"""Properties that hold for any token split of a streamed reply."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Any

from fishaudio import FlushEvent, TextEvent
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from fish_audio_suite_kit import CueMode, CueStyle, ends_sentence
from fish_audio_suite_voice.tts.text import delta_events, text_events

# Pieces that exercise the scrubber's state machine: spans that hold text until
# they close, line and sentence starts, cues, and the break characters it folds.
_FRAGMENTS = (
    "Hello",
    " there",
    " friend",
    ".",
    "!",
    "?",
    ",",
    " ",
    "  ",
    "\n",
    "\n\n",
    "\r",
    "\r\n",
    " ",
    "*",
    "**",
    "_",
    "`",
    "```",
    "~~~",
    "(",
    ")",
    "[",
    "]",
    "[calm]",
    "[whispering] ",
    "<think>",
    "</think>",
    " (https://example.com)",
    "http://a.b/c",
    "Excited",
    "Happy,",
    "Dr.",
    "- item",
    "你好",
    "。",
)

_texts = st.lists(st.sampled_from(_FRAGMENTS), max_size=14).map("".join)


@st.composite
def _token_streams(draw: st.DrawFn) -> list[str]:
    text = draw(_texts)
    cuts = sorted(draw(st.sets(st.integers(0, len(text)), max_size=8)))
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in pairwise(bounds)]


_partial = st.integers(min_value=1, max_value=60)
_flags = st.booleans()

_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _run(
    tokens: list[str],
    *,
    partial: int,
    mood_lead: bool = False,
    cancelled: bool = False,
) -> list[Any]:
    """The events the speaker sends Fish for ``tokens``, with its early flush."""
    cancel = threading.Event()
    if cancelled:
        cancel.set()

    async def collect() -> list[Any]:
        return [
            ev
            async for ev in delta_events(
                tokens,
                cancel,
                partial_chars=partial,
                style=CueStyle(lead=mood_lead),
            )
        ]

    return asyncio.run(collect())


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_only_text_and_flush_events_come_out_and_never_an_empty_one(
    tokens: list[str], partial: int, mood: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood)
    for event in events:
        assert isinstance(event, TextEvent | FlushEvent)
        if isinstance(event, TextEvent):
            assert event.text.strip()


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_a_reply_with_text_ends_in_a_flush_and_one_without_sends_nothing(
    tokens: list[str], partial: int, mood: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood)
    texts = [e for e in events if isinstance(e, TextEvent)]
    if texts:
        assert isinstance(events[-1], FlushEvent)
    else:
        assert events == []


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_an_early_flush_never_leads_and_two_flushes_are_never_adjacent(
    tokens: list[str], partial: int, mood: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood)
    if events:
        assert isinstance(events[0], TextEvent)
    for first, second in pairwise(events):
        assert not (isinstance(first, FlushEvent) and isinstance(second, FlushEvent))
    assert len([e for e in events if isinstance(e, FlushEvent)]) <= 2


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_a_cancelled_stream_sends_nothing(tokens: list[str], partial: int, mood: bool) -> None:
    assert _run(tokens, partial=partial, mood_lead=mood, cancelled=True) == []


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_the_same_tokens_give_the_same_events(tokens: list[str], partial: int, mood: bool) -> None:
    def shape(events: list[Any]) -> list[tuple[str, str]]:
        return [("text", e.text) if isinstance(e, TextEvent) else ("flush", "") for e in events]

    first = _run(tokens, partial=partial, mood_lead=mood)
    second = _run(tokens, partial=partial, mood_lead=mood)
    assert shape(first) == shape(second)


async def _events_of(tokens: list[str], *, partial: int = 40) -> list[tuple[str, str]]:
    return [
        ("text", ev.text) if isinstance(ev, TextEvent) else ("flush", "")
        async for ev in delta_events(
            tokens, threading.Event(), partial_chars=partial, style=CueStyle()
        )
    ]


def test_the_early_flush_waits_for_the_end_of_the_first_sentence() -> None:
    # "[curious] Oh, I can definitely hear you! " is 41 characters, one over the
    # window, so the cut drops "you!" into the next piece. Flushing after the
    # first piece sent "...hear" to Fish as a finished utterance, and it paused
    # before "you!".
    events = asyncio.run(
        _events_of(["[curious] Oh, I can definitely hear you! [excited", "] What's on your mind?"])
    )
    first_flush = events.index(("flush", ""))
    spoken_before = "".join(text for kind, text in events[:first_flush] if kind == "text")
    assert spoken_before == "[curious] Oh, I can definitely hear you! "


def test_a_long_first_sentence_without_a_stop_still_flushes_early() -> None:
    words = " ".join(["word"] * 60)
    events = asyncio.run(_events_of([words], partial=40))
    flushes = [i for i, event in enumerate(events) if event == ("flush", "")]
    # The reply is one sentence with no stop. Waiting for the end would hold the
    # first audio until the model finished, so a long run flushes anyway.
    assert len(flushes) == 2
    spoken_before = "".join(t for k, t in events[: flushes[0]] if k == "text")
    assert len(spoken_before) >= 80


@_SETTINGS
@example(tokens=["Dr. friendh", "ttp://a.b/c"], partial=18, mood=False)  # a lone "." piece
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_the_early_flush_never_cuts_a_sentence_short(
    tokens: list[str], partial: int, mood: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood)
    flushes = [i for i, event in enumerate(events) if isinstance(event, FlushEvent)]
    if len(flushes) < 2:
        return  # no early flush: the reply ended before one was due
    before = "".join(e.text for e in events[: flushes[0]] if isinstance(e, TextEvent))
    assert ends_sentence(before) or len(before) >= 2 * partial


def test_a_stop_split_from_its_abbreviation_is_not_a_sentence_end() -> None:
    # "Mr" and "." arrive as separate pieces. The "." alone looks like a sentence
    # end, but "Mr." is not one, so the early flush must wait for the real stop.
    events = asyncio.run(
        _events_of(["Hello Mr", ".", " Smith is here today friend. More text."], partial=8)
    )
    first_flush = events.index(("flush", ""))
    spoken_before = "".join(text for kind, text in events[:first_flush] if kind == "text")
    assert "Smith" in spoken_before


# Prose as a model writes it: words, some wrapped in emphasis with a colon or comma
# inside or outside the marks, an optional mood lead, sentences joined by a space, a
# tab or a line break. Gaps between words are spaces or tabs.
_WORD = st.sampled_from(("hello", "there", "friend", "Note", "today", "works"))


@st.composite
def _phrase(draw: st.DrawFn) -> str:
    word = draw(_WORD)
    mark = draw(st.sampled_from(("", "", "*", "**")))
    punct = draw(st.sampled_from(("", "", ":", ",")))
    if draw(st.booleans()):
        return f"{mark}{word}{punct}{mark}"
    return f"{mark}{word}{mark}{punct}"


@st.composite
def _sentence(draw: st.DrawFn) -> str:
    lead = draw(st.sampled_from(("", "", "Excited, ", "Happy, ", "Calm, ")))
    phrases = draw(st.lists(_phrase(), min_size=1, max_size=4))
    body = phrases[0]
    for phrase in phrases[1:]:
        body += draw(st.sampled_from((" ", " ", "\t"))) + phrase
    return lead + body + draw(st.sampled_from((".", "!", "?")))


@st.composite
def _prose_streams(draw: st.DrawFn) -> list[str]:
    sentences = draw(st.lists(_sentence(), min_size=1, max_size=3))
    text = sentences[0]
    for sentence in sentences[1:]:
        text += draw(st.sampled_from((" ", "\t", "\n", "\n\n"))) + sentence
    cuts = sorted(draw(st.sets(st.integers(0, len(text)), max_size=10)))
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in pairwise(bounds)]


@_SETTINGS
@given(
    tokens=_prose_streams(),
    partial=st.integers(min_value=20, max_value=300),
    mode=st.sampled_from(("official", "free", "directed")),
    lead=_flags,
)
def test_streamed_prose_sends_the_words_and_cues_the_whole_reply_sends(
    tokens: list[str], partial: int, mode: CueMode, lead: bool
) -> None:
    style = CueStyle(mode, lead=lead)

    async def texts(events: AsyncIterator[Any]) -> str:
        return "".join([e.text async for e in events if isinstance(e, TextEvent)])

    streamed = asyncio.run(
        texts(delta_events(tokens, threading.Event(), partial_chars=partial, style=style))
    )
    whole = asyncio.run(
        texts(text_events(style.shape("".join(tokens)), threading.Event(), partial, style=style))
    )
    # Fish reads any run of spaces, tabs and line breaks as one gap.
    assert streamed.split() == whole.split()
