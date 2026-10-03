"""Properties that hold for any token split of a streamed reply."""

from __future__ import annotations

import asyncio
import threading
from itertools import pairwise
from typing import Any

from fishaudio import FlushEvent, TextEvent
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fish_audio_suite_kit import skip_empty_delta
from fish_audio_suite_voice.stream_scrub import delta_events

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
    early_flush: bool = False,
    cancelled: bool = False,
) -> list[Any]:
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
                mood_lead=mood_lead,
                early_flush=early_flush,
            )
        ]

    return asyncio.run(collect())


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags, early=_flags)
def test_only_text_and_flush_events_come_out_and_never_an_empty_one(
    tokens: list[str], partial: int, mood: bool, early: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood, early_flush=early)
    for event in events:
        assert isinstance(event, TextEvent | FlushEvent)
        if isinstance(event, TextEvent):
            assert not skip_empty_delta(event.text)


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_a_reply_with_text_ends_in_one_flush_and_one_without_sends_nothing(
    tokens: list[str], partial: int, mood: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood)
    texts = [e for e in events if isinstance(e, TextEvent)]
    flushes = [e for e in events if isinstance(e, FlushEvent)]
    if texts:
        assert isinstance(events[-1], FlushEvent)
        assert len(flushes) == 1
    else:
        assert events == []


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags)
def test_an_early_flush_never_leads_and_two_flushes_are_never_adjacent(
    tokens: list[str], partial: int, mood: bool
) -> None:
    events = _run(tokens, partial=partial, mood_lead=mood, early_flush=True)
    if events:
        assert isinstance(events[0], TextEvent)
    for first, second in pairwise(events):
        assert not (isinstance(first, FlushEvent) and isinstance(second, FlushEvent))
    assert len([e for e in events if isinstance(e, FlushEvent)]) <= 2


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags, early=_flags)
def test_a_cancelled_stream_sends_nothing(
    tokens: list[str], partial: int, mood: bool, early: bool
) -> None:
    assert _run(tokens, partial=partial, mood_lead=mood, early_flush=early, cancelled=True) == []


@_SETTINGS
@given(tokens=_token_streams(), partial=_partial, mood=_flags, early=_flags)
def test_the_same_tokens_give_the_same_events(
    tokens: list[str], partial: int, mood: bool, early: bool
) -> None:
    def shape(events: list[Any]) -> list[tuple[str, str]]:
        return [("text", e.text) if isinstance(e, TextEvent) else ("flush", "") for e in events]

    first = _run(tokens, partial=partial, mood_lead=mood, early_flush=early)
    second = _run(tokens, partial=partial, mood_lead=mood, early_flush=early)
    assert shape(first) == shape(second)
