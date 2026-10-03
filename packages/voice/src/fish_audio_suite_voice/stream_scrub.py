"""Incremental scrub for a streamed reply: hold unclosed spans, cut, then send.

The duplex loop waits for the whole reply and does not use this. It backs
``IsolatedFishTts.speak_deltas`` for callers that forward model tokens as they
arrive.
"""

from __future__ import annotations

import re
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from typing import Any, Final

from fishaudio import FlushEvent, TextEvent

from fish_audio_suite_kit import (
    ends_sentence,
    hold_tts,
    normalize_cues,
    scrub_tts,
    skip_empty_delta,
    split_tts_piece,
)
from fish_audio_suite_voice.wire import as_async, flush_if_sent

__all__ = [
    "delta_events",
]


def _text_event(piece: str) -> TextEvent | None:
    # The piece was already scrubbed, including one edge space. Scrubbing
    # again strips that space and the next cut is spoken as one word.
    if skip_empty_delta(piece):
        return None
    return TextEvent(text=piece)


def _hold_at(
    text: str,
    *,
    line_start: bool,
    sentence_start: bool,
    before: str = "",
    lead: bool = False,
) -> int:
    return hold_tts(
        text,
        line_start=line_start,
        sentence_start=sentence_start,
        before=before,
        lead=lead,
    )


def _at_line_start(ready: str) -> bool:
    text = ready.rstrip(" \t")
    return not text or text.endswith("\n")


def _stable_prefix(
    text: str,
    *,
    line_start: bool,
    sentence_start: bool,
    before: str = "",
    lead: bool = False,
) -> tuple[str, str]:
    cut = _hold_at(
        text, line_start=line_start, sentence_start=sentence_start, before=before, lead=lead
    )
    return text[:cut], text[cut:]


# A closer split from its word ("words" then "** ") is not an operator.
# " * " still is: the mark does not start the chunk.
_ORPHAN_CLOSER_RE: Final = re.compile(r"^[*_`~]+(?=\s)")


def _fold_stream_breaks(text: str) -> str:
    # A carriage return or a Unicode line separator is a newline only after
    # scrub. Until then the next mood looks mid-sentence and is spoken.
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )


def _glue_sentence_stop(ready: str, piece: str) -> str:
    # "(https://example.com)" is removed after "door " was already buffered.
    # The period arrives next, and Fish says "door" and then "dot".
    if (
        piece
        and piece[0] in ".!?…。！？,;:，；："
        and ready[-1:].isspace()
        and ready[-1:] not in "\n\r"
    ):
        ready = ready[:-1]
    return ready + piece


def _drop_orphan_closer(stable: str, ready: str) -> str:
    if not ready or not ready[-1].isalnum():
        return stable
    # "~~~" then a newline is a code fence, not a leftover closer.
    # Stripping it spoke the code.
    if stable.startswith(("~~~", "```")):
        return stable
    return _ORPHAN_CLOSER_RE.sub("", stable)


def _continues_sentence(ready: str, incoming: str = "") -> bool:
    if not ready.rstrip():
        return False
    # A new line is its own sentence, even when the previous line has no stop.
    # "List\nExcited," is a cue. The newline was already sent, so the next
    # token would otherwise look mid-sentence and the mood would be spoken.
    if ready.rstrip(" \t").endswith("\n"):
        return False
    if ends_sentence(ready):
        return False
    # "Hello؟" + " Excited" — the space is still in this token, not in ready.
    # "Dr." + " Happy" is not a new sentence.
    return not (incoming[:1].isspace() and ends_sentence(ready + incoming[:1]))


def _scrub_chunk(
    text: str,
    *,
    lead: bool = True,
    line_start: bool = True,
    before: str = "",
    after: str = "",
    continued: bool = False,
) -> str:
    """Scrub one stable stream chunk with the text already accepted around it."""
    if not text:
        return ""
    # A chunk edge is the middle of the reply. Whole-string scrub strips
    # that space, and cue rewrite strips it again. The newline is how the
    # next chunk knows it starts a line.
    lead_space = text[0] == " "
    trail_space = text[-1] == " "
    body = text.lstrip(" \t")
    lead_breaks = min(len(body) - len(body.lstrip("\n")), 2)
    ended_line = text.rstrip(" \t").endswith("\n")
    cleaned = scrub_tts(
        text,
        line_start=line_start,
        continued=continued,
        before=before[-1:],
        after=after[:1],
    )
    spoken = normalize_cues(cleaned, lead=lead)
    if (lead_space or cleaned[:1] == " ") and spoken[:1] != " ":
        spoken = f" {spoken}"
    if (trail_space or cleaned[-1:] == " ") and spoken[-1:] != " ":
        spoken = f"{spoken} "
    if lead_breaks and not spoken.startswith("\n"):
        spoken = ("\n" * lead_breaks) + spoken.lstrip(" ")
    elif ended_line and not spoken.endswith("\n"):
        spoken = f"{spoken.rstrip(' ')}\n"
    return spoken


class _StreamScrubber:
    """State machine behind ``delta_events``: tokens in, Fish events out.

    ``feed`` and ``finish`` are generators, so ``cancel`` is read at the same
    points the consumer pulls an event, and a cancel set between two events
    stops the next one.
    """

    def __init__(
        self,
        cancel: threading.Event,
        *,
        partial_chars: int,
        mood_lead: bool,
        early_flush: bool,
    ) -> None:
        self._cancel = cancel
        self._partial_chars = partial_chars
        self._mood_lead = mood_lead
        self._early_flush = early_flush
        self._raw = ""
        self._ready = ""
        # The last accepted character. ready is empty once that text is sent,
        # and the next span still needs the neighbor so its gap survives.
        self._last = ""
        self._sent = 0
        self._flushed_early = False
        self._since_flush = 0
        # Text already sent on this line. An empty ready buffer is not a new
        # line, so a star after "Hello." must not be stripped as a bullet.
        self._sent_line = ""

    def _take_ready(self) -> str | None:
        split = split_tts_piece(self._ready, self._partial_chars, flush_rest=False)
        if split is None:
            return None
        piece, self._ready = split
        return piece

    def _emit(self, piece: str) -> Iterator[Any]:
        event = _text_event(piece)
        if event is None:
            return
        self._sent += 1
        yield event
        self._since_flush += 1
        if self._early_flush and not self._flushed_early and not self._cancel.is_set():
            self._flushed_early = True
            self._since_flush = 0
            yield FlushEvent()

    def feed(self, tok: str) -> Iterator[Any]:
        """Take one model token and yield the events it makes ready."""
        # A space-only token is not a TextEvent, but it is the boundary
        # between words. Dropping it here joins those words.
        self._raw = _fold_stream_breaks(self._raw + tok)
        # The line already sent counts too: once a piece leaves, ``ready`` is empty
        # and a mood word inside a long sentence would look like a new one.
        sentence_start = not _continues_sentence(self._sent_line + self._ready, self._raw)
        stable, self._raw = _stable_prefix(
            self._raw,
            line_start=_at_line_start(self._sent_line + self._ready),
            sentence_start=sentence_start,
            before=self._ready,
            lead=self._mood_lead,
        )
        if stable:
            stable = _drop_orphan_closer(stable, self._ready)
            self._ready = _glue_sentence_stop(
                self._ready,
                _scrub_chunk(
                    stable,
                    lead=sentence_start and self._mood_lead,
                    line_start=_at_line_start(self._sent_line + self._ready),
                    before=self._ready[-1:] or self._last,
                    after=self._raw[:1],
                    continued=bool((self._sent_line + self._ready).strip()),
                ),
            )
            if self._ready:
                self._last = self._ready[-1]
        while (piece := self._take_ready()) is not None:
            self._sent_line = (self._sent_line + piece).rsplit("\n", 1)[-1]
            yield from self._emit(piece)

    def finish(self) -> Iterator[Any]:
        """Scrub what is still held, send it, and yield the closing flush."""
        if not self._cancel.is_set() and self._raw:
            lead = not _continues_sentence(self._sent_line + self._ready, self._raw)
            self._raw = _drop_orphan_closer(self._raw, self._ready)
            self._ready = _glue_sentence_stop(
                self._ready,
                _scrub_chunk(
                    self._raw,
                    lead=lead,
                    line_start=_at_line_start(self._sent_line + self._ready),
                    before=self._ready[-1:] or self._last,
                    continued=bool((self._sent_line + self._ready).strip()),
                ),
            )
        # A span held until the end can be longer than the send window.
        # One cut would speak the first words and drop the rest.
        while self._ready and not self._cancel.is_set():
            split = split_tts_piece(self._ready, self._partial_chars, flush_rest=True)
            if split is None:
                break
            piece, self._ready = split
            yield from self._emit(piece)
        # A flush with no text after the early one would ask Fish to flush nothing.
        flush = flush_if_sent(
            self._since_flush if self._flushed_early else self._sent, self._cancel
        )
        if flush is not None:
            yield flush


async def delta_events(
    deltas: Iterable[str] | AsyncIterable[str],
    cancel: threading.Event,
    *,
    partial_chars: int,
    mood_lead: bool = False,
    early_flush: bool = False,
) -> AsyncIterator[Any]:
    """Cut model deltas into Fish text events, scrubbing as each span closes.

    Parameters
    ----------
    deltas : Iterable or AsyncIterable of str
        Token stream. Empty pieces are skipped.
    cancel : threading.Event
        Stops the stream.
    partial_chars : int
        Size of one Fish text event.
    mood_lead : bool, optional
        Rewrite a sentence-leading mood word into a ``[cue]``. Default False.
    early_flush : bool, optional
        Flush once right after the first piece, so Fish speaks it while the
        model is still writing. Default False. Fish holds text until a chunk
        fills or a flush arrives, so without this a streamed reply is silent
        until the model finishes.

    Yields
    ------
    Any
        ``TextEvent`` pieces, then one ``FlushEvent`` when any text was sent.
        With ``early_flush`` there is one more flush after the first piece,
        and the final flush is sent only if text followed it.

    Notes
    -----
    A thought, parenthesis, bracket, or URL stays buffered until it closes, so
    a cut cannot speak the inside of a span the closer would remove.
    """
    scrubber = _StreamScrubber(
        cancel, partial_chars=partial_chars, mood_lead=mood_lead, early_flush=early_flush
    )
    async for tok in as_async(deltas):
        if cancel.is_set():
            break
        for event in scrubber.feed(tok):
            yield event
    for event in scrubber.finish():
        yield event
