"""The text events Fish is sent for one turn, from a whole reply or a token stream.

``text_events`` cuts a whole reply that is already shaped. ``delta_events`` shapes a
streamed reply as it arrives: it holds an unclosed span, shapes each stable chunk with the
turn's ``CueStyle`` and cuts it, so Fish speaks the first sentence while the model is still
writing. Both send each piece through the same step (skip a blank one, carry the mood
unless the model is directed) and flush only after text.
"""

from __future__ import annotations

import re
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final

from fishaudio import FlushEvent, TextEvent

from fish_audio_suite_kit import CueStyle, MoodCarry, ends_sentence, split_tts_piece

__all__ = [
    "WholeText",
    "as_async",
    "delta_events",
    "flush_if_sent",
    "text_events",
]


def flush_if_sent(sent: int, cancel: threading.Event) -> FlushEvent | None:
    """Return a flush only when text was sent and the turn was not cancelled.

    Parameters
    ----------
    sent : int
        How many ``TextEvent`` values were yielded.
    cancel : threading.Event
        When set, return None so Fish is not asked to flush a dead turn.

    Returns
    -------
    FlushEvent or None
        None when ``sent`` is 0. An empty turn plus a bare flush is invalid.
    """
    if sent and not cancel.is_set():
        return FlushEvent()
    return None


async def as_async(deltas: Iterable[str] | AsyncIterable[str]) -> AsyncIterator[str]:
    """Yield text deltas from either an async or a plain iterable.

    Parameters
    ----------
    deltas : Iterable or AsyncIterable of str
        The text pieces.

    Yields
    ------
    str
        Each piece, in order.
    """
    if isinstance(deltas, AsyncIterable):
        async for item in deltas:
            yield item
        return
    for item in deltas:
        yield item


class _Pieces:
    """Turn shaped pieces into text events, one at a time, as Fish is sent them."""

    def __init__(self, style: CueStyle) -> None:
        self._carry = MoodCarry() if style.carries_mood else None

    def event(self, piece: str) -> TextEvent | None:
        # The piece is already shaped, including one edge space. Shaping it again
        # would strip that space, and the next cut would be spoken as one word.
        if not piece.strip():
            return None
        # Each piece goes to Fish on its own, so a sentence the model wrote without a cue
        # would start cold: it gets the mood the last sentence had. A directed model reads
        # each direction where it is written, so it gets none.
        return TextEvent(text=piece if self._carry is None else self._carry.apply(piece))


async def text_events(
    prepared: str,
    cancel: threading.Event,
    partial_chars: int,
    *,
    style: CueStyle,
) -> AsyncIterator[Any]:
    """Yield Fish text events for a whole reply, then one flush.

    Parameters
    ----------
    prepared : str
        The reply, already shaped with ``style``.
    cancel : threading.Event
        Stops before the next piece. A cancel after zero pieces yields no flush.
    partial_chars : int
        Passed to ``split_tts_piece``. ``flush_rest`` is True, so the tail
        is not left buffered.
    style : CueStyle
        The turn's cue style. Only whether it carries a mood is read here.

    Yields
    ------
    TextEvent or FlushEvent
        Empty pieces are skipped. ``FlushEvent`` is yielded only after at
        least one ``TextEvent``. A bare flush on an empty turn is invalid.
    """
    buf = prepared
    sent = 0
    pieces = _Pieces(style)
    while buf:
        if cancel.is_set():
            break
        split = split_tts_piece(buf, partial_chars, flush_rest=True)
        if split is None:
            break
        piece, buf = split
        event = pieces.event(piece)
        if event is None:
            continue
        yield event
        sent += 1
    flush = flush_if_sent(sent, cancel)
    if flush is not None:
        yield flush


@dataclass(frozen=True, slots=True)
class WholeText:
    """A whole reply, already shaped, that a turn can send again after a failed attempt.

    Attributes
    ----------
    text : str
        The reply as Fish reads it, already shaped with ``style``.
    partial_chars : int
        The cut window for ``text_events``.
    style : CueStyle
        The turn's cue style.
    """

    text: str
    partial_chars: int
    style: CueStyle

    def events(self, cancel: threading.Event) -> AsyncIterator[Any]:
        """Return a fresh ``text_events`` stream over ``text``.

        Parameters
        ----------
        cancel : threading.Event
            Stops the stream.

        Returns
        -------
        AsyncIterator
            Text events, then one flush.
        """
        return text_events(self.text, cancel, self.partial_chars, style=self.style)


def _at_line_start(ready: str) -> bool:
    text = ready.rstrip(" \t")
    return not text or text.endswith("\n")


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
    # Stripping it would speak the code.
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
    # "Hello؟" + " Excited": the space is still in this token, not in ready.
    # "Dr." + " Happy" is not a new sentence.
    return not (incoming[:1].isspace() and ends_sentence(ready + incoming[:1]))


def _shape_chunk(
    text: str,
    style: CueStyle,
    *,
    sentence_start: bool,
    line_start: bool,
    before: str,
    after: str = "",
    continued: bool,
) -> str:
    """Shape one stable stream chunk with the text already accepted around it."""
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
    spoken = style.shape(
        text,
        line_start=line_start,
        continued=continued,
        before=before[-1:],
        after=after[:1],
        sentence_start=sentence_start,
    )
    if lead_space and spoken[:1] != " ":
        spoken = f" {spoken}"
    if trail_space and spoken[-1:] != " ":
        spoken = f"{spoken} "
    if lead_breaks and not spoken.startswith("\n"):
        spoken = ("\n" * lead_breaks) + spoken.lstrip(" ")
    elif ended_line and not spoken.endswith("\n"):
        spoken = f"{spoken.rstrip(' ')}\n"
    return spoken


# The early flush gives up waiting for a sentence end after this many send windows.
_EARLY_FLUSH_WINDOWS: Final = 2


class _StreamShaper:
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
        style: CueStyle,
        early_flush: bool,
    ) -> None:
        self._cancel = cancel
        self._style = style
        self._pieces = _Pieces(style)
        self._partial_chars = partial_chars
        self._early_flush = early_flush
        self._raw = ""
        self._ready = ""
        # The last accepted character. ready is empty once that text is sent,
        # and the next span still needs the neighbor so its gap survives.
        self._last = ""
        self._sent = 0
        self._flushed_early = False
        self._since_flush = 0
        # Everything sent before the early flush. The flush waits for a sentence
        # end in this whole text, not in the last piece alone: a piece that is
        # just "." after "Mr" would otherwise look like one. It gives up after
        # this many characters, or a long run-on would hold the audio.
        self._sent_text = ""
        self._early_flush_cap = _EARLY_FLUSH_WINDOWS * partial_chars
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
        event = self._pieces.event(piece)
        if event is None:
            return
        self._sent += 1
        yield event
        self._since_flush += 1
        if self._early_flush and not self._flushed_early:
            self._sent_text += event.text
            if not self._cancel.is_set() and (
                ends_sentence(self._sent_text) or len(self._sent_text) >= self._early_flush_cap
            ):
                self._flushed_early = True
                self._since_flush = 0
                self._sent_text = ""
                yield FlushEvent()

    def _accept(self, stable: str, *, sentence_start: bool, after: str = "") -> None:
        accepted = self._sent_line + self._ready
        self._ready = _glue_sentence_stop(
            self._ready,
            _shape_chunk(
                stable,
                self._style,
                sentence_start=sentence_start,
                line_start=_at_line_start(accepted),
                before=self._ready[-1:] or self._last,
                after=after,
                continued=bool(accepted.strip()),
            ),
        )

    def feed(self, tok: str) -> Iterator[Any]:
        """Take one model token and yield the events it makes ready."""
        # A space-only token is not a TextEvent, but it is the boundary
        # between words. Dropping it here joins those words.
        self._raw = _fold_stream_breaks(self._raw + tok)
        # The line already sent counts too: once a piece leaves, ``ready`` is empty
        # and a mood word inside a long sentence would look like a new one.
        sentence_start = not _continues_sentence(self._sent_line + self._ready, self._raw)
        cut = self._style.hold_at(
            self._raw,
            line_start=_at_line_start(self._sent_line + self._ready),
            sentence_start=sentence_start,
            before=self._ready,
        )
        stable, self._raw = self._raw[:cut], self._raw[cut:]
        if stable:
            stable = _drop_orphan_closer(stable, self._ready)
            self._accept(stable, sentence_start=sentence_start, after=self._raw[:1])
            if self._ready:
                self._last = self._ready[-1]
        while (piece := self._take_ready()) is not None:
            self._sent_line = (self._sent_line + piece).rsplit("\n", 1)[-1]
            yield from self._emit(piece)

    def finish(self) -> Iterator[Any]:
        """Shape what is still held, send it, and yield the closing flush."""
        if not self._cancel.is_set() and self._raw:
            sentence_start = not _continues_sentence(self._sent_line + self._ready, self._raw)
            self._raw = _drop_orphan_closer(self._raw, self._ready)
            self._accept(self._raw, sentence_start=sentence_start)
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
    style: CueStyle,
    early_flush: bool = True,
) -> AsyncIterator[Any]:
    """Cut model deltas into Fish text events, shaping each span as it closes.

    Parameters
    ----------
    deltas : Iterable or AsyncIterable of str
        Token stream. Empty pieces are skipped.
    cancel : threading.Event
        Stops the stream.
    partial_chars : int
        Size of one Fish text event.
    style : CueStyle
        How cues are shaped, the same style a whole reply is shaped with.
    early_flush : bool, optional
        Flush once at the end of the first sentence, so Fish speaks it while
        the model is still writing. Default True, which is what the speaker
        sends; False sends one flush at the end only. Fish holds text until a
        chunk fills or a flush arrives, so without this a streamed reply is
        silent until the model finishes. The flush waits for a sentence end,
        because a flush mid-sentence makes Fish end that fragment as a finished
        utterance and pause before the rest. After two send windows of text
        with no sentence end it flushes anyway.

    Yields
    ------
    Any
        ``TextEvent`` pieces, then one ``FlushEvent`` when any text was sent.
        With ``early_flush`` there is one more flush after the first sentence,
        and the final flush is sent only if text followed it.

    Notes
    -----
    A thought, parenthesis, bracket, or URL stays buffered until it closes, so
    a cut cannot speak the inside of a span the closer would remove.
    """
    shaper = _StreamShaper(
        cancel, partial_chars=partial_chars, style=style, early_flush=early_flush
    )
    async for tok in as_async(deltas):
        if cancel.is_set():
            break
        for event in shaper.feed(tok):
            yield event
    for event in shaper.finish():
        yield event
