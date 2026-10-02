"""Incremental scrub for a streamed reply: hold unclosed spans, cut, then send.

The duplex loop waits for the whole reply and does not use this. It backs
``IsolatedFishTts.speak_deltas`` for callers that forward model tokens as they
arrive.
"""

from __future__ import annotations

import re
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import Any

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
_ORPHAN_CLOSER_RE = re.compile(r"^[*_`~]+(?=\s)")


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
    raw = ""
    ready = ""
    # The last accepted character. ready is empty once that text is sent,
    # and the next span still needs the neighbor so its gap survives.
    last = ""
    sent = 0
    flushed_early = False
    since_flush = 0
    # Text already sent on this line. An empty ready buffer is not a new
    # line, so a star after "Hello." must not be stripped as a bullet.
    sent_line = ""

    def counted(piece: str) -> TextEvent | None:
        nonlocal sent
        event = _text_event(piece)
        if event is None:
            return None
        sent += 1
        return event

    def take_ready() -> str | None:
        nonlocal ready
        split = split_tts_piece(ready, partial_chars, flush_rest=False)
        if split is None:
            return None
        piece, ready = split
        return piece

    async for tok in as_async(deltas):
        if cancel.is_set():
            break
        # A space-only token is not a TextEvent, but it is the boundary
        # between words. Dropping it here joins those words.
        raw = _fold_stream_breaks(raw + tok)
        sentence_start = not _continues_sentence(ready, raw)
        stable, raw = _stable_prefix(
            raw,
            line_start=_at_line_start(sent_line + ready),
            sentence_start=sentence_start,
            before=ready,
            lead=mood_lead,
        )
        if stable:
            stable = _drop_orphan_closer(stable, ready)
            ready = _glue_sentence_stop(
                ready,
                _scrub_chunk(
                    stable,
                    lead=sentence_start and mood_lead,
                    line_start=_at_line_start(sent_line + ready),
                    before=ready[-1:] or last,
                    after=raw[:1],
                    continued=bool((sent_line + ready).strip()),
                ),
            )
            if ready:
                last = ready[-1]
        while True:
            piece = take_ready()
            if piece is None:
                break
            sent_line = (sent_line + piece).rsplit("\n", 1)[-1]
            event = counted(piece)
            if event is not None:
                yield event
                since_flush += 1
                if early_flush and not flushed_early:
                    flushed_early = True
                    since_flush = 0
                    yield FlushEvent()
    if not cancel.is_set() and raw:
        lead = not _continues_sentence(ready, raw)
        raw = _drop_orphan_closer(raw, ready)
        ready = _glue_sentence_stop(
            ready,
            _scrub_chunk(
                raw,
                lead=lead,
                line_start=_at_line_start(sent_line + ready),
                before=ready[-1:] or last,
                continued=bool((sent_line + ready).strip()),
            ),
        )
    # A span held until the end can be longer than the send window.
    # One cut would speak the first words and drop the rest.
    while ready and not cancel.is_set():
        split = split_tts_piece(ready, partial_chars, flush_rest=True)
        if split is None:
            break
        piece, ready = split
        event = counted(piece)
        if event is not None:
            yield event
            since_flush += 1
            if early_flush and not flushed_early:
                flushed_early = True
                since_flush = 0
                yield FlushEvent()
    # A flush with no text after the early one would ask Fish to flush nothing.
    flush = flush_if_sent(since_flush if flushed_early else sent, cancel)
    if flush is not None:
        yield flush
