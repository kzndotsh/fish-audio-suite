"""Shapes of the JSON bodies the kit reads and builds. Types only."""

from __future__ import annotations

from typing import NotRequired, TypedDict

__all__ = ["AsrBody", "AsrSegment", "AsrWord", "OpenAIErrorBody", "OpenAIErrorDetail"]


class AsrWord(TypedDict, total=False):
    """One word with its timing, in seconds.

    Fish ``/v1/asr`` does not send a ``words`` field; its ``segments`` carry
    the word timings. This shape is for a body that does hold one.
    """

    text: str
    start: float
    end: float


class AsrSegment(TypedDict, total=False):
    """One segment of a Fish transcript: a single word with its timing, in seconds.

    Fish segments are word-level, not phrases. Fish does not send ``words``
    inside a segment; the key is typed only so a body that holds one still fits.
    """

    text: str
    start: float
    end: float
    words: list[AsrWord]


class AsrBody(TypedDict, total=False):
    """A Fish ``/v1/asr`` response. Only ``text`` is checked, and Fish may omit any key.

    ``segments`` hold one word each and ``duration`` is in seconds.
    ``transcribe-1-pro`` also sends ``speaker_turns`` and ``request_id``, which
    are not typed here.
    """

    text: str
    segments: list[AsrSegment]
    duration: float
    language: str
    language_code: str


class OpenAIErrorDetail(TypedDict):
    """The ``error`` object inside an OpenAI-style error response."""

    code: int
    message: str
    type: str
    metadata: NotRequired[dict[str, str]]


class OpenAIErrorBody(TypedDict):
    """An OpenAI-style error response: ``{"error": {...}}``."""

    error: OpenAIErrorDetail
