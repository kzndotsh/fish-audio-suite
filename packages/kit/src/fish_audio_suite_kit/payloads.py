"""Shapes of the JSON bodies the kit reads and builds. Types only."""

from __future__ import annotations

from typing import NotRequired, TypedDict

__all__ = ["AsrBody", "AsrSegment", "AsrWord", "OpenAIErrorBody", "OpenAIErrorDetail"]


class AsrWord(TypedDict, total=False):
    """One word with its timing, when Fish returns word timings."""

    text: str
    start: float
    end: float


class AsrSegment(TypedDict, total=False):
    """One transcribed phrase with its timing, in seconds."""

    text: str
    start: float
    end: float
    words: list[AsrWord]


class AsrBody(TypedDict, total=False):
    """A Fish ``/v1/asr`` response. Only ``text`` is checked, and Fish may omit any key."""

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
