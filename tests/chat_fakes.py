"""A chat backend for the script tests: each model answers through a reply function, offline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager

import eval_models
import pytest

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.llm.tune import LlmTune

type Reply = Callable[[str, list[ChatMessage]], AsyncIterator[str]]
"""Streams the answer a model (the first argument) gives to the messages."""


class ScriptedChat:
    """A ``ChatBackend`` for one model, whose every answer comes from ``reply``."""

    def __init__(self, model: str, reply: Reply) -> None:
        self.model = model
        self._reply = reply

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        cancel: asyncio.Event | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[str]:
        del cancel, trace_id
        return self._reply(self.model, messages)

    async def aclose(self) -> None:
        return None


def install_chat(monkeypatch: pytest.MonkeyPatch, reply: Reply) -> None:
    """Make the evaluation scripts open a ``ScriptedChat`` for each model they run."""

    @asynccontextmanager
    async def opened(
        tune: LlmTune, *, session_id: str | None = None
    ) -> AsyncGenerator[ScriptedChat, None]:
        del session_id
        yield ScriptedChat(tune.model, reply)

    monkeypatch.setattr(eval_models, "open_chat_backend", opened)
