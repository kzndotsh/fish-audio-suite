"""Process-wide duplex cancel flags."""

from __future__ import annotations

import asyncio
import threading

HISTORY_TURNS = 20

STOP_RECORD = threading.Event()


class _TurnSignals:
    cancel: threading.Event | None = None
    llm_cancel: asyncio.Event | None = None

    def bind(self, cancel: threading.Event, llm_cancel: asyncio.Event) -> None:
        self.cancel = cancel
        self.llm_cancel = llm_cancel

    def fire(self) -> None:
        if self.cancel is not None:
            self.cancel.set()
        if self.llm_cancel is not None:
            self.llm_cancel.set()

    def clear(self) -> None:
        self.cancel = None
        self.llm_cancel = None


TURN = _TurnSignals()


def request_quit() -> None:
    """SIGINT: stop mic listen and cancel in-flight LLM/TTS. Sticky until process exit."""
    STOP_RECORD.set()
    TURN.fire()
