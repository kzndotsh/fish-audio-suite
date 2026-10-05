"""Print a session to the terminal from its events."""

from __future__ import annotations

from collections.abc import Callable

from fish_audio_suite_voice.console import console_print, end_reply_line, write_reply_token
from fish_audio_suite_voice.debug import conversation, debug, debug_enabled, trace
from fish_audio_suite_voice.events import (
    Bye,
    Event,
    EventBus,
    Heard,
    Listening,
    Notice,
    ReplyEnd,
    ReplyToken,
    TurnEnded,
)
from fish_audio_suite_voice.reply import turn_summary

__all__ = [
    "ConsoleSink",
]


class ConsoleSink:
    """The plain terminal display: one subscriber that prints what the events say.

    Parameters
    ----------
    bus : EventBus
        The bus to print from.

    Notes
    -----
    A display that draws its own screen leaves this out. With debug on, the
    reply is held and printed whole on one line after the log lines around it,
    and the turn summary goes to the log instead of stdout.
    """

    def __init__(self, bus: EventBus) -> None:
        self._reply_open = False
        self._unsubscribe: Callable[[], None] = bus.subscribe(self._on_event)

    def close(self) -> None:
        """Stop printing."""
        self._unsubscribe()

    def _on_event(self, event: Event) -> None:
        match event:
            case Listening():
                if not debug_enabled():
                    console_print("listening…")
            case Heard(text=text):
                conversation("you", text)
            case ReplyToken(text=text):
                if not debug_enabled():
                    if not self._reply_open:
                        write_reply_token("llm ▸ ")
                        self._reply_open = True
                    write_reply_token(text)
            case ReplyEnd(text=text):
                end_reply_line()
                self._reply_open = False
                if text and debug_enabled():
                    conversation("llm", text)
            case Notice(text=text):
                console_print(f"  {text}", flush=True)
            case TurnEnded(snapshot=snapshot):
                summary = turn_summary(snapshot)
                if debug_enabled():
                    debug("turn.summary {}", summary.strip().removeprefix("↳ "))
                    trace("turn.timing {}", snapshot.log_line())
                else:
                    console_print(summary, flush=True)
            case Bye():
                console_print("\nbye")
            case _:
                return
