"""Stdout that carries the conversation: status lines and the streaming reply line."""

from __future__ import annotations

import sys
from typing import Any, Final

__all__ = [
    "console_print",
    "end_reply_line",
    "write_reply_token",
]


class _ReplyLine:
    open: bool = False


_REPLY: Final = _ReplyLine()


def console_print(*args: object, **kwargs: Any) -> None:
    """Print a status line. A closed stdout must not drop the spoken reply."""
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        return


def write_reply_token(text: str) -> None:
    """Stream one LLM token. The line stays open until a debug log or the turn ends."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        return
    _REPLY.open = True


def end_reply_line() -> None:
    """Close the open token line with a newline."""
    if not _REPLY.open:
        return
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        pass
    _REPLY.open = False
