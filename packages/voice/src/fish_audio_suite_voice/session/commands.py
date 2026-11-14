"""Changes a display asks the session to make to the chat, as plain values.

A display never changes the ``Conversation`` itself. It sends one of these through its turn
source (``session.inputs.LiveInput.send``), and the session loop applies it between turns,
saves the chat, and answers with a ``session.events.ChatShown``. The loop stays the record's
only writer, so a reply being recorded and an edit never race.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ChatCommand",
    "DeleteTurn",
    "EditReply",
    "NewChat",
    "OpenChat",
    "SetHidden",
    "Undo",
]


@dataclass(frozen=True, slots=True)
class EditReply:
    """Send the model ``text`` in place of a turn's reply from now on.

    Attributes
    ----------
    turn_id : str
        The turn whose reply changes.
    text : str
        The new reply. Not blank: hide or delete the turn instead.
    """

    turn_id: str
    text: str


@dataclass(frozen=True, slots=True)
class DeleteTurn:
    """Remove a turn from the chat.

    Attributes
    ----------
    turn_id : str
        The turn to remove.
    """

    turn_id: str


@dataclass(frozen=True, slots=True)
class SetHidden:
    """Leave a turn out of what the model is sent, or put it back.

    Attributes
    ----------
    turn_id : str
        The turn to change.
    hidden : bool
        True hides it from the model. The chat keeps it either way.
    """

    turn_id: str
    hidden: bool


@dataclass(frozen=True, slots=True)
class Undo:
    """Reverse the last edit, hide or delete in this chat."""


@dataclass(frozen=True, slots=True)
class OpenChat:
    """Go on with a saved chat instead of this one.

    Attributes
    ----------
    path : Path
        The saved chat's file, which the session goes on writing to.
    """

    path: Path


@dataclass(frozen=True, slots=True)
class NewChat:
    """Start a new chat, in a new file."""


type ChatCommand = EditReply | DeleteTurn | SetHidden | Undo | OpenChat | NewChat
"""Every change a display can ask for."""
