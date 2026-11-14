"""The full-screen app's dialogs: rewriting a reply, and choosing a saved chat.

Neither changes the chat. Each returns what the user chose, and the caller sends it to the
session as a ``ChatCommand``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Footer, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from fish_audio_suite_voice.session.chat_store import ChatSummary, list_chats
from fish_audio_suite_voice.session.commands import ChatCommand, NewChat, OpenChat

__all__ = [
    "ChatPicker",
    "EditReplyScreen",
]

_NEW = "new"


class EditReplyScreen(ModalScreen[str | None]):
    """Rewrite one reply. Dismissed with the new text, or None to keep the reply as it was.

    Parameters
    ----------
    text : str
        The reply as the model is sent it now.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        """Lay out the reply in an editor, with the keys below it."""
        with Vertical(id="edit-box") as box:
            box.border_title = "edit reply"
            yield Static("The model is sent this reply from now on.", classes="dialog-hint")
            yield TextArea(self._text, id="edit-text", soft_wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        """Put the cursor at the end of the reply."""
        editor = self.query_one(TextArea)
        editor.focus()
        editor.move_cursor(editor.document.end)

    def action_save(self) -> None:
        """Keep the new text, unless it is blank."""
        text = self.query_one(TextArea).text.strip()
        if not text:
            self.notify(
                "A reply cannot be blank. Hide or delete the turn instead.", severity="warning"
            )
            return
        self.dismiss(text)

    def action_cancel(self) -> None:
        """Keep the reply as it was."""
        self.dismiss(None)


def _when(moment: datetime) -> str:
    return f"{moment:%Y-%m-%d %H:%M}"


def _label(chat: ChatSummary, *, current: bool) -> Content:
    turns = f"{chat.turns} turn{'' if chat.turns == 1 else 's'}"
    detail = f"started {_when(chat.started)} · updated {_when(chat.updated)} · {turns}"
    title = chat.title or "(no turns yet)"
    parts: list[tuple[str, str] | str] = [(title, "bold")]
    if current:
        parts.append(("  open now", "$text-success"))
    parts += ["\n", (detail, "$text-muted")]
    return Content.assemble(*parts)


class ChatPicker(ModalScreen[ChatCommand | None]):
    """Choose a saved chat to go on with, or start a new one.

    Parameters
    ----------
    folder : Path or None
        The chat folder (``FISH_VOICE_CHAT_DIR``). None offers only a new chat.
    current : Path or None
        The file of the chat open now, which is marked, and choosing it changes nothing.

    Notes
    -----
    Dismissed with ``OpenChat`` or ``NewChat``, or None when nothing changes. The chats are
    listed newest first. A file that does not read as a chat is left out.
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, folder: Path | None, current: Path | None) -> None:
        super().__init__()
        self._folder = folder
        self._current = current
        self._paths: dict[str, Path] = {}

    def compose(self) -> ComposeResult:
        """Lay out the list of chats, with the keys below it."""
        with Vertical(id="picker-box") as box:
            box.border_title = "chats"
            yield Static("Enter opens a chat between turns.", classes="dialog-hint")
            yield OptionList(Option(Content.assemble(("+ new chat", "bold")), id=_NEW), id="chats")
        yield Footer()

    async def on_mount(self) -> None:
        """Read the chat folder off the screen's thread, then list what it holds."""
        options = self.query_one(OptionList)
        options.focus()
        if self._folder is None:
            return
        chats = await asyncio.to_thread(list_chats, self._folder)
        for index, chat in enumerate(chats):
            key = str(index)
            self._paths[key] = chat.path
            current = self._current is not None and chat.path == self._current
            options.add_option(Option(_label(chat, current=current), id=key))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Go on with the chosen chat, or start a new one."""
        key = event.option.id
        if key == _NEW:
            self.dismiss(NewChat())
            return
        path = self._paths.get(key or "")
        self.dismiss(None if path is None or path == self._current else OpenChat(path))

    def action_cancel(self) -> None:
        """Close the list and change nothing."""
        self.dismiss(None)
