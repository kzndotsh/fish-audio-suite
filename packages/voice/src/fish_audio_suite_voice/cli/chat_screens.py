"""The full-screen app's dialogs: the saved chats, and the keys.

Neither changes the chat itself. The chats dialog sends each change the user asks for to
the session as a ``ChatCommand``, through the callable it is given.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from fish_audio_suite_kit import split_cues
from fish_audio_suite_voice.session.chat_store import ChatSummary, list_chats
from fish_audio_suite_voice.session.commands import (
    ChatCommand,
    DeleteChat,
    NewChat,
    OpenChat,
    RenameChat,
)

__all__ = [
    "KEYS",
    "ChatPicker",
    "KeysScreen",
    "printable",
    "spoken",
    "when_label",
]

_NEW = "new"
_ROW_WIDTH = 56  # characters of a chat's row before the dialog is laid out
_WEEK = 7

# Every key of the app, in plain words, as the keys dialog lists them.
KEYS: tuple[tuple[str, str], ...] = (
    ("Enter", "Send what you typed"),
    ("↑ ↓", "Previous messages you typed"),
    ("Esc", "Stop speaking, else close, else back to typing"),
    ("F2", "Mic on or off"),
    ("Tab", "Move between typing and messages"),
    ("j k  ↑ ↓", "Pick a message"),
    ("g  G", "The first or the newest message"),
    ("e", "Edit the reply"),
    ("E", "Edit my message and send it again"),
    ("r", "Try again (a new reply)"),
    ("y", "Copy the reply"),
    ("h", "Hide from the AI, or show again"),
    ("d", "Delete"),
    ("z", "Undo"),
    ("Ctrl+O", "Chats"),
    ("Ctrl+P", "All commands"),
    ("?", "Keys (from the messages)"),
    ("F12", "Diagnostics"),
    ("Ctrl+Q", "Quit"),
)


def printable(text: str) -> str:
    """Return ``text`` without control characters, which would reach the terminal as escapes.

    Parameters
    ----------
    text : str
        A reply, transcript or log line, which is untrusted: an ESC in one could clear the
        screen or retitle the window.

    Returns
    -------
    str
        The text with every control character but newline and tab removed.
    """
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def spoken(text: str) -> str:
    """Return a reply as it is heard: without its ``[cue]`` tags or control characters.

    Parameters
    ----------
    text : str
        A reply as the model wrote it or as it was sent to Fish.

    Returns
    -------
    str
        The words, each line's spaces collapsed, blank lines kept between paragraphs.

    Examples
    --------
    >>> spoken("[warm, teasing] Bart it is. [playful] Does he look like a Bart?")
    'Bart it is. Does he look like a Bart?'
    """
    words = "".join(piece for piece, is_cue in split_cues(printable(text)) if not is_cue)
    return "\n".join(" ".join(line.split()) for line in words.split("\n")).strip()


def when_label(moment: datetime, now: datetime) -> str:
    """Say when a chat was last used, in plain words.

    Parameters
    ----------
    moment : datetime
        When it was last saved.
    now : datetime
        The time now, in the same time zone.

    Returns
    -------
    str
        ``Today 01:33``, ``Yesterday``, a weekday (``Mon``) within the last week, else the
        day (``Nov 9``), with the year when it is not this one.

    Examples
    --------
    >>> now = datetime(2026, 11, 15, 9, 0)
    >>> [when_label(datetime(2026, 11, d, 1, 33), now) for d in (15, 14, 9, 2)]
    ['Today 01:33', 'Yesterday', 'Mon', 'Nov 2']
    >>> when_label(datetime(2025, 12, 31), now)
    'Dec 31, 2025'
    """
    days = (now.date() - moment.date()).days
    if days <= 0:
        return f"Today {moment:%H:%M}"
    if days == 1:
        return "Yesterday"
    if days < _WEEK:
        return f"{moment:%a}"
    day = f"{moment:%b} {moment.day}"
    return day if moment.year == now.year else f"{day}, {moment.year}"


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def _row(chat: ChatSummary, *, width: int, now: datetime, current: bool) -> Content:
    """Draw a chat as the list shows it: its title and when it was used, then its last reply."""
    when = when_label(chat.updated, now)
    title = _cut(printable(chat.title) or "(no turns yet)", max(1, width - len(when) - 2))
    gap = " " * max(2, width - len(title) - len(when))
    last = spoken(chat.last_reply).split("\n")[0]
    return Content.assemble(
        (title, "bold" if current else ""),
        gap,
        (when, "$text-muted"),
        "\n",
        (_cut(last, width), "$text-muted"),
    )


def _now() -> datetime:
    return datetime.now().astimezone()


class _ChatList(OptionList):
    """The chats. Enter opens the highlighted one."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("enter", "select", "Open")]


class ChatPicker(ModalScreen[None]):
    """The saved chats: search them, open one, start a new one, rename or delete one.

    Parameters
    ----------
    folder : Path or None
        The chat folder (``FISH_VOICE_CHAT_DIR``). None offers only a new chat.
    current : Path or None
        The file of the chat open now, which is shown in bold; opening it changes nothing.
    send : Callable
        Takes each change, such as ``LiveInput.send``.
    now : Callable, optional
        Returns the time now, which says how long ago each chat was used.

    Notes
    -----
    The chats are listed newest first, each with its title, when it was last used and its
    last reply as it is heard. A file that does not read as a chat is left out. Opening a
    chat or starting one closes the dialog; renaming and deleting keep it open, and the
    list shows the change at once, while the session applies it between turns.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("n", "new", "New"),
        Binding("r", "rename", "Rename"),
        Binding("d,delete", "delete", "Delete"),
        Binding("slash", "search", "Search", key_display="/"),
        Binding("y", "confirm", "Yes", show=False),
        Binding("down", "to_list", "Chats", show=False),
        Binding("escape", "close", "Close"),
    ]

    def __init__(
        self,
        folder: Path | None,
        current: Path | None,
        send: Callable[[ChatCommand], None],
        *,
        now: Callable[[], datetime] = _now,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._current = current
        self._send = send
        self._now = now
        self._chats: list[ChatSummary] = []
        self._shown: list[ChatSummary] = []
        self._confirming: ChatSummary | None = None
        self._renaming: ChatSummary | None = None

    def compose(self) -> ComposeResult:
        """Lay out the search, the list, and the line a question or a new name goes on."""
        with Vertical(id="picker-box") as box:
            box.border_title = "Chats"
            yield Input(placeholder="Search", id="chat-search")
            yield _ChatList(id="chats")
            yield Static(id="chat-ask")
            yield Input(id="chat-rename")
        yield Footer()

    async def on_mount(self) -> None:
        """Read the chat folder off the screen's thread, then list what it holds."""
        self.query_one("#chat-ask").display = False
        self.query_one("#chat-rename").display = False
        self._fill()
        self.query_one(_ChatList).focus()
        if self._folder is not None:
            self._chats = await asyncio.to_thread(list_chats, self._folder)
        self._fill()

    def _fill(self, keep: Path | None = None) -> None:
        """List the chats that match the search, keeping the highlight on ``keep``."""
        options = self.query_one(_ChatList)
        query = self.query_one("#chat-search", Input).value.strip().casefold()
        self._shown = [
            chat
            for chat in self._chats
            if not query
            or query in chat.title.casefold()
            or query in spoken(chat.last_reply).casefold()
        ]
        width = options.scrollable_content_region.width - 2
        width = width if width > 0 else _ROW_WIDTH
        now = self._now()
        options.clear_options()
        options.add_option(Option(Content.assemble(("+ New chat", "$text-accent")), id=_NEW))
        for index, chat in enumerate(self._shown):
            row = _row(chat, width=width, now=now, current=chat.path == self._current)
            options.add_option(Option(row, id=str(index)))
        at = next((i for i, c in enumerate(self._shown) if c.path == keep), None)
        options.highlighted = 0 if at is None else at + 1

    @property
    def _highlighted(self) -> ChatSummary | None:
        options = self.query_one(_ChatList)
        index = options.highlighted
        if index is None or index == 0 or index > len(self._shown):
            return None
        return self._shown[index - 1]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Offer only the keys that apply: a question waiting takes y, n and Escape.

        Parameters
        ----------
        action : str
            The action's name.
        parameters : tuple
            Its parameters.

        Returns
        -------
        bool or None
            False hides the key.
        """
        del parameters
        if self._confirming is not None:
            return action in ("confirm", "new", "close")
        if action in ("rename", "delete"):
            return self._renaming is None and self._highlighted is not None
        return action != "confirm"

    def on_option_list_option_highlighted(self) -> None:
        """Offer rename and delete only on a chat."""
        self.refresh_bindings()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the chats as the search is typed."""
        if event.input.id == "chat-search":
            self._fill()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search goes to the first match; in a new name, keeps it."""
        event.stop()
        if event.input.id == "chat-search":
            self.action_to_list()
            return
        chat, self._renaming = self._renaming, None
        event.input.display = False
        title = event.value.strip()
        if chat is not None and title and title != chat.title:
            self._send(RenameChat(chat.path, title))
            self._chats = [replace(c, title=title) if c is chat else c for c in self._chats]
            self.notify("Chat renamed")
        self._fill(keep=None if chat is None else chat.path)
        self.query_one(_ChatList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Go on with the chosen chat, or start a new one."""
        if event.option.id == _NEW:
            self.action_new()
            return
        chat = self._highlighted
        if chat is not None and chat.path != self._current:
            self._send(OpenChat(chat.path))
        self.dismiss(None)

    def action_new(self) -> None:
        """Start a new chat, or answer no to the question waiting."""
        if self._confirming is not None:
            self._ask(None)
            return
        self._send(NewChat())
        self.dismiss(None)

    def action_search(self) -> None:
        """Type into the search."""
        self.query_one("#chat-search", Input).focus()

    def action_to_list(self) -> None:
        """Move from the search to the first chat that matches."""
        options = self.query_one(_ChatList)
        options.focus()
        options.highlighted = 1 if self._shown else 0

    def action_rename(self) -> None:
        """Type a new name for the highlighted chat."""
        chat = self._highlighted
        if chat is None:
            return
        self._renaming = chat
        field = self.query_one("#chat-rename", Input)
        field.border_title = "New name"
        field.value = chat.title
        field.display = True
        field.focus()
        self.refresh_bindings()

    def action_delete(self) -> None:
        """Ask before deleting the highlighted chat."""
        chat = self._highlighted
        if chat is not None:
            self._ask(chat)

    def _ask(self, chat: ChatSummary | None) -> None:
        self._confirming = chat
        ask = self.query_one("#chat-ask", Static)
        ask.display = chat is not None
        if chat is not None:
            name = printable(chat.title) or "this chat"
            ask.update(
                Content.assemble(
                    f"Delete '{name}'? ",
                    ("y", "bold $text-accent"),
                    " / ",
                    ("n", "bold $text-accent"),
                )
            )
        self.refresh_bindings()

    def action_confirm(self) -> None:
        """Delete the chat the question is about."""
        chat = self._confirming
        self._ask(None)
        if chat is None:
            return
        self._send(DeleteChat(chat.path))
        self._chats = [c for c in self._chats if c is not chat]
        if chat.path == self._current:
            self._current = None  # the session starts a new chat in its place
        self._fill()
        self.notify("Chat deleted")

    def action_close(self) -> None:
        """Drop the question or the new name, else close the dialog."""
        if self._confirming is not None:
            self._ask(None)
        elif self._renaming is not None:
            self._renaming = None
            self.query_one("#chat-rename").display = False
            self.query_one(_ChatList).focus()
            self.refresh_bindings()
        else:
            self.dismiss(None)


class KeysScreen(ModalScreen[None]):
    """Every key of the app, in plain words. Any of Escape, ``?`` or ``q`` closes it."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,question_mark,q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        """Lay out the keys, one to a line."""
        width = max(len(key) for key, _ in KEYS)
        lines = [
            Content.assemble((key.ljust(width), "bold $text-accent"), "  ", does)
            for key, does in KEYS
        ]
        with Vertical(id="keys-box") as box:
            box.border_title = "keys"
            yield Static(Content("\n").join(lines))
        yield Footer()

    def action_close(self) -> None:
        """Close the list."""
        self.dismiss(None)
