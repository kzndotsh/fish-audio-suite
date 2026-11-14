"""Saved chats: one JSON file per conversation, written whole and atomically.

A file is ``Conversation.to_json`` with ``started`` and ``updated`` times beside it, in UTF-8
with an indent, so it can be read and edited by hand. It is named for when the chat started
and the start of its id, so a folder of them sorts by time. The folder is made private and
each file is readable only by its owner, because they hold what was said.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from fish_audio_suite_voice.logutil.debug import debug
from fish_audio_suite_voice.session.conversation import SCHEMA_VERSION, Conversation

__all__ = [
    "ChatFile",
    "ChatLoadError",
    "ChatSummary",
    "SavedChat",
    "find_chat",
    "list_chats",
    "load_chat",
]

# The first user line, cut to this many characters, is a chat's title.
_TITLE_CHARS: Final = 60
# Characters of the id in a file name: enough to tell chats started in one second apart.
_NAME_ID_CHARS: Final = 8


class ChatLoadError(Exception):
    """A saved chat cannot be found or read. The message names the file and the reason."""


@dataclass(frozen=True, slots=True)
class ChatFile:
    """Where one conversation is saved.

    Attributes
    ----------
    path : Path
        The file. A resumed chat keeps the path it was read from.
    started : datetime
        When the chat started, with its time zone.
    """

    path: Path
    started: datetime

    @classmethod
    def new(cls, folder: Path, conversation: Conversation) -> ChatFile:
        """Name the file for a chat that starts now.

        Parameters
        ----------
        folder : Path
            The chat folder. ``~`` is expanded. Nothing is written until ``save``.
        conversation : Conversation
            The chat, whose id the name ends with.

        Returns
        -------
        ChatFile
            ``<folder>/<YYYYmmdd-HHMMSS>-<first 8 of the id>.json``, started now.
        """
        started = datetime.now().astimezone().replace(microsecond=0)
        name = f"{started:%Y%m%d-%H%M%S}-{conversation.id[:_NAME_ID_CHARS]}.json"
        return cls(folder.expanduser() / name, started)

    def save(self, conversation: Conversation) -> None:
        """Write the conversation to the file, replacing what was there in one step.

        Parameters
        ----------
        conversation : Conversation
            The record to write. ``updated`` is set to now.

        Notes
        -----
        The text goes to a temporary file in the same folder, is flushed to disk, then
        renamed over the old file, so a crash or a full disk leaves the old file whole.
        The folder is made with mode 0o700 when it is missing, and the file is 0o600.
        """
        data = {
            **conversation.to_json(),
            "started": self.started.isoformat(timespec="seconds"),
            "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkstemp makes the file readable and writable only by its owner.
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        temp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
            raise


@dataclass(frozen=True, slots=True)
class ChatSummary:
    """One saved chat, as a list of chats shows it.

    Attributes
    ----------
    id : str
        The conversation's id.
    path : Path
        Its file.
    title : str
        Its first user line, shortened. Empty when it has no turns.
    started : datetime
        When it started.
    updated : datetime
        When it was last saved.
    turns : int
        How many turns it has, hidden ones included.
    """

    id: str
    path: Path
    title: str
    started: datetime
    updated: datetime
    turns: int


@dataclass(frozen=True, slots=True)
class SavedChat:
    """A chat read back from its file.

    Attributes
    ----------
    file : ChatFile
        Where it is saved, which a resumed session keeps writing to.
    conversation : Conversation
        The record as saved, with its own system prompt.
    updated : datetime
        When it was last saved.
    """

    file: ChatFile
    conversation: Conversation
    updated: datetime

    def summary(self) -> ChatSummary:
        """Return what a list of chats shows of this one.

        Returns
        -------
        ChatSummary
            Its id, file, title, times and number of turns.
        """
        turns = self.conversation.turns
        first = turns[0].user.split("\n")[0].strip() if turns else ""
        title = first if len(first) <= _TITLE_CHARS else first[: _TITLE_CHARS - 1] + "…"
        return ChatSummary(
            self.conversation.id,
            self.file.path,
            title,
            self.file.started,
            self.updated,
            len(turns),
        )


def load_chat(path: Path) -> SavedChat:
    """Read a saved chat.

    Parameters
    ----------
    path : Path
        The file. ``~`` is expanded.

    Returns
    -------
    SavedChat
        The record, where it is saved, and when it was last saved.

    Raises
    ------
    ChatLoadError
        When the file cannot be read, is not JSON, was written by a newer version, or a
        field is missing or has the wrong type.
    """
    path = path.expanduser()
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatLoadError(f"cannot read the saved chat {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ChatLoadError(f"{path} is not a saved chat: it holds no JSON object")
    fields: dict[str, object] = {str(key): value for key, value in data.items()}
    version = fields.get("version")
    if isinstance(version, int) and not isinstance(version, bool) and version > SCHEMA_VERSION:
        raise ChatLoadError(
            f"{path} was saved by a newer fish-voice (version {version}, this one reads "
            f"{SCHEMA_VERSION}); update fish-voice to open it"
        )
    try:
        conversation = Conversation.from_json(fields)
        started = _time(fields, "started")
        updated = _time(fields, "updated")
    except (TypeError, ValueError) as exc:
        raise ChatLoadError(f"{path} is not a saved chat: {exc}") from exc
    return SavedChat(ChatFile(path, started), conversation, updated)


def list_chats(folder: Path) -> list[ChatSummary]:
    """List the saved chats in a folder, the most recently updated first.

    Parameters
    ----------
    folder : Path
        The chat folder. ``~`` is expanded.

    Returns
    -------
    list of ChatSummary
        Every ``*.json`` file that reads as a chat. Empty when the folder is missing. A
        file that does not read is left out, with a debug line saying why.
    """
    folder = folder.expanduser()
    if not folder.is_dir():
        return []
    found: list[ChatSummary] = []
    for path in folder.glob("*.json"):
        try:
            found.append(load_chat(path).summary())
        except ChatLoadError as exc:
            debug("chat.skip {}", exc)
    return sorted(found, key=lambda chat: chat.updated, reverse=True)


def find_chat(folder: Path | None, ref: str) -> Path:
    """Find the file of the chat ``--resume`` names.

    Parameters
    ----------
    folder : Path or None
        The chat folder. None looks only at ``ref`` as a path.
    ref : str
        Empty for the most recently updated chat. Otherwise a path to a file, a file name
        in the folder, or the start of a chat's id.

    Returns
    -------
    Path
        The chat's file.

    Raises
    ------
    ChatLoadError
        When there is no saved chat, nothing matches, or more than one chat's id starts
        with ``ref``.
    """
    if ref:
        in_folder = None if folder is None else folder.expanduser() / ref
        for candidate in (Path(ref).expanduser(), in_folder):
            if candidate is not None and candidate.is_file():
                return candidate
    if folder is None:
        raise ChatLoadError(f"no saved chat at {ref!r}")
    chats = list_chats(folder)
    if not ref:
        if not chats:
            raise ChatLoadError(f"no saved chats in {folder}")
        return chats[0].path
    matches = [chat for chat in chats if chat.id.startswith(ref.lower())]
    if not matches:
        raise ChatLoadError(f"no saved chat in {folder} has an id starting with {ref!r}")
    if len(matches) > 1:
        ids = ", ".join(chat.id[: len(ref) + 4] for chat in matches)
        raise ChatLoadError(
            f"{len(matches)} saved chats have an id starting with {ref!r} ({ids}); give more of it"
        )
    return matches[0].path


def _time(data: dict[str, object], key: str) -> datetime:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} is {type(value).__name__}, not a time")
    # A time edited by hand without a zone is taken as local, so times still compare.
    return datetime.fromisoformat(value).astimezone()
