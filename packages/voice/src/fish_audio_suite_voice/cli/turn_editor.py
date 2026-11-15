"""The editor the conversation opens in place of a line: a reply, or the user's own line.

It changes nothing itself. Enter posts ``TurnEditor.Saved`` with the text and Escape posts
``TurnEditor.Cancelled``; the pane decides what to send the session and closes it. Ctrl+E
hands the text to the user's editor (``$VISUAL`` or ``$EDITOR``, read by the settings) and
takes back what was saved there.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar, Literal

from textual import events
from textual.app import ComposeResult, SuspendNotSupported
from textual.containers import VerticalGroup
from textual.content import Content
from textual.message import Message
from textual.widgets import Static, TextArea

__all__ = [
    "EditKind",
    "TurnEditor",
]

type EditKind = Literal["reply", "message"]
"""Which line of a turn is edited: the reply, or the user's own line."""

_KEY = "bold $text-accent"
# Keys that put a line break in the text. Enter saves, so these are the only way to a new
# line; a terminal sends Shift+Enter only if it reports modified keys.
_NEW_LINE_KEYS = frozenset({"shift+enter", "ctrl+j"})


def _hint() -> Content:
    pairs = (("enter", "Save"), ("esc", "Cancel"), ("^e", "Open in editor"))
    parts: list[tuple[str, str] | str] = []
    for key, label in pairs:
        parts += [(key, _KEY), (f" {label}   ", "dim")]
    return Content.assemble(*parts).rstrip()


class _Area(TextArea):
    """The text itself. Enter saves instead of starting a new line."""

    def __init__(self, text: str, editor: str) -> None:
        super().__init__(text, soft_wrap=True, compact=True, highlight_cursor_line=False)
        self._editor = editor

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        editor = self.parent
        if not isinstance(editor, TurnEditor) or key not in _HANDLED:
            return
        event.stop()
        event.prevent_default()
        if key == "enter":
            editor.post_message(TurnEditor.Saved(editor, self.text))
        elif key == "escape":
            editor.post_message(TurnEditor.Cancelled(editor))
        elif key == "ctrl+e":
            self._open_in_editor()
        elif key in _NEW_LINE_KEYS:
            start, end = self.selection
            self.replace("\n", start, end, maintain_selection_offset=False)
        # Tab is held here: the editor keeps the focus until it is saved or cancelled.

    def _open_in_editor(self) -> None:
        """Hand the text to the user's editor, and take back what they saved."""
        command = self._editor
        # mkstemp makes the file readable only by its owner: it holds what was said.
        fd, name = tempfile.mkstemp(prefix="fish-voice-", suffix=".txt")
        path = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.text)
            try:
                with self.app.suspend():
                    # The user's own editor, from their own environment, with no shell.
                    subprocess.run([*shlex.split(command), str(path)], check=False)  # noqa: S603
            except SuspendNotSupported:
                self.notify("This terminal cannot hand over to an editor.", severity="warning")
                return
            except (OSError, ValueError) as exc:
                self.notify(f"Cannot open the editor {command!r}: {exc}", severity="warning")
                return
            text = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
        finally:
            path.unlink(missing_ok=True)
        self.load_text(text)
        self.move_cursor(self.document.end)


_HANDLED = frozenset({"enter", "escape", "ctrl+e", "tab", *_NEW_LINE_KEYS})


class TurnEditor(VerticalGroup):
    """A bordered editor in place of one line of a turn, with its keys under the text.

    Parameters
    ----------
    turn_id : str
        The turn being edited.
    kind : EditKind
        Which of its lines.
    text : str
        The text to start from: a reply as it is heard, without its cues, or the user's line.
    editor : str
        The command Ctrl+E hands the text to, with any arguments.
    """

    DEFAULT_CSS: ClassVar[str] = """
    TurnEditor {
        border: round $accent;
        padding: 0 1;
        margin-left: 2;

        _Area {
            height: auto;
            max-height: 10;
            background: transparent;
        }

        .editor-keys {
            color: $text-muted;
        }
    }
    """

    class Saved(Message):
        """Enter was pressed: keep ``text``.

        Parameters
        ----------
        editor : TurnEditor
            The editor.
        text : str
            Its text as it stands.
        """

        def __init__(self, editor: TurnEditor, text: str) -> None:
            super().__init__()
            self.editor = editor
            self.text = text

    class Cancelled(Message):
        """Escape was pressed: keep the line as it was.

        Parameters
        ----------
        editor : TurnEditor
            The editor.
        """

        def __init__(self, editor: TurnEditor) -> None:
            super().__init__()
            self.editor = editor

    def __init__(self, turn_id: str, kind: EditKind, text: str, *, editor: str) -> None:
        super().__init__(classes="editor")
        self.turn_id = turn_id
        self.kind: EditKind = kind
        self._text = text
        self._editor = editor

    def compose(self) -> ComposeResult:
        """Lay out the text, then its keys."""
        yield _Area(self._text, self._editor)
        yield Static(_hint(), classes="editor-keys")

    def on_mount(self) -> None:
        """Take the focus, with the cursor at the end of the text."""
        self.focus_text()

    def focus_text(self) -> None:
        """Give the text the focus, with the cursor at the end."""
        area = self.query_one(_Area)
        area.focus()
        area.move_cursor(area.document.end)

    @property
    def text(self) -> str:
        """The text as it stands."""
        return self.query_one(_Area).text
