"""The conversation pane of the full-screen app: the chat's turns, and the turn in progress.

Between turns it draws the record the session sends (``ChatShown``): one row per turn,
marked when the model is not sent it. While a turn happens it adds the user's words and
the reply as they arrive, and the next record takes their place. With the pane focused, a
turn can be selected and changed. Each change goes to the session as a ``ChatCommand``;
the pane redraws only from the record the session sends back.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static

from fish_audio_suite_kit import split_cues
from fish_audio_suite_voice.cli.chat_screens import EditReplyScreen
from fish_audio_suite_voice.cli.session_view import wave_dots
from fish_audio_suite_voice.session.commands import (
    ChatCommand,
    DeleteTurn,
    EditReply,
    SetHidden,
    Undo,
)
from fish_audio_suite_voice.session.conversation import Turn
from fish_audio_suite_voice.session.events import ChatShown

__all__ = [
    "Conversation",
    "Seen",
    "TurnRow",
    "cue_content",
    "printable",
]

_HEARING_HOLD_S = 3.0  # how long "you" keeps pulsing after the last loud moment, with no transcript
_HEARING_TICK_S = 0.1
_PULSE_CELLS = 5  # characters wide, two bars each
_YOU = ("you ▸ ", "bold $text-primary")
_LLM = ("llm ▸ ", "bold $text-accent")


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


def cue_content(text: str) -> Content:
    """Return ``text`` as content with each ``[cue]`` styled and nothing read as markup.

    Parameters
    ----------
    text : str
        A reply.

    Returns
    -------
    Content
        The printable text, its cues in the accent colour.
    """
    parts: list[tuple[str, str] | str] = [
        (piece, "italic $text-accent") if is_cue else piece
        for piece, is_cue in split_cues(printable(text))
    ]
    return Content.assemble(*parts)


class Seen(StrEnum):
    """Whether the model is sent a turn, and if not, why."""

    SENT = "sent"
    HIDDEN = "hidden"
    OLDER = "older"


def _seen(turn: Turn, shown: ChatShown) -> Seen:
    if turn.hidden:
        return Seen.HIDDEN
    return Seen.SENT if turn.id in shown.sent else Seen.OLDER


_SEEN_NOTES = {
    Seen.SENT: "",
    Seen.HIDDEN: "hidden from the model",
    Seen.OLDER: "older than the model is sent",
}


class TurnRow(Static):
    """One recorded turn: the user's line, the reply the model is sent, and what is marked."""

    turn: Turn
    seen: Seen

    def __init__(self, turn: Turn, seen: Seen) -> None:
        super().__init__(classes="message turn")
        self.show(turn, seen)

    def show(self, turn: Turn, seen: Seen) -> None:
        """Draw the turn as it is now.

        Parameters
        ----------
        turn : Turn
            The turn.
        seen : Seen
            Whether the model is sent it.
        """
        self.turn = turn
        self.seen = seen
        lines = [Content.assemble(_YOU, printable(turn.user))]
        reply = turn.reply
        if reply is not None:
            lines.append(Content.assemble(_LLM, cue_content(reply.content)))
        notes = [
            note
            for note, on in (
                ("edited", reply is not None and reply.edit is not None),
                ("cut off", reply is not None and reply.interrupted),
                (_SEEN_NOTES[seen], seen is not Seen.SENT),
            )
            if on
        ]
        if notes:
            lines.append(Content.assemble((" · ".join(notes), "italic $text-muted")))
        self.update(Content("\n").join(lines))
        self.set_class(seen is Seen.HIDDEN, "hidden")
        self.set_class(seen is Seen.OLDER, "older")


class Conversation(VerticalScroll):
    """The conversation: the chat's turns, the turn in progress, and short notes.

    Parameters
    ----------
    send : Callable
        Takes each change the user asks for, such as ``LiveInput.send``.
    **kwargs : Any
        Passed to ``VerticalScroll``.

    Notes
    -----
    The app tells it what happened and it keeps the rest to itself, including which widget
    is the reply being written and which turn is selected, so the app holds no widget state
    for it.
    """

    BORDER_TITLE: ClassVar[str] = "conversation"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "move(-1)", "Up", show=False),
        Binding("down,j", "move(1)", "Down", show=False),
        Binding("e", "edit", "Edit"),
        Binding("h", "hide", "Hide/show"),
        Binding("d,delete", "delete", "Delete"),
        Binding("u", "undo", "Undo"),
    ]

    def __init__(self, *, send: Callable[[ChatCommand], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._send = send
        self._user: Static | None = None
        self._reply: Static | None = None
        self._pending: Static | None = None
        self._words = ""
        self._pulse: Timer | None = None
        self._expiry: Timer | None = None
        self._phase = 0
        self._chat_id: str | None = None
        self._rows: dict[str, TurnRow] = {}
        self._order: list[str] = []
        self._selected: str | None = None

    @property
    def selected(self) -> Turn | None:
        """The selected turn, if any."""
        row = None if self._selected is None else self._rows.get(self._selected)
        return None if row is None else row.turn

    def show_record(self, shown: ChatShown) -> None:
        """Draw the chat's record, in place of the turn in progress.

        Parameters
        ----------
        shown : ChatShown
            The record as the session holds it now. Another chat's record clears the pane.

        Notes
        -----
        Rows already drawn are updated where they stand, so notes stay where they were.
        A new row goes where the turn in progress was drawn, or beside its neighbour.
        """
        if shown.chat_id != self._chat_id:
            self._chat_id = shown.chat_id
            self._stop_pending()
            self.remove_children()
            self._rows, self._user, self._reply, self._selected = {}, None, None, None
        ids = [turn.id for turn in shown.turns]
        for turn_id in set(self._rows) - set(ids):
            self._rows.pop(turn_id).remove()
        previous: TurnRow | None = None
        for turn in shown.turns:
            seen = _seen(turn, shown)
            row = self._rows.get(turn.id)
            if row is not None:
                row.show(turn, seen)
            else:
                row = self._rows[turn.id] = TurnRow(turn, seen)
                self._mount_row(row, previous)
            previous = row
        for live in (self._user, self._reply):
            if live is not None:
                live.remove()
        self._user = self._reply = None
        self._order = ids
        if self._selected not in self._rows:
            self._selected = None
        self._mark_selected()
        visible = sum(1 for turn in shown.turns if turn.id in shown.sent)
        # Set through the base type: an assignment on ``self`` reads to the public-API type
        # check as a new attribute with no declared type.
        pane: VerticalScroll = self
        pane.border_subtitle = f"model sees {visible} of {len(ids)} turns"
        self.refresh_bindings()

    def _mount_row(self, row: TurnRow, previous: TurnRow | None) -> None:
        if self._user is not None:
            self.mount(row, before=self._user)
        elif previous is not None:
            self.mount(row, after=previous)
        elif self.children:
            self.mount(row, before=0)
        else:
            self.mount(row)

    def follow(self) -> None:
        """Scroll to the newest line, unless the user is looking through the turns."""
        if not (self.has_focus and self._selected is not None):
            self.scroll_end(animate=False)

    def on_focus(self) -> None:
        """Select the newest turn when the pane is focused with nothing selected."""
        if self._selected is None and self._order:
            self._select(self._order[-1])

    def _select(self, turn_id: str | None) -> None:
        self._selected = turn_id
        self._mark_selected()
        row = None if turn_id is None else self._rows.get(turn_id)
        if row is not None:
            self.scroll_to_widget(row, animate=False)
        self.refresh_bindings()

    def _mark_selected(self) -> None:
        for turn_id, row in self._rows.items():
            row.set_class(turn_id == self._selected, "selected")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Offer the changes that apply to the selected turn.

        Parameters
        ----------
        action : str
            The action's name.
        parameters : tuple
            Its parameters.

        Returns
        -------
        bool or None
            None shows a change that does not apply now as unavailable.
        """
        del parameters
        turn = self.selected
        if action == "edit":
            return True if turn is not None and turn.reply is not None else None
        if action in ("hide", "delete"):
            return True if turn is not None else None
        return True

    def action_move(self, step: int) -> None:
        """Select the turn ``step`` after the selected one, or the newest when none is.

        Parameters
        ----------
        step : int
            -1 for the turn before, 1 for the turn after.
        """
        if not self._order:
            return
        if self._selected not in self._order:
            self._select(self._order[-1])
            return
        index = self._order.index(self._selected) + step
        self._select(self._order[min(len(self._order) - 1, max(0, index))])

    def action_edit(self) -> None:
        """Open the selected reply to rewrite it."""
        turn = self.selected
        if turn is None or turn.reply is None:
            return
        before = turn.reply.content
        turn_id = turn.id

        def edited(text: str | None) -> None:
            if text is not None and text != before:
                self._send(EditReply(turn_id, text))

        self.app.push_screen(EditReplyScreen(before), edited)

    def action_hide(self) -> None:
        """Hide the selected turn from the model, or show it again."""
        turn = self.selected
        if turn is not None:
            self._send(SetHidden(turn.id, hidden=not turn.hidden))

    def action_delete(self) -> None:
        """Remove the selected turn from the chat, and select the one after it."""
        turn = self.selected
        if turn is None:
            return
        index = self._order.index(turn.id)
        rest = self._order[index + 1 :] or self._order[:index][-1:]
        self._send(DeleteTurn(turn.id))
        self._select(rest[0] if rest else None)

    def action_undo(self) -> None:
        """Reverse the last change to the chat."""
        self._send(Undo())

    def show_hearing(self) -> None:
        """Show that the user is speaking: a pulsing "you" line, until the words arrive.

        Call it on every loud moment. The line is added on the first call, and goes away
        by itself if no words follow, as after a cough.
        """
        if self._pending is None:
            self._phase = 0
            self._pending = self._add("you pending", self._pulse_content())
            if self.app.animation_level != "none":  # Textual's reduced-motion setting
                self._pulse = self.set_interval(_HEARING_TICK_S, self._advance_pulse)
            self.follow()
        if self._expiry is not None:
            self._expiry.stop()
        self._expiry = self.set_timer(_HEARING_HOLD_S, self._drop_pending)

    def show_words(self, text: str) -> None:
        """Show what the user has said so far, on the line that is about to become their words.

        Parameters
        ----------
        text : str
            Everything recognised in the turn up to now. Each call replaces the last.
        """
        self._words = printable(text).strip()
        self.show_hearing()
        if self._pending is not None:
            self._pending.update(self._pulse_content())

    def _pulse_content(self) -> Content:
        levels = [(math.sin(self._phase * 0.7 + i * 0.6) + 1) / 2 for i in range(_PULSE_CELLS * 2)]
        (dots,) = wave_dots(levels, 1)
        if self._words:
            return Content.assemble(_YOU, (dots, "$text-success"), " ", self._words)
        return Content.assemble(_YOU, (dots, "$text-success"))

    def _advance_pulse(self) -> None:
        if self._pending is not None:
            self._phase += 1
            self._pending.update(self._pulse_content())

    def _stop_pending(self) -> Static | None:
        pending, self._pending = self._pending, None
        self._words = ""
        for timer in (self._pulse, self._expiry):
            if timer is not None:
                timer.stop()
        self._pulse = self._expiry = None
        return pending

    def _drop_pending(self) -> None:
        pending = self._stop_pending()
        if pending is not None:
            pending.remove()

    def add_user(self, text: str) -> None:
        """Add a line the user said or typed. The reply that follows starts fresh.

        If the "you" line was pulsing, it becomes this line where it stands.

        Parameters
        ----------
        text : str
            The line.
        """
        self._reply = None
        content = Content.assemble(_YOU, printable(text))
        pending = self._stop_pending()
        if pending is None:
            self._user = self._add("you", content)
        else:
            pending.remove_class("pending")
            pending.update(content)
            self._user = pending

    def add_note(self, text: str) -> None:
        """Add a dim status line, such as a retry or an interruption.

        Parameters
        ----------
        text : str
            The note.
        """
        self._add("note", Content.assemble((printable(text), "dim")))

    def show_reply(self, text: str) -> None:
        """Show the reply so far, adding its line on the first call and updating it after.

        Parameters
        ----------
        text : str
            The reply so far.
        """
        if self._reply is None and not text:
            return
        content = Content.assemble(_LLM, cue_content(text))
        if self._reply is None:
            self._reply = self._add("llm", content)
        else:
            self._reply.update(content)

    def _add(self, role: str, content: Content) -> Static:
        message = Static(content, classes=f"message {role}")
        self.mount(message)
        return message
