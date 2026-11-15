"""The conversation pane of the full-screen app: the chat's turns, and the turn in progress.

Between turns it draws the record the session sends (``ChatShown``): one row per turn, the
user's line and the reply as it is heard, without its cues. While a turn happens it adds the
user's words and the reply as they arrive, and the next record takes their place. With the
pane focused, a turn can be selected and changed. Each change goes to the session as a
``ChatCommand``; the pane redraws only from the record the session sends back.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import VerticalGroup, VerticalScroll
from textual.content import Content
from textual.timer import Timer
from textual.widgets import Input, Static

from fish_audio_suite_kit import split_cues
from fish_audio_suite_voice.cli.chat_screens import EditReplyScreen, printable
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
    "TurnRow",
    "spoken",
]

_HEARING_HOLD_S = 3.0  # how long "you" keeps pulsing after the last loud moment, with no transcript
_HEARING_TICK_S = 0.1
_PULSE_CELLS = 5  # characters wide, two bars each
_YOU = ("› ", "bold $text-primary")  # noqa: RUF001 - the quote mark is the design
_CUT = ("…", "dim")  # ends a reply that was cut off
_KEY = "bold $text-accent"
_UNDO_HINT = " · z to undo"


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


def _settled(text: str) -> str:
    # A reply still streaming can end in the first half of a cue ("[war"). Leave it out
    # until it closes, so it never flashes up as words.
    head, bracket, tail = text.rpartition("[")
    return head if bracket and "]" not in tail and "\n" not in tail else text


def _hint(*pairs: tuple[str, str]) -> Content:
    parts: list[tuple[str, str] | str] = []
    for index, (key, label) in enumerate(pairs):
        if index:
            parts.append("   ")
        parts += [(key, _KEY), (f" {label}", "dim")]
    return Content.assemble(*parts)


class TurnRow(VerticalGroup):
    """One recorded turn: the user's line, the reply as heard, its notes and, when selected, its keys.

    Parameters
    ----------
    turn : Turn
        The turn to draw.
    """

    turn: Turn

    def __init__(self, turn: Turn) -> None:
        self._user = Static(classes="you")
        self._reply = Static(classes="reply")
        self._note = Static(classes="turn-note")
        self._keys = Static(classes="actions")
        super().__init__(self._user, self._reply, self._note, self._keys, classes="message turn")
        self.show(turn)

    def show(self, turn: Turn) -> None:
        """Draw the turn as it is now.

        Parameters
        ----------
        turn : Turn
            The turn.
        """
        self.turn = turn
        reply = turn.reply
        self._user.update(Content.assemble(_YOU, printable(turn.user)))
        self._reply.display = reply is not None
        if reply is not None:
            cut = [_CUT] if reply.interrupted else []
            self._reply.update(Content.assemble(spoken(reply.content), *cut))
        notes = [
            note
            for note, on in (
                ("hidden from AI", turn.hidden),
                ("edited", reply is not None and reply.edit is not None),
            )
            if on
        ]
        self._note.display = bool(notes)
        self._note.update(Content.assemble((" · ".join(notes), "dim italic")))
        keys = [("e", "Edit reply"), ("y", "Copy")] if reply is not None else []
        keys += [("h", "Show" if turn.hidden else "Hide"), ("d", "Delete")]
        self._keys.update(_hint(*keys))
        self.set_class(turn.hidden, "hidden")


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
    for it. While the newest turn is selected, a new turn is selected as it arrives, so the
    pane follows the chat; selecting an older turn stops that until End.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("k,up", "move(-1)", "Move", show=False),
        Binding("j,down", "move(1)", "Move", key_display="j/k"),
        Binding("g,home", "first", "First", show=False),
        Binding("G,end", "newest", "Newest", show=False),
        Binding("e", "edit", "Edit"),
        Binding("y", "copy", "Copy", show=False),
        Binding("h", "hide", "Hide or show", show=False),
        Binding("d,delete", "delete", "Delete", show=False),
        Binding("z", "undo", "Undo"),
        Binding("question_mark", "app.keys", "Keys", show=False),
        Binding("escape,tab", "leave", "Back to typing"),
    ]

    def __init__(self, *, send: Callable[[ChatCommand], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._send = send
        self._user: Static | None = None
        self._reply: Static | None = None
        self._reply_words = ""
        self._reply_cut = False
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

    @property
    def _following(self) -> bool:
        return not self._order or self._selected in (None, self._order[-1])

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
        following = self._following
        ids = [turn.id for turn in shown.turns]
        for turn_id in set(self._rows) - set(ids):
            self._rows.pop(turn_id).remove()
        previous: TurnRow | None = None
        for turn in shown.turns:
            row = self._rows.get(turn.id)
            if row is not None:
                row.show(turn)
            else:
                row = self._rows[turn.id] = TurnRow(turn)
                self._mount_row(row, previous)
            previous = row
        for live in (self._user, self._reply):
            if live is not None:
                live.remove()
        self._user = self._reply = None
        self._order = ids
        if self._selected not in self._rows:
            self._selected = None
        if self.has_focus and following and ids:
            self._selected = ids[-1]
        self._mark_selected()
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
        """Scroll to the newest line, unless the user is looking through older turns."""
        if not self.has_focus or self._following:
            self.scroll_end(animate=False)

    def on_focus(self) -> None:
        """Select the newest turn whenever the pane is focused, where the chat is now."""
        if self._order:
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
        """Offer only the changes that apply to the selected turn.

        Parameters
        ----------
        action : str
            The action's name.
        parameters : tuple
            Its parameters.

        Returns
        -------
        bool or None
            False hides a key that does not apply now, and lets it go on to the app.
        """
        del parameters
        turn = self.selected
        if action in ("edit", "copy"):
            return turn is not None and turn.reply is not None
        if action in ("hide", "delete"):
            return turn is not None
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

    def action_first(self) -> None:
        """Select the first turn of the chat."""
        if self._order:
            self._select(self._order[0])

    def action_newest(self) -> None:
        """Select the newest turn, and follow new turns as they arrive."""
        if self._order:
            self._select(self._order[-1])
        self.scroll_end(animate=False)

    def action_leave(self) -> None:
        """Go back to typing."""
        self.screen.focus_next(Input)

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
                self.notify("Reply edited" + _UNDO_HINT)

        self.app.push_screen(EditReplyScreen(before), edited)

    def action_copy(self) -> None:
        """Copy the selected reply as it is heard, without its cues."""
        turn = self.selected
        if turn is not None and turn.reply is not None:
            self.app.copy_to_clipboard(spoken(turn.reply.content))
            self.notify("Copied")

    def action_hide(self) -> None:
        """Hide the selected turn from the model, or show it again."""
        turn = self.selected
        if turn is not None:
            self._send(SetHidden(turn.id, hidden=not turn.hidden))
            done = "shown to AI again" if turn.hidden else "hidden from AI"
            self.notify(f"Message {done}{_UNDO_HINT}")

    def action_delete(self) -> None:
        """Remove the selected turn from the chat, and select the one after it."""
        turn = self.selected
        if turn is None:
            return
        index = self._order.index(turn.id)
        rest = self._order[index + 1 :] or self._order[:index][-1:]
        self._send(DeleteTurn(turn.id))
        self._select(rest[0] if rest else None)
        self.notify("Message deleted" + _UNDO_HINT)

    def action_undo(self) -> None:
        """Reverse the last change to the chat."""
        self._send(Undo())

    def show_hearing(self) -> None:
        """Show that the user is speaking: a pulsing line, until the words arrive.

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

        If the pulsing line was showing, it becomes this line where it stands.

        Parameters
        ----------
        text : str
            The line.
        """
        self._reply, self._reply_words, self._reply_cut = None, "", False
        content = Content.assemble(_YOU, printable(text))
        pending = self._stop_pending()
        if pending is None:
            self._user = self._add("you", content)
        else:
            pending.remove_class("pending")
            pending.update(content)
            self._user = pending

    def add_note(self, text: str) -> None:
        """Add a dim status line, such as a retry.

        Parameters
        ----------
        text : str
            The note.
        """
        self._add("note", Content.assemble((printable(text), "dim")))

    def show_reply(self, text: str) -> None:
        """Show the reply so far as it is heard, adding its line on the first call.

        Parameters
        ----------
        text : str
            The reply so far, cues and all. They are left out of what is shown.
        """
        self._reply_words = spoken(_settled(text))
        if self._reply is None and not self._reply_words:
            return
        cut = [_CUT] if self._reply_cut else []
        content = Content.assemble(self._reply_words, *cut)
        if self._reply is None:
            self._reply = self._add("reply", content)
        else:
            self._reply.update(content)

    def cut_reply(self) -> None:
        """Mark the reply in progress as cut off, when the user talks over it."""
        if self._reply is not None:
            self._reply_cut = True
            self.show_reply(self._reply_words)

    def _add(self, role: str, content: Content) -> Static:
        message = Static(content, classes=f"message {role}")
        self.mount(message)
        return message
