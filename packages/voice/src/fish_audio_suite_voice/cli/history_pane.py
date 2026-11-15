"""The conversation pane of the full-screen app: the chat's turns, and the turn in progress.

Between turns it draws the record the session sends (``ChatShown``): one row per turn, the
user's line and the reply as it is heard, without its cues. While a turn happens it adds the
user's words and the reply as they arrive, and the next record takes their place. With the
pane focused, a turn can be selected and changed. Each change goes to the session as a
``ChatCommand``; the pane redraws only from the record the session sends back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import VerticalGroup, VerticalScroll
from textual.content import Content
from textual.timer import Timer
from textual.widgets import Input, Static

from fish_audio_suite_kit import split_cues
from fish_audio_suite_voice.cli.chat_screens import printable, spoken
from fish_audio_suite_voice.cli.turn_editor import EditKind, TurnEditor
from fish_audio_suite_voice.session.commands import (
    ChatCommand,
    DeleteTurn,
    EditMessage,
    EditReply,
    SetHidden,
    TryAgain,
    Undo,
)
from fish_audio_suite_voice.session.conversation import Turn
from fish_audio_suite_voice.session.events import ChatShown

__all__ = [
    "Conversation",
    "TurnRow",
]

# How long the words so far stay up after the last update with no turn made of them, as
# when the recogniser drops the turn as noise.
_HEARING_HOLD_S = 3.0
_YOU = ("› ", "bold $text-primary")  # noqa: RUF001 - the quote mark is the design
_CUT = ("…", "dim")  # ends a reply that was cut off
_UNHEARD = "dim"  # the words of a reply not played yet
_KEY = "bold $text-accent"
_UNDO_HINT = " · z to undo"


def _with_lead_cue(before: str, text: str) -> str:
    """Return an edited reply led by the cue the reply opened with, unless it has its own.

    The editor shows a reply without its cues, so an edit saves none of them. The cue a
    reply opens with sets its mood, and keeping it is the one cue that maps without doubt
    onto the new text.
    """
    if text.startswith("["):
        return text
    for piece, is_cue in split_cues(before):
        if piece.strip():
            return f"{piece} {text}" if is_cue else text
    return text


def _heard_end(words: str, chars: int) -> int:
    """Return where the heard part of ``words`` ends: at ``chars``, back to a whole word."""
    if chars >= len(words):
        return len(words)
    end = max(0, chars)
    while end > 0 and not words[end].isspace():
        end -= 1
    return end


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
        self._editing: EditKind | None = None
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
        self._user.display = self._editing != "message"
        self._reply.display = reply is not None and self._editing != "reply"
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
        self.set_class(turn.hidden, "hidden")

    def open_editor(self, editor: TurnEditor) -> None:
        """Put ``editor`` in place of the line it edits.

        Parameters
        ----------
        editor : TurnEditor
            The editor, not yet mounted.
        """
        self._editing = editor.kind
        self.add_class("editing")
        self.mount(editor, after=self._reply if editor.kind == "reply" else self._user)
        self.show(self.turn)

    def close_editor(self) -> None:
        """Show the edited line again. The caller removes the editor."""
        self._editing = None
        self.remove_class("editing")
        self.show(self.turn)

    def show_keys(self, *, resend: bool, try_again: bool) -> None:
        """Show the keys that change this turn, under it while it is selected.

        Parameters
        ----------
        resend : bool
            Whether ``E`` applies: no turn is in progress.
        try_again : bool
            Whether ``r`` applies: the turn is the newest and no turn is in progress.
        """
        answered = self.turn.reply is not None
        keys = [("e", "Edit reply")] if answered else []
        keys += [("E", "Edit my message")] if resend else []
        keys += [("r", "Try again")] if try_again else []
        keys += [("y", "Copy")] if answered else []
        keys += [("h", "Show" if self.turn.hidden else "Hide"), ("d", "Delete")]
        self._keys.update(_hint(*keys))


class Conversation(VerticalScroll):
    """The conversation: the chat's turns, the turn in progress, and short notes.

    Parameters
    ----------
    send : Callable
        Takes each change the user asks for, such as ``LiveInput.send``.
    hold_mic : Callable
        Called with True when a line opens in the editor, and False when it closes, so the
        mic is off while the user edits.
    editor : str, optional
        The command Ctrl+E hands a line being edited to. ``vi`` by default.
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
        Binding("E", "edit_message", "Edit my message", show=False),
        Binding("r", "try_again", "Try again"),
        Binding("y", "copy", "Copy", show=False),
        Binding("h", "hide", "Hide or show", show=False),
        Binding("d,delete", "delete", "Delete", show=False),
        Binding("z", "undo", "Undo"),
        Binding("question_mark", "app.keys", "Keys", show=False),
        Binding("escape,tab", "leave", "Back to typing"),
    ]

    def __init__(
        self,
        *,
        send: Callable[[ChatCommand], None],
        hold_mic: Callable[[bool], None],
        editor: str = "vi",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._editor_command = editor
        self._send = send
        self._hold_mic = hold_mic
        self._editor: TurnEditor | None = None
        self._back_from: str | None = None  # the turn an editor was closed on
        self._user: Static | None = None
        self._reply: Static | None = None
        self._reply_words = ""
        self._reply_cut = False
        # Characters of the reply heard so far while it plays; None shows it all as heard.
        self._heard: int | None = None
        self._pending: Static | None = None
        self._expiry: Timer | None = None
        self._chat_id: str | None = None
        self._rows: dict[str, TurnRow] = {}
        self._order: list[str] = []
        self._selected: str | None = None
        # A turn is in progress: from its line until the record that holds it.
        self._busy = False

    @property
    def selected(self) -> Turn | None:
        """The selected turn, if any."""
        row = None if self._selected is None else self._rows.get(self._selected)
        return None if row is None else row.turn

    @property
    def can_try_again(self) -> bool:
        """Whether the newest turn's reply can be tried again now."""
        return self._try_again_target() is not None

    def _try_again_target(self) -> Turn | None:
        row = self._rows.get(self._order[-1]) if self._order else None
        turn = None if row is None else row.turn
        if self._busy or turn is None or turn.reply is None or turn.hidden:
            return None
        return turn

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
        editor = self._editor
        if editor is not None and (
            shown.chat_id != self._chat_id or editor.turn_id not in {t.id for t in shown.turns}
        ):
            self._close_editor()  # its turn is gone
        if shown.chat_id != self._chat_id:
            self._chat_id = shown.chat_id
            self._stop_pending()
            self.remove_children()
            self._rows, self._user, self._reply, self._selected = {}, None, None, None
        self._busy = False
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
        """Select the newest turn whenever the pane is focused, or go back to an open editor."""
        if self._editor is not None:
            self._editor.focus_text()
            return
        back_from, self._back_from = self._back_from, None
        if back_from in self._rows:
            self._select(back_from)  # the turn just edited stays selected
        elif self._order:
            self._select(self._order[-1])

    def _select(self, turn_id: str | None) -> None:
        self._selected = turn_id
        self._mark_selected()
        row = None if turn_id is None else self._rows.get(turn_id)
        if row is not None:
            self.scroll_to_widget(row, animate=False)
        self.refresh_bindings()

    def _mark_selected(self) -> None:
        target = self._try_again_target()
        for turn_id, row in self._rows.items():
            row.set_class(turn_id == self._selected, "selected")
            if turn_id == self._selected:
                row.show_keys(
                    resend=not self._busy,
                    try_again=target is not None and target.id == turn_id,
                )

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
        if action == "edit_message":
            return turn is not None and not self._busy
        if action == "try_again":
            target = self._try_again_target()
            return turn is not None and target is not None and target.id == turn.id
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
        """Rewrite the selected reply in place, as it is heard."""
        turn = self.selected
        if turn is not None and turn.reply is not None:
            self._open_editor(turn, "reply", spoken(turn.reply.content))

    def action_edit_message(self) -> None:
        """Rewrite the user's line of the selected turn in place, then answer it afresh."""
        turn = self.selected
        if turn is not None and not self._busy:
            self._open_editor(turn, "message", turn.user)

    @property
    def editing(self) -> bool:
        """Whether a line is open in the editor."""
        return self._editor is not None

    def _open_editor(self, turn: Turn, kind: EditKind, text: str) -> None:
        self._close_editor()
        row = self._rows.get(turn.id)
        if row is None:
            return
        self._editor = TurnEditor(turn.id, kind, text, editor=self._editor_command)
        row.open_editor(self._editor)
        self._hold_mic(True)
        self.refresh_bindings()

    def _close_editor(self) -> None:
        editor, self._editor = self._editor, None
        if editor is None:
            return
        row = self._rows.get(editor.turn_id)
        if row is not None:
            row.close_editor()
        editor.remove()
        self._hold_mic(False)
        self._back_from = editor.turn_id
        self.focus()
        self.refresh_bindings()

    def on_turn_editor_cancelled(self, event: TurnEditor.Cancelled) -> None:
        """Close the editor and keep the line as it was."""
        event.stop()
        self._close_editor()

    def on_turn_editor_saved(self, event: TurnEditor.Saved) -> None:
        """Send the change the editor holds, unless it is blank, then close it."""
        event.stop()
        row = self._rows.get(event.editor.turn_id)
        turn = None if row is None else row.turn
        text = event.text.strip()
        if turn is None:
            self._close_editor()
            return
        if not text:
            refused = (
                "A message cannot be blank. Delete the turn instead."
                if event.editor.kind == "message"
                else "A reply cannot be blank. Hide or delete the turn instead."
            )
            self.notify(refused, severity="warning")
            return
        self._close_editor()
        if event.editor.kind == "message":
            if text != turn.user:
                self._send(EditMessage(turn.id, text))
                self._busy = True
                self._mark_selected()
                self.refresh_bindings()
                self.notify("Message edited and sent again" + _UNDO_HINT)
        elif turn.reply is not None and text != spoken(turn.reply.content):
            self._send(EditReply(turn.id, _with_lead_cue(turn.reply.content, text)))
            self.notify("Reply edited" + _UNDO_HINT)

    def action_try_again(self) -> None:
        """Answer the selected turn's line again, in place of its reply."""
        if self.check_action("try_again", ()):
            self.try_again()

    def try_again(self) -> None:
        """Answer the newest turn's line again, in place of its reply. ``z`` brings it back."""
        turn = self._try_again_target()
        if turn is None:
            return
        self._send(TryAgain(turn.id))
        self._busy = True
        self._mark_selected()
        self.refresh_bindings()
        self.notify("Trying again" + _UNDO_HINT)

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

    def show_words(self, text: str) -> None:
        """Show what the user has said so far, dim, on the line that is about to become their words.

        Parameters
        ----------
        text : str
            Everything recognised in the turn up to now. Each call replaces the last.

        Notes
        -----
        The line is added with the first words, and goes away by itself if no line follows
        them, as when the recogniser drops the turn as noise. Blank text adds nothing.
        """
        words = printable(text).strip()
        if not words:
            return
        content = Content.assemble(_YOU, words)
        if self._pending is None:
            self._pending = self._add("you pending", content)
            self.follow()
        else:
            self._pending.update(content)
        if self._expiry is not None:
            self._expiry.stop()
        self._expiry = self.set_timer(_HEARING_HOLD_S, self._drop_pending)

    def _stop_pending(self) -> Static | None:
        pending, self._pending = self._pending, None
        if self._expiry is not None:
            self._expiry.stop()
        self._expiry = None
        return pending

    def _drop_pending(self) -> None:
        pending = self._stop_pending()
        if pending is not None:
            pending.remove()

    def add_user(self, text: str) -> None:
        """Add a line the user said or typed. The reply that follows starts fresh.

        If the words so far were showing, their line becomes this line where it stands.

        Parameters
        ----------
        text : str
            The line.
        """
        self._reply, self._reply_words, self._reply_cut = None, "", False
        self._heard = None
        self._busy = True
        self._mark_selected()
        self.refresh_bindings()
        content = Content.assemble(_YOU, printable(text))
        pending = self._stop_pending()
        if pending is None and self._open_line() == text:
            return  # a line answered again: the record already shows it
        if pending is None:
            self._user = self._add("you", content)
        else:
            pending.remove_class("pending")
            pending.update(content)
            self._user = pending

    def _open_line(self) -> str | None:
        row = self._rows.get(self._order[-1]) if self._order else None
        if row is None or row.turn.reply is not None or row.turn.hidden:
            return None
        return row.turn.user

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
        content = self._reply_content()
        if self._reply is None:
            self._reply = self._add("reply", content)
        else:
            self._reply.update(content)

    def _reply_content(self) -> Content:
        words, heard = self._reply_words, self._heard
        if heard is None:
            return Content.assemble(words, *([_CUT] if self._reply_cut else []))
        end = _heard_end(words, heard)
        if self._reply_cut:
            return Content.assemble(words[:end].rstrip(), _CUT)  # the rest was never heard
        return Content.assemble(words[:end], (words[end:], _UNHEARD))

    def show_heard(self, chars: int) -> None:
        """Show how much of the reply has been heard: the rest is dim until it plays.

        Parameters
        ----------
        chars : int
            Characters of the reply heard so far, its cues left out (``ReplyHeard``).
        """
        if self._reply_cut:
            return  # what plays after a barge-in is not this reply's
        self._heard = chars
        if self._reply is not None:
            self._reply.update(self._reply_content())

    def end_playback(self) -> None:
        """Show the whole reply as heard: it has played to the end."""
        if self._heard is not None and not self._reply_cut:
            self._heard = None
            if self._reply is not None:
                self._reply.update(self._reply_content())

    def cut_reply(self) -> None:
        """Mark the reply in progress as cut off, when the user talks over it.

        Words not heard yet go, when it is known how much was heard.
        """
        if self._reply is not None:
            self._reply_cut = True
            self._reply.update(self._reply_content())

    def _add(self, role: str, content: Content) -> Static:
        message = Static(content, classes=f"message {role}")
        self.mount(message)
        return message
