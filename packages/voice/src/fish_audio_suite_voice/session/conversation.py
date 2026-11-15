"""The conversation record, and the messages a request sends the model from it.

``Conversation`` keeps every turn of a session: the user's line, what the model wrote of
the part the user heard, and the user's edits. It converts to and from plain JSON. Turns
can be edited, hidden, deleted, and each change undone.

``HistoryView`` builds the request from it: the system prompt, then the last visible turns
with edits applied. It drops old turns in blocks, so the start of the prompt stays the
same over several turns, and it can strip the cues from older replies, because the model
copies the cues it sees in its own history.
"""

from __future__ import annotations

import dataclasses
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Self

from fish_audio_suite_kit import (
    ChatMessage,
    is_same_utterance,
    strip_cue_tags,
)

__all__ = [
    "SCHEMA_VERSION",
    "Conversation",
    "HistoryView",
    "Reply",
    "Turn",
]

# The ``version`` field of ``Conversation.to_json``. Raise it when the shape changes.
SCHEMA_VERSION: Final = 1
# Unanswered user lines kept in one turn. While the model keeps failing, the oldest drop
# off, so the merged line cannot grow without bound.
_UNANSWERED_LINES: Final = 4
# Over the cap, a quarter of the turns are dropped at once. See HistoryView.
_TRIM_BLOCK: Final = 4
# Replies in a row that open with one word before the next request asks for another.
_OPENING_RUN: Final = 2
# A lead set off from the sentence, such as an exclamation, a sound or a form of address:
# one to three words, then a comma, period, "!", "?" or ellipsis. A sentence that merely
# starts with a common word ("I think...") has none, so it never asks for another opening.
_LEAD: Final = re.compile(
    r"\W*([^\W\d_]+(?:'[^\W\d_]+)?)(?:\s+[^\W\d_]+(?:'[^\W\d_]+)?){0,2}[,.!?…]"
)
_STRETCHED: Final = re.compile(r"(\w)\1+")
# Words that start a short sentence ("I know.", "That's it!") rather than an exclamation.
_SENTENCE_STARTS: Final = frozenset(
    {"i", "i'm", "i'd", "i'll", "i've", "you", "we", "it", "it's", "that", "that's", "this"}
    | {"they", "he", "she", "there", "here", "what", "who", "how", "why"}
)


@dataclass(frozen=True, slots=True)
class Reply:
    """What the user heard of one reply.

    Attributes
    ----------
    text : str
        What the model wrote, of the part the user heard: none of the cues added before
        Fish read it.
    interrupted : bool
        The user did not hear all of it: a barge-in or a failure cut it off.
    edit : str or None
        The user's text in place of ``text``, which is kept so the edit can be undone.
    """

    text: str
    interrupted: bool = False
    edit: str | None = None

    @property
    def content(self) -> str:
        """The reply as the model is sent it: the edit when there is one."""
        return self.text if self.edit is None else self.edit


@dataclass(frozen=True, slots=True)
class Turn:
    """One user line and the reply to it.

    Attributes
    ----------
    id : str
        Stable for the life of the record, so an edit names the turn it changes.
    user : str
        What the user said. Lines the model never answered are joined by newlines.
    reply : Reply or None
        None until a reply was heard.
    hidden : bool
        Kept in the record, but left out of what the model is sent.
    """

    id: str
    user: str
    reply: Reply | None = None
    hidden: bool = False


@dataclass(frozen=True, slots=True)
class _Undo:
    """How to reverse one edit: put ``before`` back at ``index``."""

    index: int
    before: Turn
    removed: bool


@dataclass(slots=True)
class Conversation:
    """A session's conversation: the prompt and every turn.

    Attributes
    ----------
    system : str
        The system prompt.
    turns : list of Turn
        Every turn, oldest first, hidden ones included. ``HistoryView`` decides which the
        model is sent.
    id : str
        Stable for the life of the record, saved with it, so a saved chat can be named
        and found again.

    Notes
    -----
    A turn is immutable: each change replaces it, and the turn it replaced goes on an undo
    stack. The stack lives only in memory and is not part of ``to_json``.
    """

    system: str
    turns: list[Turn] = field(default_factory=list[Turn])
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    _undo: list[_Undo] = field(default_factory=list[_Undo], repr=False, compare=False)

    def copy(self) -> Conversation:
        """Return a copy whose turns can change without touching this one.

        Returns
        -------
        Conversation
            The same id and turns, which are immutable, in a new list, with no undo
            history.
        """
        return Conversation(self.system, list(self.turns), id=self.id)

    def reopened(self, system_prompt: str) -> Conversation:
        """Return this record under another system prompt, to go on with a saved chat.

        Parameters
        ----------
        system_prompt : str
            The prompt the session runs with now, which may differ from the saved one.

        Returns
        -------
        Conversation
            The same id and turns, the given prompt, and no undo history.
        """
        return Conversation(system_prompt, list(self.turns), id=self.id)

    def last_user(self) -> str:
        """Return the line the next reply answers, or ``""`` when the last turn has a reply."""
        last = self._open_turn()
        return "" if last is None else last.user

    def add_user_line(self, text: str) -> None:
        """Start a turn with what the user said.

        Parameters
        ----------
        text : str
            What the user said or typed.

        Notes
        -----
        A previous line that got no reply (the model failed, or a barge-in cut it off before
        any audio) is still open. The new line joins it instead of starting a turn, so roles
        keep alternating, and a repeat of it is kept once. At most the last four unanswered
        lines are kept.
        """
        last = self._open_turn()
        if last is None:
            self.turns.append(Turn(uuid.uuid4().hex, text))
            return
        if is_same_utterance(text, last.user):
            joined = text
        else:
            joined = "\n".join([*last.user.split("\n"), text][-_UNANSWERED_LINES:])
        self.turns[-1] = dataclasses.replace(last, user=joined)

    def add_reply(self, text: str, *, interrupted: bool) -> None:
        """Record what the user heard of a reply to the last turn.

        Parameters
        ----------
        text : str
            What the model wrote, of the part that was heard. Empty records nothing.
        interrupted : bool
            The user did not hear all of it.

        Notes
        -----
        A reply that only repeats the line it answers is not recorded: kept, it would teach
        the next turn to repeat the user again. When the last turn already has a reply, this
        is the rest of it, spoken again after a false barge-in, and it joins that reply.
        """
        if not text or not self.turns:
            return
        last = self.turns[-1]
        if last.reply is None:
            if is_same_utterance(text, last.user):
                return
            reply = Reply(text, interrupted)
        else:
            reply = Reply(f"{last.reply.text} {text}", interrupted, last.reply.edit)
        self.turns[-1] = dataclasses.replace(last, reply=reply)

    def edit_reply(self, turn_id: str, text: str) -> None:
        """Replace the text of a turn's reply, keeping what the model wrote.

        Parameters
        ----------
        turn_id : str
            The turn whose reply changes.
        text : str
            The reply the model is sent from now on.

        Raises
        ------
        ValueError
            When ``text`` is blank or the turn has no reply. Hide or delete the turn
            instead.
        """
        index, turn = self._find(turn_id)
        if turn.reply is None or not text.strip():
            raise ValueError(f"turn {turn_id} has no reply to edit, or the edit is blank")
        self._change(
            index, dataclasses.replace(turn, reply=dataclasses.replace(turn.reply, edit=text))
        )

    def set_hidden(self, turn_id: str, *, hidden: bool) -> None:
        """Hide a turn from the model, or show it again. The record keeps it either way.

        Parameters
        ----------
        turn_id : str
            The turn to change.
        hidden : bool
            True leaves the turn out of what the model is sent.
        """
        index, turn = self._find(turn_id)
        self._change(index, dataclasses.replace(turn, hidden=hidden))

    def delete_turn(self, turn_id: str) -> None:
        """Remove a turn from the record. ``undo`` puts it back.

        Parameters
        ----------
        turn_id : str
            The turn to remove.
        """
        index, turn = self._find(turn_id)
        del self.turns[index]
        self._undo.append(_Undo(index, turn, removed=True))

    def undo(self) -> bool:
        """Reverse the last edit, hide or delete that was not undone yet.

        Returns
        -------
        bool
            False when there was nothing to undo.
        """
        if not self._undo:
            return False
        step = self._undo.pop()
        if step.removed:
            self.turns.insert(step.index, step.before)
        else:
            self.turns[self._find(step.before.id)[0]] = step.before
        return True

    def to_json(self) -> dict[str, object]:
        """Return the record as plain JSON values.

        Returns
        -------
        dict
            ``version``, ``id``, ``system`` and ``turns``. ``from_json`` reads it back.
        """
        return {
            "version": SCHEMA_VERSION,
            "id": self.id,
            "system": self.system,
            "turns": [_turn_json(turn) for turn in self.turns],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> Self:
        """Read a record written by ``to_json``.

        Parameters
        ----------
        data : Mapping
            The parsed JSON object.

        Returns
        -------
        Conversation
            The record, with no undo history.

        Raises
        ------
        TypeError
            When a field is missing or has the wrong type.
        ValueError
            When the version is not ``SCHEMA_VERSION``, the id is blank, two turns share
            an id, or a reply's edit is blank.
        """
        if data.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"conversation version {data.get('version')!r} is not {SCHEMA_VERSION}"
            )
        turns = [_turn(item) for item in _list(data, "turns")]
        if len({turn.id for turn in turns}) != len(turns):
            raise ValueError("two turns share an id")
        record_id = _str(data, "id")
        if not record_id.strip():
            raise ValueError("the conversation id is blank")
        return cls(_str(data, "system"), turns, id=record_id)

    def _open_turn(self) -> Turn | None:
        # A hidden turn takes no more lines: the user could not see them join it.
        if self.turns and self.turns[-1].reply is None and not self.turns[-1].hidden:
            return self.turns[-1]
        return None

    def _find(self, turn_id: str) -> tuple[int, Turn]:
        for index, turn in enumerate(self.turns):
            if turn.id == turn_id:
                return index, turn
        raise KeyError(turn_id)

    def _change(self, index: int, turn: Turn) -> None:
        self._undo.append(_Undo(index, self.turns[index], removed=False))
        self.turns[index] = turn


@dataclass(frozen=True, slots=True)
class HistoryView:
    """Which turns the model is sent, and how much of their cues.

    Attributes
    ----------
    turns : int
        Turns sent at most (``FISH_VOICE_HISTORY_TURNS``). Over it, a quarter goes at once.
    cue_turns : int or None
        How many of the latest replies keep their cues (``FISH_VOICE_HISTORY_CUE_TURNS``).
        Older replies are sent without them. None keeps the cues in every reply.

    Notes
    -----
    Dropping a quarter at once keeps the start of the prompt the same for the next several
    turns, so a provider that caches prompts reuses it. Dropping one turn per turn would
    change it every time. A cap under four turns drops only what is over it.
    """

    turns: int
    cue_turns: int | None = None

    def messages(self, conversation: Conversation, tone: tuple[str, ...] = ()) -> list[ChatMessage]:
        """Return the messages one request sends the model.

        Parameters
        ----------
        conversation : Conversation
            The record. It is not changed.
        tone : tuple of str, optional
            What Fish ASR heard in the voice, such as ``("laughter", "tired")``. It is noted
            on the last user line for this request only, so it is never sent again.

        Returns
        -------
        list of ChatMessage
            The system prompt, then the visible turns the cap keeps,
            with edits applied. Roles alternate: a visible turn with no reply before the
            last one joins its line to the next. When the last two replies sent opened
            with the same word, a note on the last user line names it and asks for
            another opening, for this request only.
        """
        sent = self.sent(conversation)
        out: list[ChatMessage] = [{"role": "system", "content": conversation.system}]
        out.extend(self._turn_messages(sent))
        notes = [note for note in (_tone_note(tone), _opening_note(sent)) if note]
        return _with_notes(out, notes)

    def sent(self, conversation: Conversation) -> list[Turn]:
        """Return the turns a request sends the model, oldest first.

        Parameters
        ----------
        conversation : Conversation
            The record. It is not changed.

        Returns
        -------
        list of Turn
            The visible turns the cap keeps. A turn left out is hidden or older than the
            window.
        """
        visible = [turn for turn in conversation.turns if not turn.hidden]
        return visible[_window_start(len(visible), self.turns) :]

    def _turn_messages(self, turns: Sequence[Turn]) -> Iterator[ChatMessage]:
        replies = sum(1 for turn in turns if turn.reply is not None)
        plain = 0 if self.cue_turns is None else max(0, replies - self.cue_turns)
        pending: list[str] = []
        for turn in turns:
            pending.append(turn.user)
            if turn.reply is None:
                continue
            yield {"role": "user", "content": "\n".join(pending)}
            pending = []
            content = turn.reply.content
            if plain > 0:
                plain -= 1
                content = strip_cue_tags(content)
            yield {"role": "assistant", "content": content}
        if pending:
            yield {"role": "user", "content": "\n".join(pending)}


def _window_start(count: int, cap: int) -> int:
    """Return the first of ``count`` turns sent under a cap of ``cap``.

    Each time the count goes over the cap, the start moves on by a quarter of the cap and
    one more, then holds until the count is over the cap again.
    """
    if count <= cap:
        return 0
    step = cap // _TRIM_BLOCK + 1
    return -(-(count - cap) // step) * step


def _tone_note(tone: tuple[str, ...]) -> str:
    """Say how the last user line sounded, and that it is not something they said."""
    if not tone:
        return ""
    return (
        f"(Note on how they sounded: {', '.join(tone)}. This is not something they said. "
        "Do not repeat or mention it, and never write it as a [cue] or in brackets.)"
    )


def _opening_note(turns: Sequence[Turn]) -> str:
    """Name the lead word the latest replies all opened with, so the next one opens otherwise.

    The model copies its own recent replies more than it follows the system prompt, so a
    reply that opens like the last few makes the next one likelier to. A note on the line
    it answers is the closest place to break the run.
    """
    replies = [turn.reply.content for turn in turns if turn.reply is not None]
    recent = [_lead_word(reply) for reply in replies[-_OPENING_RUN:]]
    if len(recent) < _OPENING_RUN or not recent[-1]:
        return ""
    if len({_word_key(word) for word in recent}) != 1:
        return ""
    return (
        f'(Note: your last {_OPENING_RUN} replies all began with "{recent[-1]}". '
        "Begin this one another way, without that word. Do not mention this note.)"
    )


def _lead_word(reply: str) -> str:
    found = _LEAD.match(strip_cue_tags(reply))
    if found is None or found.group(1).casefold() in _SENTENCE_STARTS:
        return ""
    return found.group(1)


def _word_key(word: str) -> str:
    """Fold case and stretched letters, so drawn-out spellings of a word count as one."""
    return _STRETCHED.sub(r"\1", word.casefold())


def _with_notes(history: list[ChatMessage], notes: list[str]) -> list[ChatMessage]:
    """Add notes to the last user line, in brackets ``scrub_tts`` removes if echoed.

    The notes are for this request only. If the model echoes one anyway, it is not spoken.
    """
    if not notes or history[-1]["role"] != "user":
        return history
    last = history[-1]
    content = "\n\n".join([last["content"], *notes])
    return [*history[:-1], {"role": "user", "content": content}]


def _turn_json(turn: Turn) -> dict[str, object]:
    reply = turn.reply
    return {
        "id": turn.id,
        "user": turn.user,
        "reply": None
        if reply is None
        else {"text": reply.text, "interrupted": reply.interrupted, "edit": reply.edit},
        "hidden": turn.hidden,
    }


def _turn(item: object) -> Turn:
    data = _object(item, "turn")
    reply = data.get("reply")
    return Turn(
        _str(data, "id"),
        _str(data, "user"),
        None if reply is None else _reply(reply),
        _bool(data, "hidden"),
    )


def _reply(item: object) -> Reply:
    data = _object(item, "reply")
    edit = data.get("edit")
    if edit is not None and not isinstance(edit, str):
        raise TypeError(f"reply edit is {type(edit).__name__}, not a string or null")
    if edit is not None and not edit.strip():
        raise ValueError("a reply edit is blank: remove it, or hide or delete the turn")
    return Reply(_str(data, "text"), _bool(data, "interrupted"), edit)


def _object(item: object, what: str) -> Mapping[str, object]:
    if not isinstance(item, Mapping):
        raise TypeError(f"a {what} is {type(item).__name__}, not an object")
    return {str(key): value for key, value in item.items()}


def _list(data: Mapping[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} is {type(value).__name__}, not a list")
    return list(value)


def _str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} is {type(value).__name__}, not a string")
    return value


def _bool(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{key} is {type(value).__name__}, not true or false")
    return value
