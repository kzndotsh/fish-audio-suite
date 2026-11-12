"""Chat history kept between turns.

It holds the pinned opening exchange, merges unanswered user lines, adds the tone
note for one request, and trims old turns in blocks so a cached prompt start
survives several turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Self

from fish_audio_suite_kit import DEFAULT_SEED_EXCHANGE, ChatMessage, is_same_utterance

__all__ = [
    "History",
    "with_tone_hint",
]

KEEP_SYSTEM: Final = 1
_ROLES_PER_TURN: Final = 2
# Unanswered user lines kept in one message. While the model keeps failing, the
# oldest drop off, so the merged message cannot grow without bound.
_UNANSWERED_LINES: Final = 4
# Over the cap, a quarter of the turns are dropped at once. See trim_history.
_TRIM_BLOCK: Final = 4


@dataclass(slots=True)
class History:
    """The chat so far, the messages trimming never drops, and the cap on turns.

    Attributes
    ----------
    messages : list of ChatMessage
        The system prompt, the pinned opening exchange if any, then the turns.
    turns : int
        User and assistant pairs kept (``FISH_VOICE_HISTORY_TURNS``).
    pinned : int
        How many leading messages trimming never drops.
    """

    messages: list[ChatMessage]
    turns: int
    pinned: int = KEEP_SYSTEM

    @classmethod
    def opening(cls, system_prompt: str, *, seed: bool, turns: int) -> Self:
        """Start a session's history.

        Parameters
        ----------
        system_prompt : str
            The system prompt for this session.
        seed : bool
            Pin one opening exchange with cues after the system message, because the model
            copies the pattern of the replies it sees. ``config.SystemPrompt.pin_seed`` decides
            it: the default prompt and a character file (which keeps the voice rules) get
            it, and a prompt written out in full does not, so it stays in control of how
            the model replies.
        turns : int
            User and assistant pairs to keep.

        Returns
        -------
        History
            The system message and the seed, all pinned.
        """
        messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
        if seed:
            for user, assistant in DEFAULT_SEED_EXCHANGE:
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
        return cls(messages, turns, len(messages))

    def copy(self) -> History:
        """Return a copy whose messages can change without touching this one.

        Returns
        -------
        History
            Each message is copied too: adding a line merges it into an unanswered user
            line in place.
        """
        return History([message.copy() for message in self.messages], self.turns, self.pinned)

    def last_user(self) -> str:
        """Return the last message when it is the user's, which the reply answers, else ``""``."""
        if self.messages and self.messages[-1]["role"] == "user":
            return self.messages[-1]["content"]
        return ""

    def remember_user(self, text: str) -> None:
        """Append a user line and trim to the cap.

        Parameters
        ----------
        text : str
            What the user said or typed.

        Notes
        -----
        A previous user line that got no reply (the model failed, or a barge-in cut it
        off before any audio) is still the last message. The new line joins it instead
        of following it, so roles keep alternating, and a repeat of it is kept once.
        At most the last four unanswered lines are kept.
        """
        messages = self.messages
        last = messages[-1] if len(messages) > self.pinned else None
        if last is not None and last["role"] == "user":
            if is_same_utterance(text, last["content"]):
                last["content"] = text
            else:
                lines = [*last["content"].split("\n"), text]
                last["content"] = "\n".join(lines[-_UNANSWERED_LINES:])
        else:
            messages.append({"role": "user", "content": text})
        trim_history(messages, self.turns, self.pinned)

    def remember_assistant(self, spoken: str, *, resumes: bool = False) -> None:
        """Record what the user heard of a reply.

        Parameters
        ----------
        spoken : str
            The part of the reply that was heard. Empty records nothing.
        resumes : bool, optional
            The rest of a reply a false barge-in cut off: it joins the last assistant
            message instead of starting a new one.

        Notes
        -----
        A reply that only repeats the line it answers is not recorded: saved, it would teach
        the next turn to repeat the user again. Nothing is trimmed here; the next user line
        trims.
        """
        if not spoken or is_same_utterance(spoken, self.last_user()):
            return
        messages = self.messages
        if resumes and messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] = f"{messages[-1]['content']} {spoken}"
        else:
            messages.append({"role": "assistant", "content": spoken})


def with_tone_hint(history: list[ChatMessage], tone: tuple[str, ...]) -> list[ChatMessage]:
    """Return the history for one request, with a note on how the last user line sounded.

    Parameters
    ----------
    history : list of ChatMessage
        The session history. It is not changed.
    tone : tuple of str
        What Fish ASR heard in the voice, such as ``("laughter", "tired")``. Empty means no
        hint.

    Returns
    -------
    list of ChatMessage
        ``history`` itself when there is no tone or the last message is not the user's.
        Otherwise a copy whose last user message ends with a parenthesised note. The note is
        for this request only: the history keeps the plain line, so it is never sent again
        on a later turn.

    Notes
    -----
    The note is in round brackets, which ``scrub_tts`` removes, and it tells the model not
    to repeat it or turn it into a ``[cue]``. If the model echoes it anyway, it is not
    spoken.
    """
    if not tone or not history or history[-1]["role"] != "user":
        return history
    note = (
        f"(Note on how they sounded: {', '.join(tone)}. This is not something they said. "
        "Do not repeat or mention it, and never write it as a [cue] or in brackets.)"
    )
    last = history[-1]
    return [*history[:-1], {"role": "user", "content": f"{last['content']}\n\n{note}"}]


def trim_history(
    history: list[ChatMessage],
    turns: int,
    pinned: int = KEEP_SYSTEM,
) -> None:
    """Keep at most ``turns`` user and assistant pairs, dropping the oldest in blocks.

    Notes
    -----
    Over the cap, a quarter of ``turns`` goes at once, so the start of the prompt
    stays the same for the next several turns. A provider that caches prompts
    reuses that start. Dropping one pair per turn would change it every time. A
    cap under four turns drops only what is over the cap.
    """
    cap = pinned + turns * _ROLES_PER_TURN
    if len(history) <= cap:
        return
    target = cap - (turns // _TRIM_BLOCK) * _ROLES_PER_TURN
    while len(history) > target:
        # Drop the oldest user and assistant together. Popping one message
        # leaves that assistant answering the next user.
        paired = (
            len(history) > pinned + 1
            and history[pinned]["role"] == "user"
            and history[pinned + 1]["role"] == "assistant"
        )
        if paired:
            del history[pinned : pinned + _ROLES_PER_TURN]
            continue
        del history[pinned]
