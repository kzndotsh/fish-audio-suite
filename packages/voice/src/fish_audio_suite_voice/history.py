"""Chat history kept between turns: the opening exchange and trimming by pairs."""

from __future__ import annotations

from typing import Final

from fish_audio_suite_kit import DEFAULT_SEED_EXCHANGE, DEFAULT_SYSTEM_PROMPT, ChatMessage

__all__ = [
    "KEEP_SYSTEM",
    "opening_history",
    "remember_user",
    "trim_history",
]

KEEP_SYSTEM: Final = 1
_ROLES_PER_TURN: Final = 2


def opening_history(system_prompt: str) -> tuple[list[ChatMessage], int]:
    """Build the starting history and say how many leading messages stay pinned.

    Parameters
    ----------
    system_prompt : str
        The system prompt for this session.

    Returns
    -------
    tuple of list and int
        The history and the count of messages that trimming never drops. With
        the default prompt, one opening exchange with several cues is pinned
        after the system message, because the model copies the pattern of the
        replies it sees. A custom prompt gets no seed, so it stays in control
        of how the model replies.
    """
    history: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    if system_prompt == DEFAULT_SYSTEM_PROMPT:
        for user, assistant in DEFAULT_SEED_EXCHANGE:
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": assistant})
    return history, len(history)


def trim_history(
    history: list[ChatMessage],
    turns: int,
    pinned: int = KEEP_SYSTEM,
) -> None:
    """Drop the oldest user and assistant pairs until the history fits ``turns``."""
    cap = pinned + turns * _ROLES_PER_TURN
    while len(history) > cap:
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


def remember_user(
    history: list[ChatMessage],
    text: str,
    turns: int,
    pinned: int = KEEP_SYSTEM,
) -> None:
    """Append a user line and trim the history to its cap."""
    history.append({"role": "user", "content": text})
    trim_history(history, turns, pinned)
