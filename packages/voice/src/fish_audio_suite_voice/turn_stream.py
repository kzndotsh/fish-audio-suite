"""What a streaming recogniser says about a turn, and the connection both of ours share.

Deepgram Flux (``deepgram``) and Fish's realtime speech to text (``fish_realtime``) each take
audio as it is spoken and decide for themselves when the speaker has finished. Each reads its
own protocol into these messages, so ``streaming`` hears a turn the same way through either.
"""

from __future__ import annotations

import ssl
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

__all__ = [
    "EagerTurnEnd",
    "StreamError",
    "StreamWarning",
    "SttStreamError",
    "TurnDropped",
    "TurnEnded",
    "TurnMessage",
    "TurnResumed",
    "TurnStarted",
    "TurnStream",
    "TurnUpdate",
    "is_closed",
    "websockets_connect",
]

_OPEN_TIMEOUT_S: Final = 10.0
_CLOSE_TIMEOUT_S: Final = 2.0
_AUTH_STATUSES: Final = frozenset({401, 402, 403})


class SttStreamError(Exception):
    """The recogniser could not be reached or refused the connection.

    Attributes
    ----------
    fatal : bool
        True when trying again cannot help: a missing, wrong or unpaid key.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


@dataclass(frozen=True, slots=True)
class TurnStarted:
    """The speaker began a turn.

    Attributes
    ----------
    text : str
        The words so far, which may be empty.
    """

    text: str


@dataclass(frozen=True, slots=True)
class TurnUpdate:
    """The words of the turn so far, sent about four times a second while it goes on.

    Attributes
    ----------
    text : str
        Everything heard so far in this turn.
    """

    text: str


@dataclass(frozen=True, slots=True)
class TurnEnded:
    """The recogniser judged that the speaker has finished.

    Attributes
    ----------
    text : str
        The whole turn.
    confidence : float
        How sure it is that the turn is over, from 0 to 1. A recogniser that gives no
        figure reports 1.
    trigger : str
        What ended it: ``"model"``, ``"manual"`` (asked for) or ``"timeout"``.
    """

    text: str
    confidence: float
    trigger: str


@dataclass(frozen=True, slots=True)
class EagerTurnEnd:
    """The recogniser thinks the speaker has probably finished, a little before it is sure.

    Only sent when the early end of turn is asked for. The speaker may still carry on
    (``TurnResumed``), so a reply started from this must be held back until
    ``TurnEnded`` confirms it.

    Attributes
    ----------
    text : str
        The words so far. A confirming ``TurnEnded`` carries the same text.
    """

    text: str


@dataclass(frozen=True, slots=True)
class TurnResumed:
    """The speaker carried on after an ``EagerTurnEnd``, so that guess was wrong."""


@dataclass(frozen=True, slots=True)
class StreamError:
    """The recogniser reported an error on an open connection.

    Attributes
    ----------
    code : str
        Its error code.
    description : str
        What it says went wrong.
    """

    code: str
    description: str


@dataclass(frozen=True, slots=True)
class StreamWarning:
    """The recogniser flagged something on an open connection and kept it open.

    Attributes
    ----------
    code : str
        Its code for the condition.
    description : str
        What it says.
    """

    code: str
    description: str


@dataclass(frozen=True, slots=True)
class TurnDropped:
    """The recogniser closed the turn without taking it for the user's: noise, another voice.

    Whatever words arrived for it are not to be answered.
    """


type TurnMessage = (
    TurnStarted
    | TurnUpdate
    | TurnEnded
    | EagerTurnEnd
    | TurnResumed
    | TurnDropped
    | StreamError
    | StreamWarning
)


class TurnStream(Protocol):
    """One connection to a recogniser that streams: audio goes up, turn messages come back."""

    async def open(self) -> None:
        """Connect. Raises ``SttStreamError`` when it cannot be reached or says no."""
        ...

    async def send_audio(self, pcm: bytes) -> None:
        """Send 16-bit mono samples."""
        ...

    async def force_end_turn(self) -> None:
        """Ask for the turn to be ended now."""
        ...

    def messages(self) -> AsyncIterator[TurnMessage]:
        """Read messages until the connection closes."""
        ...

    async def close(self) -> None:
        """Close. Safe to call twice, or on a connection that never opened."""
        ...


def is_closed(exc: Exception) -> bool:
    """Say whether ``exc`` is the connection closing, which is the end of the stream.

    Parameters
    ----------
    exc : Exception
        What reading the connection raised.

    Returns
    -------
    bool
        True for a closed connection. False for anything else, or without ``websockets``.
    """
    try:
        from websockets.exceptions import ConnectionClosed  # noqa: PLC0415 - optional extra
    except ImportError:
        return False
    return isinstance(exc, ConnectionClosed)


def _tls_context() -> ssl.SSLContext:
    """Trust the same certificate authorities as the HTTP client: certifi's, when installed.

    ``websockets`` uses the system store alone, which some installs (Nix, some Pythons from
    uv) do not have, so the handshake fails with ``CERTIFICATE_VERIFY_FAILED`` while the Fish
    and LLM calls, which use certifi, work.
    """
    try:
        import certifi  # noqa: PLC0415 - a dependency of httpx, so normally present
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


async def websockets_connect(
    url: str, headers: Mapping[str, str], *, service: str, extra: str
) -> Any:
    """Open a real connection. ``websockets`` is imported here, so only this needs the extra.

    Parameters
    ----------
    url : str
        The WebSocket address.
    headers : Mapping
        Request headers, the key among them. The key never goes in the address.
    service : str
        The recogniser's name, for error messages.
    extra : str
        The package extra that installs ``websockets`` for it.

    Returns
    -------
    Any
        The open ``websockets`` connection.

    Raises
    ------
    SttStreamError
        When ``websockets`` is missing or the key is refused (both fatal), or the service
        cannot be reached.
    """
    try:
        from websockets.asyncio.client import connect  # noqa: PLC0415 - optional extra
        from websockets.exceptions import InvalidStatus  # noqa: PLC0415
    except ImportError as exc:
        msg = f"{service} needs the {extra} extra: pip install 'fish-audio-suite-voice[{extra}]'"
        raise SttStreamError(msg, fatal=True) from exc
    try:
        return await connect(
            url,
            ssl=_tls_context(),
            additional_headers=dict(headers),
            open_timeout=_OPEN_TIMEOUT_S,
            close_timeout=_CLOSE_TIMEOUT_S,
            max_size=None,
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        raise SttStreamError(
            f"{service} refused the connection (HTTP {status})", fatal=status in _AUTH_STATUSES
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise SttStreamError(f"could not reach {service}: {exc}") from exc
