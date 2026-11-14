"""What a streaming recogniser is, what it says about a turn, and the connection they share.

A streaming recogniser takes audio as it is spoken and decides for itself when the speaker
has finished. Each one is an adapter module (``deepgram``, ``fish_realtime``) that declares a
``Recogniser``: its settings name, how it is opened from the settings, and what it needs. It
reads its own protocol into the messages below, so ``streaming`` hears a turn the same way
through any of them.
"""

from __future__ import annotations

import contextlib
import importlib.util
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_voice.config.tune import SttTune

__all__ = [
    "EagerTurnEnd",
    "Recogniser",
    "RecogniserSettings",
    "SocketStream",
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


@dataclass(frozen=True, slots=True)
class RecogniserSettings:
    """The settings a streaming recogniser is opened with, read once at start-up.

    Attributes
    ----------
    stt : SttTune
        Which recogniser, its own settings, the early end of turn and where to save the audio.
    fish_key : str
        The Fish API key, for a recogniser on the Fish account.
    fish_base : str
        The Fish API base, for a recogniser on the Fish account.
    """

    stt: SttTune
    fish_key: str = field(default="", repr=False)
    fish_base: str = SuiteDefaults().fish_base


@dataclass(frozen=True, slots=True)
class Recogniser:
    """What a streaming recogniser declares about itself, so nothing else names it.

    Attributes
    ----------
    name : str
        Its ``FISH_VOICE_STT`` value, such as ``"deepgram"``.
    label : str
        What log lines call it, such as ``"Flux"``.
    extra : str
        The package extra that installs its client.
    key_name : str
        The environment variable its key comes from.
    key : Callable
        Reads its key from the settings. Empty when it is not set.
    url : Callable
        Builds the address to connect to from the settings.
    stream : Callable
        Opens a ``TurnStream`` given ``(url, key, settings)``. Nothing is sent until ``open``.
    min_silence_frames : int
        The least local silence, in 30 ms frames, before the app ends a turn itself. A
        recogniser that needs longer than the user's silence limit to end a turn on its own
        sets it, so its judgment is not cut short. 0 keeps the user's limit.
    """

    name: str
    label: str
    extra: str
    key_name: str
    key: Callable[[RecogniserSettings], str]
    url: Callable[[RecogniserSettings], str]
    stream: Callable[[str, str, RecogniserSettings], TurnStream]
    min_silence_frames: int = 0

    def problem(self, settings: RecogniserSettings) -> str | None:
        """Say why this recogniser cannot be used, before the session starts.

        Parameters
        ----------
        settings : RecogniserSettings
            The settings it would be opened with.

        Returns
        -------
        str or None
            What is missing, the key or the extra that installs ``websockets``, or None when
            nothing is.
        """
        if not self.key(settings):
            return f"{self.key_name} (FISH_VOICE_STT={self.name}) missing"
        if importlib.util.find_spec("websockets") is None:
            return (
                f"FISH_VOICE_STT={self.name} needs the {self.extra} extra: "
                f"pip install 'fish-audio-suite-voice[{self.extra}]' (or uv sync --extra {self.extra})"
            )
        return None


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


class SocketStream:
    """The part of a ``TurnStream`` every recogniser shares: the socket, read and closed.

    An adapter adds its own ``send_audio`` and ``force_end_turn``, and may extend ``open``.

    Parameters
    ----------
    url : str
        The WebSocket address.
    key : str
        The API key. Sent as a header, never in the address.
    connect : Callable
        Opens the connection, given ``(url, key=...)``.
    parse : Callable
        Reads one frame into a turn message, or None for one that needs no action.
    goodbye : str or bytes
        Sent before the socket is closed, to end the session.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        connect: Callable[..., Awaitable[Any]],
        parse: Callable[[str | bytes], TurnMessage | None],
        goodbye: str | bytes,
    ) -> None:
        self._url = url
        self._key = key
        self._connect = connect
        self._parse = parse
        self._goodbye = goodbye
        self._socket: Any = None

    async def open(self) -> None:
        """Connect. Raises ``SttStreamError`` when the recogniser cannot be reached or says no."""
        self._socket = await self._connect(self._url, key=self._key)

    async def messages(self) -> AsyncIterator[TurnMessage]:
        """Read messages until the connection closes.

        Yields
        ------
        TurnStarted or TurnUpdate or TurnEnded or EagerTurnEnd or TurnResumed or TurnDropped \
        or StreamError or StreamWarning
            What the recogniser says, in order. Messages that need no action are skipped.
        """
        if self._socket is None:
            return
        try:
            async for raw in self._socket:
                if (message := self._parse(raw)) is not None:
                    yield message
        except Exception as exc:
            if not is_closed(exc):
                raise

    async def close(self) -> None:
        """Say goodbye and close. Safe to call twice, or on a connection that never opened."""
        socket, self._socket = self._socket, None
        if socket is None:
            return
        with contextlib.suppress(Exception):  # it may be closed already
            await socket.send(self._goodbye)
        await socket.close()
