"""Deepgram Flux: speech recognition that streams, and decides for itself when a turn is over.

Flux takes raw audio as it is spoken and sends back the words so far, then a final
``EndOfTurn`` when it judges the speaker has finished. That replaces recording until a
silence and then sending a clip. This module is the protocol and the connection; it knows
nothing about the microphone or the session, so it is tested against a fake connection.

``websockets`` is imported when a connection is opened, so the voice package imports without
the ``deepgram`` extra.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlencode

__all__ = [
    "FLUX_URL",
    "DeepgramError",
    "FluxStream",
    "StreamError",
    "StreamWarning",
    "TurnEnded",
    "TurnStarted",
    "TurnUpdate",
    "flux_base",
    "flux_url",
    "parse_flux_message",
]

FLUX_URL: Final = "wss://api.deepgram.com/v2/listen"
# The regional hosts process the audio inside their region and fail rather than send it elsewhere.
_REGION_HOSTS: Final = {
    "eu": "api.eu.deepgram.com",
    "au": "api.au.deepgram.com",
    "in": "api.in.deepgram.com",
}
DEFAULT_EOT_TIMEOUT_MS: Final = 5000  # Flux ends a turn after this much silence at the latest
_OPEN_TIMEOUT_S: Final = 10.0
_CLOSE_TIMEOUT_S: Final = 2.0
_AUTH_STATUSES: Final = frozenset({401, 402, 403})


class DeepgramError(Exception):
    """Deepgram could not be reached or refused the connection.

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
    """Flux judged that the speaker has finished.

    Attributes
    ----------
    text : str
        The whole turn.
    confidence : float
        How sure Flux is that the turn is over, from 0 to 1.
    trigger : str
        What ended it: ``"model"``, ``"manual"`` (asked for) or ``"timeout"``.
    """

    text: str
    confidence: float
    trigger: str


@dataclass(frozen=True, slots=True)
class StreamError:
    """Deepgram reported an error on an open connection.

    Attributes
    ----------
    code : str
        Deepgram's error code.
    description : str
        What it says went wrong.
    """

    code: str
    description: str


@dataclass(frozen=True, slots=True)
class StreamWarning:
    """Deepgram flagged something on an open connection and kept it open.

    Attributes
    ----------
    code : str
        Deepgram's code for the condition, in ``SCREAMING_SNAKE_CASE``.
    description : str
        What it says.
    """

    code: str
    description: str


type FluxMessage = TurnStarted | TurnUpdate | TurnEnded | StreamError | StreamWarning


def flux_base(region: str) -> str:
    """Return the Flux endpoint for a region.

    Parameters
    ----------
    region : str
        ``"eu"``, ``"au"`` or ``"in"``. Anything else, such as ``"global"``, is the default.

    Returns
    -------
    str
        The WebSocket address, without a query.

    Examples
    --------
    >>> flux_base("eu")
    'wss://api.eu.deepgram.com/v2/listen'
    >>> flux_base("global") == FLUX_URL
    True
    """
    host = _REGION_HOSTS.get(region)
    return FLUX_URL if host is None else f"wss://{host}/v2/listen"


def flux_url(
    model: str,
    *,
    sample_rate: int,
    eot_threshold: float,
    eot_timeout_ms: int = DEFAULT_EOT_TIMEOUT_MS,
    base: str = FLUX_URL,
) -> str:
    """Build the Flux connection address for 16-bit mono PCM.

    Parameters
    ----------
    model : str
        ``flux-general-en`` or ``flux-general-multi``.
    sample_rate : int
        Samples a second of the audio that will be sent.
    eot_threshold : float
        How sure Flux must be that a turn is over, from 0.5 to 1.
    eot_timeout_ms : int, optional
        The longest silence before a turn is ended anyway.
    base : str, optional
        The endpoint, to use a regional one.

    Returns
    -------
    str
        The address, with ``mip_opt_out=true`` so Deepgram keeps neither the audio nor the text.

    Examples
    --------
    >>> flux_url("flux-general-en", sample_rate=16000, eot_threshold=0.7)  # doctest: +ELLIPSIS
    'wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000&...'
    """
    query = urlencode(
        {
            "model": model,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "eot_threshold": eot_threshold,
            "eot_timeout_ms": eot_timeout_ms,
            "mip_opt_out": "true",
        }
    )
    return f"{base}?{query}"


def _number(value: object) -> float:
    """Read a number that Deepgram's schema calls a string but that may arrive as either."""
    try:
        return float(value)  # type: ignore[arg-type]  # a bad value is the except below
    except (TypeError, ValueError):
        return 0.0


def parse_flux_message(raw: str | bytes) -> FluxMessage | None:
    """Read one message from Flux.

    Parameters
    ----------
    raw : str or bytes
        A text frame from the connection.

    Returns
    -------
    TurnStarted or TurnUpdate or TurnEnded or StreamError or StreamWarning or None
        What the message says, or None for one that needs no action: the greeting, an
        acknowledgement, an early end-of-turn guess, malformed JSON or anything new.

    Examples
    --------
    >>> parse_flux_message(
    ...     '{"type":"TurnInfo","event":"EndOfTurn","transcript":"hi","end_of_turn_confidence":0.9,"trigger":"model"}'
    ... )
    TurnEnded(text='hi', confidence=0.9, trigger='model')
    >>> parse_flux_message('{"type":"Connected"}') is None
    True
    """
    try:
        message = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(message, dict):
        return None
    kind = message.get("type")
    if kind == "Error":
        return StreamError(str(message.get("code", "")), str(message.get("description", "")))
    if kind == "Warning":
        return StreamWarning(str(message.get("code", "")), str(message.get("description", "")))
    if kind != "TurnInfo":
        return None
    text = str(message.get("transcript") or "")
    match message.get("event"):
        case "StartOfTurn":
            return TurnStarted(text)
        case "Update":
            return TurnUpdate(text)
        case "EndOfTurn":
            return TurnEnded(
                text,
                _number(message.get("end_of_turn_confidence")),
                str(message.get("trigger") or "model"),
            )
        case _:
            return None  # EagerEndOfTurn and TurnResumed are not used yet


def _is_closed(exc: Exception) -> bool:
    """Say whether ``exc`` is the connection closing, which is the end of the stream."""
    try:
        from websockets.exceptions import ConnectionClosed  # noqa: PLC0415 - optional extra
    except ImportError:
        return False
    return isinstance(exc, ConnectionClosed)


async def _websockets_connect(url: str, *, key: str) -> Any:
    """Open a real connection. Imported here so the extra is only needed when this runs."""
    try:
        from websockets.asyncio.client import connect  # noqa: PLC0415 - optional extra
        from websockets.exceptions import InvalidStatus  # noqa: PLC0415
    except ImportError as exc:
        msg = "Deepgram needs the deepgram extra: pip install 'fish-audio-suite-voice[deepgram]'"
        raise DeepgramError(msg, fatal=True) from exc
    try:
        return await connect(
            url,
            additional_headers={"Authorization": f"Token {key}"},
            open_timeout=_OPEN_TIMEOUT_S,
            close_timeout=_CLOSE_TIMEOUT_S,
            max_size=None,
        )
    except InvalidStatus as exc:
        status = exc.response.status_code
        raise DeepgramError(
            f"Deepgram refused the connection (HTTP {status})", fatal=status in _AUTH_STATUSES
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise DeepgramError(f"could not reach Deepgram: {exc}") from exc


class FluxStream:
    """One Flux connection: send audio up, read turn messages back.

    Parameters
    ----------
    url : str
        From ``flux_url``.
    key : str
        The Deepgram API key. Sent as a header, never in the address.
    connect : Callable, optional
        Opens the connection, given ``(url, key=...)``. A fake in tests.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        connect: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._connect = connect or _websockets_connect
        self._socket: Any = None

    @property
    def connected(self) -> bool:
        """Whether there is a connection that has not closed."""
        return self._socket is not None and getattr(self._socket, "close_code", None) is None

    async def open(self) -> None:
        """Connect. Raises ``DeepgramError`` when Deepgram cannot be reached or says no."""
        self._socket = await self._connect(self._url, key=self._key)

    async def send_audio(self, pcm: bytes) -> None:
        """Send audio.

        Parameters
        ----------
        pcm : bytes
            16-bit mono samples. Empty bytes are skipped, because Deepgram closes the
            connection on them.
        """
        if pcm and self._socket is not None:
            await self._socket.send(pcm)

    async def force_end_turn(self) -> None:
        """Ask Flux to end the turn now, for when the caller decides the speaker has finished."""
        if self._socket is not None:
            await self._socket.send(json.dumps({"type": "ForceEndTurn"}))

    async def messages(self) -> AsyncIterator[FluxMessage]:
        """Read messages until the connection closes.

        Yields
        ------
        TurnStarted or TurnUpdate or TurnEnded or StreamError or StreamWarning
            What Flux says, in order. Messages that need no action are skipped.
        """
        if self._socket is None:
            return
        try:
            async for raw in self._socket:
                if (message := parse_flux_message(raw)) is not None:
                    yield message
        except Exception as exc:
            if not _is_closed(exc):
                raise

    async def close(self) -> None:
        """Say goodbye and close. Safe to call twice, or on a connection that never opened."""
        socket, self._socket = self._socket, None
        if socket is None:
            return
        with contextlib.suppress(Exception):  # it may be closed already
            await socket.send(json.dumps({"type": "CloseStream"}))
        await socket.close()
