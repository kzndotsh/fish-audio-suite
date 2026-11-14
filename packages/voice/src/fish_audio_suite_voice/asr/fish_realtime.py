"""Fish Audio's realtime speech to text: it streams, and decides when a turn is over.

``transcribe-stream-1`` (in preview) takes raw audio as it is spoken and sends back drafts of
the words so far, then one ``committed`` message when it judges the speaker has finished. It
is the same shape as Deepgram Flux, on the Fish key. This module is the protocol and the
connection; it knows nothing about the microphone or the session, so it is tested against a
fake connection.

Every message in both directions is one binary frame holding one MessagePack map. Fish
rejects text frames and JSON. ``websockets`` is imported when a connection is opened, so the
voice package imports without the ``fish-realtime`` extra.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Final

import ormsgpack

from fish_audio_suite_kit import MS_PER_S
from fish_audio_suite_voice.asr.turn_stream import (
    EagerTurnEnd,
    Recogniser,
    RecogniserSettings,
    SocketStream,
    StreamError,
    StreamWarning,
    TurnDropped,
    TurnEnded,
    TurnMessage,
    TurnResumed,
    TurnStarted,
    TurnUpdate,
    websockets_connect,
)
from fish_audio_suite_voice.capture.pcm import FRAME_MS, SAMPLE_RATE

__all__ = [
    "FISH_REALTIME",
    "FishRealtimeStream",
    "parse_realtime_message",
    "realtime_config",
    "realtime_url",
]

_EXTRA: Final = "fish-realtime"
_MODEL: Final = "transcribe-stream-1"
_PATH: Final = "/v1/asr/realtime"
# Fish ends a turn about 1.3 s after the speech does, and about 2 s after a speaker who stopped
# mid-sentence. The local silence limit has to outlast both, or it ends every turn first and
# Fish's own judgment is never used. Measured on the live API, 2026-11-11.
_SILENCE_FLOOR_S: Final = 3
# Fish's ``reason`` on a committed turn, as the trigger names the session logs.
_TRIGGERS: Final = {"finalize": "manual", "audio_done": "manual", "silence_timeout": "timeout"}


def realtime_url(fish_base: str) -> str:
    """Return the realtime socket's address for a Fish API base.

    Parameters
    ----------
    fish_base : str
        The REST base, such as ``https://api.fish.audio``.

    Returns
    -------
    str
        The WebSocket address: ``wss`` for an ``https`` base, ``ws`` for ``http``.

    Examples
    --------
    >>> realtime_url("https://api.fish.audio/")
    'wss://api.fish.audio/v1/asr/realtime'
    >>> realtime_url("http://127.0.0.1:8080")
    'ws://127.0.0.1:8080/v1/asr/realtime'
    """
    scheme, _, rest = fish_base.strip().rstrip("/").partition("://")
    return f"{'ws' if scheme == 'http' else 'wss'}://{rest}{_PATH}"


def realtime_config(*, sample_rate: int, eager: bool = False) -> dict[str, object]:
    """Build the first message of a session, for 16-bit mono PCM.

    Parameters
    ----------
    sample_rate : int
        Samples a second of the audio that will be sent.
    eager : bool, optional
        Ask Fish to say when a turn is probably over, before it is sure.

    Returns
    -------
    dict
        The ``config`` message. Fish ends the turn itself (``auto``), sends drafts, and leaves
        out speaker labels and sound tags, which one person talking to the app does not need.
        Fish closes the session on any field it does not know, so nothing else is sent.
    """
    return {
        "type": "config",
        "audio": {"format": "pcm_s16le", "sample_rate": sample_rate, "channels": 1},
        "turn_detection": {"mode": "auto", "eager": eager},
        "results": {"drafts": True, "diarize": False, "tag_audio_events": False},
    }


def _committed(message: dict[str, Any]) -> TurnMessage:
    # A missing user_turn means the user's. Noise, another voice and a listener's "mm-hmm"
    # are committed too, marked as not the user's turn.
    if message.get("user_turn") is False:
        return TurnDropped()
    reason = str(message.get("reason") or "eot")
    return TurnEnded(str(message.get("text") or ""), 1.0, _TRIGGERS.get(reason, "model"))


def parse_realtime_message(raw: str | bytes) -> TurnMessage | None:
    """Read one message from Fish's realtime socket.

    Parameters
    ----------
    raw : str or bytes
        A frame from the connection. Fish sends binary MessagePack only.

    Returns
    -------
    TurnStarted or TurnUpdate or TurnEnded or EagerTurnEnd or TurnResumed or TurnDropped \
    or StreamError or StreamWarning or None
        What the message says, or None for one that needs no action: the greeting, progress,
        a text frame, malformed MessagePack or anything new.

    Examples
    --------
    >>> import ormsgpack
    >>> parse_realtime_message(ormsgpack.packb({"type": "committed", "text": "hi", "reason": "eot"}))
    TurnEnded(text='hi', confidence=1.0, trigger='model')
    >>> parse_realtime_message(ormsgpack.packb({"type": "session_started"})) is None
    True
    """
    if isinstance(raw, str):
        return None
    try:
        message = ormsgpack.unpackb(raw)
    except ormsgpack.MsgpackDecodeError:
        return None
    if not isinstance(message, dict):
        return None
    text = str(message.get("text") or "")
    match message.get("type"):
        case "speech_start":
            return TurnStarted("")
        case "draft":
            return TurnUpdate(text)
        case "eager_end_of_turn":
            return EagerTurnEnd(text)
        case "turn_resumed":
            return TurnResumed()
        case "committed":
            return _committed(message)
        case "session_done":
            # Reached only when no turn was committed first: there was nothing to end.
            return TurnDropped()
        case "error":
            kind = str(message.get("error_type") or message.get("code") or "")
            said = str(message.get("message") or "")
            if message.get("recoverable"):
                return StreamWarning(kind, said)
            return StreamError(kind, said)
        case _:
            return None


async def _websockets_connect(url: str, *, key: str) -> Any:
    """Open a real connection to Fish, with the key and the model as headers."""
    return await websockets_connect(
        url,
        {"Authorization": f"Bearer {key}", "model": _MODEL},
        service="Fish realtime speech to text",
        extra=_EXTRA,
    )


class FishRealtimeStream(SocketStream):
    """One realtime session: send audio up, read turn messages back.

    Parameters
    ----------
    url : str
        From ``realtime_url``.
    key : str
        The Fish API key. Sent as a header, never in the address.
    sample_rate : int
        Samples a second of the audio that will be sent.
    eager : bool, optional
        Ask for the early end of turn.
    connect : Callable, optional
        Opens the connection, given ``(url, key=...)``. A fake in tests.

    Notes
    -----
    Closing ends the session and its billing.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        sample_rate: int,
        eager: bool = False,
        connect: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        super().__init__(
            url,
            key,
            connect=connect or _websockets_connect,
            parse=parse_realtime_message,
            goodbye=ormsgpack.packb({"type": "audio_done"}),
        )
        self._config = realtime_config(sample_rate=sample_rate, eager=eager)

    async def _send(self, message: dict[str, object]) -> None:
        if self._socket is not None:
            await self._socket.send(ormsgpack.packb(message))

    async def open(self) -> None:
        """Connect and send the settings. Raises ``SttStreamError`` when Fish says no.

        Audio may follow at once: Fish keeps what arrives before it confirms the settings.
        """
        await super().open()
        await self._send(self._config)

    async def send_audio(self, pcm: bytes) -> None:
        """Send audio.

        Parameters
        ----------
        pcm : bytes
            16-bit mono samples. Empty bytes are skipped.
        """
        if pcm:
            await self._send({"type": "audio", "data": pcm})

    async def force_end_turn(self) -> None:
        """Say the audio is over, so Fish commits the open turn, or reports that there is none.

        ``finalize`` would be ignored with no turn open, and the caller would wait for an
        answer that never comes. A session carries one turn here, so ending it loses nothing.
        """
        await self._send({"type": "audio_done"})


def _stream(url: str, key: str, settings: RecogniserSettings) -> FishRealtimeStream:
    return FishRealtimeStream(url, key, sample_rate=SAMPLE_RATE, eager=settings.stt.eager)


FISH_REALTIME: Final = Recogniser(
    name="fish-realtime",
    label="Fish",
    extra=_EXTRA,
    key_name="FISH_API_KEY",
    key=lambda settings: settings.fish_key,
    url=lambda settings: realtime_url(settings.fish_base),
    stream=_stream,
    min_silence_frames=_SILENCE_FLOOR_S * MS_PER_S // FRAME_MS,
)
