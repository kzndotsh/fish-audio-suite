"""Fish Audio's realtime speech to text: it streams, and decides when a turn is over.

``transcribe-stream-1`` (in preview) takes raw audio as it is spoken and sends back drafts of
the words so far, then one ``committed`` message when it judges the speaker has finished. It
is the same shape as Deepgram Flux, on the Fish key. This module is the protocol and the
connection; it knows nothing about the microphone or the session, so it is tested against a
fake connection.

Every message in both directions is one binary frame holding one MessagePack map. Fish
rejects text frames and JSON. ``websockets`` is imported when a connection is opened, so the
voice package imports without the ``fish-realtime`` extra.

A session keeps one connection across turns (``FishRealtimeSession``): the connection is
opened when the first turn's speech starts, told the conversation so far, and lent to each
turn after it. Between turns only keepalives and new lines of the conversation are sent, never
audio, and it closes after ``SttTune.idle_s`` without a turn, since an open connection is
billed. A turn that ends any way but with Fish committing it closes the connection, so nothing
Fish says about it can reach the next turn; the next turn opens a new one, told the
conversation again.

With ``FISH_VOICE_BARGE_JUDGE=fish`` the same connection also hears the mic while a reply
plays (``FishRealtimeSession.judge``): Fish is told the reply is playing (``agent_speaking``),
and its ``user_backchannel`` and ``user_turn_taken`` messages say whether speech over the reply
was an acknowledgement or an interruption.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from typing import Any, Final

import ormsgpack

from fish_audio_suite_kit import MS_PER_S
from fish_audio_suite_voice.asr.turn_stream import (
    BargeJudge,
    BargeVerdict,
    ContextLine,
    ContextSource,
    EagerTurnEnd,
    Recogniser,
    RecogniserSettings,
    SocketStream,
    StreamError,
    StreamWarning,
    SttStreamError,
    TurnDropped,
    TurnEnded,
    TurnMessage,
    TurnResumed,
    TurnStarted,
    TurnStream,
    TurnUpdate,
    is_closed,
    websockets_connect,
)
from fish_audio_suite_voice.capture.pcm import FRAME_MS, SAMPLE_RATE
from fish_audio_suite_voice.logutil.debug import debug, trace, warn

__all__ = [
    "FISH_REALTIME",
    "FishRealtimeSession",
    "FishRealtimeStream",
    "parse_realtime_message",
    "realtime_config",
    "realtime_url",
    "unsaid_lines",
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
# Fish's limits on what it is told of the conversation: lines at once (in the settings or one
# ``context_add``), and characters a line.
_CONTEXT_LINES: Final = 32
_CONTEXT_CHARS: Final = 2000
# Fish closes a session after 30 s without a message from the client.
_KEEPALIVE_S: Final = 10.0
# Fish ends a session at 3600 s. One older than this is replaced before a turn, not during it.
_MAX_AGE_S: Final = 3300.0
# Fish's messages about speech over a reply, as what they mean.
_VERDICTS: Final[dict[str, BargeVerdict]] = {
    "user_turn_taken": "turn",
    "user_backchannel": "backchannel",
}


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


def realtime_config(
    *,
    sample_rate: int,
    eager: bool = False,
    history: Sequence[ContextLine] = (),
    terms: Sequence[str] = (),
) -> dict[str, object]:
    """Build the first message of a session, for 16-bit mono PCM.

    Parameters
    ----------
    sample_rate : int
        Samples a second of the audio that will be sent.
    eager : bool, optional
        Ask Fish to say when a turn is probably over, before it is sure.
    history : Sequence of ContextLine, optional
        The conversation so far. Only the last 32 lines go, each cut to its last 2,000
        characters, as Fish allows.
    terms : Sequence of str, optional
        Words to listen out for, already held to Fish's limits by ``SttTune``.

    Returns
    -------
    dict
        The ``config`` message. Fish ends the turn itself (``auto``), sends drafts, and leaves
        out speaker labels and sound tags, which one person talking to the app does not need.
        ``context`` is there only when there is history or terms. Fish closes the session on
        any field it does not know, so nothing else is sent.
    """
    config: dict[str, object] = {
        "type": "config",
        "audio": {"format": "pcm_s16le", "sample_rate": sample_rate, "channels": 1},
        "turn_detection": {"mode": "auto", "eager": eager},
        "results": {"drafts": True, "diarize": False, "tag_audio_events": False},
    }
    context: dict[str, object] = {}
    if history:
        context["history"] = _context_items(history[-_CONTEXT_LINES:])
    if terms:
        context["terms"] = list(terms)
    if context:
        config["context"] = context
    return config


def _context_items(lines: Iterable[ContextLine]) -> list[dict[str, str]]:
    """Write lines as Fish's context items, each cut to the last characters Fish takes."""
    return [{"role": line.role, "text": line.text[-_CONTEXT_CHARS:]} for line in lines]


def unsaid_lines(
    told: Sequence[ContextLine], wanted: Sequence[ContextLine]
) -> list[ContextLine] | None:
    """Say which lines of the conversation Fish has not been told yet.

    Parameters
    ----------
    told : Sequence of ContextLine
        What Fish knows: the history it was opened with, the lines added since, and each
        user turn it committed, which it adds itself.
    wanted : Sequence of ContextLine
        The conversation as it stands.

    Returns
    -------
    list of ContextLine or None
        The lines at the end of ``wanted`` that Fish lacks. A user line Fish heard that the
        conversation does not keep (an "mm-hmm" it skipped) is passed over. None when what Fish
        knows no longer leads up to the conversation, after an edit, a delete or a new chat,
        so it has to be told again from the start.

    Examples
    --------
    >>> user, bot = ContextLine("user", "hi"), ContextLine("assistant", "hello")
    >>> unsaid_lines([user], [user, bot])
    [ContextLine(role='assistant', text='hello')]
    >>> unsaid_lines([bot], [user]) is None
    True
    """
    at = 0
    for line in told:
        if at < len(wanted) and wanted[at] == line:
            at += 1
        elif line.role != "user":
            return None
    return list(wanted[at:])


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
    message = _unpack(raw)
    return None if message is None else _turn_message(message)


def _unpack(raw: str | bytes) -> dict[str, Any] | None:
    """Read one frame into its map, or None for a text frame or one that is not a map."""
    if isinstance(raw, str):
        return None
    try:
        message = ormsgpack.unpackb(raw)
    except ormsgpack.MsgpackDecodeError:
        return None
    return message if isinstance(message, dict) else None


def _turn_message(message: dict[str, Any]) -> TurnMessage | None:
    """Read one of Fish's messages into what it means for the turn."""
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


class FishRealtimeSession:
    """One realtime connection kept open across turns, and told the conversation.

    Parameters
    ----------
    url : str
        From ``realtime_url``.
    key : str
        The Fish API key. Sent as a header, never in the address.
    context : ContextSource
        Returns the conversation so far. Read when the connection opens, and before each
        turn after that, to tell Fish the lines it has not heard.
    sample_rate : int
        Samples a second of the audio that will be sent.
    eager : bool, optional
        Ask for the early end of turn.
    terms : Sequence of str, optional
        Words to listen out for.
    idle_s : int, optional
        Seconds without a turn after which the connection is closed. 0 closes it when each
        turn ends.
    connect : Callable, optional
        Opens the connection, given ``(url, key=...)``. A fake in tests.
    clock : Callable, optional
        Seconds from some fixed point. A fake in tests.

    Notes
    -----
    Everything runs on the event loop that opens the first turn. A reader task drains the
    connection the whole time it is open, so its keepalive answers are never held up, and
    hands each message to the turn that is open, or drops it when none is.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        context: ContextSource,
        sample_rate: int,
        eager: bool = False,
        terms: Sequence[str] = (),
        idle_s: int = 0,
        connect: Callable[..., Awaitable[Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._key = key
        self._context = context
        self._sample_rate = sample_rate
        self._eager = eager
        self._terms = tuple(terms)
        self._idle_s = idle_s
        self._connect_fn = connect or _websockets_connect
        self._clock = clock
        self._socket: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._told: list[ContextLine] = []  # what Fish knows of the conversation
        self._inbox: asyncio.Queue[TurnMessage | None] | None = None  # the open turn's
        self._busy = False  # a turn is open, or opening
        self._opened_at = 0.0
        self._last_turn = 0.0
        self._seen = -1  # the highest turn index Fish has named
        self._committed = -1  # the highest turn index Fish has committed
        self._floor = 0  # the lowest turn index the open turn takes
        self._judging = False  # a reply is playing and Fish has been told
        self._verdict_lock = threading.Lock()  # the barge-in thread takes the verdicts
        self._verdicts: list[tuple[float, BargeVerdict]] = []

    def turn(self) -> TurnStream:
        """Return the next turn's stream.

        Returns
        -------
        TurnStream
            ``open`` connects, or reuses the connection and tells Fish the new lines. ``close``
            keeps the connection for the next turn when Fish committed this one, and closes it
            otherwise.
        """
        return _SessionTurn(self)

    def judge(self) -> BargeJudge:
        """Return a judge for one reply, on this connection.

        Returns
        -------
        BargeJudge
            Bound to the running event loop, which every other use of the session is on.
        """
        return _ReplyJudge(self, asyncio.get_running_loop())

    async def close(self) -> None:
        """End the session and its billing. Safe to call twice, or when it never opened."""
        await self._drop("the session ended")

    async def _ready(self) -> None:
        """Have a connection told the conversation, reusing the open one when it can be.

        Marks the session busy, so it is not closed as idle, and leaves it so on success.
        """
        self._busy = True
        try:
            if self._socket is not None:
                await self._reuse()
            if self._socket is None:
                await self._open()
        except BaseException:
            self._busy = False
            await self._drop("it could not be opened")
            raise

    @property
    def turn_heard(self) -> bool:
        """Whether Fish has opened a turn since the open turn began."""
        return self._seen >= self._floor

    async def begin_turn(self) -> asyncio.Queue[TurnMessage | None]:
        """Have a connection ready for a turn, and return the queue its messages go to.

        Returns
        -------
        asyncio.Queue
            Gets each of Fish's messages about the turn, then None when the connection closes.

        Raises
        ------
        SttStreamError
            When Fish cannot be reached or refuses the connection.
        """
        await self._ready()
        # Anything Fish still says about a turn before this one is not this turn's.
        self._floor = self._seen + 1
        self._inbox = asyncio.Queue()
        return self._inbox

    async def _reuse(self) -> None:
        """Tell an open connection the new lines, or close it when it cannot be reused."""
        if self._clock() - self._opened_at >= _MAX_AGE_S:
            await self._drop("it is near Fish's hour limit")
            return
        if self._seen > self._committed:
            # Audio sent after the last turn ended opened another, which nobody reads.
            await self._drop("Fish has a turn open that no one is reading")
            return
        unsaid = unsaid_lines(self._told, self._context())
        if unsaid is None:
            await self._drop("the conversation changed under it")
            return
        for start in range(0, len(unsaid), _CONTEXT_LINES):
            batch = unsaid[start : start + _CONTEXT_LINES]
            await self.send({"type": "context_add", "items": _context_items(batch)})
        self._told += unsaid
        if unsaid:
            debug("stt.context told Fish {} new line(s)", len(unsaid))

    async def _open(self) -> None:
        """Connect, send the settings with the conversation so far, and start reading."""
        lines = list(self._context())
        await self._drop("a new one replaces it")  # the tasks of one Fish closed
        socket = await self._connect_fn(self._url, key=self._key)
        self._socket = socket
        self._told = lines
        self._opened_at = self._clock()
        self._seen = self._committed = -1
        await self.send(
            realtime_config(
                sample_rate=self._sample_rate, eager=self._eager, history=lines, terms=self._terms
            )
        )
        self._tasks = [
            asyncio.create_task(self._read(socket)),
            asyncio.create_task(self._keep_alive(socket)),
        ]
        debug("stt.session opened, told {} line(s) and {} term(s)", len(lines), len(self._terms))

    async def begin_reply(self) -> None:
        """Tell Fish a reply is playing, connecting first if needed.

        Raises
        ------
        SttStreamError
            When Fish cannot be reached or refuses the connection.
        """
        await self._ready()
        with self._verdict_lock:
            self._verdicts.clear()
        self._judging = True
        await self.send({"type": "agent_speaking", "speaking": True})

    async def end_reply(self) -> None:
        """Tell Fish the reply is over, and let the connection go idle."""
        if self._judging:
            await self.send({"type": "agent_speaking", "speaking": False})
        self._judging = False
        self._busy = False
        self._last_turn = self._clock()
        if self._idle_s == 0:
            await self._drop("FISH_VOICE_STT_IDLE is 0")

    def take_verdict(self, within_s: float) -> BargeVerdict | None:
        """Return and forget what Fish said about speech over the reply lately.

        Parameters
        ----------
        within_s : float
            How far back a verdict still counts, in seconds.

        Returns
        -------
        {"turn", "backchannel"} or None
            ``"turn"`` when Fish said the user took the turn in that time, even after an
            acknowledgement, else ``"backchannel"`` when it heard one, else None. Safe from any
            thread.
        """
        now = self._clock()
        with self._verdict_lock:
            recent: list[BargeVerdict] = [v for at, v in self._verdicts if now - at <= within_s]
            self._verdicts.clear()
        if "turn" in recent:
            return "turn"
        return recent[-1] if recent else None

    async def end_turn(self, *, committed: bool) -> None:
        """End the open turn, keeping the connection only when Fish committed it.

        Parameters
        ----------
        committed : bool
            Whether Fish committed the turn. When it did not, it may still say something about
            it, which must not reach the next turn, so the connection is closed.
        """
        self._busy = False
        self._inbox = None
        self._last_turn = self._clock()
        if not committed:
            await self._drop("the turn ended before Fish committed it")
        elif self._idle_s == 0:
            await self._drop("FISH_VOICE_STT_IDLE is 0")

    async def send(self, message: dict[str, object]) -> None:
        """Send one message, or nothing when the connection is closed.

        Parameters
        ----------
        message : dict
            Packed as MessagePack.
        """
        socket = self._socket
        if socket is None:
            return
        try:
            await socket.send(ormsgpack.packb(message))
        except Exception as exc:
            if not is_closed(exc):
                raise
            # The reader sees the same close and ends the turn.

    async def _read(self, socket: Any) -> None:
        try:
            async for raw in socket:
                message = _unpack(raw)
                if message is None:
                    continue
                self._note(message)
                turn_message = _turn_message(message)
                if turn_message is None:
                    continue
                index = message.get("turn_index")
                if self._inbox is None or (isinstance(index, int) and index < self._floor):
                    trace("stt.stale {} for no open turn", message.get("type"))
                    continue
                self._inbox.put_nowait(turn_message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a background task: the turn hears the end
            if not is_closed(exc):
                warn(f"[stt] reading Fish failed: {exc}")
        if self._socket is socket:
            # Fish closed it: the open turn hears the end, and the next turn opens a new one.
            debug("stt.session closed by Fish")
            self._socket = None
            with contextlib.suppress(Exception):
                await socket.close()
        if self._inbox is not None:
            self._inbox.put_nowait(None)

    def _note(self, message: dict[str, Any]) -> None:
        """Keep count of Fish's turns, and of the user turns it adds to what it knows."""
        index = message.get("turn_index")
        if isinstance(index, int):
            self._seen = max(self._seen, index)
        kind = message.get("type")
        if self._judging and (verdict := _VERDICTS.get(str(kind))) is not None:
            debug("stt.judge Fish heard {} over the reply", verdict)
            with self._verdict_lock:
                self._verdicts.append((self._clock(), verdict))
        if kind != "committed":
            return
        if isinstance(index, int):
            self._committed = max(self._committed, index)
        text = str(message.get("text") or "").strip()
        if message.get("user_turn") is not False and text:
            self._told.append(ContextLine("user", text))

    async def _keep_alive(self, socket: Any) -> None:
        while self._socket is socket:
            await asyncio.sleep(_KEEPALIVE_S)
            if self._socket is not socket:
                return
            # With idle_s 0, end_turn and end_reply close the connection, so it is never idle here.
            if not self._busy and self._clock() - self._last_turn >= self._idle_s:
                await self._drop(f"no turn for {self._idle_s} s")
                return
            await self.send({"type": "keepalive"})

    async def _drop(self, why: str) -> None:
        """Close the connection, if one is open, and stop its tasks. Never raises."""
        socket, self._socket = self._socket, None
        tasks, self._tasks = self._tasks, []
        current = asyncio.current_task()
        for task in tasks:
            if task is not current:
                task.cancel()
        await asyncio.gather(*(t for t in tasks if t is not current), return_exceptions=True)
        # The reader may have put its own None already; the turn reads up to the first, and
        # each turn gets a new queue, so a second one is never read.
        if self._inbox is not None:
            self._inbox.put_nowait(None)
        if socket is None:
            return
        debug("stt.session closed: {}", why)
        with contextlib.suppress(Exception):  # it may be closed already
            await socket.send(ormsgpack.packb({"type": "audio_done"}))
        with contextlib.suppress(Exception):
            await socket.close()


class _ReplyJudge:
    """A ``BargeJudge`` for one reply, on a ``FishRealtimeSession``.

    Mic frames from the barge-in thread are queued onto the event loop and sent in order by
    one task, so they reach Fish as they were heard.
    """

    def __init__(self, session: FishRealtimeSession, loop: asyncio.AbstractEventLoop) -> None:
        self._session = session
        self._loop = loop
        self._live = threading.Event()  # Fish knows the reply is playing
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Tell Fish the reply is playing. An outage leaves the gate to decide alone."""
        try:
            await self._session.begin_reply()
        except SttStreamError as exc:
            warn(f"[barge-in] Fish cannot judge this reply, the gate decides alone: {exc}")
            return
        self._pump = asyncio.create_task(self._send_frames())
        self._live.set()

    async def _send_frames(self) -> None:
        try:
            while (frame := await self._frames.get()) is not None:
                await self._session.send({"type": "audio", "data": frame})
        except Exception as exc:  # noqa: BLE001 - a background task: the gate decides on its own
            warn(f"[barge-in] sending the mic to Fish failed, the gate decides alone: {exc}")
            self._live.clear()

    def hear(self, frame: bytes) -> None:
        """Queue one mic frame for Fish. Dropped while Fish does not know the reply plays.

        Parameters
        ----------
        frame : bytes
            16-bit mono samples.
        """
        if self._live.is_set():
            with contextlib.suppress(RuntimeError):  # the loop closed as the session ended
                self._loop.call_soon_threadsafe(self._frames.put_nowait, frame)

    def take(self, within_s: float) -> BargeVerdict | None:
        """Return and forget Fish's verdict of the last ``within_s`` seconds.

        Parameters
        ----------
        within_s : float
            How far back a verdict still counts.

        Returns
        -------
        {"turn", "backchannel"} or None
            What Fish said, ``"turn"`` when it is not connected, or None when it has said
            nothing yet.
        """
        if not self._live.is_set():
            return "turn"
        return self._session.take_verdict(within_s)

    async def finish(self) -> None:
        """Stop sending the mic and tell Fish the reply is over."""
        self._live.clear()
        if self._pump is not None:
            self._frames.put_nowait(None)
            await self._pump
            self._pump = None
        await self._session.end_reply()


class _SessionTurn:
    """One turn on a ``FishRealtimeSession``: a ``TurnStream`` that borrows its connection."""

    def __init__(self, session: FishRealtimeSession) -> None:
        self._session = session
        self._inbox: asyncio.Queue[TurnMessage | None] | None = None
        self._committed = False
        self._closed = False

    async def open(self) -> None:
        """Have the connection ready. Raises ``SttStreamError`` when Fish says no."""
        self._inbox = await self._session.begin_turn()

    async def send_audio(self, pcm: bytes) -> None:
        """Send audio.

        Parameters
        ----------
        pcm : bytes
            16-bit mono samples. Empty bytes are skipped.
        """
        # Audio after Fish committed the turn would open another that no one reads.
        if pcm and self._inbox is not None and not self._closed and not self._committed:
            await self._session.send({"type": "audio", "data": pcm})

    async def force_end_turn(self) -> None:
        """Ask Fish to commit the open turn, or say there is none.

        Fish ignores ``finalize`` when it has no turn open, so a turn it never heard start
        is dropped here at once, instead of being waited on.
        """
        if self._inbox is None or self._closed:
            return
        if not self._session.turn_heard:
            self._inbox.put_nowait(TurnDropped())
            return
        await self._session.send({"type": "finalize"})

    async def messages(self) -> AsyncIterator[TurnMessage]:
        """Read this turn's messages until Fish commits it or the connection closes.

        Yields
        ------
        TurnStarted or TurnUpdate or TurnEnded or EagerTurnEnd or TurnResumed or TurnDropped \
        or StreamError or StreamWarning
            What Fish says about this turn, in order.
        """
        inbox = self._inbox
        if inbox is None:
            return
        while (message := await inbox.get()) is not None:
            # A TurnDropped is Fish committing the turn as not the user's (noise, another
            # voice), or none having started: either way Fish has no turn of ours left open.
            if isinstance(message, TurnEnded | TurnDropped):
                self._committed = True
            yield message
            if self._committed:
                return

    async def close(self) -> None:
        """End the turn. Safe to call twice, or on a turn that never opened."""
        if self._closed or self._inbox is None:
            self._closed = True
            return
        self._closed = True
        await self._session.end_turn(committed=self._committed)


def _stream(url: str, key: str, settings: RecogniserSettings) -> FishRealtimeStream:
    return FishRealtimeStream(url, key, sample_rate=SAMPLE_RATE, eager=settings.stt.eager)


def _shared(
    url: str, key: str, settings: RecogniserSettings, context: ContextSource
) -> FishRealtimeSession:
    return FishRealtimeSession(
        url,
        key,
        context=context,
        sample_rate=SAMPLE_RATE,
        eager=settings.stt.eager,
        terms=settings.stt.terms,
        idle_s=settings.stt.idle_s,
    )


FISH_REALTIME: Final = Recogniser(
    name="fish-realtime",
    label="Fish",
    extra=_EXTRA,
    key_name="FISH_API_KEY",
    key=lambda settings: settings.fish_key,
    url=lambda settings: realtime_url(settings.fish_base),
    stream=_stream,
    min_silence_frames=_SILENCE_FLOOR_S * MS_PER_S // FRAME_MS,
    shared=_shared,
)
