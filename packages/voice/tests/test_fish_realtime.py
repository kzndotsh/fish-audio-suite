"""Fish realtime speech to text: the protocol, and a connection driven by a fake socket."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import ormsgpack
import pytest

from fish_audio_suite_voice.asr.fish_realtime import (
    FishRealtimeStream,
    parse_realtime_message,
    realtime_config,
    realtime_url,
)
from fish_audio_suite_voice.asr.turn_stream import (
    EagerTurnEnd,
    StreamError,
    StreamWarning,
    SttStreamError,
    TurnDropped,
    TurnEnded,
    TurnResumed,
    TurnStarted,
    TurnUpdate,
)


def _pack(kind: str, **fields: Any) -> bytes:
    return ormsgpack.packb({"type": kind, "seq": 1, **fields})


class _FakeSocket:
    def __init__(self, incoming: list[bytes | str]) -> None:
        self.sent: list[Any] = []
        self.closed = False
        self._incoming = incoming

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    def __aiter__(self) -> AsyncIterator[bytes | str]:
        async def generate() -> AsyncIterator[bytes | str]:
            for frame in self._incoming:
                yield frame

        return generate()

    async def close(self) -> None:
        self.closed = True


def _stream(
    socket: _FakeSocket, *, eager: bool = False
) -> tuple[FishRealtimeStream, dict[str, str]]:
    seen: dict[str, str] = {}

    async def connect(url: str, *, key: str) -> _FakeSocket:
        seen.update(url=url, key=key)
        return socket

    stream = FishRealtimeStream(
        "wss://example/v1/asr/realtime", "secret", sample_rate=16000, eager=eager, connect=connect
    )
    return stream, seen


def test_the_address_follows_the_fish_base_and_its_scheme() -> None:
    assert realtime_url("https://api.fish.audio") == "wss://api.fish.audio/v1/asr/realtime"
    assert realtime_url(" https://api.fish.audio/ ") == "wss://api.fish.audio/v1/asr/realtime"
    assert realtime_url("http://127.0.0.1:8080") == "ws://127.0.0.1:8080/v1/asr/realtime"


def test_the_settings_ask_fish_to_end_turns_and_send_drafts_of_plain_text() -> None:
    config = realtime_config(sample_rate=16000)
    assert config == {
        "type": "config",
        "audio": {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        "turn_detection": {"mode": "auto", "eager": False},
        "results": {"drafts": True, "diarize": False, "tag_audio_events": False},
    }
    eager = realtime_config(sample_rate=16000, eager=True)
    assert eager["turn_detection"] == {"mode": "auto", "eager": True}


def test_each_turn_message_is_read_into_what_it_means() -> None:
    assert parse_realtime_message(_pack("speech_start", turn_index=0)) == TurnStarted("")
    assert parse_realtime_message(_pack("draft", text="hi the")) == TurnUpdate("hi the")
    assert parse_realtime_message(_pack("eager_end_of_turn", text="hi")) == EagerTurnEnd("hi")
    assert parse_realtime_message(_pack("turn_resumed")) == TurnResumed()


@pytest.mark.parametrize(
    ("reason", "trigger"),
    [
        ("eot", "model"),
        ("finalize", "manual"),
        ("audio_done", "manual"),
        ("silence_timeout", "timeout"),
        ("max_turn_duration", "model"),
        ("something_new", "model"),
    ],
)
def test_a_committed_user_turn_ends_the_turn_with_what_ended_it(reason: str, trigger: str) -> None:
    raw = _pack("committed", text="hi there", reason=reason, user_turn=True, turn_kind="user")
    assert parse_realtime_message(raw) == TurnEnded("hi there", 1.0, trigger)
    # Fish leaves user_turn out when the turn is the user's.
    assert parse_realtime_message(_pack("committed", text="hi there", reason=reason)) == TurnEnded(
        "hi there", 1.0, trigger
    )


def test_a_committed_turn_that_is_not_the_users_is_dropped_whatever_words_it_has() -> None:
    for kind in ("backchannel", "other"):
        raw = _pack("committed", text="mm-hmm", reason="eot", user_turn=False, turn_kind=kind)
        assert parse_realtime_message(raw) == TurnDropped()


def test_a_session_that_ends_with_no_turn_committed_is_a_dropped_turn() -> None:
    assert parse_realtime_message(_pack("session_done", stats={"turns": 0})) == TurnDropped()


def test_a_recoverable_error_is_a_warning_and_any_other_ends_the_stream() -> None:
    recoverable = _pack(
        "error", code=400, error_type="invalid_request", message="dropped", recoverable=True
    )
    assert parse_realtime_message(recoverable) == StreamWarning("invalid_request", "dropped")
    fatal = _pack(
        "error", code=408, error_type="idle_timeout", message="too quiet", recoverable=False
    )
    assert parse_realtime_message(fatal) == StreamError("idle_timeout", "too quiet")


@pytest.mark.parametrize(
    "raw",
    [
        _pack("session_started", session_id="s"),
        _pack("progress"),
        _pack("user_turn_taken"),
        _pack("user_backchannel"),
        _pack("a_new_message"),
        ormsgpack.packb(["not", "a", "map"]),
        b"\xc1 not msgpack",
        '{"type":"draft","text":"a text frame"}',
    ],
)
def test_a_message_that_needs_no_action_or_cannot_be_read_is_skipped(raw: bytes | str) -> None:
    assert parse_realtime_message(raw) is None


def test_a_stream_sends_the_settings_then_audio_as_msgpack_and_reads_the_turn() -> None:
    socket = _FakeSocket(
        [
            _pack("session_started"),
            _pack("speech_start"),
            _pack("draft", text="hi"),
            _pack("committed", text="hi there", reason="eot"),
        ]
    )
    stream, seen = _stream(socket, eager=True)
    audio = b"\x01\x02" * 480

    async def run() -> list[object]:
        await stream.open()
        await stream.send_audio(audio)
        await stream.send_audio(b"")
        await stream.force_end_turn()
        heard = [message async for message in stream.messages()]
        await stream.close()
        await stream.close()  # twice is harmless
        return list(heard)

    heard = asyncio.run(run())
    assert heard == [TurnStarted(""), TurnUpdate("hi"), TurnEnded("hi there", 1.0, "model")]
    assert seen == {"url": "wss://example/v1/asr/realtime", "key": "secret"}
    assert all(isinstance(frame, bytes) for frame in socket.sent)  # Fish rejects text frames
    sent = [ormsgpack.unpackb(frame) for frame in socket.sent]
    assert sent[0] == realtime_config(sample_rate=16000, eager=True)
    assert sent[1] == {"type": "audio", "data": audio}  # raw bytes, not base64
    assert sent[2:] == [{"type": "audio_done"}, {"type": "audio_done"}]
    assert socket.closed


def test_a_stream_that_never_opened_does_nothing_and_closes_quietly() -> None:
    socket = _FakeSocket([])
    stream, _ = _stream(socket)

    async def run() -> list[object]:
        await stream.send_audio(b"\x00\x00")
        await stream.force_end_turn()
        heard = [message async for message in stream.messages()]
        await stream.close()
        return list(heard)

    assert asyncio.run(run()) == []
    assert socket.sent == []


def test_a_stream_ends_when_the_connection_closes_and_raises_on_anything_else() -> None:
    from websockets.exceptions import ConnectionClosedError

    class Dropped(_FakeSocket):
        def __aiter__(self) -> AsyncIterator[bytes | str]:
            async def generate() -> AsyncIterator[bytes | str]:
                yield _pack("draft", text="hi")
                raise ConnectionClosedError(None, None)

            return generate()

    class Broken(_FakeSocket):
        def __aiter__(self) -> AsyncIterator[bytes | str]:
            async def generate() -> AsyncIterator[bytes | str]:
                raise RuntimeError("a bug, not a closed connection")
                yield b""

            return generate()

    async def read(socket: _FakeSocket) -> list[object]:
        stream, _ = _stream(socket)
        await stream.open()
        return [message async for message in stream.messages()]

    assert asyncio.run(read(Dropped([]))) == [TurnUpdate("hi")]
    with pytest.raises(RuntimeError, match="a bug"):
        asyncio.run(read(Broken([])))


def _default_stream(url: str, key: str) -> FishRealtimeStream:
    """A stream on the real connection code, whose websocket library a test replaces."""
    return FishRealtimeStream(url, key, sample_rate=16000)


def test_the_key_and_the_model_go_in_headers_and_a_refusal_names_fish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    seen: dict[str, Any] = {}

    async def connect(url: str, **kwargs: Any) -> _FakeSocket:
        seen.update(url=url, **kwargs)
        return _FakeSocket([])

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    asyncio.run(_default_stream("wss://x/v1/asr/realtime", "sekrit").open())
    assert seen["url"] == "wss://x/v1/asr/realtime"  # the key is never in the address
    assert seen["additional_headers"] == {
        "Authorization": "Bearer sekrit",
        "model": "transcribe-stream-1",
    }

    def refuse(status: int) -> Any:
        async def refused(*_args: object, **_kwargs: object) -> object:
            raise InvalidStatus(Response(status, "no", Headers(), b""))

        return refused

    for status, fatal in ((401, True), (402, True), (429, False), (503, False)):
        monkeypatch.setattr("websockets.asyncio.client.connect", refuse(status))
        with pytest.raises(SttStreamError, match="Fish realtime speech to text") as raised:
            asyncio.run(_default_stream("wss://x/y", "k").open())
        assert raised.value.fatal is fatal
