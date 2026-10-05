"""Deepgram Flux: the protocol, and a connection driven by a fake socket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from fish_audio_suite_voice.deepgram import (
    DeepgramError,
    FluxStream,
    StreamError,
    StreamWarning,
    TurnEnded,
    TurnStarted,
    TurnUpdate,
    _websockets_connect,
    flux_base,
    flux_url,
    parse_flux_message,
)


def _turn(event: str, text: str = "", **extra: Any) -> str:
    return json.dumps(
        {"type": "TurnInfo", "event": event, "transcript": text, "turn_index": 0, **extra}
    )


def test_the_address_asks_for_16_bit_mono_pcm_and_no_retention() -> None:
    url = flux_url("flux-general-en", sample_rate=16000, eot_threshold=0.7, eot_timeout_ms=4000)
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "wss://api.deepgram.com/v2/listen"
    query = {key: value[0] for key, value in parse_qs(parts.query).items()}
    assert query == {
        "model": "flux-general-en",
        "encoding": "linear16",
        "sample_rate": "16000",
        "eot_threshold": "0.7",
        "eot_timeout_ms": "4000",
        "mip_opt_out": "true",  # Deepgram keeps neither the audio nor the text
    }
    assert "key" not in parts.query.lower()  # the key goes in a header, never the address


def test_a_regional_endpoint_can_replace_the_default_host() -> None:
    url = flux_url(
        "flux-general-multi",
        sample_rate=16000,
        eot_threshold=0.8,
        base="wss://api.eu.deepgram.com/v2/listen",
    )
    assert url.startswith("wss://api.eu.deepgram.com/v2/listen?model=flux-general-multi")


def test_the_turn_messages_are_read_as_deepgram_documents_them() -> None:
    assert parse_flux_message(_turn("StartOfTurn", "hello")) == TurnStarted("hello")
    assert parse_flux_message(_turn("Update", "hello there")) == TurnUpdate("hello there")
    ended = _turn("EndOfTurn", "hello there", end_of_turn_confidence=0.72, trigger="model")
    assert parse_flux_message(ended) == TurnEnded("hello there", 0.72, "model")
    # Bytes are accepted too, and a missing trigger or confidence has a sensible stand-in.
    bare = _turn("EndOfTurn", "hi").encode()
    assert parse_flux_message(bare) == TurnEnded("hi", 0.0, "model")
    forced = _turn("EndOfTurn", "hi", trigger="manual")
    assert parse_flux_message(forced) == TurnEnded("hi", 0.0, "manual")


def test_an_error_message_carries_its_code_and_description() -> None:
    raw = json.dumps({"type": "Error", "code": "INTERNAL_SERVER_ERROR", "description": "oops"})
    assert parse_flux_message(raw) == StreamError("INTERNAL_SERVER_ERROR", "oops")


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"Connected","request_id":"x","sequence_id":0}',
        '{"type":"ConfigureSuccess"}',
        _turn("EagerEndOfTurn", "maybe done"),  # an early guess is not used yet
        _turn("TurnResumed", "still talking"),
        _turn("SomethingNew"),
        "not json at all",
        "[1, 2, 3]",
        "null",
        "",
    ],
)
def test_messages_that_need_no_action_are_skipped(raw: str) -> None:
    assert parse_flux_message(raw) is None


class _FakeSocket:
    """What ``FluxStream`` needs of a connection: send, iterate, close."""

    def __init__(self, incoming: list[str]) -> None:
        self.sent: list[str | bytes] = []
        self.closed = False
        self._incoming = incoming

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

    def __aiter__(self) -> AsyncIterator[str]:
        async def generate() -> AsyncIterator[str]:
            for raw in self._incoming:
                yield raw

        return generate()

    async def close(self) -> None:
        self.closed = True


def _stream(socket: _FakeSocket) -> tuple[FluxStream, dict[str, Any]]:
    seen: dict[str, Any] = {}

    async def connect(url: str, *, key: str) -> _FakeSocket:
        seen["url"], seen["key"] = url, key
        return socket

    return FluxStream("wss://example/v2/listen?x=1", "secret", connect=connect), seen


def test_a_stream_sends_audio_and_commands_and_reads_the_turn_back() -> None:
    socket = _FakeSocket(
        [
            '{"type":"Connected"}',
            _turn("StartOfTurn", "hi"),
            _turn("Update", "hi there"),
            _turn("EndOfTurn", "hi there", end_of_turn_confidence=0.8, trigger="model"),
        ]
    )
    stream, seen = _stream(socket)

    async def run() -> list[object]:
        await stream.open()
        await stream.send_audio(b"\x01\x02" * 480)
        await stream.send_audio(b"")  # Deepgram closes on an empty frame, so it is never sent
        await stream.force_end_turn()
        heard = [message async for message in stream.messages()]
        await stream.close()
        await stream.close()  # twice is harmless
        return list(heard)

    heard = asyncio.run(run())
    assert heard == [TurnStarted("hi"), TurnUpdate("hi there"), TurnEnded("hi there", 0.8, "model")]
    assert seen == {"url": "wss://example/v2/listen?x=1", "key": "secret"}
    assert socket.sent[0] == b"\x01\x02" * 480
    assert json.loads(socket.sent[1]) == {"type": "ForceEndTurn"}
    assert json.loads(socket.sent[2]) == {"type": "CloseStream"}
    assert len(socket.sent) == 3
    assert socket.closed


def test_a_stream_that_never_opened_does_nothing_and_closes_quietly() -> None:
    stream, _ = _stream(_FakeSocket([]))

    async def run() -> list[object]:
        await stream.send_audio(b"\x00\x00")
        await stream.force_end_turn()
        heard = [message async for message in stream.messages()]
        await stream.close()
        return list(heard)

    assert asyncio.run(run()) == []


def test_a_stream_ends_when_the_connection_closes_and_raises_on_anything_else() -> None:
    from websockets.exceptions import ConnectionClosedError

    class Dropped(_FakeSocket):
        def __aiter__(self) -> AsyncIterator[str]:
            async def generate() -> AsyncIterator[str]:
                yield _turn("Update", "hi")
                raise ConnectionClosedError(None, None)

            return generate()

    class Broken(_FakeSocket):
        def __aiter__(self) -> AsyncIterator[str]:
            async def generate() -> AsyncIterator[str]:
                raise RuntimeError("a bug, not a closed connection")
                yield ""

            return generate()

    async def read(socket: _FakeSocket) -> list[object]:
        stream, _ = _stream(socket)
        await stream.open()
        return [message async for message in stream.messages()]

    assert asyncio.run(read(Dropped([]))) == [TurnUpdate("hi")]  # a drop is the end of the stream
    with pytest.raises(RuntimeError, match="a bug"):
        asyncio.run(read(Broken([])))


def test_a_refused_connection_says_whether_trying_again_could_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    def refuse(status: int) -> Any:
        async def connect(*_args: object, **_kwargs: object) -> object:
            raise InvalidStatus(Response(status, "no", Headers(), b""))

        return connect

    for status, fatal in ((401, True), (402, True), (403, True), (500, False), (503, False)):
        monkeypatch.setattr("websockets.asyncio.client.connect", refuse(status))
        with pytest.raises(DeepgramError) as raised:
            asyncio.run(_websockets_connect("wss://x/y", key="k"))
        assert raised.value.fatal is fatal, status
        assert str(status) in str(raised.value)


def test_an_unreachable_server_is_a_retryable_error_not_a_fatal_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def connect(*_args: object, **_kwargs: object) -> object:
        raise OSError("no route to host")

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    with pytest.raises(DeepgramError, match="no route") as raised:
        asyncio.run(_websockets_connect("wss://x/y", key="k"))
    assert not raised.value.fatal


def test_the_real_connection_sends_the_key_as_a_header_and_never_in_the_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def connect(url: str, **kwargs: Any) -> object:
        seen["url"] = url
        seen.update(kwargs)
        return object()

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    asyncio.run(_websockets_connect("wss://x/y?model=m", key="sekrit"))
    assert seen["additional_headers"] == {"Authorization": "Token sekrit"}
    assert "sekrit" not in seen["url"]
    assert seen["open_timeout"] > 0


def test_a_warning_is_read_so_it_can_be_logged() -> None:
    raw = json.dumps({"type": "Warning", "code": "SOME_CODE", "description": "heads up"})
    assert parse_flux_message(raw) == StreamWarning("SOME_CODE", "heads up")


def test_a_full_turn_info_message_as_the_schema_describes_it_is_read() -> None:
    """Every field of the published TurnInfo schema, with its numbers written as strings."""
    raw = json.dumps(
        {
            "type": "TurnInfo",
            "request_id": "3f1c7c1e-9b6c-4d56-8f2f-0b3c0c1d2e3f",
            "sequence_id": 7,
            "event": "EndOfTurn",
            "turn_index": 2,
            "audio_window_start": "0.0",
            "audio_window_end": "1.7",
            "transcript": "Hi I need to cancel my subscription please.",
            "words": [{"word": "Hi", "confidence": "0.99", "start": 0.1, "end": 0.3}],
            "end_of_turn_confidence": "0.72",
            "trigger": "timeout",
            "languages": ["en"],
            "languages_hinted": ["en"],
        }
    )
    assert parse_flux_message(raw) == TurnEnded(
        "Hi I need to cancel my subscription please.", 0.72, "timeout"
    )
    assert parse_flux_message(raw.replace('"0.72"', "null")) == TurnEnded(
        "Hi I need to cancel my subscription please.", 0.0, "timeout"
    )
    assert parse_flux_message(
        raw.replace('"trigger": "timeout"', '"trigger": "brand-new"')
    ) == TurnEnded(
        "Hi I need to cancel my subscription please.", 0.72, "brand-new"
    )  # an open enum: a value we do not know is kept, not rejected


@pytest.mark.parametrize(
    ("region", "host"),
    [
        ("global", "api.deepgram.com"),
        ("eu", "api.eu.deepgram.com"),
        ("au", "api.au.deepgram.com"),
        ("in", "api.in.deepgram.com"),
        ("unknown", "api.deepgram.com"),  # never a made-up host: an unknown region is the default
    ],
)
def test_each_region_has_its_own_flux_endpoint_and_an_unknown_one_is_the_default(
    region: str, host: str
) -> None:
    assert flux_base(region) == f"wss://{host}/v2/listen"
