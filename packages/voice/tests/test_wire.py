from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any, cast

import httpx
import ormsgpack
import pytest
from fishaudio.exceptions import RateLimitError, ValidationError, WebSocketError
from wire_helpers import (
    RecordingSink,
    ScriptedSocket,
    collect_events,
    make_run,
    play_stream,
    spec_sent,
    stream_events,
    stream_text,
)

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    CueStyle,
    FishAuthError,
    FishLatency,
    SuiteDefaults,
)
from fish_audio_suite_voice.playback import MpvSink, PlaybackSink
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tts_text import WholeText, text_events
from fish_audio_suite_voice.tts_tune import TtsTune
from fish_audio_suite_voice.tts_turn import run_turn, turn_failure
from fish_audio_suite_voice.wire import (
    FlushEvent,
    TextEvent,
    TtsResult,
    TurnSpec,
    tts_result,
)


@pytest.fixture
def spec_of(monkeypatch: pytest.MonkeyPatch) -> Callable[..., TurnSpec]:
    """Speak with a ``FishSpeaker`` and return the ``TurnSpec`` its websocket opened with."""

    def spec(tts: FishSpeaker, sink: PlaybackSink | None = None) -> TurnSpec:
        return spec_sent(monkeypatch, tts, sink)

    return spec


def test_text_events_flush_only_after_text() -> None:
    cancel = threading.Event()
    assert collect_events("", cancel, 40) == []
    assert collect_events("   ", cancel, 40) == []
    events = collect_events("word " * 30, cancel, 20)
    texts = [event.text for event in events if isinstance(event, TextEvent)]
    assert len(texts) > 1
    assert all(len(piece) <= 20 for piece in texts[:-1])
    assert isinstance(events[-1], FlushEvent)


def test_text_events_cancel_drops_the_flush() -> None:
    cancel = threading.Event()

    async def collect() -> list[TextEvent | FlushEvent]:
        stream = text_events("word " * 30, cancel, 20, style=CueStyle())
        first = await anext(stream)
        cancel.set()
        rest = [event async for event in stream]
        return [first, *rest]

    events = asyncio.run(collect())
    assert len(events) == 1
    assert isinstance(events[0], TextEvent)


def test_split_strikethrough_is_not_spoken() -> None:
    whole = stream_text(
        ["I meant ", "~~Tuesday~~", " Wednesday for the meeting."],
        40,
    )
    assert "Tuesday" not in whole
    assert "Wednesday" in whole
    split = stream_text(
        ["I meant ~~Tues", "day~~ Wednesday for the meeting."],
        40,
    )
    assert "Tuesday" not in split
    assert "Wednesday" in split
    kept = stream_text(["Meet ", "~softly~", " on Wednesday please."], 40)
    assert "softly" in kept


def test_partial_cut_does_not_split_a_cue() -> None:
    # A 40-character hard cut used to send "[whi" and then "spering]".
    # A period inside the cue used to send "[hello." and then "there]".
    tts = FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(partial_chars=40))

    async def pieces_of(sample: str) -> list[str]:
        events = [event async for event in stream_events(tts, [sample], threading.Event())]
        return [event.text for event in events if isinstance(event, TextEvent)]

    pieces = asyncio.run(pieces_of("x" * 36 + "[whispering] come closer today friend please."))
    assert pieces
    assert all(piece.count("[") == piece.count("]") for piece in pieces)
    assert any("[whispering]" in piece for piece in pieces)
    noted = asyncio.run(pieces_of("Note [hello. there] and then more words please today friend."))
    assert noted
    assert all(piece.count("[") == piece.count("]") for piece in noted)
    assert any("[hello. there]" in piece for piece in noted)


def test_sdk_limits_are_applied_before_the_socket(spec_of: Callable[..., TurnSpec]) -> None:
    tts = FishSpeaker(
        api_key="k",
        voice_id="v",
        tune=TtsTune(
            latency="low",
            chunk_length=800,
            volume=25,
            speed=4,
            temperature=1.5,
            top_p=-0.2,
        ),
    )
    spec = spec_of(tts)
    assert spec.latency == "balanced"
    # Out-of-contract spellings that a caller reading an env var could pass.
    loud = spec_of(
        FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(latency=cast(FishLatency, " LOW ")))
    )
    assert loud.latency == "balanced"
    spaced = spec_of(
        FishSpeaker(
            api_key="k", voice_id="v", tune=TtsTune(latency=cast(FishLatency, " balanced "))
        )
    )
    assert spaced.latency == "balanced"
    assert spec.speed == 2.0
    assert spec.config.temperature == 1.0
    assert spec.config.top_p == 0.0
    assert spec.config.chunk_length == 300
    assert spec.config.prosody is not None
    assert spec.config.prosody.volume == 20.0
    quiet = FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(volume=-40))
    prosody = spec_of(quiet).config.prosody
    assert prosody is not None
    assert prosody.volume == -20.0


def test_voice_id_and_model_survive_the_socket_start(spec_of: Callable[..., TurnSpec]) -> None:
    spec = spec_of(
        FishSpeaker(
            api_key="k",
            voice_id="voice-\ud800",
            model="s2.1-pro\nbad",
        )
    )
    assert "\ud800" not in spec.voice_id
    assert spec.voice_id.startswith("voice-")
    assert spec.model == SuiteDefaults().tts_model
    assert "\n" not in spec.model
    ormsgpack.packb({"reference_id": spec.voice_id})
    kept = spec_of(FishSpeaker(api_key="k", voice_id="v", model="MyModel"))
    assert kept.model == "MyModel"


def test_zero_sample_rate_is_replaced_before_the_socket(spec_of: Callable[..., TurnSpec]) -> None:
    for rate in (0, 2**32):
        spec = spec_of(FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(sample_rate=rate)))
        assert spec.sample_rate == SuiteDefaults().sample_rate
        assert spec.config.sample_rate == SuiteDefaults().sample_rate


def test_fish_is_asked_for_the_format_the_sink_takes(spec_of: Callable[..., TurnSpec]) -> None:
    tts = FishSpeaker(api_key="k", voice_id="v")
    # A sink that does not say, such as one written before sinks declared a format, gets PCM.
    spec = spec_of(tts, RecordingSink())
    assert spec.audio_format == "pcm"
    assert spec.config.format == "pcm"
    mp3 = spec_of(tts, MpvSink())
    assert mp3.audio_format == "mp3"
    assert mp3.config.format == "mp3"

    class OggSink(RecordingSink):
        audio_format = "ogg"

    with pytest.raises(ValueError, match="'ogg'"):
        spec_of(tts, OggSink())


def test_turn_failure_retries_only_before_audio(capsys: pytest.CaptureFixture[str]) -> None:
    slow = RateLimitError(429, "slow down", None)
    cancel = threading.Event()
    retryable = turn_failure(slow, attempt=0, sent_text="hello", got_audio=False, cancel=cancel)
    assert retryable.retry is True
    heard = turn_failure(slow, attempt=0, sent_text="hello", got_audio=True, cancel=cancel)
    assert heard.retry is False
    assert heard.error_status == 429
    empty = turn_failure(slow, attempt=0, sent_text="", got_audio=False, cancel=cancel)
    assert empty.retry is False
    last = turn_failure(
        slow,
        attempt=FISH_RETRY_ATTEMPTS - 1,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert last.retry is False
    cancel.set()
    barged = turn_failure(slow, attempt=0, sent_text="hello", got_audio=False, cancel=cancel)
    assert barged.retry is False
    assert barged.error_status is None
    assert "no audio" not in capsys.readouterr().err


def test_turn_failure_classifies_socket_validation_and_groups(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cancel = threading.Event()
    socket = turn_failure(
        WebSocketError("dropped"),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert socket.retry is True
    invalid = turn_failure(
        ValidationError("bad voice"),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert invalid.retry is False
    assert invalid.error_status == 400
    grouped = turn_failure(
        BaseExceptionGroup("turn", [RateLimitError(503, "down", None)]),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert grouped.retry is True
    hidden = turn_failure(
        BaseExceptionGroup(
            "turn",
            [asyncio.CancelledError(), RateLimitError(503, "down", None)],
        ),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert hidden.retry is True
    only_cancel = turn_failure(
        BaseExceptionGroup("turn", [asyncio.CancelledError()]),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert only_cancel.retry is False
    refused = turn_failure(
        httpx.ConnectError("refused"),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert refused.retry is True
    assert "status=502" in capsys.readouterr().err


def test_pump_does_not_play_audio_that_arrives_after_cancel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sink = RecordingSink()
    cancel = threading.Event()

    async def chunks() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        await asyncio.sleep(0.05)
        cancel.set()
        yield b"\x03\x04"
        yield b""

    result, socket = asyncio.run(play_stream(chunks(), sink, cancel))
    assert sink.chunks == [b"\x01\x02"]
    assert result.got_audio is True
    assert result.tts_first_audio_ms is not None
    assert result.cancelled is True
    assert socket.closed >= 1
    assert "[tts first audio ttfa]" not in capsys.readouterr().out


def test_cancel_scope_after_audio_keeps_only_the_played_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket(
        [
            b"\x00" * 33_075,
            RuntimeError(
                "Attempted to exit cancel scope in a different task than it was entered in"
            ),
        ]
    )
    run, sink = make_run()

    result = asyncio.run(
        run_turn(
            run.spec,
            WholeText("hello there friend", 40, CueStyle()),
            sink,
            threading.Event(),
            socket=socket,
        )
    )
    assert socket.streams == 1
    assert result.got_audio is True
    assert result.cancelled is False
    assert result.spoken_so_far == "hello"


def test_cancelled_encoded_turn_leaves_history_empty() -> None:
    run, sink = make_run()
    run.spec = replace(run.spec, config=run.spec.config.model_copy(update={"format": "mp3"}))
    run.sent_text = "hello there friend"
    run.audio.got_audio = True
    run.cancel.set()
    sink.chunks.append(b"x" * 100)
    result = tts_result(run)
    assert result.cancelled
    assert result.spoken_so_far == ""


def test_spoken_estimate_uses_speed_and_the_sinks_output_latency() -> None:
    run, sink = make_run()
    run.sent_text = "hello there friend how are you today"
    run.audio.got_audio = True
    run.cancel.set()
    sink.chunks.append(b"\x00" * 44100 * 2)  # one second at 44.1 kHz int16
    base = tts_result(run).spoken_so_far
    sink.output_latency_s = 0.5
    buffered = tts_result(run).spoken_so_far
    assert 0 < len(buffered) < len(base)
    sink.output_latency_s = 2.0
    assert tts_result(run).spoken_so_far == ""


def test_tts_spec_warns_once_when_it_swaps_a_latency(
    spec_of: Callable[..., TurnSpec], capsys: pytest.CaptureFixture[str]
) -> None:
    spec_of(
        FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(latency=cast(FishLatency, "turbo")))
    )
    spec_of(
        FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(latency=cast(FishLatency, "turbo")))
    )
    spec_of(FishSpeaker(api_key="k", voice_id="v", tune=TtsTune(latency="low")))
    err = capsys.readouterr().err
    assert err.count("'turbo'") == 1
    assert "'low'" in err


def test_pump_does_not_let_the_reader_run_far_ahead_of_playback() -> None:
    produced = 0
    lead: list[int] = []

    class SlowSink(RecordingSink):
        def write(self, chunk: bytes) -> None:
            lead.append(produced - len(self.chunks))
            time.sleep(0.01)  # a device write blocks while the reader is free to run on
            super().write(chunk)

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal produced
        for index in range(12):
            produced += 1
            yield bytes([index + 1]) * 4

    sink = SlowSink()
    asyncio.run(play_stream(chunks(), sink, threading.Event()))
    assert len(sink.chunks) == 12
    # One queued chunk plus the one being written plus one held by the reader.
    assert max(lead) <= 4


def test_pump_waits_for_the_cancelled_reader_to_unwind() -> None:
    cleaned: list[str] = []
    cancel = threading.Event()

    class CancellingSink(RecordingSink):
        def write(self, chunk: bytes) -> None:
            super().write(chunk)
            cancel.set()

    async def chunks() -> AsyncIterator[bytes]:
        try:
            yield b"\x01\x02"
            await asyncio.sleep(30)
        finally:
            # A slow teardown. The turn must not return before it ends.
            await asyncio.sleep(0.05)
            cleaned.append("done")

    async def turn() -> None:
        await play_stream(chunks(), CancellingSink(), cancel)
        assert cleaned == ["done"]

    asyncio.run(turn())


def test_a_turn_spec_copies_and_freezes_its_trace_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = {"traceparent": "00-a"}
    spec = spec_sent(monkeypatch, FishSpeaker(api_key="k", voice_id="v"), trace_headers=headers)
    headers["traceparent"] = "00-changed"
    assert spec.trace_headers["traceparent"] == "00-a"
    with pytest.raises(TypeError):
        cast(Any, spec.trace_headers)["x"] = "y"


def test_speak_takes_cancel_by_keyword_only() -> None:
    tts = FishSpeaker(api_key="k", voice_id="v")
    with pytest.raises(TypeError):
        cast(Any, tts.speak)("hi", RecordingSink(), threading.Event())
    with pytest.raises(TypeError):
        cast(Any, tts.speak_stream)(["hi"], RecordingSink(), threading.Event())


def test_a_sink_without_output_latency_still_finishes_the_turn() -> None:
    class OldSink:
        """Written against the protocol before output_latency_s existed."""

        def start(self) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            del chunk

        def finish(self, *, kill: bool = False) -> None:
            del kill

        def bytes_played(self) -> int:
            return 0

    run, _sink = make_run()
    # The cast is the point: this sink deliberately lacks a member of the protocol.
    run.sink = cast("PlaybackSink", OldSink())
    result = tts_result(run)
    assert result.bytes_played == 0


def test_a_result_built_from_a_status_gets_the_matching_error() -> None:

    fatal = TtsResult("", 0, False, False, None, None, error_status=401, error_message="no")
    assert isinstance(fatal.error, FishAuthError)
    plain = TtsResult("hi", 4, True, False, 1.0, 1.0)
    assert plain.error is None


def test_pump_runs_the_whole_stream_in_one_task() -> None:
    sink = RecordingSink()
    tasks: set[asyncio.Task[object] | None] = set()

    async def chunks() -> AsyncIterator[bytes]:
        for piece in (b"\x01\x02", b"\x03\x04"):
            tasks.add(asyncio.current_task())
            yield piece
        tasks.add(asyncio.current_task())

    asyncio.run(play_stream(chunks(), sink, threading.Event()))
    assert sink.chunks == [b"\x01\x02", b"\x03\x04"]
    assert len(tasks) == 1


def test_a_cancel_raised_by_the_stream_itself_is_passed_on() -> None:
    sink = RecordingSink()

    async def chunks() -> AsyncIterator[bytes]:
        yield b"\x01\x02"
        raise asyncio.CancelledError

    async def turn() -> None:
        # The turn's cancel flag is not set, so the cancel is not its own: it goes up rather
        # than ending the turn quietly. The timeout only keeps a hang from stalling the suite.
        await asyncio.wait_for(play_stream(chunks(), sink, threading.Event()), timeout=5.0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(turn())
    assert sink.chunks == [b"\x01\x02"]


def test_a_slow_device_write_does_not_stall_the_turn_loop() -> None:
    # Each write waits until the turn's event loop has run something else. A write made
    # on the loop itself could never see that, and would give up after the timeout.
    writing = threading.Event()
    loop_ran = threading.Event()
    released: list[bool] = []

    class BlockingSink(RecordingSink):
        def write(self, chunk: bytes) -> None:
            loop_ran.clear()
            writing.set()
            released.append(loop_ran.wait(timeout=5))
            writing.clear()
            super().write(chunk)

    async def chunks() -> AsyncIterator[bytes]:
        yield b"\x01\x00" * 4
        yield b"\x02\x00" * 4

    async def main() -> None:
        async def loop_side() -> None:
            while True:
                if writing.is_set():
                    loop_ran.set()
                await asyncio.sleep(0.001)

        watcher = asyncio.create_task(loop_side())
        sink = BlockingSink()
        await play_stream(chunks(), sink, threading.Event())
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        assert len(sink.chunks) == 2

    asyncio.run(main())
    assert released == [True, True]


@pytest.mark.perf
def test_cancel_while_waiting_for_audio_stops_the_pump_at_once() -> None:
    sink = RecordingSink()
    cancel = threading.Event()

    async def silent() -> AsyncIterator[bytes]:
        await asyncio.sleep(10)
        yield b"\x01\x00"

    async def main() -> float:
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        loop.call_later(0.05, cancel.set)
        _result, socket = await play_stream(silent(), sink, cancel)
        assert socket.closed
        return time.perf_counter() - started

    elapsed = asyncio.run(main())
    assert not sink.chunks
    # Polling waited up to 0.25 s after the cancel. Now it takes one loop turn.
    assert elapsed < 0.2


def test_a_whole_reply_for_a_directed_model_carries_no_mood_between_sentences() -> None:
    text = "[happy] You sound so relaxed. Just perfect. Truly."
    cancel = threading.Event()

    async def run(*, directed: bool) -> str:
        style = CueStyle("directed" if directed else "free")
        events = text_events(text, cancel, 30, style=style)
        return "".join([event.text async for event in events if isinstance(event, TextEvent)])

    assert asyncio.run(run(directed=True)).count("[happy]") == 1
    assert asyncio.run(run(directed=False)).count("[happy]") > 1


class _Websocket:
    """An audio iterator like the SDK's, which must never be closed with ``aclose()``."""

    def __init__(self) -> None:
        self.sent = 0
        self.aclosed = False

    def __aiter__(self) -> _Websocket:
        return self

    async def __anext__(self) -> bytes:
        if self.sent:
            await asyncio.sleep(30)  # Fish has more to say
        self.sent += 1
        return b"\x01\x02"

    async def aclose(self) -> None:
        self.aclosed = True


def test_a_cancelled_turn_closes_the_client_and_never_acloses_the_websocket_iterator() -> None:
    cancel = threading.Event()

    class BargedSink(RecordingSink):
        def write(self, chunk: bytes) -> None:
            super().write(chunk)
            cancel.set()  # the user talks over the first chunk

    websocket = _Websocket()
    result, socket = asyncio.run(play_stream(websocket, BargedSink(), cancel))
    assert result.cancelled
    assert socket.closed >= 1
    assert websocket.aclosed is False
