"""One reply: collecting the model's text, speaking it whole or as it streams, and what the reply leaves for the history and the next listen."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable

import pytest
from fishaudio import TextEvent
from session_fakes import (
    cut_off_speech,
    make_ctx,
)
from voice_fakes import FakeGate, install_gate, make_result, set_tts

from fish_audio_suite_voice.asr.outcome import HeardLine
from fish_audio_suite_voice.asr.result import BargeIn
from fish_audio_suite_voice.capture.portaudio import PortAudioMissingError
from fish_audio_suite_voice.cli.console_sink import ConsoleSink
from fish_audio_suite_voice.llm.reply import collect_reply, speak_reply, stream_turn
from fish_audio_suite_voice.session.events import EVENTS
from fish_audio_suite_voice.session.latency import LatencySnapshot
from fish_audio_suite_voice.session.state import DuplexContext
from fish_audio_suite_voice.tts.playback import PlaybackSink
from fish_audio_suite_voice.tts.played_text import Unheard, spoken_prefix
from fish_audio_suite_voice.tts.text import delta_events
from fish_audio_suite_voice.tts.websocket import TtsResult


def _asked(tokens: Callable[..., AsyncIterator[str]] | None = None) -> DuplexContext:
    """A context whose user said "hi", so the reply has a turn to land in."""
    ctx = make_ctx(tokens)
    ctx.chat.history.add_user_line("hi")
    return ctx


def _reply(ctx: DuplexContext) -> str | None:
    """What the conversation recorded of the last turn's reply."""
    reply = ctx.chat.history.turns[-1].reply
    return None if reply is None else reply.content


def _speak_once(
    monkeypatch: pytest.MonkeyPatch, ctx: DuplexContext, result: TtsResult
) -> tuple[LatencySnapshot, int | None]:
    """Speak one whole reply, with Fish standing in to return ``result``."""
    set_tts(monkeypatch, ctx.tts, speak=lambda *_a, **_k: result)
    return asyncio.run(
        speak_reply(
            ctx,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id=None,
        )
    )


def test_a_spoken_reply_records_only_audio_that_was_played(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_gate(monkeypatch)
    ctx = _asked()
    fatal, code = _speak_once(monkeypatch, ctx, make_result(error_status=401, error_message="no"))
    assert code == 2
    assert _reply(ctx) is None
    assert fatal.tts_first_audio_ms is None

    ctx = _asked()
    _speak_once(monkeypatch, ctx, make_result(cancelled=True))
    assert _reply(ctx) is None

    ctx = _asked()
    heard_part = make_result(
        "hello",
        bytes_played=8,
        got_audio=True,
        cancelled=True,
        tts_first_audio_ms=12.0,
        tts_first_text_ms=4.0,
    )
    updated, code = _speak_once(monkeypatch, ctx, heard_part)
    assert code is None
    assert _reply(ctx) == "Hello"  # as the model wrote it
    assert updated.tts_first_audio_ms == 12.0

    ctx = make_ctx()
    ctx.chat.history.add_user_line("Hey! Can you hear me?")
    echo = make_result(
        "Hey! Can you hear me?",
        bytes_played=8,
        got_audio=True,
        tts_first_audio_ms=12.0,
        tts_first_text_ms=4.0,
    )
    _speak_once(monkeypatch, ctx, echo)
    assert _reply(ctx) is None
    answer = make_result(
        "Yeah. What's up?",
        bytes_played=8,
        got_audio=True,
        tts_first_audio_ms=12.0,
        tts_first_text_ms=4.0,
    )
    _speak_once(monkeypatch, ctx, answer)
    assert _reply(ctx) == "Yeah. What's up?"


class _ClosedStdout:
    def write(self, _text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


def test_collect_reply_survives_a_closed_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Open the door today friend."

    ctx = make_ctx(tokens)
    monkeypatch.setattr(sys, "stdout", _ClosedStdout())
    printer = ConsoleSink(EVENTS)  # the display that writes the tokens to the closed stdout
    try:
        reply, ttft = asyncio.run(
            collect_reply(ctx.chat, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
        )
    finally:
        printer.close()
    assert reply == "Open the door today friend."
    assert ttft is not None


def test_collect_reply_keeps_partial_text_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cancel = asyncio.Event()

    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hi"
        # A barge-in sets the flag, then the stream unwinds with a cancel.
        cancel.set()
        raise asyncio.CancelledError

    ctx = make_ctx(tokens)
    reply, ttft = asyncio.run(
        collect_reply(ctx.chat, llm_cancel=cancel, trace_id=None, started=0.0)
    )
    assert reply == "Hi"
    assert ttft is not None
    assert "[llm]" not in capsys.readouterr().err


def test_collect_reply_lets_an_outside_cancel_propagate() -> None:
    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hi"
        await asyncio.sleep(30)

    ctx = make_ctx(tokens)

    async def run() -> None:
        task = asyncio.create_task(
            collect_reply(ctx.chat, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    asyncio.run(run())


def test_collect_reply_lets_asyncio_timeout_fire() -> None:
    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hi"
        await asyncio.sleep(30)

    ctx = make_ctx(tokens)

    async def run() -> None:
        async with asyncio.timeout(0.1):
            await collect_reply(ctx.chat, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)

    with pytest.raises(TimeoutError):
        asyncio.run(run())


def test_speak_reply_exits_when_playback_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _asked()
    install_gate(monkeypatch)

    def speak(*_args: object, **_kwargs: object) -> TtsResult:
        raise PortAudioMissingError("missing")

    set_tts(monkeypatch, ctx.tts, speak=speak)
    _snapshot, code = asyncio.run(
        speak_reply(
            ctx,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id=None,
        )
    )
    assert code == 2
    assert _reply(ctx) is None


def test_cancelled_speak_keeps_the_audio_that_tripped_barge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _asked()

    gate = FakeGate(captured=b"clip", resume_at=42, min_rms=90.0)  # FISH_VOICE_BARGE_RMS=90
    install_gate(monkeypatch, gate)

    def speak(
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event | None = None,
        on_first_audio: object = None,
        trace_headers: object = None,
    ) -> TtsResult:
        assert text == "[calm] Hello there friend"
        return make_result(
            "hello",
            bytes_played=4,
            got_audio=True,
            cancelled=True,
            tts_first_audio_ms=3.0,
            tts_first_text_ms=2.0,
        )

    set_tts(monkeypatch, ctx.tts, speak=speak)
    _snapshot, code = asyncio.run(
        speak_reply(
            ctx,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id="abc",
        )
    )
    assert code is None
    # The next listen reads on from just after the clip, scoring it at the gate's own level.
    assert ctx.turn_input.carry.clip == BargeIn(b"clip", 42, 90.0)
    assert _reply(ctx) == "Hello"


def test_speak_reply_releases_the_mic_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _asked()
    gate = install_gate(monkeypatch, FakeGate(hold=True))

    def speak(*_args: object, **_kwargs: object) -> TtsResult:
        return make_result(
            "hello there",
            bytes_played=8,
            got_audio=True,
            tts_first_audio_ms=3.0,
            tts_first_text_ms=2.0,
        )

    set_tts(monkeypatch, ctx.tts, speak=speak)
    _snapshot, code = asyncio.run(
        speak_reply(
            ctx,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id=None,
        )
    )
    assert code is None
    assert _reply(ctx) == "Hello there"
    # The watcher waits up to 30 s for the cancel flag, and the caller only joins
    # for about a second, so a thread that is already finished proves the caller
    # set the flag before it returned.
    assert gate.thread is not None
    assert gate.thread.is_alive() is False


def test_each_token_reaches_fish_while_the_model_is_still_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TTS thread is woken by each token, not only when the reply is finished."""
    fish_reading = threading.Event()
    first_reached_fish = threading.Event()

    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        # Write once the TTS thread is waiting, so the token has to wake it.
        await asyncio.to_thread(fish_reading.wait, 5)
        await asyncio.sleep(0.05)
        yield "Hi "
        # The model holds its next token until Fish has the first one.
        yield "there." if await asyncio.to_thread(first_reached_fish.wait, 5) else "too late."

    ctx = make_ctx(tokens)
    install_gate(monkeypatch)
    sent: list[str] = []

    def speak_stream(
        deltas: AsyncIterator[str], sink: PlaybackSink, cancel: threading.Event, **_kw: object
    ) -> TtsResult:
        async def drain() -> None:
            fish_reading.set()
            async for token in deltas:
                sent.append(token)
                first_reached_fish.set()

        asyncio.run(drain())
        return make_result("Hi there.", bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    set_tts(monkeypatch, ctx.tts, speak_stream=speak_stream)
    _snapshot, code = asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert code is None
    assert sent == ["Hi ", "there."]


async def _two_tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
    for token in ("Hello ", "there friend."):
        yield token
        await asyncio.sleep(0)


def test_stream_turn_feeds_tokens_to_tts_and_arms_barge_at_first_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _asked(_two_tokens)
    gate = install_gate(monkeypatch)
    heard_tokens: list[str] = []
    armed_before_audio: list[int] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: object = None,
    ) -> TtsResult:
        async def drain() -> None:
            heard_tokens.extend([token async for token in deltas])

        asyncio.run(drain())
        armed_before_audio.append(gate.armed)
        assert on_first_audio is not None
        on_first_audio()
        return make_result(
            "Hello there friend.",
            bytes_played=4,
            got_audio=True,
            tts_first_audio_ms=3.0,
            tts_first_text_ms=2.0,
        )

    set_tts(monkeypatch, ctx.tts, speak_stream=speak_stream)
    snapshot, code = asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert code is None
    assert heard_tokens == ["Hello ", "there friend."]
    assert armed_before_audio == [0]
    assert gate.armed == 1
    assert snapshot.first_audio_ms is not None
    assert snapshot.llm_first_token_ms is not None
    assert _reply(ctx) == "Hello there friend."


def test_stream_turn_speaks_the_finished_reply_when_fish_fails_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _asked(_two_tokens)
    install_gate(monkeypatch)
    spoken: list[str] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: object = None,
    ) -> TtsResult:
        async def drain() -> None:
            async for _token in deltas:
                pass

        asyncio.run(drain())
        return make_result(error_status=503)

    def speak_whole(
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: object = None,
    ) -> TtsResult:
        spoken.append(text)
        return make_result(
            "Hello there friend.",
            bytes_played=4,
            got_audio=True,
            tts_first_audio_ms=3.0,
            tts_first_text_ms=2.0,
        )

    set_tts(monkeypatch, ctx.tts, speak=speak_whole, speak_stream=speak_stream)
    _snapshot, code = asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert code is None
    assert len(spoken) == 1
    assert "Hello" in spoken[0]
    assert _reply(ctx) == "Hello there friend."


@pytest.mark.parametrize(("status", "stops_the_model"), [(401, True), (503, False)])
def test_a_fatal_fish_status_stops_the_llm_but_a_transient_one_does_not(
    monkeypatch: pytest.MonkeyPatch, status: int, *, stops_the_model: bool
) -> None:
    fish_gave_up = threading.Event()
    model_stopped: list[bool] = []

    async def tokens(
        *_args: object, cancel: asyncio.Event | None = None, **_kwargs: object
    ) -> AsyncIterator[str]:
        assert cancel is not None
        yield "Hello "
        await asyncio.to_thread(fish_gave_up.wait, 5)
        for _ in range(50):  # the stop arrives on the loop just after Fish returns
            if cancel.is_set():
                break
            await asyncio.sleep(0.01)
        model_stopped.append(cancel.is_set())
        yield "there friend."

    ctx = make_ctx(tokens)
    install_gate(monkeypatch)

    def speak_stream(*_args: object, **_kwargs: object) -> TtsResult:
        fish_gave_up.set()
        return make_result(error_status=status)

    def speak_whole(text: str, *_args: object, **_kwargs: object) -> TtsResult:
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    set_tts(monkeypatch, ctx.tts, speak=speak_whole, speak_stream=speak_stream)
    asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert model_stopped == [stops_the_model]


def test_stream_turn_with_an_empty_reply_speaks_nothing_and_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def no_tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        if False:
            yield ""

    ctx = _asked(no_tokens)
    install_gate(monkeypatch)
    seen_cancel: list[bool] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: object = None,
    ) -> TtsResult:
        async def drain() -> None:
            async for _token in deltas:
                pass

        asyncio.run(drain())
        seen_cancel.append(cancel.is_set())
        return make_result(cancelled=True)

    def never(*_args: object, **_kwargs: object) -> TtsResult:
        raise AssertionError("an empty reply must not be spoken")

    set_tts(monkeypatch, ctx.tts, speak=never, speak_stream=speak_stream)
    history_before = ctx.chat.messages()
    _snapshot, code = asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert code is None
    assert seen_cancel == [True]
    assert ctx.chat.messages() == history_before
    assert "silent" not in capsys.readouterr().out


def test_quit_during_the_stream_fallback_rebind_does_not_speak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx(_two_tokens)
    install_gate(monkeypatch)

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: object = None,
    ) -> TtsResult:
        async def drain() -> None:
            async for _token in deltas:
                pass

        asyncio.run(drain())
        return make_result(error_status=503)

    started_cancelled: list[bool] = []

    def speak_whole(
        text: str, sink: PlaybackSink, cancel: threading.Event | None = None, **_kwargs: object
    ) -> TtsResult:
        del text, sink
        started_cancelled.append(cancel is not None and cancel.is_set())
        return make_result(cancelled=True)

    bind = ctx.turn_input.session.turn.bind

    def bind_then_quit(cancel: threading.Event, llm_cancel: asyncio.Event, *a: object) -> None:
        bind(cancel, llm_cancel)
        # Ctrl+C lands right after the old flag was replaced.
        ctx.turn_input.session.request_quit()

    monkeypatch.setattr(ctx.turn_input.session.turn, "bind", bind_then_quit)
    set_tts(monkeypatch, ctx.tts, speak=speak_whole, speak_stream=speak_stream)
    _snapshot, code = asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert code is None
    assert started_cancelled == [True]  # quit reaches the retried turn: nothing is played
    assert ctx.turn_input.session.turn.cancel is not None
    assert ctx.turn_input.session.turn.cancel.is_set()


def _cut_off_ctx(monkeypatch: pytest.MonkeyPatch, spoken: str) -> tuple[DuplexContext, list[str]]:
    """A context whose first reply is cut off by a barge-in after ``spoken`` was heard."""
    ctx = make_ctx()
    install_gate(monkeypatch, FakeGate(captured=b"clip"))
    return ctx, cut_off_speech(monkeypatch, ctx.tts, spoken)


def test_a_cut_off_reply_is_remembered_for_resuming(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _said = _cut_off_ctx(monkeypatch, "Hello there.")
    asyncio.run(
        speak_reply(
            ctx,
            "Hello there. How was your day today?",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id=None,
        )
    )
    # Fish is sent the rest led by the mood from before the cut; the record gets it as written.
    assert ctx.turn_input.carry.resume == Unheard(
        "[calm] How was your day today?", "How was your day today?"
    )


def test_a_streamed_reply_is_recorded_as_written_though_fish_got_the_mood_in_each_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def tokens(*_a: object, **_k: object) -> AsyncIterator[str]:
        for token in ("[calm] The first thing. ", "The second thing. ", "The third thing."):
            yield token

    ctx = _asked(tokens)
    install_gate(monkeypatch)
    sent: list[str] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: object = None,
    ) -> TtsResult:
        async def pieces() -> None:
            events = delta_events(deltas, cancel, partial_chars=20, style=ctx.config.cue_style)
            sent.extend([ev.text async for ev in events if isinstance(ev, TextEvent)])

        asyncio.run(pieces())
        heard = spoken_prefix(
            "".join(sent),
            bytes_played=4,
            sample_rate=16_000,
            audio_format="pcm",
            got_audio=True,
            cancelled=False,
        )
        return make_result(heard, bytes_played=4, got_audio=True)

    set_tts(monkeypatch, ctx.tts, speak_stream=speak_stream)
    asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert "".join(sent).count("[calm]") == 3
    assert _reply(ctx) == "[calm] The first thing. The second thing. The third thing."
