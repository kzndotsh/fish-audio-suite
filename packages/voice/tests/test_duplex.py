from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from voice_fakes import FakeGate, FakeSink, install_audio, make_result, set_tts

from fish_audio_suite_kit import ChatMessage, FishHttpError, LatencySnapshot
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.hearing import HeardLine, classify_transcript, recognize
from fish_audio_suite_voice.history import opening_history, remember_user, trim_history
from fish_audio_suite_voice.live import IsolatedFishTts, TtsResult
from fish_audio_suite_voice.playback import PortAudioMissingError
from fish_audio_suite_voice.reply import (
    _TokenPipe,
    after_speech,
    collect_reply,
    speak_reply,
    stream_turn,
)
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.tune import DEFAULT_HISTORY_TURNS, LlmSettings

HISTORY_TURNS = DEFAULT_HISTORY_TURNS


def _config() -> VoiceCliConfig:
    return VoiceCliConfig(
        fish_api_key="k",
        fish_base="https://api.fish.audio",
        fish_voice_id="voice",
        fish_asr_language="",
        tts_model="s2.1-pro",
        latency="balanced",
        speed=1.0,
        temperature=0.7,
        top_p=0.7,
        repetition_penalty=1.2,
        chunk_length=200,
        min_chunk_length=50,
        volume=0.0,
        sample_rate=44100,
        playback="stdout",
        system_prompt="be brief",
        device=None,
        llm=LlmSettings(backend="openai", base="https://example.test/v1", api_key="lk", model="m"),
    )


class _FakeBackend:
    def __init__(self, tokens: Callable[..., AsyncIterator[str]] | None = None) -> None:
        self._tokens = tokens

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        cancel: asyncio.Event | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[str]:
        if self._tokens is not None:
            return self._tokens(messages, cancel=cancel, trace_id=trace_id)
        return self._empty()

    async def _empty(self) -> AsyncIterator[str]:
        if False:
            yield ""

    async def aclose(self) -> None:
        return None


def _ctx(
    tokens: Callable[..., AsyncIterator[str]] | None = None,
    session: DuplexSession | None = None,
) -> DuplexContext:
    return DuplexContext(
        config=_config(),
        tts=IsolatedFishTts(api_key="k", voice_id="voice"),
        device=None,
        backend=_FakeBackend(tokens),
        session=session or DuplexSession(),
        asr_http=httpx.AsyncClient(),
        history=[{"role": "system", "content": "be brief"}],
    )


def test_classify_transcript_drops_echoes_and_keeps_a_real_line() -> None:
    assert classify_transcript("yeah", "") == "skip"
    assert classify_transcript("thanks for watching", "") == "skip"
    assert classify_transcript("goodbye", "") == "quit"
    assert classify_transcript("hello there", "hello there") == "skip"
    assert classify_transcript("hello there.", "hello there") == "skip"
    assert classify_transcript("Hello There!", "hello there") == "skip"
    assert classify_transcript("bye.", "bye") == "quit"
    assert classify_transcript("I said hello there", "hello there") == "ok"
    assert classify_transcript("hello there", "earlier") == "ok"


def test_history_keeps_the_system_prompt_and_drops_the_oldest_turn() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS * 2 + 1):
        remember_user(history, f"u{i}", HISTORY_TURNS)
    assert history[0] == {"role": "system", "content": "be brief"}
    assert history[1]["content"] == "u1"
    assert history[-1]["content"] == f"u{HISTORY_TURNS * 2}"
    assert len(history) == 1 + HISTORY_TURNS * 2


def test_history_drops_a_whole_turn_so_roles_stay_paired() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS):
        remember_user(history, f"u{i}", HISTORY_TURNS)
        history.append({"role": "assistant", "content": f"a{i}"})
    remember_user(history, "newest", HISTORY_TURNS)
    assert history[1] == {"role": "user", "content": "u1"}
    assert history[2] == {"role": "assistant", "content": "a1"}
    history.append({"role": "assistant", "content": "answer"})
    assert history[-2] == {"role": "user", "content": "newest"}
    assert history[-1] == {"role": "assistant", "content": "answer"}
    assert len(history) == 1 + HISTORY_TURNS * 2


def test_after_speech_records_only_audio_that_was_played() -> None:
    ctx = _ctx()
    snapshot = LatencySnapshot()
    fatal, code = after_speech(
        ctx,
        snapshot,
        make_result(error_status=401, error_message="no"),
        started=0.0,
    )
    assert code == 2
    assert len(ctx.history) == 1
    assert fatal.tts_first_audio_ms is None

    ctx = _ctx()
    after_speech(
        ctx,
        snapshot,
        make_result(cancelled=True),
        started=0.0,
    )
    assert len(ctx.history) == 1

    ctx = _ctx()
    updated, code = after_speech(
        ctx,
        snapshot,
        make_result(
            "hello",
            bytes_played=8,
            got_audio=True,
            cancelled=True,
            tts_first_audio_ms=12.0,
            tts_first_text_ms=4.0,
        ),
        started=0.0,
    )
    assert code is None
    assert ctx.history[-1] == {"role": "assistant", "content": "hello"}
    assert updated.tts_first_audio_ms == 12.0

    ctx = _ctx()
    ctx.history.append({"role": "user", "content": "Hey! Can you hear me?"})
    after_speech(
        ctx,
        snapshot,
        make_result(
            "Hey! Can you hear me?",
            bytes_played=8,
            got_audio=True,
            tts_first_audio_ms=12.0,
            tts_first_text_ms=4.0,
        ),
        started=0.0,
    )
    assert ctx.history[-1]["role"] == "user"
    after_speech(
        ctx,
        snapshot,
        make_result(
            "Yeah. What's up?",
            bytes_played=8,
            got_audio=True,
            tts_first_audio_ms=12.0,
            tts_first_text_ms=4.0,
        ),
        started=0.0,
    )
    assert ctx.history[-1] == {"role": "assistant", "content": "Yeah. What's up?"}


def test_recognize_fatal_again_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx()

    async def denied(*_args: object, **_kwargs: object) -> str:
        raise FishHttpError.from_status(401, "nope")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", denied)
    fatal = asyncio.run(recognize(ctx, b"wav", ""))
    assert fatal.kind == "fatal"
    assert fatal.code == 2

    async def down(*_args: object, **_kwargs: object) -> str:
        raise TimeoutError("late")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", down)
    assert asyncio.run(recognize(ctx, b"wav", "")).kind == "again"

    async def bye_text(*_args: object, **_kwargs: object) -> str:
        return "goodbye"

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", bye_text)
    assert asyncio.run(recognize(ctx, b"wav", "")).kind == "bye"


def test_quit_during_asr_does_not_ask_the_llm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = DuplexSession()
    asked = {"n": 0}

    def wav(*_args: object, **_kwargs: object) -> bytes:
        return b"RIFFwav"

    async def asr(*_args: object, **_kwargs: object) -> str:
        session.quit_requested.set()
        return "hello there friend"

    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        asked["n"] += 1
        if False:
            yield ""

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", wav)
    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", asr)
    code = asyncio.run(
        asyncio.wait_for(
            duplex_turns(
                _config(),
                IsolatedFishTts(api_key="k", voice_id="voice"),
                None,
                _FakeBackend(tokens),
                session,
            ),
            1,
        )
    )
    assert code == 0
    assert asked["n"] == 0
    assert "you:" not in capsys.readouterr().out


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

    ctx = _ctx(tokens)
    monkeypatch.setattr(sys, "stdout", _ClosedStdout())
    reply, ttft = asyncio.run(
        collect_reply(ctx, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
    )
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

    ctx = _ctx(tokens)
    reply, ttft = asyncio.run(collect_reply(ctx, llm_cancel=cancel, trace_id=None, started=0.0))
    assert reply == "Hi"
    assert ttft is not None
    assert "[llm]" not in capsys.readouterr().err


def test_collect_reply_lets_an_outside_cancel_propagate() -> None:
    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hi"
        await asyncio.sleep(30)

    ctx = _ctx(tokens)

    async def run() -> None:
        task = asyncio.create_task(
            collect_reply(ctx, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
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

    ctx = _ctx(tokens)

    async def run() -> None:
        async with asyncio.timeout(0.1):
            await collect_reply(ctx, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)

    with pytest.raises(TimeoutError):
        asyncio.run(run())


def test_speak_reply_exits_when_playback_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()
    install_audio(monkeypatch)

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
    assert len(ctx.history) == 1


def test_cancelled_speak_keeps_the_audio_that_tripped_barge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()

    install_audio(monkeypatch, gate=FakeGate(captured=b"clip"))

    def speak(
        text: str,
        sink: FakeSink,
        cancel: threading.Event | None = None,
        on_first_audio: object = None,
    ) -> TtsResult:
        assert text == "Hello there friend"
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
    assert ctx.barge_prefix == b"clip"
    assert ctx.history[-1]["content"] == "hello"


def test_speak_reply_releases_the_mic_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx()
    gate, _sink = install_audio(monkeypatch, gate=FakeGate(hold=True), sink=FakeSink(played=8))

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
    assert ctx.history[-1]["content"] == "hello there"
    # The watcher waits up to 30 s for the cancel flag, and the caller only joins
    # for about a second, so a thread that is already finished proves the caller
    # set the flag before it returned.
    assert gate.thread is not None
    assert gate.thread.is_alive() is False


def test_history_cap_follows_the_configured_turns() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(5):
        remember_user(history, f"u{i}", 2)
        history.append({"role": "assistant", "content": f"a{i}"})
    remember_user(history, "newest", 2)
    # The oldest user and assistant leave together, so roles stay paired.
    assert [m["content"] for m in history] == ["be brief", "u4", "a4", "newest"]


def test_a_deliberate_repeat_is_kept_but_a_stale_copy_is_dropped() -> None:
    assert classify_transcript("no", "no", stale=False) == "ok"
    assert classify_transcript("No.", "no", stale=False) == "ok"
    assert classify_transcript("no", "no", stale=True) == "skip"
    assert classify_transcript("no", "yes", stale=True) == "ok"


def test_token_pipe_hands_tokens_across_in_order() -> None:
    pipe = _TokenPipe()
    for token in ("Hello ", "there"):
        pipe.push(token)
    pipe.close()

    async def drain() -> list[str]:
        return [token async for token in pipe]

    assert asyncio.run(drain()) == ["Hello ", "there"]


async def _two_tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
    for token in ("Hello ", "there friend."):
        yield token
        await asyncio.sleep(0)


def test_stream_turn_feeds_tokens_to_tts_and_arms_barge_at_first_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(_two_tokens)
    gate, _sink = install_audio(monkeypatch)
    heard_tokens: list[str] = []
    armed_before_audio: list[int] = []

    def speak_stream(
        deltas: _TokenPipe,
        sink: FakeSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
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
    assert ctx.history[-1] == {"role": "assistant", "content": "Hello there friend."}


def test_stream_turn_speaks_the_finished_reply_when_fish_fails_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(_two_tokens)
    install_audio(monkeypatch)
    spoken: list[str] = []

    def speak_stream(
        deltas: _TokenPipe,
        sink: FakeSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        async def drain() -> None:
            async for _token in deltas:
                pass

        asyncio.run(drain())
        return make_result(error_status=503)

    def speak_whole(
        text: str,
        sink: FakeSink,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
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
    assert ctx.history[-1]["role"] == "assistant"


def test_default_prompt_starts_with_a_pinned_multi_cue_exchange() -> None:
    from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT

    history, pinned = opening_history(DEFAULT_SYSTEM_PROMPT)
    assert pinned == 3
    assert [m["role"] for m in history] == ["system", "user", "assistant"]
    assert history[2]["content"].count("[") >= 2


def test_a_custom_prompt_gets_no_seed_exchange() -> None:
    history, pinned = opening_history("You are a pirate.")
    assert pinned == 1
    assert history == [{"role": "system", "content": "You are a pirate."}]


def test_trimming_never_drops_the_pinned_seed() -> None:
    from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT

    history, pinned = opening_history(DEFAULT_SYSTEM_PROMPT)
    for index in range(30):
        remember_user(history, f"question {index}", 3, pinned)
        history.append({"role": "assistant", "content": f"[calm] answer {index}"})
    assert history[:pinned] == opening_history(DEFAULT_SYSTEM_PROMPT)[0]
    assert len(history) == pinned + 3 * 2
    assert history[-2]["content"] == "question 29"


def test_trim_history_still_pairs_user_and_assistant_without_a_seed() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    for index in range(5):
        history.append({"role": "user", "content": f"u{index}"})
        history.append({"role": "assistant", "content": f"a{index}"})
    trim_history(history, 2)
    assert [m["content"] for m in history] == ["s", "u3", "a3", "u4", "a4"]


def test_a_fatal_fish_status_stops_the_llm_but_a_transient_one_does_not() -> None:
    from fish_audio_suite_voice.reply import _end_llm_when_tts_stops

    async def scenario(status: int) -> bool:
        loop = asyncio.get_running_loop()
        done: asyncio.Future[TtsResult] = loop.create_future()
        done.set_result(make_result(error_status=status))
        llm_cancel = asyncio.Event()
        _end_llm_when_tts_stops(done, llm_cancel)
        return llm_cancel.is_set()

    assert asyncio.run(scenario(401)) is True
    assert asyncio.run(scenario(503)) is False


def test_stream_turn_with_an_empty_reply_speaks_nothing_and_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def no_tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        if False:
            yield ""

    ctx = _ctx(no_tokens)
    install_audio(monkeypatch)
    seen_cancel: list[bool] = []

    def speak_stream(
        deltas: _TokenPipe,
        sink: FakeSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
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
    history_before = list(ctx.history)
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
    assert ctx.history == history_before
    assert "silent" not in capsys.readouterr().out


def test_quit_during_the_stream_fallback_rebind_does_not_speak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _ctx(_two_tokens)
    install_audio(monkeypatch)

    def speak_stream(
        deltas: _TokenPipe,
        sink: FakeSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        async def drain() -> None:
            async for _token in deltas:
                pass

        asyncio.run(drain())
        return make_result(error_status=503)

    def never(*_args: object, **_kwargs: object) -> TtsResult:
        raise AssertionError("nothing may be spoken after quit")

    bind = ctx.session.turn.bind

    def bind_then_quit(cancel: threading.Event, llm_cancel: asyncio.Event, *a: object) -> None:
        bind(cancel, llm_cancel)
        # Ctrl+C lands right after the old flag was replaced.
        ctx.session.quit_requested.set()

    monkeypatch.setattr(ctx.session.turn, "bind", bind_then_quit)
    set_tts(monkeypatch, ctx.tts, speak=never, speak_stream=speak_stream)
    _snapshot, code = asyncio.run(
        stream_turn(
            ctx,
            HeardLine("line", text="hi", started=time.perf_counter()),
            threading.Event(),
            asyncio.Event(),
        )
    )
    assert code is None
    assert ctx.session.turn.cancel is not None
    assert ctx.session.turn.cancel.is_set()


def test_recognize_listens_again_after_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx()

    async def unreachable(*_args: object, **_kwargs: object) -> str:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", unreachable)
    assert asyncio.run(recognize(ctx, b"wav", "")).kind == "again"


@pytest.mark.parametrize("bug", [RuntimeError("bug"), ValueError("bad"), KeyError("k")])
def test_recognize_does_not_hide_a_programming_error(
    monkeypatch: pytest.MonkeyPatch, bug: Exception
) -> None:
    ctx = _ctx()

    async def broken(*_args: object, **_kwargs: object) -> str:
        raise bug

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", broken)
    with pytest.raises(type(bug)):
        asyncio.run(recognize(ctx, b"wav", ""))
