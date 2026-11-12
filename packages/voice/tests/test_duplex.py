from __future__ import annotations

import asyncio
import itertools
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from session_fakes import (
    FakeBackend,
    ScriptedHearing,
    barge_clip,
    make_config,
    make_ctx,
    quick_config,
    recognize,
)
from voice_fakes import FakeGate, FakeSink, install_audio, make_result, set_tts

from fish_audio_suite_kit import ChatMessage, FishHttpError
from fish_audio_suite_voice.asr import AsrResult
from fish_audio_suite_voice.config import DEFAULT_HISTORY_TURNS, SystemPrompt
from fish_audio_suite_voice.console_sink import ConsoleSink
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.events import EVENTS
from fish_audio_suite_voice.heard import HeardLine, classify_transcript
from fish_audio_suite_voice.hearing import hear_line
from fish_audio_suite_voice.history import History, trim_history
from fish_audio_suite_voice.latency import LatencySnapshot
from fish_audio_suite_voice.portaudio import PortAudioMissingError
from fish_audio_suite_voice.recognised import BargeIn
from fish_audio_suite_voice.reply import collect_reply, speak_reply, stream_turn
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.wire import TtsResult

HISTORY_TURNS = DEFAULT_HISTORY_TURNS


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


def test_a_lone_yeah_answers_on_its_own_turn_but_not_over_a_reply() -> None:
    # Said over the reply, "yeah" means "go on". On its own turn it answers.
    assert classify_transcript("Yeah.", "", over_reply=True) == "skip"
    assert classify_transcript("Yeah.", "", over_reply=False) == "ok"
    assert classify_transcript("uh-huh", "", over_reply=False) == "ok"
    # A hesitation is never an answer.
    assert classify_transcript("Um.", "", over_reply=False) == "skip"
    assert classify_transcript("hmm", "", over_reply=False) == "skip"


def test_history_keeps_the_system_prompt_and_drops_the_oldest_turns() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS + 1):
        History(history, HISTORY_TURNS).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    assert history[0] == {"role": "system", "content": "be brief"}
    kept = [m["content"] for m in history[1::2]]
    assert kept == [f"u{i}" for i in range(HISTORY_TURNS + 1 - len(kept), HISTORY_TURNS + 1)]
    assert history[-1]["content"] == f"a{HISTORY_TURNS}"
    assert len(history) <= 1 + HISTORY_TURNS * 2


def test_history_drops_whole_turns_so_roles_stay_paired() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS):
        History(history, HISTORY_TURNS).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    History(history, HISTORY_TURNS).remember_user("newest")
    history.append({"role": "assistant", "content": "answer"})
    roles = [m["role"] for m in history[1:]]
    assert roles == ["user", "assistant"] * (len(roles) // 2)
    first = history[1]["content"]
    assert history[2]["content"] == "a" + first.removeprefix("u")
    assert history[-2:] == [
        {"role": "user", "content": "newest"},
        {"role": "assistant", "content": "answer"},
    ]
    assert len(history) <= 1 + HISTORY_TURNS * 2


def test_the_history_is_trimmed_in_blocks_so_its_start_stays_the_same_between_trims() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    floor = HISTORY_TURNS - HISTORY_TURNS // 4
    starts: list[list[ChatMessage]] = []
    for i in range(HISTORY_TURNS * 3):
        History(history, HISTORY_TURNS).remember_user(f"u{i}")
        turns = (len(history) - 1) // 2  # whole pairs before the newest line
        assert turns <= HISTORY_TURNS
        if i >= HISTORY_TURNS:
            assert turns >= floor - 1
        starts.append([*history[:3]])
        history.append({"role": "assistant", "content": f"a{i}"})
    # A provider caches the start of the prompt. Dropping one turn at a time changed it
    # on every turn once the cap was reached. Now it changes once per block.
    changes = sum(1 for before, after in itertools.pairwise(starts) if before != after)
    assert changes <= 2 * HISTORY_TURNS // (HISTORY_TURNS // 4)


def test_a_small_history_cap_still_drops_one_turn_at_a_time() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(4):
        History(history, 3).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    assert [m["content"] for m in history[1:]] == ["u1", "a1", "u2", "a2", "u3", "a3"]


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
    install_audio(monkeypatch)
    ctx = make_ctx()
    fatal, code = _speak_once(monkeypatch, ctx, make_result(error_status=401, error_message="no"))
    assert code == 2
    assert len(ctx.chat.history.messages) == 1
    assert fatal.tts_first_audio_ms is None

    ctx = make_ctx()
    _speak_once(monkeypatch, ctx, make_result(cancelled=True))
    assert len(ctx.chat.history.messages) == 1

    ctx = make_ctx()
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
    assert ctx.chat.history.messages[-1] == {"role": "assistant", "content": "hello"}
    assert updated.tts_first_audio_ms == 12.0

    ctx = make_ctx()
    ctx.chat.history.messages.append({"role": "user", "content": "Hey! Can you hear me?"})
    echo = make_result(
        "Hey! Can you hear me?",
        bytes_played=8,
        got_audio=True,
        tts_first_audio_ms=12.0,
        tts_first_text_ms=4.0,
    )
    _speak_once(monkeypatch, ctx, echo)
    assert ctx.chat.history.messages[-1]["role"] == "user"
    answer = make_result(
        "Yeah. What's up?",
        bytes_played=8,
        got_audio=True,
        tts_first_audio_ms=12.0,
        tts_first_text_ms=4.0,
    )
    _speak_once(monkeypatch, ctx, answer)
    assert ctx.chat.history.messages[-1] == {"role": "assistant", "content": "Yeah. What's up?"}


def test_recognize_fatal_again_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = make_ctx()

    async def denied(*_args: object, **_kwargs: object) -> AsrResult:
        raise FishHttpError(401, "nope")

    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", denied)
    fatal = asyncio.run(recognize(ctx, b"wav", ""))
    assert fatal.kind == "fatal"
    assert fatal.code == 2

    async def down(*_args: object, **_kwargs: object) -> AsrResult:
        raise TimeoutError("late")

    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", down)
    assert asyncio.run(recognize(ctx, b"wav", "")).kind == "again"

    async def bye_text(*_args: object, **_kwargs: object) -> AsrResult:
        return AsrResult("goodbye")

    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", bye_text)
    assert asyncio.run(recognize(ctx, b"wav", "")).kind == "bye"


def test_quit_during_asr_does_not_ask_the_llm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = DuplexSession()
    asked = {"n": 0}

    def wav(*_args: object, **_kwargs: object) -> bytes:
        return b"RIFFwav"

    async def asr(*_args: object, **_kwargs: object) -> AsrResult:
        session.quit_requested.set()
        return AsrResult("hello there friend")

    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        asked["n"] += 1
        if False:
            yield ""

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", wav)
    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", asr)
    code = asyncio.run(
        asyncio.wait_for(
            duplex_turns(
                make_config(),
                FishSpeaker(api_key="k", voice_id="voice"),
                None,
                FakeBackend(tokens),
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
    ctx = make_ctx()
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
    assert len(ctx.chat.history.messages) == 1


def test_cancelled_speak_keeps_the_audio_that_tripped_barge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx()

    gate = FakeGate(captured=b"clip", min_rms=90.0)  # FISH_VOICE_BARGE_RMS=90
    gate.resume_at = 42
    install_audio(monkeypatch, gate=gate)

    def speak(
        text: str,
        sink: FakeSink,
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
    assert ctx.chat.history.messages[-1]["content"] == "hello"


def test_speak_reply_releases_the_mic_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx()
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
    assert ctx.chat.history.messages[-1]["content"] == "hello there"
    # The watcher waits up to 30 s for the cancel flag, and the caller only joins
    # for about a second, so a thread that is already finished proves the caller
    # set the flag before it returned.
    assert gate.thread is not None
    assert gate.thread.is_alive() is False


def test_history_cap_follows_the_configured_turns() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "be brief"}]
    for i in range(5):
        History(history, 2).remember_user(f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    History(history, 2).remember_user("newest")
    # The oldest user and assistant leave together, so roles stay paired.
    assert [m["content"] for m in history] == ["be brief", "u4", "a4", "newest"]


def test_a_deliberate_repeat_is_kept_but_a_stale_copy_is_dropped() -> None:
    assert classify_transcript("no", "no", stale=False) == "ok"
    assert classify_transcript("No.", "no", stale=False) == "ok"
    assert classify_transcript("no", "no", stale=True) == "skip"
    assert classify_transcript("no", "yes", stale=True) == "ok"


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
    install_audio(monkeypatch)
    sent: list[str] = []

    def speak_stream(
        deltas: AsyncIterator[str], sink: FakeSink, cancel: threading.Event, **_kw: object
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
    ctx = make_ctx(_two_tokens)
    gate, _sink = install_audio(monkeypatch)
    heard_tokens: list[str] = []
    armed_before_audio: list[int] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: FakeSink,
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
    assert ctx.chat.history.messages[-1] == {"role": "assistant", "content": "Hello there friend."}


def test_stream_turn_speaks_the_finished_reply_when_fish_fails_before_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx(_two_tokens)
    install_audio(monkeypatch)
    spoken: list[str] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: FakeSink,
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
        sink: FakeSink,
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
    assert ctx.chat.history.messages[-1]["role"] == "assistant"


def _opening(text: str) -> History:
    """Open a session's history for a prompt written out in full, as the config decides it."""
    prompt = SystemPrompt.written(text)
    return History.opening(prompt.text, seed=prompt.pin_seed, turns=DEFAULT_HISTORY_TURNS)


def test_default_prompt_starts_with_a_pinned_multi_cue_exchange() -> None:
    from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT

    opened = _opening(DEFAULT_SYSTEM_PROMPT)
    history, pinned = opened.messages, opened.pinned
    assert pinned == 3
    assert [m["role"] for m in history] == ["system", "user", "assistant"]
    assert history[2]["content"].count("[") >= 2


def test_a_custom_prompt_gets_no_seed_exchange() -> None:
    opened = _opening("You are a pirate.")
    history, pinned = opened.messages, opened.pinned
    assert pinned == 1
    assert history == [{"role": "system", "content": "You are a pirate."}]


def test_trimming_never_drops_the_pinned_seed() -> None:
    from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT

    opened = _opening(DEFAULT_SYSTEM_PROMPT)
    history, pinned = opened.messages, opened.pinned
    for index in range(30):
        History(history, 3, pinned).remember_user(f"question {index}")
        history.append({"role": "assistant", "content": f"[calm] answer {index}"})
    assert history[:pinned] == _opening(DEFAULT_SYSTEM_PROMPT).messages
    assert len(history) == pinned + 3 * 2
    assert history[-2]["content"] == "question 29"


def test_trim_history_still_pairs_user_and_assistant_without_a_seed() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    for index in range(5):
        history.append({"role": "user", "content": f"u{index}"})
        history.append({"role": "assistant", "content": f"a{index}"})
    trim_history(history, 2)
    assert [m["content"] for m in history] == ["s", "u3", "a3", "u4", "a4"]


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
    install_audio(monkeypatch)

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

    ctx = make_ctx(no_tokens)
    install_audio(monkeypatch)
    seen_cancel: list[bool] = []

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: FakeSink,
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
    history_before = list(ctx.chat.history.messages)
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
    assert ctx.chat.history.messages == history_before
    assert "silent" not in capsys.readouterr().out


def test_quit_during_the_stream_fallback_rebind_does_not_speak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = make_ctx(_two_tokens)
    install_audio(monkeypatch)

    def speak_stream(
        deltas: AsyncIterator[str],
        sink: FakeSink,
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
        text: str, sink: FakeSink, cancel: threading.Event | None = None, **_kwargs: object
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


def test_recognize_listens_again_after_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = make_ctx()

    async def unreachable(*_args: object, **_kwargs: object) -> AsrResult:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", unreachable)
    assert asyncio.run(recognize(ctx, b"wav", "")).kind == "again"


@pytest.mark.parametrize("bug", [RuntimeError("bug"), ValueError("bad"), KeyError("k")])
def test_recognize_does_not_hide_a_programming_error(
    monkeypatch: pytest.MonkeyPatch, bug: Exception
) -> None:
    ctx = make_ctx()

    async def broken(*_args: object, **_kwargs: object) -> AsrResult:
        raise bug

    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", broken)
    with pytest.raises(type(bug)):
        asyncio.run(recognize(ctx, b"wav", ""))


def test_an_unanswered_user_line_is_merged_with_the_next_one() -> None:
    # A turn the model never answered (a 429, a refusal from the provider) left its
    # user line last. Two user messages in a row reached the model before.
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    History(history, HISTORY_TURNS).remember_user("Swear on it.")
    History(history, HISTORY_TURNS).remember_user("Swear on it.")
    assert history == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Swear on it."},
    ]
    History(history, HISTORY_TURNS).remember_user("Are you there?")
    assert history[-1] == {"role": "user", "content": "Swear on it.\nAre you there?"}
    assert [m["role"] for m in history] == ["system", "user"]


def test_a_new_line_after_a_reply_is_its_own_message() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    History(history, HISTORY_TURNS).remember_user("hi")
    history.append({"role": "assistant", "content": "hello"})
    History(history, HISTORY_TURNS).remember_user("hi")
    assert [m["role"] for m in history] == ["system", "user", "assistant", "user"]


def test_a_long_run_of_unanswered_lines_keeps_only_the_latest() -> None:
    history: list[ChatMessage] = [{"role": "system", "content": "s"}]
    for i in range(10):
        History(history, HISTORY_TURNS).remember_user(f"line {i}")
    assert len(history) == 2
    assert history[-1]["content"] == "line 6\nline 7\nline 8\nline 9"


def _cut_off_ctx(
    monkeypatch: pytest.MonkeyPatch, spoken: str, tts: FishSpeaker | None = None
) -> tuple[DuplexContext, list[str]]:
    """A context whose first reply is cut off by a barge-in after ``spoken`` was heard.

    Later replies play ``spoken`` to the end. ``tts`` is the speaker to fake, the
    context's own when None.
    """
    ctx = make_ctx()
    install_audio(monkeypatch, gate=FakeGate(captured=b"clip"))
    said: list[str] = []

    def speak(
        text: str,
        sink: FakeSink,
        cancel: threading.Event | None = None,
        on_first_audio: object = None,
        trace_headers: object = None,
    ) -> TtsResult:
        said.append(text)
        return make_result(
            spoken,
            bytes_played=4,
            got_audio=True,
            cancelled=len(said) == 1,
            tts_first_audio_ms=3.0,
            tts_first_text_ms=2.0,
        )

    set_tts(monkeypatch, tts or ctx.tts, speak=speak)
    return ctx, said


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
    assert ctx.turn_input.carry.resume == "[calm] How was your day today?"


async def _cut_off_reply(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
    yield "Hello there. How was your day today?"


def _cut_off_session(
    monkeypatch: pytest.MonkeyPatch, *after: HeardLine
) -> tuple[list[str], list[ChatMessage]]:
    """Run a session whose first reply is cut off after "Hello there.", then hears ``after``."""
    tts = FishSpeaker(api_key="k", voice_id="voice")
    _ctx, said = _cut_off_ctx(monkeypatch, "Hello there.", tts)
    hearing = ScriptedHearing(monkeypatch, HeardLine("line", text="hi", started=0.0), *after)
    session = DuplexSession()
    asyncio.run(duplex_turns(quick_config(), tts, None, FakeBackend(_cut_off_reply), session))
    assert hearing.ctx is not None
    return said, hearing.ctx.chat.history.messages


def test_noise_after_a_barge_in_speaks_the_rest_and_joins_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    said, history = _cut_off_session(monkeypatch, HeardLine("noise"))
    assert said == ["[calm] Hello there. How was your day today?", "[calm] How was your day today?"]
    assert history[-1] == {"role": "assistant", "content": "Hello there. Hello there."}


def test_a_real_line_after_a_barge_in_drops_the_rest_for_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    said, _history = _cut_off_session(
        monkeypatch, HeardLine("line", text="Tell me a joke.", started=0.0), HeardLine("noise")
    )
    # The joke request is answered, and the noise after it must not bring back the reply
    # the user talked over.
    assert said == ["[calm] Hello there. How was your day today?"] * 2


@pytest.mark.parametrize(("barge_prefix", "kind"), [(b"", "line"), (b"clip", "noise")])
def test_hear_line_answers_a_yeah_unless_it_was_said_over_the_reply(
    monkeypatch: pytest.MonkeyPatch, barge_prefix: bytes, kind: str
) -> None:
    ctx = make_ctx()
    if barge_prefix:
        ctx.turn_input.carry.keep(barge_clip(barge_prefix), "")

    def wav(*_args: object, **_kwargs: object) -> bytes:
        return b"RIFFwav"

    async def asr(*_args: object, **_kwargs: object) -> AsrResult:
        return AsrResult("Yeah.")

    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", wav)
    monkeypatch.setattr("fish_audio_suite_voice.asr.fish_asr", asr)
    assert asyncio.run(hear_line(ctx, "")).kind == kind
