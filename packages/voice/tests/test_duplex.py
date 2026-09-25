from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest

from fish_audio_suite_kit import FishHttpError, LatencySnapshot
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.duplex import (
    HISTORY_TURNS,
    _accept_asr,
    _after_speech,
    _collect_reply,
    _Loop,
    _recognize,
    _remember_user,
    _speak_reply,
    duplex_turns,
)
from fish_audio_suite_voice.live import IsolatedFishTts, IsolatedResult
from fish_audio_suite_voice.playback import PortAudioMissingError
from fish_audio_suite_voice.signals import STOP_RECORD


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
        llm_backend="openrouter",
        llm_base="https://example.test/v1",
        llm_key="lk",
        llm_model="m",
    )


def _loop() -> _Loop:
    return _Loop(
        config=_config(),
        tts=IsolatedFishTts(api_key="k", voice_id="voice"),
        device=None,
        or_client=None,
        history=[{"role": "system", "content": "be brief"}],
        llm_session="s",
    )


def test_accept_asr_drops_echoes_and_keeps_a_real_line() -> None:
    assert _accept_asr("yeah", "") == "skip"
    assert _accept_asr("thanks for watching", "") == "skip"
    assert _accept_asr("goodbye", "") == "quit"
    assert _accept_asr("hello there", "hello there") == "skip"
    assert _accept_asr("hello there.", "hello there") == "skip"
    assert _accept_asr("Hello There!", "hello there") == "skip"
    assert _accept_asr("bye.", "bye") == "quit"
    assert _accept_asr("I said hello there", "hello there") == "ok"
    assert _accept_asr("hello there", "earlier") == "ok"


def test_history_keeps_the_system_prompt_and_drops_the_oldest_turn() -> None:
    history = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS * 2 + 1):
        _remember_user(history, f"u{i}")
    assert history[0] == {"role": "system", "content": "be brief"}
    assert history[1]["content"] == "u1"
    assert history[-1]["content"] == f"u{HISTORY_TURNS * 2}"
    assert len(history) == 1 + HISTORY_TURNS * 2


def test_history_drops_a_whole_turn_so_roles_stay_paired() -> None:
    history = [{"role": "system", "content": "be brief"}]
    for i in range(HISTORY_TURNS):
        _remember_user(history, f"u{i}")
        history.append({"role": "assistant", "content": f"a{i}"})
    _remember_user(history, "newest")
    assert history[1] == {"role": "user", "content": "u1"}
    assert history[2] == {"role": "assistant", "content": "a1"}
    history.append({"role": "assistant", "content": "answer"})
    assert history[-2] == {"role": "user", "content": "newest"}
    assert history[-1] == {"role": "assistant", "content": "answer"}
    assert len(history) == 1 + HISTORY_TURNS * 2


def test_after_speech_records_only_audio_that_was_played() -> None:
    loop = _loop()
    snapshot = LatencySnapshot()
    fatal, code = _after_speech(
        loop,
        snapshot,
        IsolatedResult("", 0, False, False, None, None, error_status=401, error_message="no"),
        started=0.0,
    )
    assert code == 2
    assert len(loop.history) == 1
    assert fatal.ttfa is None

    loop = _loop()
    _after_speech(
        loop,
        snapshot,
        IsolatedResult("", 0, False, True, None, None),
        started=0.0,
    )
    assert len(loop.history) == 1

    loop = _loop()
    updated, code = _after_speech(
        loop,
        snapshot,
        IsolatedResult("hello", 8, True, True, 12.0, 4.0),
        started=0.0,
    )
    assert code is None
    assert loop.history[-1] == {"role": "assistant", "content": "hello"}
    assert updated.ttfa == 12.0

    loop = _loop()
    loop.history.append({"role": "user", "content": "Hey! Can you hear me?"})
    _after_speech(
        loop,
        snapshot,
        IsolatedResult("Hey! Can you hear me?", 8, True, False, 12.0, 4.0),
        started=0.0,
    )
    assert loop.history[-1]["role"] == "user"
    _after_speech(
        loop,
        snapshot,
        IsolatedResult("Yeah. What's up?", 8, True, False, 12.0, 4.0),
        started=0.0,
    )
    assert loop.history[-1] == {"role": "assistant", "content": "Yeah. What's up?"}


def test_recognize_fatal_again_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _loop()

    async def denied(*_args: object, **_kwargs: object) -> str:
        raise FishHttpError(401, "nope")

    monkeypatch.setattr("fish_audio_suite_voice.duplex.fish_asr", denied)
    fatal = asyncio.run(_recognize(loop, b"wav", ""))
    assert fatal.kind == "fatal"
    assert fatal.code == 2

    async def down(*_args: object, **_kwargs: object) -> str:
        raise TimeoutError("late")

    monkeypatch.setattr("fish_audio_suite_voice.duplex.fish_asr", down)
    assert asyncio.run(_recognize(loop, b"wav", "")).kind == "again"

    async def bye_text(*_args: object, **_kwargs: object) -> str:
        return "goodbye"

    monkeypatch.setattr("fish_audio_suite_voice.duplex.fish_asr", bye_text)
    assert asyncio.run(_recognize(loop, b"wav", "")).kind == "bye"


def test_quit_during_asr_does_not_ask_the_llm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    STOP_RECORD.clear()
    asked = {"n": 0}

    def wav(*_args: object, **_kwargs: object) -> bytes:
        return b"RIFFwav"

    async def asr(*_args: object, **_kwargs: object) -> str:
        STOP_RECORD.set()
        return "hello there friend"

    async def tokens(*_args: object, **_kwargs: object):
        asked["n"] += 1
        if False:
            yield ""

    monkeypatch.setattr("fish_audio_suite_voice.duplex.record_utterance", wav)
    monkeypatch.setattr("fish_audio_suite_voice.duplex.fish_asr", asr)
    monkeypatch.setattr("fish_audio_suite_voice.duplex.llm_token_stream", tokens)
    try:
        code = asyncio.run(
            asyncio.wait_for(
                duplex_turns(_config(), IsolatedFishTts(api_key="k", voice_id="voice"), None, None),
                1,
            )
        )
    finally:
        STOP_RECORD.clear()
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
    loop = _loop()

    async def tokens(*_args: object, **_kwargs: object):
        yield "Open the door today friend."

    monkeypatch.setattr("fish_audio_suite_voice.duplex.llm_token_stream", tokens)
    monkeypatch.setattr(sys, "stdout", _ClosedStdout())
    reply, ttft = asyncio.run(
        _collect_reply(loop, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
    )
    assert reply == "Open the door today friend."
    assert ttft is not None


def test_collect_reply_keeps_partial_text_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loop = _loop()

    async def tokens(*_args: object, **_kwargs: object):
        yield "Hi"
        raise asyncio.CancelledError

    monkeypatch.setattr("fish_audio_suite_voice.duplex.llm_token_stream", tokens)
    reply, ttft = asyncio.run(
        _collect_reply(loop, llm_cancel=asyncio.Event(), trace_id=None, started=0.0)
    )
    assert reply == "Hi"
    assert ttft is not None
    assert "[llm]" not in capsys.readouterr().err


def test_speak_reply_exits_when_playback_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop()

    class Gate:
        captured = b""

        def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
            thread = threading.Thread(target=lambda: None)
            thread.start()
            return thread

    monkeypatch.setattr("fish_audio_suite_voice.duplex.BargeGate", lambda **_kwargs: Gate())
    monkeypatch.setattr("fish_audio_suite_voice.duplex.make_sink", lambda *_a, **_k: object())

    def speak(*_args: object, **_kwargs: object) -> IsolatedResult:
        raise PortAudioMissingError("missing")

    loop.tts.speak_isolated = speak
    _snapshot, code = asyncio.run(
        _speak_reply(
            loop,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id=None,
        )
    )
    assert code == 2
    assert len(loop.history) == 1


def test_cancelled_speak_keeps_the_audio_that_tripped_barge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop()

    class Gate:
        captured = b"clip"

        def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
            thread = threading.Thread(target=lambda: None)
            thread.start()
            return thread

    class Sink:
        def start(self) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            return None

        def finish(self, *, kill: bool = False) -> None:
            return None

        def bytes_played(self) -> int:
            return 0

    monkeypatch.setattr("fish_audio_suite_voice.duplex.BargeGate", lambda **_kwargs: Gate())
    monkeypatch.setattr("fish_audio_suite_voice.duplex.make_sink", lambda *_a, **_k: Sink())

    def speak(
        text: str,
        sink: Sink,
        cancel: threading.Event | None = None,
    ) -> IsolatedResult:
        assert text.startswith("[")
        return IsolatedResult("hello", 4, True, True, 3.0, 2.0)

    loop.tts.speak_isolated = speak
    _snapshot, code = asyncio.run(
        _speak_reply(
            loop,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id="abc",
        )
    )
    assert code is None
    assert loop.barge_prefix == b"clip"
    assert loop.history[-1]["content"] == "hello"


def test_speak_reply_releases_the_mic_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = _loop()
    held: dict[str, threading.Thread] = {}

    class Gate:
        captured = b""

        def start_after_bleed(self, cancel: threading.Event) -> threading.Thread:
            def _run() -> None:
                cancel.wait(timeout=30)

            thread = threading.Thread(target=_run)
            held["thread"] = thread
            thread.start()
            return thread

    class Sink:
        def start(self) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            return None

        def finish(self, *, kill: bool = False) -> None:
            return None

        def bytes_played(self) -> int:
            return 8

    monkeypatch.setattr("fish_audio_suite_voice.duplex.BargeGate", lambda **_kwargs: Gate())
    monkeypatch.setattr("fish_audio_suite_voice.duplex.make_sink", lambda *_a, **_k: Sink())

    def speak(*_args: object, **_kwargs: object) -> IsolatedResult:
        return IsolatedResult("hello there", 8, True, False, 3.0, 2.0)

    loop.tts.speak_isolated = speak
    started = time.monotonic()
    _snapshot, code = asyncio.run(
        _speak_reply(
            loop,
            "Hello there friend",
            threading.Event(),
            LatencySnapshot(),
            started=0.0,
            trace_id=None,
        )
    )
    assert code is None
    assert loop.history[-1]["content"] == "hello there"
    assert held["thread"].is_alive() is False
    assert time.monotonic() - started < 1
