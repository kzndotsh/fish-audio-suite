"""Fakes for driving a whole voice session without a mic, a speaker or a network.

``make_ctx`` builds a context, ``ScriptedHearing`` scripts what the mic heard,
``run_session`` runs ``duplex_turns`` to its end, and ``speak_with_fakes`` stands in for
Fish so a test can read what was spoken. Test modules share them from here, instead of
importing each other's private helpers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

import httpx
import pytest
from voice_fakes import FakeSink, make_result, set_tts

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.inputs import TurnSource
from fish_audio_suite_voice.llm_tune import LlmTune
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.wire import TtsResult


def make_config() -> VoiceCliConfig:
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
        llm=LlmTune(backend="openai", base="https://example.test/v1", api_key="lk", model="m"),
    )


class FakeBackend:
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


def make_ctx(
    tokens: Callable[..., AsyncIterator[str]] | None = None,
    session: DuplexSession | None = None,
) -> DuplexContext:
    return DuplexContext(
        config=make_config(),
        tts=FishSpeaker(api_key="k", voice_id="voice"),
        device=None,
        backend=FakeBackend(tokens),
        session=session or DuplexSession(),
        asr_http=httpx.AsyncClient(),
        history=[{"role": "system", "content": "be brief"}],
    )


def quick_config(*, stream: bool = False, cooldown_s: float = 0.0) -> VoiceCliConfig:
    cfg = make_config()
    return replace(cfg, stream_tts=stream, barge=replace(cfg.barge, cooldown_s=cooldown_s))


async def hello_tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
    for token in ("Hello ", "there friend."):
        yield token


class ScriptedHearing:
    """Scripted ``hear_line`` results, and what the loop asked of each."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *lines: HeardLine) -> None:
        self._lines = iter([*lines, HeardLine("bye")])
        self.last_user: list[str] = []
        self.ctx: DuplexContext | None = None
        monkeypatch.setattr("fish_audio_suite_voice.duplex.hear_line", self._hear)

    async def _hear(self, ctx: DuplexContext, last_user: str) -> HeardLine:
        self.ctx = ctx
        self.last_user.append(last_user)
        return next(self._lines)


def run_session(
    cfg: VoiceCliConfig,
    tts: FishSpeaker,
    tokens: Callable[..., AsyncIterator[str]] | None = hello_tokens,
    session: DuplexSession | None = None,
) -> int:
    return run_session_with(cfg, tts, tokens, None, session)


def run_session_with(
    cfg: VoiceCliConfig,
    tts: FishSpeaker,
    tokens: Callable[..., AsyncIterator[str]] | None,
    source: TurnSource | None,
    session: DuplexSession | None = None,
) -> int:
    return asyncio.run(
        asyncio.wait_for(
            duplex_turns(cfg, tts, None, FakeBackend(tokens), session, source=source), 5
        )
    )


def heard_line(text: str = "hi there") -> HeardLine:
    return HeardLine("line", text=text)


def speak_with_fakes(
    monkeypatch: pytest.MonkeyPatch, tts: FishSpeaker, *, stream: bool = False
) -> list[str]:
    said: list[str] = []

    def speak(
        text: str, sink: FakeSink, cancel: object = None, on_first_audio: object = None
    ) -> TtsResult:
        said.append(text)
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    def speak_stream(
        deltas: object, sink: FakeSink, cancel: object, on_first_audio: object = None
    ) -> TtsResult:
        async def drain() -> str:
            return "".join([token async for token in deltas])  # type: ignore[attr-defined]

        text = asyncio.run(drain())
        said.append(text)
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    set_tts(monkeypatch, tts, speak=speak, speak_stream=speak_stream if stream else None)
    return said
