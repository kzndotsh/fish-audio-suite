"""Fakes for driving a whole voice session without a mic, a speaker or a network.

``make_ctx`` builds a context through the session's own assembly, ``ScriptedTurns`` scripts
where each turn comes from, ``ScriptedRecordings`` and ``ScriptedAsr`` stand in for the mic
and Fish ASR at their boundaries, ``run_session`` runs ``duplex_turns`` to its end, and
``speak_with_fakes`` stands in for Fish TTS so a test can read what was spoken. Test modules
share them from here, instead of importing each other's private helpers.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any, Final, Literal

import httpx
import pytest
from voice_fakes import ScriptedMic, make_result, set_tts

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.asr.outcome import HeardLine
from fish_audio_suite_voice.asr.result import BargeIn
from fish_audio_suite_voice.cli.console_sink import ConsoleSink
from fish_audio_suite_voice.config.settings import SystemPrompt, VoiceCliConfig
from fish_audio_suite_voice.config.tune import DEFAULT_BARGE_RMS
from fish_audio_suite_voice.llm.tune import LlmTune
from fish_audio_suite_voice.session.cancel import StopFlag
from fish_audio_suite_voice.session.commands import ChatCommand
from fish_audio_suite_voice.session.duplex import duplex_turns
from fish_audio_suite_voice.session.events import EVENTS
from fish_audio_suite_voice.session.inputs import Hear, TurnSource
from fish_audio_suite_voice.session.signals import DuplexSession
from fish_audio_suite_voice.session.state import (
    DuplexContext,
    TurnInput,
    session_context,
    session_speaker,
)
from fish_audio_suite_voice.tts.playback import PlaybackSink
from fish_audio_suite_voice.tts.speaker import FishSpeaker
from fish_audio_suite_voice.tts.tune import TtsTune
from fish_audio_suite_voice.tts.websocket import TtsResult


def make_config() -> VoiceCliConfig:
    return VoiceCliConfig(
        fish_api_key="k",
        fish_base="https://api.fish.audio",
        fish_voice_id="voice",
        fish_asr_language="",
        tts_model="s2.1-pro",
        tts=TtsTune(latency="balanced", temperature=0.7, top_p=0.7, repetition_penalty=1.2),
        playback="stdout",
        prompt=SystemPrompt("be brief", pin_seed=False),  # as for a prompt written out in full
        device=None,
        llm=LlmTune(base="https://example.test/v1", api_key="lk", model="m"),
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
    tokens: Callable[..., AsyncIterator[str]] | None = None, *, asr: ScriptedAsr | None = None
) -> DuplexContext:
    """Build a session's context as ``duplex_turns`` does, with a scripted mic and LLM.

    ``asr`` answers the Fish ASR requests the context's client makes. Without it the client
    is a plain one, which the blocked network refuses.
    """
    config = make_config()
    return session_context(
        config,
        session=DuplexSession(),
        open_mic=lambda aec: ScriptedMic(aec=aec),
        backend=FakeBackend(tokens),
        tts=session_speaker(config),
        asr_http=asr.client() if asr is not None else httpx.AsyncClient(),
    )


def quick_config(*, stream: bool = False, cooldown_s: float = 0.0) -> VoiceCliConfig:
    cfg = make_config()
    return replace(cfg, stream_tts=stream, barge=replace(cfg.barge, cooldown_s=cooldown_s))


async def hello_tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
    for token in ("Hello ", "there friend."):
        yield token


MIC: Final = "mic"
"""A scripted turn that listens with the session's own ``hear``: the real listen."""

type ScriptedTurn = (
    HeardLine | Literal["mic"] | Callable[[TurnInput], HeardLine | Awaitable[HeardLine]]
)


class ScriptedTurns:
    """A ``TurnSource`` that plays a script of turns, then says bye.

    Each turn is a ``HeardLine``, ``MIC`` to listen with the session's own ``hear`` (the
    real listen, with the mic and Fish ASR scripted at their boundaries), or a function of
    the session's ``TurnInput`` that returns one, to act on the mic side first: keep a
    barge-in clip, leave a reply to resume, quit. ``asked`` counts the turns the loop asked
    for, and ``muted`` is the mute switch the session reads. A step may add to ``commands``,
    the changes to the chat the loop takes before it asks for the next turn.
    """

    def __init__(self, *turns: ScriptedTurn) -> None:
        self._turns = list(turns)
        self.asked = 0
        self.muted = False
        self.commands: list[ChatCommand] = []

    def take_commands(self) -> list[ChatCommand]:
        commands, self.commands = self.commands, []
        return commands

    async def next_turn(self, turn: TurnInput, hear: Hear) -> HeardLine:
        self.asked += 1
        step = self._turns.pop(0) if self._turns else HeardLine("bye")
        if isinstance(step, HeardLine):
            return step
        if isinstance(step, str):
            return await hear(threading.Event())
        got = step(turn)
        return await got if inspect.isawaitable(got) else got


def run_session(
    cfg: VoiceCliConfig,
    tts: FishSpeaker,
    tokens: Callable[..., AsyncIterator[str]] | None = hello_tokens,
    session: DuplexSession | None = None,
    *,
    source: TurnSource | None = None,
) -> int:
    """Run ``duplex_turns`` to its end with the plain display printing it, as ``fish-voice`` does."""
    return asyncio.run(asyncio.wait_for(_with_console(cfg, tts, tokens, source, session), 5))


async def _with_console(
    cfg: VoiceCliConfig,
    tts: FishSpeaker,
    tokens: Callable[..., AsyncIterator[str]] | None,
    source: TurnSource | None,
    session: DuplexSession | None,
) -> int:
    with ConsoleSink(EVENTS):
        return await duplex_turns(cfg, tts, None, FakeBackend(tokens), session, source=source)


WAIT_FOR_STOP: Final = "wait"
"""A scripted recording that keeps listening until its stop flag is set, then hears nothing."""

type Recording = bytes | BaseException | Literal["wait"] | Callable[[StopFlag], bytes | None] | None


class ScriptedRecordings:
    """The batch listen's recordings, scripted at the mic boundary (``record_utterance``).

    Each recording plays the next step: the clip it returns, None for one that heard
    nothing, an exception it raises, ``WAIT_FOR_STOP``, or a function of its stop flag.
    Once the script is used up every recording plays ``then``. ``calls`` keeps each
    recording's stop flag and its keyword arguments (``carried``, ``tune``,
    ``give_up_frames``).
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *steps: Recording, then: Recording = b""
    ) -> None:
        self._steps = list(steps)
        self._then = then
        self.calls: list[dict[str, Any]] = []
        monkeypatch.setattr("fish_audio_suite_voice.asr.pipeline.record_utterance", self._record)

    def _record(self, mic: object, stop: StopFlag, **kwargs: Any) -> bytes | None:
        del mic
        self.calls.append({"stop": stop, **kwargs})
        step = self._steps.pop(0) if self._steps else self._then
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, str):
            while not stop.is_set():
                time.sleep(0.001)
            return None
        return step(stop) if callable(step) else step


type AsrReply = str | int | BaseException
"""How Fish ASR answers one request: the raw transcript (``[laughter]`` and the like are
Fish's tone annotations), an HTTP error status, or what the request raises."""


class ScriptedAsr:
    """Fish ASR at the network boundary, answering the requests the real ``fish_asr`` sends.

    Replies are used in order, the last one for every request after it (as each retry
    of a failure asks again), or looked up by the clip a request carries when ``by_clip``
    is given. A reply may be a function, asked when the request arrives. Each request waits
    ``delay_s`` first, as a real one takes a while. The Fish backoff between retries is
    zero, so a retried status or network error costs no time. ``clips`` keeps the audio of
    each request.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *replies: AsrReply | Callable[[], AsrReply],
        by_clip: Mapping[bytes, AsrReply | Callable[[], AsrReply]] | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._replies = list(replies)
        self._by_clip = dict(by_clip or {})
        self._delay_s = delay_s
        self._monkeypatch = monkeypatch
        self.clips: list[bytes] = []
        monkeypatch.setattr(
            "fish_audio_suite_voice.session.pause.fish_backoff_s", lambda *_a, **_k: 0.0
        )

    async def _answer(self, request: httpx.Request) -> httpx.Response:
        clip = _clip_in(request.content)
        self.clips.append(clip)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._by_clip:
            reply = self._by_clip[clip]
        else:
            reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if callable(reply):
            reply = reply()
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"status": reply, "message": "scripted failure"})
        return httpx.Response(200, json={"text": reply})

    def client(self) -> httpx.AsyncClient:
        """Return an ASR client whose requests this script answers."""
        return httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

    def serve_sessions(self) -> None:
        """Answer the ASR client that a session run through ``duplex_turns`` opens."""
        self._monkeypatch.setattr("fish_audio_suite_voice.session.duplex.asr_client", self.client)


def _clip_in(body: bytes) -> bytes:
    """Return the WAV in a Fish ASR multipart request body."""
    part = body.index(b'filename="utterance.wav"')
    start = body.index(b"\r\n\r\n", part) + 4
    return body[start : body.index(b"\r\n--", start)]


def heard_line(text: str = "hi there") -> HeardLine:
    return HeardLine("line", text=text)


def barge_clip(pcm: bytes = b"clip", since: int | None = None) -> BargeIn:
    """A barge-in clip as the reply keeps it, admitted at the default gate level."""
    return BargeIn(pcm, since, DEFAULT_BARGE_RMS)


def speak_with_fakes(
    monkeypatch: pytest.MonkeyPatch, tts: FishSpeaker, *, stream: bool = False
) -> list[str]:
    said: list[str] = []

    def speak(
        text: str,
        sink: PlaybackSink,
        cancel: object = None,
        on_first_audio: object = None,
        trace_headers: object = None,
    ) -> TtsResult:
        said.append(text)
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    def speak_stream(
        deltas: object,
        sink: PlaybackSink,
        cancel: object,
        on_first_audio: object = None,
        trace_headers: object = None,
    ) -> TtsResult:
        async def drain() -> str:
            return "".join([token async for token in deltas])  # type: ignore[attr-defined]

        text = asyncio.run(drain())
        said.append(text)
        return make_result(text, bytes_played=4, got_audio=True, tts_first_audio_ms=3.0)

    set_tts(monkeypatch, tts, speak=speak, speak_stream=speak_stream if stream else None)
    return said


def cut_off_speech(monkeypatch: pytest.MonkeyPatch, tts: FishSpeaker, spoken: str) -> list[str]:
    """Stand in for Fish so the first reply is cut off by a barge-in after ``spoken`` was heard.

    Later replies play ``spoken`` to the end. Returns the list each text spoken is added to.
    """
    said: list[str] = []

    def speak(
        text: str,
        sink: PlaybackSink,
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

    set_tts(monkeypatch, tts, speak=speak)
    return said
