"""The turn loop in ``duplex_turns``, driven by scripted listen results and fake audio."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

import pytest
from test_duplex import _config, _FakeBackend  # pyright: ignore[reportPrivateUsage]
from voice_fakes import FakeSink, install_audio, make_result, set_tts

from fish_audio_suite_voice.config import VoiceCliConfig
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.inputs import TurnSource
from fish_audio_suite_voice.playback import PortAudioMissingError
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.wire import TtsResult


def _quick(*, stream: bool = False, cooldown_s: float = 0.0) -> VoiceCliConfig:
    cfg = _config()
    return replace(cfg, stream_tts=stream, barge=replace(cfg.barge, cooldown_s=cooldown_s))


async def _hello(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
    for token in ("Hello ", "there friend."):
        yield token


class _Loop:
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


def _run(
    cfg: VoiceCliConfig,
    tts: FishSpeaker,
    tokens: Callable[..., AsyncIterator[str]] | None = _hello,
    session: DuplexSession | None = None,
) -> int:
    return _run_with(cfg, tts, tokens, None, session)


def _run_with(
    cfg: VoiceCliConfig,
    tts: FishSpeaker,
    tokens: Callable[..., AsyncIterator[str]] | None,
    source: TurnSource | None,
    session: DuplexSession | None = None,
) -> int:
    return asyncio.run(
        asyncio.wait_for(
            duplex_turns(cfg, tts, None, _FakeBackend(tokens), session, source=source), 5
        )
    )


def _line(text: str = "hi there") -> HeardLine:
    return HeardLine("line", text=text)


def _spoken_by(
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


@pytest.mark.parametrize("stream", [False, True])
def test_a_heard_line_is_answered_by_the_llm_and_spoken_then_the_loop_says_bye(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    said = _spoken_by(monkeypatch, tts, stream=stream)
    loop = _Loop(monkeypatch, _line("hi there"))
    assert _run(_quick(stream=stream), tts) == 0
    assert said == ["Hello there friend."]
    assert loop.ctx is not None
    assert loop.ctx.history[-2:] == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello there friend."},
    ]
    assert "first audio" in capsys.readouterr().out


def test_each_listen_step_is_told_the_previous_line(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    _spoken_by(monkeypatch, tts)
    loop = _Loop(monkeypatch, _line("first"), HeardLine("again"), _line("second"))
    assert _run(_quick(), tts) == 0
    assert loop.last_user == ["", "first", "first", "second"]


def test_a_fatal_listen_step_ends_the_loop_with_its_code(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    _Loop(monkeypatch, HeardLine("fatal", code=2))
    assert _run(_quick(), tts) == 2


def test_an_empty_reply_is_not_spoken_and_the_loop_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def nothing(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        if False:
            yield ""

    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    said = _spoken_by(monkeypatch, tts)
    loop = _Loop(monkeypatch, _line("anyone there"))
    assert _run(_quick(), tts, nothing) == 0
    assert said == []
    assert loop.ctx is not None
    assert loop.ctx.history[-1] == {"role": "user", "content": "anyone there"}


def test_quit_while_the_llm_answers_ends_the_loop_without_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = DuplexSession()

    async def quit_midway(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hello "
        session.quit_requested.set()
        yield "there."

    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    said = _spoken_by(monkeypatch, tts)
    _Loop(monkeypatch, _line())
    assert _run(_quick(), tts, quit_midway, session) == 0
    assert said == []


@pytest.mark.parametrize("stream", [False, True])
def test_playback_that_cannot_open_ends_the_loop_with_code_2(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)

    def broken(*_args: object, **_kwargs: object) -> TtsResult:
        raise PortAudioMissingError("missing")

    set_tts(monkeypatch, tts, speak=broken, speak_stream=broken)
    loop = _Loop(monkeypatch, _line("hello"), _line("never reached"))
    assert _run(_quick(stream=stream), tts) == 2
    assert loop.last_user == [""]


def test_quit_during_the_cooldown_after_a_reply_says_bye_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = DuplexSession()
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    _spoken_by(monkeypatch, tts)
    loop = _Loop(monkeypatch, _line("hello"), _line("never reached"))
    timer = threading.Timer(0.1, session.quit_requested.set)
    timer.start()
    try:
        assert _run(_quick(cooldown_s=30.0), tts, session=session) == 0
    finally:
        timer.cancel()
    assert loop.last_user == [""]


def test_a_barge_in_skips_the_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    _spoken_by(monkeypatch, tts)
    heard = _Loop(monkeypatch)
    lines = iter([_line("hello"), HeardLine("bye")])

    async def hear(ctx: DuplexContext, last_user: str) -> HeardLine:
        heard.last_user.append(last_user)
        ctx.barge_prefix = b"clip"  # the user spoke over the reply
        return next(lines)

    monkeypatch.setattr("fish_audio_suite_voice.duplex.hear_line", hear)
    # A 30 s cooldown would trip the 5 s timeout in _run if it were waited out.
    assert _run(_quick(cooldown_s=30.0), tts) == 0


def test_a_failing_resume_ends_the_loop_with_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)

    def broken(*_args: object, **_kwargs: object) -> TtsResult:
        raise PortAudioMissingError("missing")

    set_tts(monkeypatch, tts, speak=broken)
    lines = iter([HeardLine("noise"), HeardLine("bye")])

    async def hear(ctx: DuplexContext, _last: str) -> HeardLine:
        ctx.resume_text = "the rest of the reply"
        return next(lines)

    monkeypatch.setattr("fish_audio_suite_voice.duplex.hear_line", hear)
    assert _run(_quick(), tts) == 2


def test_noise_with_nothing_to_resume_just_listens_again(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    said = _spoken_by(monkeypatch, tts)
    loop = _Loop(monkeypatch, HeardLine("noise"), HeardLine("noise"))
    assert _run(_quick(), tts) == 0
    assert said == []
    assert loop.last_user == ["", "", ""]


def test_debug_mode_logs_the_turn_summary_instead_of_printing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    _spoken_by(monkeypatch, tts)
    monkeypatch.setattr("fish_audio_suite_voice.console_sink.debug_enabled", lambda: True)
    _Loop(monkeypatch, _line())
    assert _run(_quick(), tts) == 0
    assert "first audio" not in capsys.readouterr().out
