"""The turn loop in ``duplex_turns``, driven by scripted listen results and fake audio."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator

import pytest
from session_fakes import ScriptedHearing, heard_line, quick_config, run_session, speak_with_fakes
from voice_fakes import install_audio, set_tts

from fish_audio_suite_voice.duplex_state import DuplexContext
from fish_audio_suite_voice.hearing import HeardLine
from fish_audio_suite_voice.playback import PortAudioMissingError
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.wire import TtsResult


@pytest.mark.parametrize("stream", [False, True])
def test_a_heard_line_is_answered_by_the_llm_and_spoken_then_the_loop_says_bye(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts, stream=stream)
    loop = ScriptedHearing(monkeypatch, heard_line("hi there"))
    assert run_session(quick_config(stream=stream), tts) == 0
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
    speak_with_fakes(monkeypatch, tts)
    loop = ScriptedHearing(
        monkeypatch, heard_line("first"), HeardLine("again"), heard_line("second")
    )
    assert run_session(quick_config(), tts) == 0
    assert loop.last_user == ["", "first", "first", "second"]


def test_a_fatal_listen_step_ends_the_loop_with_its_code(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    ScriptedHearing(monkeypatch, HeardLine("fatal", code=2))
    assert run_session(quick_config(), tts) == 2


def test_an_empty_reply_is_not_spoken_and_the_loop_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def nothing(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        if False:
            yield ""

    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts)
    loop = ScriptedHearing(monkeypatch, heard_line("anyone there"))
    assert run_session(quick_config(), tts, nothing) == 0
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
    said = speak_with_fakes(monkeypatch, tts)
    ScriptedHearing(monkeypatch, heard_line())
    assert run_session(quick_config(), tts, quit_midway, session) == 0
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
    loop = ScriptedHearing(monkeypatch, heard_line("hello"), heard_line("never reached"))
    assert run_session(quick_config(stream=stream), tts) == 2
    assert loop.last_user == [""]


def test_quit_during_the_cooldown_after_a_reply_says_bye_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = DuplexSession()
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    loop = ScriptedHearing(monkeypatch, heard_line("hello"), heard_line("never reached"))
    timer = threading.Timer(0.1, session.quit_requested.set)
    timer.start()
    try:
        assert run_session(quick_config(cooldown_s=30.0), tts, session=session) == 0
    finally:
        timer.cancel()
    assert loop.last_user == [""]


def test_a_barge_in_skips_the_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    heard = ScriptedHearing(monkeypatch)
    lines = iter([heard_line("hello"), HeardLine("bye")])

    async def hear(ctx: DuplexContext, last_user: str) -> HeardLine:
        heard.last_user.append(last_user)
        ctx.barge_prefix = b"clip"  # the user spoke over the reply
        return next(lines)

    monkeypatch.setattr("fish_audio_suite_voice.duplex.hear_line", hear)
    # A 30 s cooldown would trip the 5 s timeout in _run if it were waited out.
    assert run_session(quick_config(cooldown_s=30.0), tts) == 0


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
    assert run_session(quick_config(), tts) == 2


def test_noise_with_nothing_to_resume_just_listens_again(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    said = speak_with_fakes(monkeypatch, tts)
    loop = ScriptedHearing(monkeypatch, HeardLine("noise"), HeardLine("noise"))
    assert run_session(quick_config(), tts) == 0
    assert said == []
    assert loop.last_user == ["", "", ""]


def test_debug_mode_logs_the_turn_summary_instead_of_printing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    monkeypatch.setattr("fish_audio_suite_voice.console_sink.debug_enabled", lambda: True)
    ScriptedHearing(monkeypatch, heard_line())
    assert run_session(quick_config(), tts) == 0
    assert "first audio" not in capsys.readouterr().out
