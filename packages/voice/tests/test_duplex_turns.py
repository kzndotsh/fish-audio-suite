"""The turn loop in ``duplex_turns``, driven by scripted listen results and fake audio."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from typing import cast

import pytest
from session_fakes import (
    MIC,
    AsrReply,
    FakeBackend,
    ScriptedAsr,
    ScriptedRecordings,
    ScriptedTurns,
    barge_clip,
    cut_off_speech,
    heard_line,
    hello_tokens,
    make_config,
    quick_config,
    run_session,
    speak_with_fakes,
)
from voice_fakes import FakeGate, ScriptedMic, install_gate, install_vad, set_tts

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice import reply
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.duplex_state import TurnInput
from fish_audio_suite_voice.events import ExitCode
from fish_audio_suite_voice.heard import HeardLine
from fish_audio_suite_voice.log_setup import configure_voice_logging
from fish_audio_suite_voice.portaudio import PortAudioMissingError
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.transports import ChatUnusableError
from fish_audio_suite_voice.tune import ListenTune
from fish_audio_suite_voice.wire import TtsResult


@pytest.mark.parametrize("stream", [False, True])
def test_a_heard_line_is_answered_by_the_llm_and_spoken_then_the_loop_says_bye(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts, stream=stream)
    asked: list[list[ChatMessage]] = []

    async def tokens(messages: list[ChatMessage], **kwargs: object) -> AsyncIterator[str]:
        asked.append(copy.deepcopy(messages))
        async for token in hello_tokens(**kwargs):
            yield token

    source = ScriptedTurns(heard_line("hi there"), heard_line("and then?"))
    assert run_session(quick_config(stream=stream), tts, tokens, source=source) == 0
    # Only the whole-reply path gives the line a lead cue itself. A stream gets one from the scrubber.
    assert said == ["Hello there friend." if stream else "[calm] Hello there friend."] * 2
    # The next request carries the line and what was spoken of its reply.
    assert asked[1][-3:] == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": said[0]},
        {"role": "user", "content": "and then?"},
    ]
    assert "first audio" in capsys.readouterr().out


def test_each_listen_step_is_told_the_previous_line(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts)
    ScriptedRecordings(monkeypatch, then=b"wav")
    # A copy of the last line heard as listening opens is stale and dropped, so each copy
    # below is dropped only if that listen was told the line before it. A failed request
    # ("again", the 400) does not change which line that is.
    ScriptedAsr(monkeypatch, "first", 400, "First.", "second", "second").serve_sessions()
    source = ScriptedTurns(MIC, MIC, MIC, MIC, MIC)
    assert run_session(quick_config(), tts, source=source) == 0
    assert source.asked == 6
    assert len(said) == 2  # "first" and "second" are answered, their copies are not


def test_a_fatal_listen_step_ends_the_loop_with_its_code(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    source = ScriptedTurns(HeardLine("fatal", code=ExitCode.FATAL))
    assert run_session(quick_config(), tts, source=source) == 2


def test_an_empty_reply_is_not_spoken_and_the_loop_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[list[ChatMessage]] = []

    async def nothing(messages: list[ChatMessage], **_kwargs: object) -> AsyncIterator[str]:
        asked.append(copy.deepcopy(messages))
        if False:
            yield ""

    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts)
    source = ScriptedTurns(heard_line("anyone there"), heard_line("hello?"))
    assert run_session(quick_config(), tts, nothing, source=source) == 0
    assert said == []
    # The line stayed last with no reply after it, so the next one joins it.
    assert asked[1][-1] == {"role": "user", "content": "anyone there\nhello?"}


def test_quit_while_the_llm_answers_ends_the_loop_without_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = DuplexSession()

    async def quit_midway(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        yield "Hello "
        session.quit_requested.set()
        yield "there."

    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts)
    source = ScriptedTurns(heard_line())
    assert run_session(quick_config(), tts, quit_midway, session, source=source) == 0
    assert said == []


@pytest.mark.parametrize("stream", [False, True])
def test_playback_that_cannot_open_ends_the_loop_with_code_2(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)

    def broken(*_args: object, **_kwargs: object) -> TtsResult:
        raise PortAudioMissingError("missing")

    set_tts(monkeypatch, tts, speak=broken, speak_stream=broken)
    source = ScriptedTurns(heard_line("hello"), heard_line("never reached"))
    assert run_session(quick_config(stream=stream), tts, source=source) == 2
    assert source.asked == 1


def test_quit_during_the_cooldown_after_a_reply_says_bye_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = DuplexSession()
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    source = ScriptedTurns(heard_line("hello"), heard_line("never reached"))
    timer = threading.Timer(0.1, session.quit_requested.set)
    timer.start()
    try:
        assert run_session(quick_config(cooldown_s=30.0), tts, session=session, source=source) == 0
    finally:
        timer.cancel()
    assert source.asked == 1


def test_a_barge_in_skips_the_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)

    def barged(turn: TurnInput) -> HeardLine:
        turn.carry.keep(barge_clip(), "")  # the user spoke over the reply
        return heard_line("hello")

    # A 30 s cooldown would trip the 5 s timeout in run_session if it were waited out.
    assert run_session(quick_config(cooldown_s=30.0), tts, source=ScriptedTurns(barged)) == 0


def test_a_failing_resume_ends_the_loop_with_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)

    def broken(*_args: object, **_kwargs: object) -> TtsResult:
        raise PortAudioMissingError("missing")

    set_tts(monkeypatch, tts, speak=broken)

    def interrupted_by_noise(turn: TurnInput) -> HeardLine:
        turn.carry.resume = "the rest of the reply"
        return HeardLine("noise")

    assert run_session(quick_config(), tts, source=ScriptedTurns(interrupted_by_noise)) == 2


def test_noise_with_nothing_to_resume_just_listens_again(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    said = speak_with_fakes(monkeypatch, tts)
    source = ScriptedTurns(HeardLine("noise"), HeardLine("noise"))
    assert run_session(quick_config(), tts, source=source) == 0
    assert said == []
    assert source.asked == 3


@pytest.mark.usefixtures("debug_on")
def test_debug_mode_logs_the_turn_summary_instead_of_printing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    assert run_session(quick_config(), tts, source=ScriptedTurns(heard_line())) == 0
    printed = capsys.readouterr()
    assert "first audio" not in printed.out
    assert "first audio" in printed.err


@pytest.mark.parametrize("stream", [False, True])
def test_a_model_that_cannot_answer_ends_the_session(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    said = speak_with_fakes(monkeypatch, tts, stream=stream)
    source = ScriptedTurns(heard_line("hi there"), heard_line("anyone there?"))

    async def gone(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        raise ChatUnusableError("no provider serves model org/gone")
        yield ""

    assert run_session(quick_config(stream=stream), tts, gone, source=source) == 2
    # Nothing was spoken, and the second line was never answered. The streaming path
    # has Fish listening before the first token, so it ends one empty turn.
    assert said == ([""] if stream else [])


class _LoudVad:
    def __init__(self, _mode: int) -> None:
        pass

    def is_speech(self, frame: bytes, _rate: int) -> bool:
        return frame[:2] != b"\x00\x00"


@pytest.fixture
def debug_on() -> Iterator[None]:
    """Run with ``--debug``, as the CLI configures it, then put logging back."""
    configure_voice_logging(debug=1)
    yield
    configure_voice_logging(debug=False)


def _logged_cooldowns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *, barged: bool
) -> list[str]:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)

    def first_turn(turn: TurnInput) -> HeardLine:
        if barged:
            turn.carry.keep(barge_clip(), "")  # the user spoke over the reply
        return heard_line("hi there")

    quiet = b"\x00\x00" * 480
    echo = (300).to_bytes(2, "little", signed=True) * 480
    monkeypatch.setattr(
        "fish_audio_suite_voice.duplex.SessionMic",
        lambda *_a, **_k: ScriptedMic([quiet, echo, echo, quiet]),
    )
    install_vad(monkeypatch, _LoudVad)
    assert run_session(quick_config(), tts, source=ScriptedTurns(first_turn)) == 0
    err = capsys.readouterr().err
    return [line[line.index("cooldown ") :] for line in err.splitlines() if "cooldown " in line]


@pytest.mark.usefixtures("debug_on")
def test_what_the_mic_heard_in_the_cooldown_is_logged_after_each_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logged = _logged_cooldowns(monkeypatch, capsys, barged=False)
    assert logged == [
        "cooldown frames=4 peak_rms=300 floor=200 voiced=2 over_floor=2 speech_like=2 start_needs=4"
    ]


@pytest.mark.usefixtures("debug_on")
def test_a_reply_cut_off_by_a_barge_in_has_no_cooldown_to_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _logged_cooldowns(monkeypatch, capsys, barged=True) == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_reply_watches_for_barge_in_with_the_listen_detector(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    gate = install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts, stream=stream)
    built: list[ListenTune | None] = []

    def make_gate(**kwargs: object) -> FakeGate:
        built.append(cast("ListenTune | None", kwargs.get("listen")))
        return gate

    monkeypatch.setattr(reply, "BargeGate", make_gate)
    config = quick_config(stream=stream)
    listen = replace(config.listen, vad_engine="silero", vad_aggressiveness=3)
    source = ScriptedTurns(heard_line("hi there"))
    assert run_session(replace(config, listen=listen), tts, source=source) == 0
    assert built == [listen]


@pytest.mark.usefixtures("debug_on")
def test_a_detector_that_fails_in_the_cooldown_log_does_not_end_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch)
    speak_with_fakes(monkeypatch, tts)
    monkeypatch.setattr(
        "fish_audio_suite_voice.duplex.SessionMic", lambda *_a, **_k: ScriptedMic([b"\x00" * 960])
    )
    config = quick_config()
    broken = replace(config, listen=replace(config.listen, vad_engine="not-an-engine"))
    assert run_session(broken, tts, source=ScriptedTurns(heard_line("hi there"))) == 0


def test_the_cooldown_report_judges_frames_against_the_adaptive_floor() -> None:
    from fish_audio_suite_voice.listen import cooldown_report
    from fish_audio_suite_voice.tune import ListenTune

    def level(rms: int) -> bytes:
        return rms.to_bytes(2, "little", signed=True) * 480

    # Room hiss at 150 raises the floor to 375 (2.5 x the quiet level), so a 300 rms
    # echo after it does not count, though it is over the 200 seed.
    frames = [level(150)] * 30 + [level(300)] * 10
    report = cooldown_report(frames, ListenTune(min_speech_rms=200.0), _LoudVad(1))
    fields = dict(part.split("=") for part in report.split())
    assert fields["floor"] == "375"
    assert fields["over_floor"] == "0"
    assert fields["speech_like"] == "0"


def test_quit_during_asr_does_not_ask_the_llm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = DuplexSession()
    asked = {"n": 0}

    def quit_while_heard() -> AsrReply:
        session.quit_requested.set()
        return "hello there friend"

    async def tokens(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        asked["n"] += 1
        if False:
            yield ""

    ScriptedRecordings(monkeypatch, b"RIFFwav")
    ScriptedAsr(monkeypatch, quit_while_heard).serve_sessions()
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


def _cut_off_session(
    monkeypatch: pytest.MonkeyPatch, *after: HeardLine
) -> tuple[list[str], list[ChatMessage]]:
    """Run a session whose first reply is cut off after "Hello there.", then hears ``after``.

    Returns what was spoken, and the history the model is sent after ``after``: one more
    line is heard to read it, and is left out of both.
    """
    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_gate(monkeypatch, FakeGate(captured=b"clip"))
    said = cut_off_speech(monkeypatch, tts, "Hello there.")
    asked: list[list[ChatMessage]] = []

    async def cut_off_reply(messages: list[ChatMessage], **_kwargs: object) -> AsyncIterator[str]:
        asked.append(copy.deepcopy(messages))
        yield "Hello there. How was your day today?"

    source = ScriptedTurns(
        HeardLine("line", text="hi", started=0.0),
        *after,
        HeardLine("line", text="and then?", started=0.0),
    )
    asyncio.run(duplex_turns(quick_config(), tts, None, FakeBackend(cut_off_reply), source=source))
    return said[:-1], asked[-1][:-1]


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
