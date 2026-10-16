"""User tone as context: heard by Fish ASR, shown to the model for one turn, never spoken."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest
from session_fakes import (
    ScriptedHearing,
    make_config,
    make_ctx,
    quick_config,
    run_session,
    speak_with_fakes,
)
from voice_fakes import install_audio

from fish_audio_suite_kit import ChatMessage, normalize_cues, scrub_tts
from fish_audio_suite_voice.asr import AsrResult
from fish_audio_suite_voice.hearing import HeardLine, hear_line
from fish_audio_suite_voice.history import with_tone_hint
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tune import TurnTune

HISTORY: list[ChatMessage] = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "tell me a joke"},
]


def _asr(monkeypatch: pytest.MonkeyPatch, *results: AsrResult) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    queue = iter(results)

    async def asr(wav: bytes, *_args: object, **kwargs: object) -> AsrResult:
        calls.append(dict(kwargs))
        return next(queue)

    monkeypatch.setattr("fish_audio_suite_voice.hearing.fish_asr", asr)
    monkeypatch.setattr("fish_audio_suite_voice.hearing.record_utterance", lambda *_a, **_k: b"w")
    return calls


def test_the_heard_line_carries_the_tone_and_the_clean_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _asr(monkeypatch, AsrResult("tell me a joke", ("laughter", "tired")))
    ctx = make_ctx()
    ctx.config = replace(ctx.config, hear_tone=True)
    heard = asyncio.run(hear_line(ctx, ""))
    assert (heard.text, heard.tone) == ("tell me a joke", ("laughter", "tired"))
    assert calls[0]["hear_tone"] is True


def test_tone_is_not_asked_for_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _asr(monkeypatch, AsrResult("tell me a joke"))
    heard = asyncio.run(hear_line(make_ctx(), ""))
    assert heard.tone == ()
    assert calls[0]["hear_tone"] is False


def test_tone_from_every_clip_of_a_turn_reaches_the_line(monkeypatch: pytest.MonkeyPatch) -> None:
    _asr(
        monkeypatch,
        AsrResult("tell me, and", ("tired",)),
        AsrResult("then a joke.", ("laughter", "tired")),
    )
    clips = iter([b"c1", b"c2"])
    monkeypatch.setattr(
        "fish_audio_suite_voice.hearing.record_utterance", lambda *_a, **_k: next(clips, None)
    )
    ctx = make_ctx()
    ctx.config = replace(ctx.config, hear_tone=True, turn=TurnTune(enabled=True))
    heard = asyncio.run(hear_line(ctx, ""))
    assert heard.text == "tell me, and then a joke."
    assert heard.tone == ("tired", "laughter")


def test_the_hint_is_on_a_copy_of_the_last_user_line_only() -> None:
    before = [dict(m) for m in HISTORY]
    out = with_tone_hint(HISTORY, ("laughter", "tired"))
    assert before == HISTORY
    assert out is not HISTORY
    assert out[:-1] == HISTORY[:-1]
    last = out[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("tell me a joke\n\n(")
    assert "laughter, tired" in last["content"]
    assert "never write it as a [cue]" in last["content"]


def test_no_tone_or_no_user_line_means_the_history_itself() -> None:
    assert with_tone_hint(HISTORY, ()) is HISTORY
    ended: list[ChatMessage] = [*HISTORY, {"role": "assistant", "content": "ha"}]
    assert with_tone_hint(ended, ("tired",)) is ended
    assert with_tone_hint([], ("tired",)) == []


def test_a_model_that_echoes_the_hint_is_not_heard_saying_it() -> None:
    hint = with_tone_hint(HISTORY, ("laughter", "tired"))[-1]["content"].split("\n\n", 1)[1]
    for echoed in (f"Sure. {hint} Here is one.", f"{hint} Sure. Here is one."):
        spoken = normalize_cues(scrub_tts(echoed), official=True)
        assert "laughter" not in spoken
        assert "sounded" not in spoken


@pytest.mark.parametrize("stream", [False, True])
def test_the_model_is_told_the_tone_for_that_turn_and_history_keeps_the_plain_line(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    asked: list[list[ChatMessage]] = []

    async def tokens(messages: list[ChatMessage], **_kwargs: object) -> AsyncIterator[str]:
        asked.append(copy.deepcopy(messages))
        yield "Okay."

    tts = FishSpeaker(api_key="k", voice_id="voice")
    install_audio(monkeypatch)
    speak_with_fakes(monkeypatch, tts, stream=stream)
    loop = ScriptedHearing(
        monkeypatch,
        HeardLine("line", text="tell me a joke", tone=("laughter",)),
        HeardLine("line", text="another one"),
    )
    assert run_session(quick_config(stream=stream), tts, tokens) == 0
    assert "laughter" in asked[0][-1]["content"]
    assert asked[0][-1]["content"].startswith("tell me a joke")
    # The next request has no memory of the note, and nor does the history.
    assert all("laughter" not in m["content"] for m in asked[1])
    assert loop.ctx is not None
    assert all("laughter" not in m["content"] for m in loop.ctx.history)
    assert "tell me a joke" in [m["content"] for m in loop.ctx.history if m["role"] == "user"]


def test_the_default_config_has_the_tone_off() -> None:
    assert make_config().hear_tone is False
