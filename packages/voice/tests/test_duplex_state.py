"""How a session is put together from its settings: the mic's cleanup and the speaker."""

from __future__ import annotations

from dataclasses import replace

import httpx
from session_fakes import FakeBackend, make_config
from voice_fakes import ScriptedMic

from fish_audio_suite_kit import CueStyle
from fish_audio_suite_voice.capture.aec import EchoCanceller
from fish_audio_suite_voice.config.tune import AecTune
from fish_audio_suite_voice.session.signals import DuplexSession
from fish_audio_suite_voice.session.state import session_context, session_speaker
from fish_audio_suite_voice.tts.tune import TtsTune


def test_the_session_mic_is_cleaned_the_way_the_aec_settings_say() -> None:
    quiet_mic = replace(make_config(), aec=AecTune(enabled=False, auto_gain=False, wet=0.5))
    seen: list[EchoCanceller] = []

    def open_mic(aec: EchoCanceller) -> ScriptedMic:
        seen.append(aec)
        return ScriptedMic(aec=aec)

    ctx = session_context(
        quiet_mic,
        session=DuplexSession(),
        open_mic=open_mic,
        backend=FakeBackend(),
        tts=session_speaker(quiet_mic),
        asr_http=httpx.AsyncClient(),
    )
    assert len(seen) == 1  # one echo canceller for the session, handed to the mic
    assert seen[0].tune == quiet_mic.aec
    assert ctx.turn_input.mic.aec is seen[0]  # the reply's sink reads the same one


def test_the_session_speaker_takes_its_voice_model_base_and_tune_from_the_settings() -> None:
    config = replace(
        make_config(),
        fish_api_key="fish-key",
        fish_voice_id="voice-7",
        fish_base="https://fish.example",
        tts_model="drama-3",
        tts=TtsTune(speed=1.5),
    )
    speaker = session_speaker(config)
    assert speaker.api_key == "fish-key"
    assert speaker.voice_id == "voice-7"
    assert speaker.base_url == "https://fish.example"
    assert speaker.model == "drama-3"
    assert speaker.tune.speed == 1.5
    assert speaker.cue_style == config.cue_style == CueStyle("directed")
