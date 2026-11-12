from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from wire_helpers import RecordingSink, install_fish_audio

from fish_audio_suite_voice.loudness import LoudnessMatch
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.tts_tune import TtsTune

RATE = 44100
CEILING = 0.98 * 32768 + 1  # rounding to a whole sample


def _pcm(samples: np.ndarray) -> bytes:
    return np.asarray(np.rint(samples), dtype="<i2").tobytes()


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


def _voice(seconds: float, db: float) -> np.ndarray:
    """A speech-like signal whose RMS is ``db`` below full scale."""
    t = np.arange(int(RATE * seconds)) / RATE
    wave = np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    wave *= 10 ** (db / 20) / math.sqrt(np.mean(wave**2))
    return wave * 32768


def _db(samples: np.ndarray) -> float:
    return 20 * math.log10(math.sqrt(np.mean((samples / 32768) ** 2)))


def _run(match: LoudnessMatch, pcm: bytes, step: int) -> bytes:
    return b"".join(match.process(pcm[i : i + step]) for i in range(0, len(pcm), step))


@pytest.mark.parametrize(("level", "expected"), [(-30.0, -18.0), (-12.0, -18.0), (-24.0, -18.0)])
def test_a_voice_is_brought_to_the_target_level(level: float, expected: float) -> None:
    match = LoudnessMatch(RATE, -18.0)
    out = _samples(_run(match, _pcm(_voice(3.0, level)), RATE // 10))
    assert _db(out) == pytest.approx(expected, abs=1.0)


def test_the_first_chunk_already_plays_at_the_target() -> None:
    match = LoudnessMatch(RATE, -18.0)
    first = _samples(match.process(_pcm(_voice(0.3, -28.0))))
    assert _db(first) == pytest.approx(-18.0, abs=1.0)


def test_the_gain_is_bounded() -> None:
    quiet = LoudnessMatch(RATE, -18.0)
    out = _samples(_run(quiet, _pcm(_voice(2.0, -44.0)), RATE // 10))
    assert quiet.gain_db == pytest.approx(12.0)
    assert _db(out) == pytest.approx(-32.0, abs=1.0)
    loud = LoudnessMatch(RATE, -30.0)
    _run(loud, _pcm(_voice(2.0, -6.0)), RATE // 10)
    assert loud.gain_db == pytest.approx(-12.0)


def test_silence_is_not_boosted_and_does_not_move_the_gain() -> None:
    match = LoudnessMatch(RATE, -18.0)
    assert match.process(_pcm(np.zeros(RATE))) == _pcm(np.zeros(RATE))
    assert match.gain_db == 0.0
    # Room hiss under the gate is not speech either.
    hiss = np.random.default_rng(1).normal(0, 4, RATE)
    assert match.process(_pcm(hiss)) == _pcm(hiss)
    assert match.gain_db == 0.0
    _run(match, _pcm(_voice(1.0, -30.0)), RATE // 10)
    settled = match.gain_db
    out = match.process(_pcm(np.zeros(RATE * 2)))
    assert match.gain_db == pytest.approx(settled)
    assert not np.any(_samples(out))


def test_a_loud_word_after_quiet_ones_does_not_make_the_gain_pump() -> None:
    match = LoudnessMatch(RATE, -18.0)
    _run(match, _pcm(_voice(1.0, -30.0)), RATE // 10)
    gains: list[float] = []
    block = _pcm(_voice(0.02, -6.0))
    for _ in range(25):
        match.process(block)
        gains.append(match.gain_db)
    steps = np.abs(np.diff(gains))
    assert steps.max() <= 0.13 + 1e-9  # the slew limit per 20 ms block
    assert gains[-1] > gains[0] - 3.0


def test_a_peak_after_a_boost_is_limited_not_clipped() -> None:
    match = LoudnessMatch(RATE, -18.0)
    quiet = _voice(1.0, -30.0)
    spike = np.zeros(2000)
    spike[::50] = 30000
    out = _samples(_run(match, _pcm(np.concatenate([quiet, spike])), RATE // 10))
    assert np.abs(out).max() <= CEILING
    assert np.abs(out).max() > 0.8 * 32768  # bent toward the ceiling, not flattened


def test_a_short_clip_with_too_little_speech_plays_untouched() -> None:
    match = LoudnessMatch(RATE, -18.0)
    clip = _pcm(_voice(0.05, -40.0))
    assert match.process(clip) == clip
    assert match.gain_db == 0.0


def test_an_odd_byte_is_held_for_the_next_chunk() -> None:
    match = LoudnessMatch(RATE, -18.0)
    pcm = _pcm(_voice(0.5, -30.0))
    first = match.process(pcm[:1001])
    rest = match.process(pcm[1001:])
    assert len(first) + len(rest) == len(pcm)


@settings(deadline=None, max_examples=40)
@given(
    st.lists(st.integers(min_value=-32768, max_value=32767), min_size=0, max_size=6000),
    st.integers(min_value=2, max_value=4000),
    st.floats(min_value=-40.0, max_value=-6.0),
)
def test_any_audio_keeps_its_length_stays_under_the_ceiling_and_the_gain_in_bounds(
    values: list[int], step: int, target: float
) -> None:
    pcm = _pcm(np.asarray(values, dtype=np.float64))
    match = LoudnessMatch(RATE, target)
    out = b"".join(match.process(pcm[i : i + step]) for i in range(0, len(pcm), step))
    assert len(out) <= len(pcm)
    assert len(pcm) - len(out) <= 1
    samples = _samples(out)
    if match.gain_db != 0.0:
        assert np.all(np.abs(samples) <= CEILING)
    assert -12.0 <= match.gain_db <= 12.0


def _speak_levelled(
    monkeypatch: pytest.MonkeyPatch, chunks: list[bytes], tune: TtsTune
) -> np.ndarray:
    install_fish_audio(monkeypatch, chunks)
    sink = RecordingSink()
    FishSpeaker(api_key="k", voice_id="v", tune=tune).speak("Hello there.", sink)
    return _samples(b"".join(sink.chunks))


def test_the_pump_levels_pcm_before_it_reaches_the_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_pcm(_voice(0.2, -30.0)) for _ in range(5)]
    levelled = _speak_levelled(
        monkeypatch, chunks, TtsTune(sample_rate=RATE, fade_ms=0.0, loudness_db=-18.0)
    )
    assert _db(levelled) == pytest.approx(-18.0, abs=1.5)
    # The speaker's default is no levelling: the audio plays at the level Fish sent it.
    plain = _speak_levelled(monkeypatch, chunks, TtsTune(sample_rate=RATE, fade_ms=0.0))
    assert _db(plain) == pytest.approx(-30.0, abs=0.5)


def test_the_pump_levels_what_the_fade_lets_through(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_pcm(np.zeros(1000)), _pcm(_voice(0.5, -30.0))]
    played = _speak_levelled(
        monkeypatch, chunks, TtsTune(sample_rate=RATE, fade_ms=4.0, loudness_db=-18.0)
    )
    assert played.size == 1000 + int(RATE * 0.5)
    assert abs(played[1000]) < 200  # still faded in
    assert _db(played[1000:]) > -22.0  # and levelled up


def test_the_cli_hands_the_configured_level_to_each_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from session_fakes import make_config

    from fish_audio_suite_voice.cli import smoke_test

    opened = install_fish_audio(monkeypatch, [])
    config = replace(make_config(), tts=TtsTune(loudness_db=-16.0))
    asyncio.run(smoke_test(config, tmp_path / "smoke.wav"))
    assert [spec.loudness_db for spec in opened] == [-16.0]
