from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from wire_helpers import make_run

from fish_audio_suite_voice.declick import EdgeFade
from fish_audio_suite_voice.wire import _pump_ws_audio

RATE = 44100


def _pcm(samples: list[int] | np.ndarray) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(int)


def _tone(count: int, level: int = 8000) -> np.ndarray:
    return (level * np.sin(np.arange(count) * 0.3)).astype(int)


def test_a_sentence_that_starts_loud_after_silence_is_faded_in() -> None:
    fade = EdgeFade(RATE, 4.0)
    silence = np.zeros(RATE // 2, dtype=int)  # half a second of nothing
    sound = np.full(1000, 5000)
    out = _samples(_run(fade, _pcm(np.concatenate([silence, sound])), 100000))
    start = silence.size
    assert out[start] < 100  # the step from silence is gone
    ramp = round(RATE * 0.004)
    assert np.all(np.diff(out[start : start + ramp]) > 0)  # rising, no step
    assert out[start + ramp + 5] == 5000  # the rest is untouched
    assert np.array_equal(out[:start], silence)


def test_the_start_of_the_stream_counts_as_an_onset() -> None:
    out = _samples(EdgeFade(RATE, 4.0).process(_pcm(np.full(2000, 4000))))
    assert out[0] < 100
    assert out[-1] == 4000


def test_continuous_speech_and_short_pauses_are_not_touched() -> None:
    quiet_gap = np.zeros(RATE // 400, dtype=int)  # 2.5 ms: a plosive closure, not a sentence gap
    body = np.concatenate([_tone(2000), quiet_gap, _tone(2000)])
    pcm = _pcm(np.concatenate([_tone(3000), body, _tone(3000)]))
    out = _samples(_run(EdgeFade(RATE, 4.0), pcm, 3000))
    middle = slice(3000 + 100, 3000 + body.size)
    assert np.array_equal(out[middle], _samples(pcm)[middle])


def test_an_onset_that_straddles_two_chunks_is_faded_across_both() -> None:
    fade = EdgeFade(RATE, 4.0)
    silence = np.zeros(RATE // 2, dtype=int)
    ramp = round(RATE * 0.004)
    chunks = _pcm(np.concatenate([silence, np.full(ramp * 2, 6000)]))
    cut = 2 * (silence.size + ramp // 2)  # in the middle of the fade
    whole = _samples(EdgeFade(RATE, 4.0).process(chunks))
    split = np.concatenate(
        [_samples(fade.process(chunks[:cut])), _samples(fade.process(chunks[cut:]))]
    )
    assert np.array_equal(split, whole)


def test_silence_that_spans_chunks_still_counts_as_silence() -> None:
    fade = EdgeFade(RATE, 4.0)
    fade.process(_pcm(np.full(100, 3000)))  # speech, then:
    for _ in range(5):
        fade.process(_pcm(np.zeros(2000, dtype=int)))  # 10000 samples of silence in pieces
    out = _samples(fade.process(_pcm(np.full(500, 7000))))
    assert out[0] < 100


def _run(fade: EdgeFade, pcm: bytes, step: int) -> bytes:
    out = b"".join(fade.process(pcm[i : i + step]) for i in range(0, len(pcm), step))
    return out + fade.finish()


def test_a_sentence_that_stops_loud_into_silence_is_faded_out() -> None:
    silence = np.zeros(RATE // 2, dtype=int)
    pcm = _pcm(np.concatenate([np.full(3000, 5000), silence, np.full(3000, 5000)]))
    out = _samples(_run(EdgeFade(RATE, 4.0), pcm, 4096))
    stop = 3000
    ramp = round(RATE * 0.004)
    assert out[stop - 1] < 100  # no step down to silence
    assert np.all(np.diff(out[stop - ramp : stop]) < 0)
    assert out[stop - ramp - 5] == 5000
    assert out[-1] < 100  # the end of the stream fades too


def test_the_output_is_the_same_whatever_the_chunk_size() -> None:
    pcm = _pcm(np.concatenate([_tone(3000), np.zeros(RATE // 4, dtype=int), _tone(3000)]))
    whole = _run(EdgeFade(RATE, 4.0), pcm, len(pcm))
    assert _run(EdgeFade(RATE, 4.0), pcm, 1000) == whole
    assert _run(EdgeFade(RATE, 4.0), pcm, 2) == whole
    assert len(whole) == len(pcm)


def test_an_odd_byte_passes_through_at_the_end() -> None:
    fade = EdgeFade(RATE, 4.0)
    odd = _pcm(np.full(1001, 2000)) + b"\x07"
    out = fade.process(odd) + fade.finish()
    assert len(out) == len(odd) - 1  # the stray byte waits for a partner that never comes


def test_zero_milliseconds_turns_it_off_and_an_empty_chunk_is_fine() -> None:
    chunk = _pcm(np.full(500, 9000))
    assert EdgeFade(RATE, 0.0).process(chunk) == chunk
    assert EdgeFade(RATE, 4.0).process(b"") == b""


@given(
    st.lists(st.integers(min_value=-32768, max_value=32767), min_size=0, max_size=3000),
    st.integers(min_value=1, max_value=20),
    st.floats(min_value=0.5, max_value=20.0),
)
def test_fading_never_makes_a_sample_louder_or_changes_the_length(
    values: list[int], pieces: int, fade_ms: float
) -> None:
    pcm = _pcm(values)
    fade = EdgeFade(RATE, fade_ms)
    step = max(2, (len(pcm) // pieces) // 2 * 2)
    out = _run(fade, pcm, step)
    assert len(out) == len(pcm)
    assert np.all(np.abs(_samples(out)) <= np.abs(_samples(pcm)))


def test_the_pump_fades_pcm_before_it_reaches_the_sink() -> None:
    run, sink = make_run()

    async def close_client() -> None:
        return None

    async def stream() -> AsyncIterator[bytes]:
        yield _pcm(np.zeros(RATE // 2, dtype=int))
        yield _pcm(np.full(2000, 6000))

    faded = replace(run, spec=replace(run.spec, fade_ms=4.0))
    asyncio.run(_pump_ws_audio(stream(), faded, close_client))
    played = _samples(b"".join(sink.chunks))
    assert played[RATE // 2] < 100
    assert played[-1] < 100  # the reply ends faded
    assert played[RATE // 2 + 500] == 6000
    # A turn built without a fade, as the helpers do, passes the audio through unchanged.
    plain_run, plain_sink = make_run()
    asyncio.run(_pump_ws_audio(stream(), plain_run, close_client))
    assert _samples(b"".join(plain_sink.chunks))[RATE // 2] == 6000
