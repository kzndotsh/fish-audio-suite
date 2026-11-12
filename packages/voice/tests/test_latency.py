"""``LatencySnapshot``: one turn's timings, its log line and the summaries built from it."""

from __future__ import annotations

from fish_audio_suite_voice.latency import LatencySnapshot


def test_the_timing_line_names_only_the_known_times() -> None:
    assert "first_audio=850ms" in LatencySnapshot(first_audio_ms=850).log_line()
    line = LatencySnapshot(tts_first_audio_ms=12.4).log_line()
    assert "tts_first_audio=12ms" in line
    assert "asr=" not in line
    heard = LatencySnapshot(asr_ms=40).log_line()
    assert "asr=40ms" in heard
    assert "trace=" not in line
    traced = LatencySnapshot(
        tts_first_audio_ms=12.4, trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
    ).log_line()
    assert "trace=4bf92f3577b34da6a3ce929d0e0e4736" in traced


def test_the_wait_after_the_last_word_adds_the_silence_to_the_first_audio() -> None:
    snapshot = LatencySnapshot(first_audio_ms=900, end_wait_ms=360)
    assert snapshot.since_stop_ms == 1260
    assert "end_wait=360ms" in snapshot.log_line()
    assert LatencySnapshot(first_audio_ms=900).since_stop_ms is None  # Flux ended the turn
    assert LatencySnapshot(end_wait_ms=360).since_stop_ms is None  # no sound was heard


def test_latency_snapshot_log_line_names_every_time() -> None:
    line = LatencySnapshot(
        asr_ms=1.0,
        llm_first_token_ms=2.0,
        tts_first_text_ms=3.0,
        tts_first_audio_ms=4.0,
        voice_to_voice_ms=5.0,
        trace_id="t",
        first_audio_ms=6.0,
    ).log_line()
    assert line == (
        "[timing asr=1ms llm_first_token=2ms tts_first_text=3ms tts_first_audio=4ms"
        " first_audio=6ms voice_to_voice=5ms trace=t]"
    )
