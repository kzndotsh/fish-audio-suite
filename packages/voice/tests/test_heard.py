"""Judging a transcript: echoes, quit words, backchannels and stale copies."""

from __future__ import annotations

from fish_audio_suite_voice.heard import classify_transcript


def test_classify_transcript_drops_echoes_and_keeps_a_real_line() -> None:
    assert classify_transcript("yeah", "") == "skip"
    assert classify_transcript("thanks for watching", "") == "skip"
    assert classify_transcript("goodbye", "") == "quit"
    assert classify_transcript("hello there", "hello there") == "skip"
    assert classify_transcript("hello there.", "hello there") == "skip"
    assert classify_transcript("Hello There!", "hello there") == "skip"
    assert classify_transcript("bye.", "bye") == "quit"
    assert classify_transcript("I said hello there", "hello there") == "ok"
    assert classify_transcript("hello there", "earlier") == "ok"


def test_a_lone_yeah_answers_on_its_own_turn_but_not_over_a_reply() -> None:
    # Said over the reply, "yeah" means "go on". On its own turn it answers.
    assert classify_transcript("Yeah.", "", over_reply=True) == "skip"
    assert classify_transcript("Yeah.", "", over_reply=False) == "ok"
    assert classify_transcript("uh-huh", "", over_reply=False) == "ok"
    # A hesitation is never an answer.
    assert classify_transcript("Um.", "", over_reply=False) == "skip"
    assert classify_transcript("hmm", "", over_reply=False) == "skip"


def test_a_deliberate_repeat_is_kept_but_a_stale_copy_is_dropped() -> None:
    assert classify_transcript("no", "no", stale=False) == "ok"
    assert classify_transcript("No.", "no", stale=False) == "ok"
    assert classify_transcript("no", "no", stale=True) == "skip"
    assert classify_transcript("no", "yes", stale=True) == "ok"
