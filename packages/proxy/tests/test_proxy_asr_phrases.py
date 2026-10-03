"""Fish ASR segments are words; captions and OpenAI segments are phrases built from them."""

from __future__ import annotations

import time
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from proxy_helpers import WAV_UPLOAD, AsrJson, capture_upstream

from fish_audio_suite_kit import AsrBody, CaptionCue
from fish_audio_suite_proxy.server import app
from fish_audio_suite_proxy.transcribe import caption_cues, transcription_body


def _words(*timed: tuple[str, float, float]) -> list[dict[str, Any]]:
    return [{"text": text, "start": start, "end": end} for text, start, end in timed]


def _body(data: dict[str, Any]) -> AsrBody:
    return cast(AsrBody, data)


def _cues(data: dict[str, Any], text: str) -> list[CaptionCue]:
    return caption_cues(_body(data), text, strip_speakers=False)


def test_words_are_grouped_and_take_punctuation_from_the_transcript() -> None:
    # The shape Fish transcribe-1 returns: one word per segment, no punctuation.
    data = {
        "duration": 3.4,
        "segments": _words(
            ("Hello", 0.0, 0.4),
            ("there", 0.4, 0.96),
            ("This", 0.96, 1.28),
            ("is", 1.28, 1.44),
            ("a", 1.44, 1.44),
            ("test", 1.44, 1.9),
        ),
    }
    cues = _cues(data, "Hello there, this is a test.")
    assert cues == [CaptionCue(0.0, 1.9, "Hello there, this is a test.")]


def test_a_sentence_end_starts_a_new_cue() -> None:
    data = {"segments": _words(("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 0.6, 0.9))}
    cues = _cues(data, "One two. Three.")
    assert [cue.text for cue in cues] == ["One two.", "Three."]
    assert (cues[0].start, cues[0].end) == (0.0, 0.6)


def test_an_abbreviation_does_not_end_the_cue() -> None:
    data = {"segments": _words(("Dr", 0.0, 0.2), ("Smith", 0.2, 0.6), ("arrived", 0.6, 1.0))}
    assert [cue.text for cue in _cues(data, "Dr. Smith arrived.")] == ["Dr. Smith arrived."]


def test_a_pause_starts_a_new_cue() -> None:
    data = {"segments": _words(("wait", 0.0, 0.3), ("then", 1.2, 1.5))}
    assert [cue.text for cue in _cues(data, "wait then")] == ["wait", "then"]


def test_a_long_run_is_split_before_it_gets_too_long_to_read() -> None:
    words = [(f"word{i}", i * 0.2, i * 0.2 + 0.2) for i in range(30)]
    text = " ".join(word for word, _start, _end in words)
    cues = _cues({"segments": _words(*words)}, text)
    assert len(cues) > 1
    assert all(len(cue.text) <= 84 for cue in cues)
    assert all(cue.end - cue.start <= 6.0 for cue in cues)
    assert " ".join(cue.text for cue in cues) == text


def test_a_speaker_turn_starts_a_new_cue() -> None:
    data = {
        "segments": _words(("hi", 0.0, 0.3), ("hello", 0.35, 0.7)),
        "speaker_turns": [
            {"speaker": "A", "text": "hi", "start": 0.0, "end": 0.3},
            {"speaker": "B", "text": "hello", "start": 0.35, "end": 0.7},
        ],
    }
    assert [cue.text for cue in _cues(data, "hi hello")] == ["hi", "hello"]


def test_cjk_words_join_without_spaces() -> None:
    data = {"segments": _words(("我们", 0.0, 0.3), ("今天", 0.3, 0.6), ("见面", 0.6, 0.9))}
    assert [cue.text for cue in _cues(data, "")] == ["我们今天见面"]


def test_a_word_missing_from_the_transcript_is_kept_as_fish_sent_it() -> None:
    data = {"segments": _words(("hello", 0.0, 0.3), ("um", 0.3, 0.5), ("there", 0.5, 0.9))}
    assert [cue.text for cue in _cues(data, "Hello there.")] == ["Hello um there."]


def test_verbose_json_words_come_from_the_word_segments() -> None:
    data = {"duration": 0.9, "segments": _words(("hello", 0.0, 0.4), ("there", 0.4, 0.9))}
    cues = _cues(data, "Hello there.")
    body = transcription_body(
        "verbose_json", "Hello there.", cues, _body(data), language="en", granularities=["word"]
    )
    assert isinstance(body, dict)
    assert body["segments"] == [{"id": 0, "text": "Hello there.", "start": 0.0, "end": 0.9}]
    assert body["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "there", "start": 0.4, "end": 0.9},
    ]
    without = transcription_body(
        "verbose_json", "Hello there.", cues, _body(data), language="en", granularities=["segment"]
    )
    assert isinstance(without, dict)
    assert "words" not in without


def test_a_words_array_in_the_body_still_wins() -> None:
    data = {
        "segments": _words(("hello", 0.0, 0.9)),
        "words": [{"word": "hullo", "start": 0.0, "end": 0.9}],
    }
    body = transcription_body(
        "verbose_json",
        "hello",
        _cues(data, "hello"),
        _body(data),
        language=None,
        granularities=["word"],
    )
    assert isinstance(body, dict)
    assert body["words"] == [{"word": "hullo", "start": 0.0, "end": 0.9}]


def test_srt_has_one_cue_per_phrase_not_per_word() -> None:
    data = {
        "segments": _words(("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 0.6, 0.9)),
    }
    cues = _cues(data, "One two. Three.")
    srt = transcription_body(
        "srt", "One two. Three.", cues, _body(data), language=None, granularities=[]
    )
    assert not isinstance(srt, dict)
    text = bytes(srt.body).decode()
    assert "1\n00:00:00,000 --> 00:00:00,600\nOne two." in text
    assert "2\n00:00:00,600 --> 00:00:00,900\nThree." in text
    assert "\n3\n" not in text


def _texts(data: dict[str, Any], text: str, *, strip_cues: bool = False) -> list[str]:
    cues = caption_cues(_body(data), text, strip_speakers=True, strip_cues=strip_cues)
    return [cue.text for cue in cues]


# The transcribe-1-pro example in Fish's speech-to-text guide.
_DOC_TEXT = "<|speaker:0|> 你好。 <|speaker:1|> [高兴]很开心认识你。 <|speaker:0|> 我也是。"
_DOC_BODY: dict[str, Any] = {
    "text": _DOC_TEXT,
    "duration": 6.4,
    "segments": _words(
        ("你", 0.32, 0.56),
        ("好", 0.56, 0.88),
        ("很", 1.84, 2.04),
        ("开", 2.04, 2.24),
        ("心", 2.24, 2.48),
        ("认", 2.48, 2.68),
        ("识", 2.68, 2.88),
        ("你", 2.88, 3.2),
        ("我", 4.56, 4.8),
        ("也", 4.8, 5.0),
        ("是", 5.0, 5.36),
    ),
    "speaker_turns": [
        {"speaker": "speaker:0", "text": "你好。", "start": 0.32, "end": 0.88},
        {"speaker": "speaker:1", "text": "[高兴]很开心认识你。", "start": 1.84, "end": 3.2},
        {"speaker": "speaker:0", "text": "我也是。", "start": 4.56, "end": 5.36},
    ],
    "language_code": "zh",
    "language": "Chinese",
}


def test_the_documented_chinese_example_makes_one_cue_per_sentence() -> None:
    # The cue stays, as in Fish's own speaker_turns text, unless cues are stripped.
    assert _texts(_DOC_BODY, _DOC_TEXT) == ["你好。", "[高兴]很开心认识你。", "我也是。"]
    assert _texts(_DOC_BODY, _DOC_TEXT, strip_cues=True) == ["你好。", "很开心认识你。", "我也是。"]
    cues = caption_cues(_body(_DOC_BODY), _DOC_TEXT, strip_speakers=True)
    assert [(cue.start, cue.end) for cue in cues] == [(0.32, 0.88), (1.84, 3.2), (4.56, 5.36)]


def test_cjk_sentences_split_on_the_ideographic_stop_without_speaker_turns() -> None:
    data = {"segments": _DOC_BODY["segments"]}
    assert _texts(data, "你好。很开心认识你！我也是？") == ["你好。", "很开心认识你！", "我也是？"]


def test_latin_words_take_their_commas_and_periods() -> None:
    data = {"segments": _words(("hi", 0.0, 0.2), ("there", 0.2, 0.5), ("friend", 0.5, 0.9))}
    assert _texts(data, "Hi there, friend.") == ["Hi there, friend."]


def test_a_normalized_number_shows_as_the_transcript_wrote_it() -> None:
    data = {"segments": _words(("it", 0.0, 0.2), ("is", 0.2, 0.3), ("35", 0.3, 0.6))}
    assert _texts(data, "It is 3.5.") == ["It is 3.5."]


def test_a_word_equal_to_its_neighbour_start_and_end_is_kept() -> None:
    # Fish can send start == end for a short word.
    data = {"segments": _words(("a", 1.0, 1.0), ("cat", 1.0, 1.3))}
    assert caption_cues(_body(data), "A cat.", strip_speakers=True) == [
        CaptionCue(1.0, 1.3, "A cat.")
    ]


def test_inline_speaker_markers_and_cues_are_skipped() -> None:
    data = {"segments": _words(("well", 0.0, 0.3), ("hello", 0.3, 0.6), ("there", 0.6, 0.9))}
    text = '<|speaker:0|> [laughs] "Well, <|speaker:1|> hello [sighs] there."'
    assert _texts(data, text) == ['[laughs] "Well, hello [sighs] there."']
    assert _texts(data, text, strip_cues=True) == ['"Well, hello there."']


def test_a_digit_bracket_is_speech_not_a_cue() -> None:
    data = {"segments": _words(("page", 0.0, 0.3), ("5", 0.3, 0.6))}
    assert _texts(data, "Page [5].") == ["Page [5]."]


def test_a_missing_word_does_not_move_the_alignment() -> None:
    data = {
        "segments": _words(("so", 0.0, 0.2), ("uh", 0.2, 0.3), ("yes", 0.3, 0.5), ("no", 0.5, 0.7))
    }
    assert _texts(data, "So, yes. No.") == ["So, uh yes.", "No."]


def test_a_word_does_not_match_inside_a_longer_word() -> None:
    data = {"segments": _words(("um", 0.0, 0.2), ("drum", 0.2, 0.5))}
    assert _texts(data, "Drum.") == ["um Drum."]


def test_a_repeated_word_matches_the_nearest_occurrence() -> None:
    data = {"segments": _words(("no", 0.0, 0.2), ("no", 0.2, 0.4), ("way", 0.4, 0.6))}
    assert _texts(data, "No, no, way!") == ["No, no, way!"]


def test_a_word_far_ahead_is_not_reached() -> None:
    # Only a short stretch past the last match is searched, so a word the
    # transcript dropped cannot jump the alignment to a later sentence.
    filler = " ".join(["blah"] * 20)
    data = {"segments": _words(("start", 0.0, 0.2), ("end", 0.2, 0.4))}
    assert _texts(data, f"Start. {filler} end!") == ["Start.", "end"]


def _align_seconds(size: int) -> float:
    unit = "[[<|<|a " * 2 + "word, "
    text = unit * (size // len(unit))
    # Every Fish word misses, so each one searches its whole window.
    words = _words(*(("wordy", i * 0.1, i * 0.1 + 0.1) for i in range(size // 10)))
    started = time.perf_counter()
    caption_cues(_body({"segments": words}), text, strip_speakers=True)
    return time.perf_counter() - started


def test_alignment_stays_linear_on_hostile_input() -> None:
    # Unclosed "[" and "<|" openers and words that never match. A fast run
    # passes outright; a slow one must not grow like the square of the input.
    whole = min(_align_seconds(40_000) for _ in range(2))
    if whole > 0.5:
        quarter = min(_align_seconds(10_000) for _ in range(2))
        assert whole < 10 * quarter


def test_the_documented_example_through_the_route(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Pro(AsrJson):
        def json(self) -> dict[str, Any]:
            return _DOC_BODY

    for name in ("FISH_PROXY_ASR_STRIP_CUES", "FISH_ASR_STRIP_CUES"):
        monkeypatch.delenv(name, raising=False)
    capture_upstream(monkeypatch, _Pro())
    with TestClient(app) as client:
        reply = client.post(
            "/v1/audio/transcriptions", files=WAV_UPLOAD, data={"response_format": "verbose_json"}
        )
    body = reply.json()
    assert body["text"] == "你好。 [高兴]很开心认识你。 我也是。"
    assert [row["text"] for row in body["segments"]] == [
        "你好。",
        "[高兴]很开心认识你。",
        "我也是。",
    ]


def test_a_number_fish_splits_stays_one_token_and_does_not_end_the_sentence() -> None:
    # Live Fish output for "The price is $3.5, okay? Thanks.": "3" and "5" are
    # separate words. The "." of "3.5" is not a sentence end.
    data = {
        "segments": _words(
            ("The", 0.0, 0.2),
            ("price", 0.2, 0.5),
            ("is", 0.5, 0.6),
            ("3", 0.6, 0.8),
            ("5", 0.8, 1.0),
            ("okay", 1.0, 1.3),
            ("Thanks", 1.6, 2.0),
        ),
    }
    cues = _cues(data, "The price is $3.5, okay? Thanks.")
    assert [cue.text for cue in cues] == ["The price is $3.5, okay?", "Thanks."]
    assert (cues[0].start, cues[0].end) == (0.0, 1.3)


def test_a_hyphenated_word_fish_splits_is_shown_once() -> None:
    data = {"segments": _words(("well", 0.0, 0.2), ("known", 0.2, 0.5), ("fact", 0.5, 0.8))}
    assert [cue.text for cue in _cues(data, "A well-known fact.")] == ["well-known fact."]
