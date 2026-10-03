"""Fish ASR segments are words; captions and OpenAI segments are phrases built from them."""

from __future__ import annotations

from typing import Any, cast

from fish_audio_suite_kit import AsrBody, CaptionCue
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
