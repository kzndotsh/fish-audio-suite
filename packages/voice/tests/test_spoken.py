from __future__ import annotations

from fish_audio_suite_voice.spoken import _word_prefix
from fish_audio_suite_voice.spoken import spoken_prefix as _spoken_prefix


def test_spoken_prefix_omits_before_audio() -> None:
    assert (
        _spoken_prefix(
            "hello",
            bytes_played=0,
            sample_rate=44100,
            audio_format="pcm",
            got_audio=False,
            cancelled=True,
        )
        == ""
    )


def test_spoken_prefix_ignores_audio_the_sink_did_not_play() -> None:
    assert (
        _spoken_prefix(
            "hello there",
            bytes_played=0,
            sample_rate=44100,
            audio_format="mp3",
            got_audio=True,
            cancelled=False,
        )
        == ""
    )


def test_played_line_keeps_the_finished_words_before_the_next_line() -> None:
    line = "Hello there friend.\nThe next sentence is longer today."
    # 23 ends on "The" and the next character is a space, so that word was played.
    assert _word_prefix(line, 23) == "Hello there friend. The"
    # One character into "next" is not a finished word.
    assert _word_prefix(line, 25).split() == ["Hello", "there", "friend.", "The"]
    assert _word_prefix("Hello\nthere friend today please", 6) == "Hello"


def test_spoken_prefix_keeps_played_cjk_without_spaces() -> None:
    # 8000 bytes at 16 kHz int16 is 0.25 s, about four characters.
    heard = _spoken_prefix(
        "我们今天见面",
        bytes_played=8000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert heard == "我们今天"
    assert (
        _spoken_prefix(
            "hello there friend",
            bytes_played=6000,
            sample_rate=16_000,
            audio_format="pcm",
            got_audio=True,
            cancelled=True,
        )
        == ""
    )


def test_spoken_prefix_does_not_spend_the_barge_budget_on_a_cue() -> None:
    # 16000 bytes at 16 kHz is half a second, about eight characters.
    # Those characters used to be "[clear] ", so history stored the tag
    # and the next turn said "hello" again.
    heard = _spoken_prefix(
        "[clear] hello there friend",
        bytes_played=16_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert heard == "hello"


def test_finished_turn_does_not_store_the_clear_tag() -> None:
    heard = _spoken_prefix(
        "[clear] hello there friend",
        bytes_played=160_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=False,
    )
    assert heard == "hello there friend"
    mood = _spoken_prefix(
        "[excited] hello there friend",
        bytes_played=160_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=False,
    )
    assert mood == "[excited] hello there friend"


def test_spoken_prefix_keeps_a_played_index() -> None:
    # 40000 bytes at 16 kHz is 1.25 s, about twenty characters, which
    # reaches the index. Those brackets are speech, not a Fish cue.
    heard = _spoken_prefix(
        "Use the key a[i][j] in the code today friend.",
        bytes_played=40_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert "a[i][j]" in heard


def test_spoken_prefix_keeps_the_word_before_a_nonbreaking_space() -> None:
    assert _word_prefix("Hello\u00a0there friend today", 8) == "Hello"
    assert _word_prefix("Hello there friend", 8) == "Hello"


def test_word_prefix_keeps_a_word_that_ends_on_the_cut() -> None:
    # The next character is the space, so "there" was played. Dropping it
    # made the next turn say "there" again.
    assert _word_prefix("Hello there friend", 11) == "Hello there"
    assert _word_prefix("Hello there friend", 10) == "Hello"
    assert _word_prefix("Hello there friend", 5) == "Hello"
    assert _word_prefix("Hello there friend", 3) == ""
    assert _word_prefix("Привет друг сегодня", len("Привет")) == "Привет"
    assert _word_prefix("Hello there. Friend today", 11) == "Hello there"
    assert _word_prefix("Hello there, friend today", 11) == "Hello there"
    # The next letter is still part of "друг". Recording "дру" made the
    # next turn skip the rest of the word.
    assert _word_prefix("Привет друг сегодня", len("Привет дру")) == "Привет"
    assert _word_prefix("Привет друг сегодня", len("Привет друг")) == "Привет друг"
    assert _word_prefix("Hello 你好朋友", 7) == "Hello 你"
    assert _word_prefix("Hello 你好朋友", 6) == "Hello"
    # "there." was played. The next letter is a new word, not more of "there".
    assert _word_prefix("Hello there.Friend today", len("Hello there.")) == "Hello there."
    assert _word_prefix("Hello there.Friend today", len("Hello there.Fr")) == "Hello there."
    assert _word_prefix("Hello there!Friend today", len("Hello there!Fr")) == "Hello there!"
    assert _word_prefix("Hello,there friend today", len("Hello,")) == "Hello,"
    assert _word_prefix("Привет,друг сегодня", len("Привет,")) == "Привет,"
    assert _word_prefix("don't go today", 4) == ""
    assert _word_prefix("don't go today", 5) == "don't"
    assert _word_prefix("3.14 today friend", 2) == ""
    assert _word_prefix("3.14 today friend", 3) == ""
    assert _word_prefix("Hello.Friend today", len("Hello.Fr")) == "Hello."
    assert _word_prefix("Привет.друг сегодня", len("Привет.дру")) == "Привет."


def test_spoken_prefix_keeps_cjk_played_after_an_english_word() -> None:
    assert _word_prefix("Hello你好朋友", 8) == "Hello你好朋"
    assert _word_prefix("Hello", 3) == ""
    heard = _spoken_prefix(
        "Hello你好朋友",
        bytes_played=16_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert heard == "Hello你好朋"


def test_spoken_prefix_full_when_complete() -> None:
    assert (
        _spoken_prefix(
            "hello there",
            bytes_played=100,
            sample_rate=44100,
            audio_format="pcm",
            got_audio=True,
            cancelled=False,
        )
        == "hello there"
    )


def test_cancelled_pcm_keeps_only_the_played_words() -> None:
    spoken = _spoken_prefix(
        "hello there friend",
        bytes_played=12_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert spoken == "hello"
    partial = _spoken_prefix(
        "hello there friend",
        bytes_played=10,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
    )
    assert partial == ""
    assert (
        _spoken_prefix(
            "hello there friend",
            bytes_played=10,
            sample_rate=16_000,
            audio_format="mp3",
            got_audio=True,
            cancelled=True,
        )
        == ""
    )
    assert (
        _spoken_prefix(
            "[happy] hello there friend",
            bytes_played=10,
            sample_rate=16_000,
            audio_format="mp3",
            got_audio=True,
            cancelled=False,
        )
        == "[happy] hello there friend"
    )
    failed = _spoken_prefix(
        "hello there friend",
        bytes_played=12_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=False,
        failed=True,
    )
    assert failed == "hello"


def test_failed_encoded_turn_records_nothing() -> None:
    assert (
        _spoken_prefix(
            "hello there friend",
            bytes_played=50_000,
            sample_rate=16_000,
            audio_format="mp3",
            got_audio=True,
            cancelled=False,
            failed=True,
        )
        == ""
    )


def _cancelled_pcm(text: str, *, speed: float = 1.0, output_latency_s: float = 0.0) -> str:
    return _spoken_prefix(
        text,
        bytes_played=12_000,
        sample_rate=16_000,
        audio_format="pcm",
        got_audio=True,
        cancelled=True,
        speed=speed,
        output_latency_s=output_latency_s,
    )


def test_faster_speech_covers_more_text() -> None:
    text = "hello there friend how are you today"
    slow = _cancelled_pcm(text, speed=0.5)
    normal = _cancelled_pcm(text, speed=1.0)
    fast = _cancelled_pcm(text, speed=2.0)
    assert len(slow) < len(normal) < len(fast)


def test_device_buffer_is_not_counted_as_heard() -> None:
    text = "hello there friend how are you today"
    assert _cancelled_pcm(text, output_latency_s=0.0) == "hello"
    assert _cancelled_pcm(text, output_latency_s=0.375) == ""
    assert _cancelled_pcm(text, output_latency_s=5.0) == ""
