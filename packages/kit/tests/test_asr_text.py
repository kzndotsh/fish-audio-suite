from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    DEFAULT_BACKCHANNELS,
    DEFAULT_QUIT_PHRASES,
    asr_language_hint,
    is_asr_hallucination,
    is_backchannel,
    is_caption_watermark,
    is_quit_utterance,
    is_same_utterance,
    scrub_asr,
    scrub_tts,
    without_watermark_segments,
)


def test_cjk_asr_keeps_speech_drops_thanks() -> None:
    assert is_asr_hallucination("谢谢观看")
    assert is_asr_hallucination("嗯")
    assert not is_asr_hallucination("你好，很开心认识你")
    assert not is_asr_hallucination("<|speaker:0|>你好")


def test_asr_english_not_nuked_by_script() -> None:
    assert not is_asr_hallucination("hello there friend")
    assert not is_asr_hallucination("100")
    assert not is_asr_hallucination("3.14")
    assert is_asr_hallucination("42")
    assert not is_asr_hallucination("Été")
    assert not is_asr_hallucination("Привет, как дела сегодня")
    assert not is_asr_hallucination("مرحبا يا صديقي اليوم")
    assert not is_asr_hallucination("Да")


def test_asr_emoji_and_mixed() -> None:
    assert is_asr_hallucination("🔥🔥🔥")
    assert not is_asr_hallucination("AB你")


def test_a_curly_apostrophe_is_the_same_line() -> None:
    assert is_same_utterance("I don\u2019t know the answer", "I don't know the answer")
    assert not is_same_utterance("well go home today", "we'll go home today")
    assert is_same_utterance("افتح الباب اليوم", "افتح الباب اليوم؟")
    assert is_same_utterance("hello there friend", "hello there friend،")
    assert is_same_utterance("hello there friend", "hello there friend।")
    assert is_quit_utterance("bye؟")
    assert is_backchannel("yeah؟")
    assert not is_same_utterance("open the door", "open the other door")
    assert is_same_utterance("افتح الباب اليوم", "افتح البـاب اليوم")
    assert is_same_utterance("مرحبا يا صديقي", "مرحبا يا صديـقي")
    assert not is_same_utterance("افتح الباب اليوم", "افتح النافذة اليوم")


def test_a_decomposed_accent_is_the_same_line() -> None:
    composed = "the caf\u00e9 is open today"
    decomposed = "the cafe\u0301 is open today"
    assert is_same_utterance(decomposed, composed)
    assert not is_same_utterance("the cafe is open today", composed)


def test_invisible_characters_do_not_hide_a_repeat_or_a_quit() -> None:
    heard = scrub_asr("hello \u200bthere")
    assert is_same_utterance(heard, "hello there")
    assert is_quit_utterance(scrub_asr("bye\u200b"))
    assert is_asr_hallucination(scrub_asr("Thanks for watching.\u200b"))
    assert not is_quit_utterance(scrub_asr("hello there friend"))


def test_s1_and_nospeech() -> None:
    assert "[S1]" not in scrub_tts("[S1] Hello there friend")
    spoken = scrub_tts("Open the door[S1]today friend please.")
    assert spoken.split() == ["Open", "the", "door", "today", "friend", "please."]
    paused = scrub_tts("Open the door[pause 1s]today friend please.")
    assert "doortoday" not in paused
    assert "today" in paused
    stamped = scrub_asr("Open the door[0.0 - 1.2]today friend please.")
    assert stamped.split() == ["Open", "the", "door", "today", "friend", "please."]
    assert "[1-2]" in scrub_asr("see items [1-2] today please")
    assert is_asr_hallucination("<|nospeech|>")
    assert is_asr_hallucination("NoSpeech")
    assert not is_asr_hallucination("this is not nospeech today friend")
    assert is_asr_hallucination("<|HAPPY|>")


def test_speaker_and_timestamp_asr() -> None:
    out = scrub_asr("[0.0 - 1.2] Speaker 1: hello there")
    assert "Speaker" not in out
    assert "0.0" not in out
    assert "hello there" in out
    kept = scrub_asr("see items [1-2] today please")
    assert "[1-2]" in kept
    clock = scrub_asr("Meet at Speaker 1:00 today please friend.")
    assert clock == "Meet at Speaker 1:00 today please friend."
    labeled = scrub_asr("Speaker 2: the door is open today.")
    assert labeled == "the door is open today."
    wide = scrub_asr("Speaker 1： hello there friend")
    assert wide == "hello there friend"
    wide_clock = scrub_asr("Meet at Speaker 1：00 today please friend.")
    assert wide_clock == "Meet at Speaker 1：00 today please friend."


def test_scrub_asr_replaces_lone_surrogates() -> None:
    out = scrub_asr("hello \ud800 there")
    out.encode("utf-8")
    assert out.startswith("hello ")
    assert "\ud800" not in out
    assert "there" in out


def test_scrub_asr_keeps_speakers_when_asked() -> None:
    out = scrub_asr("Speaker 1: hello there", strip_speakers=False)
    assert "Speaker 1" in out
    assert "hello there" in out


def test_backchannel() -> None:
    assert is_backchannel("yeah")
    assert not is_backchannel("okay")
    assert not is_backchannel("Okay.")
    assert is_backchannel("uh huh")
    assert is_backchannel("mm hmm")
    assert is_backchannel("嗯。")
    assert is_backchannel("嗯嗯")
    assert not is_backchannel("yeah can you repeat that")


def test_quit_utterance() -> None:
    assert is_quit_utterance("bye")
    assert is_quit_utterance("Goodbye!")
    assert is_quit_utterance("bye bye")
    assert is_quit_utterance("Bye bye!")
    assert is_quit_utterance("bye-bye")
    assert is_quit_utterance("good-bye")
    assert not is_quit_utterance("don't stop now")
    assert not is_quit_utterance("please stop talking about the door")
    assert not is_quit_utterance("I said bye to a friend")


def test_watermark_punctuation_does_not_keep_the_phrase() -> None:
    out = without_watermark_segments(
        "hello there thanks for watching",
        [{"text": "Thanks for watching."}],
    )
    assert "watching" not in out
    assert out == "hello there"
    cjk = without_watermark_segments("请开门。谢谢观看", [{"text": "谢谢观看。"}])
    assert "谢谢观看" not in cjk
    assert "请开门" in cjk
    kept = without_watermark_segments(
        "hello there thanks for watching the door",
        [{"text": "hello there thanks for watching the door"}],
    )
    assert kept == "hello there thanks for watching the door"
    sentence = without_watermark_segments(
        "I subscribe to the idea today friend.",
        [{"text": "Subscribe!"}],
    )
    assert sentence == "I subscribe to the idea today friend."
    leading = without_watermark_segments(
        "Thanks for watching the door today friend.",
        [{"text": "Thanks for watching."}],
    )
    assert leading == "Thanks for watching the door today friend."
    subscribe = without_watermark_segments(
        "Subscribe to the newsletter today friend.",
        [{"text": "Subscribe!"}],
    )
    assert subscribe == "Subscribe to the newsletter today friend."
    caption = without_watermark_segments(
        "Thanks for watching. Open the door today friend.",
        [{"text": "Thanks for watching."}],
    )
    assert caption == "Open the door today friend."
    repeated = without_watermark_segments(
        "hello there friend. Thanks for watching. Thanks for watching.",
        [{"text": "Thanks for watching."}],
    )
    assert repeated == "hello there friend."
    later = without_watermark_segments(
        "I subscribe to the idea today friend. Subscribe!",
        [{"text": "Subscribe!"}],
    )
    assert later == "I subscribe to the idea today friend."
    after_stop = without_watermark_segments(
        "افتح الباب اليوم؟ Thanks for watching. Open the door today friend.",
        [{"text": "Thanks for watching."}],
    )
    assert after_stop == "افتح الباب اليوم؟ Open the door today friend."
    after_danda = without_watermark_segments(
        "hello there friend। Thanks for watching. Open the door today friend.",
        [{"text": "Thanks for watching."}],
    )
    assert "watching" not in after_danda
    assert after_danda.startswith("hello there friend।")
    own_stop = without_watermark_segments(
        "Thanks for watching؟ Open the door today friend.",
        [{"text": "Thanks for watching؟"}],
    )
    assert own_stop == "Open the door today friend."
    after_ellipsis = without_watermark_segments(
        "hello there friend… Thanks for watching. Open the door today friend.",
        [{"text": "Thanks for watching."}],
    )
    assert after_ellipsis == "hello there friend… Open the door today friend."
    comma = without_watermark_segments(
        "hello there friend, thanks for watching the door today.",
        [{"text": "thanks for watching"}],
    )
    assert "watching" in comma


def test_thanks_for_watching() -> None:
    assert is_caption_watermark("Thanks for watching.")
    assert is_caption_watermark("谢谢观看")
    assert not is_caption_watermark("ok")
    assert not is_caption_watermark("hello there friend")
    assert is_asr_hallucination("thanks for watching")
    assert is_asr_hallucination("Thanks for watching.")
    assert is_asr_hallucination("Subtitles by the Amara.org community")
    assert is_asr_hallucination("I hope you enjoyed the video.")
    loop = "If the sentence is cut off, do not make up words. " * 12
    assert is_asr_hallucination(loop)
    assert not is_asr_hallucination("hello there friend")


def test_scrub_asr_strips_pro_cues_only_when_asked() -> None:
    raw = "<|speaker:0|> Hello there [laughter] how are you [高兴] today"
    assert "[laughter]" in scrub_asr(raw)
    out = scrub_asr(raw, strip_cues=True)
    assert out == "Hello there how are you today"
    assert (
        scrub_asr("see items [1-2] and [5] today", strip_cues=True)
        == "see items [1-2] and [5] today"
    )
    assert scrub_asr("hello [sigh].", strip_cues=True) == "hello."


@pytest.mark.parametrize("answer", ["no", "No.", "ok", "OK", "hi", "yes", "Да"])
def test_short_answers_are_not_hallucinations(answer: str) -> None:
    assert not is_asr_hallucination(answer)


@pytest.mark.parametrize("noise", ["a", "!", "", "   ", "嗯"])
def test_a_lone_letter_or_mark_is_a_hallucination(noise: str) -> None:
    assert is_asr_hallucination(noise)


def test_the_letter_floor_is_a_parameter() -> None:
    assert is_asr_hallucination("go on", min_letters=9, short_words=frozenset())
    assert not is_asr_hallucination("no", min_letters=9)
    assert not is_asr_hallucination("sure", min_letters=9, short_words=frozenset({"sure"}))
    assert is_asr_hallucination("no", min_letters=9, short_words=frozenset())


@pytest.mark.parametrize("word", ["stop", "please stop", "stop now", "Stop, please."])
def test_stop_does_not_quit_the_session_by_default(word: str) -> None:
    assert not is_quit_utterance(word)


def test_quit_phrases_are_caller_supplied() -> None:
    assert is_quit_utterance("Stop.", phrases={"stop"})
    assert not is_quit_utterance("bye", phrases={"stop"})
    assert "stop" not in DEFAULT_QUIT_PHRASES


@pytest.mark.parametrize("answer", ["right", "Sure.", "sure"])
def test_a_real_answer_is_not_a_backchannel(answer: str) -> None:
    assert not is_backchannel(answer)
    assert is_backchannel(answer, phrases={"right", "sure"})
    assert "right" not in DEFAULT_BACKCHANNELS


@pytest.mark.parametrize(
    ("raw", "hint"),
    [
        ("en", "en"),
        ("en-US", "en"),
        ("EN", "en"),
        (" zh_CN ", "zh"),
        ("English", ""),
        ("", ""),
        ("   ", ""),
        ("e", ""),
        ("eng", ""),
        ("é1", ""),
        ("日本", ""),
        ("-US", ""),
    ],
)
def test_asr_language_hint_keeps_only_a_two_letter_primary_subtag(raw: str, hint: str) -> None:
    assert asr_language_hint(raw) == hint
