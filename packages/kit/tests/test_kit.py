from __future__ import annotations

import asyncio
import json

import pytest

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    CaptionCue,
    FishHttpError,
    LatencySnapshot,
    SuiteDefaults,
    bearer,
    canonical_traceparent,
    chunk_length_hi,
    clamp_num,
    ends_sentence,
    ensure_lead_cue,
    ensure_trace_headers,
    env_base,
    env_bool,
    env_float,
    env_int,
    env_off,
    env_text,
    env_token,
    extract_quoted_speech,
    fish_backoff_seconds,
    fish_error_body,
    fish_retry_pause,
    fish_transport_error,
    format_as_srt,
    format_as_vtt,
    hold_tts,
    is_asr_hallucination,
    is_backchannel,
    is_caption_watermark,
    is_quit_utterance,
    is_tts_junk,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    known_tts_model,
    make_traceparent,
    mood_lead_hold_at,
    next_tts_cut,
    normalize_cues,
    number_or,
    parse_asr_body,
    parse_fish_error,
    same_utterance,
    scrub_asr,
    scrub_tts,
    sentence_closer_hold_at,
    should_retry_fish_status,
    skip_empty_delta,
    split_tts_piece,
    trace_id_of,
    w3c_trace_headers,
    without_watermark_segments,
)


def test_mood_lead_becomes_cue() -> None:
    assert normalize_cues("Excited, hello") == "[excited] hello"


def test_official_emotion_lead_only_at_sentence_start() -> None:
    assert normalize_cues("Anxious: wait") == "[anxious] wait"
    assert normalize_cues("I am anxious today") == "I am anxious today"
    assert normalize_cues("excited, hello", lead=False) == "excited, hello"
    assert normalize_cues("excited, hello. Sad, bye", lead=False) == "excited, hello. [sad] bye"
    assert (
        normalize_cues("你好。Excited, goodbye now please.")
        == "你好。[excited] goodbye now please."
    )
    assert (
        normalize_cues("你好。 Excited, goodbye now please.")
        == "你好。 [excited] goodbye now please."
    )
    assert (
        normalize_cues("你好！Sad, the news is bad today friend.")
        == "你好！[sad] the news is bad today friend."
    )
    assert (
        normalize_cues("Hello؟ Excited, yes today friend.") == "Hello؟ [excited] yes today friend."
    )
    assert normalize_cues("Hello؟Excited, yes today friend.") == "Hello؟Excited, yes today friend."
    assert (
        normalize_cues("今天天气很好。我们打算下午出去走走。")
        == "今天天气很好。我们打算下午出去走走。"
    )
    assert (
        normalize_cues("Dr. Happy, the patient arrived today.")
        == "Dr. Happy, the patient arrived today."
    )
    assert (
        normalize_cues("1. Excited, the first item is ready today.")
        == "1. Excited, the first item is ready today."
    )
    assert (
        normalize_cues("The visit ended. Excited, goodbye now please.")
        == "The visit ended. [excited] goodbye now please."
    )


def test_tone_lead_becomes_cue() -> None:
    assert normalize_cues("Shouting, hey") == "[shouting] hey"


def test_a_mid_line_star_is_not_a_bullet() -> None:
    kept = scrub_tts("* item today friend please.", line_start=False)
    assert kept.startswith("* item")
    assert scrub_tts("* item today friend please.").split()[0] == "item"


def test_a_continued_paren_is_an_aside() -> None:
    aside = "(Please open the door today"
    assert "door" in scrub_tts(aside)
    assert "door" not in scrub_tts(aside, continued=True)


def test_a_neighbor_letter_keeps_the_word_gap() -> None:
    out = scrub_tts("[the docs](https://example.com)", before="r", after="t")
    assert "docs" in out
    assert not out.split()[0].startswith("r")
    assert out.endswith(" ")


def test_default_scrub_still_strips_the_whole_string() -> None:
    assert scrub_tts("  Open the door today friend.  ") == "Open the door today friend."


def test_hold_tts_keeps_an_unfinished_span() -> None:
    assert hold_tts("<think", line_start=True, sentence_start=True) == 0
    partial = "See https://exa"
    assert hold_tts(partial, line_start=True, sentence_start=False) < len(partial)
    fence = "Open the door.\n```"
    assert hold_tts(fence, line_start=True, sentence_start=False) <= fence.index("`")


def test_hold_tts_keeps_a_mood_until_the_comma() -> None:
    assert hold_tts("Excited,", line_start=True, sentence_start=True) == 0
    finished = "Excited, hello"
    assert hold_tts(finished, line_start=True, sentence_start=True) == len(finished)


def test_mood_lead_holds_only_an_unfinished_word() -> None:
    assert mood_lead_hold_at("Exc", sentence_start=True) == 0
    assert mood_lead_hold_at("  Exc", sentence_start=True) == 2
    assert mood_lead_hold_at("Excited,", sentence_start=True) == 0
    assert mood_lead_hold_at("Excited, hi", sentence_start=True) is None
    assert mood_lead_hold_at("Anxious!", sentence_start=True) == 0
    assert mood_lead_hold_at("Anxious !!", sentence_start=True) == 0
    assert mood_lead_hold_at("Anxious! Hello", sentence_start=True) is None
    assert mood_lead_hold_at("Excited*", sentence_start=True) == 0
    assert mood_lead_hold_at("Example", sentence_start=True) is None
    assert mood_lead_hold_at("Happy birthday", sentence_start=True) is None
    assert mood_lead_hold_at("Happy ", sentence_start=True) == 0
    assert mood_lead_hold_at("Happ ", sentence_start=True) is None
    assert mood_lead_hold_at("* E", sentence_start=True) == 2
    assert mood_lead_hold_at("* Excited, hi", sentence_start=True) is None
    assert mood_lead_hold_at("- [x] Ex", sentence_start=True) == 6
    assert mood_lead_hold_at("### Ex", sentence_start=True) == 4
    assert mood_lead_hold_at("*Sad", sentence_start=True) == 1
    assert mood_lead_hold_at("excited", sentence_start=False) is None
    assert mood_lead_hold_at("Hello\nEx", sentence_start=False) == 6


def test_sound_effect_lead_stays_spoken() -> None:
    assert normalize_cues("Pause, wait") == "Pause, wait"


def test_unlisted_lead_stays_spoken() -> None:
    assert normalize_cues("Soft, hello") == "Soft, hello"


def test_cue_lowercase() -> None:
    assert normalize_cues("[Excited] Hello.") == "[excited] Hello."


def test_laugh_aliases() -> None:
    assert "[laughing]" in normalize_cues("[laugh] ha")
    assert "[laughing]" in normalize_cues("[laughs] ha")


def test_sigh_and_chuckle_aliases() -> None:
    assert "[sighing]" in normalize_cues("[sigh] ok")
    assert "[chuckling]" in normalize_cues("[chuckle] ha")


def test_stacked_leading_cues_kept() -> None:
    assert normalize_cues("[sad][whispering] I miss you") == "[sad] [whispering] I miss you"
    assert "[whisper in small voice]" in normalize_cues("[whisper in small voice] come closer")
    # A doubled bracket is not a cue stack. The inner "[" used to take a space.
    assert normalize_cues("[[page]] and then more words") == "[[page]] and then more words"


def test_whisper_aliases() -> None:
    assert "[whispering]" in normalize_cues("[whispers] psst")
    assert "[whispering]" in normalize_cues("[whisper] psst")
    assert "[whispering]" in normalize_cues("<whisper>psst</whisper>")


def test_inline_chuckle_and_cough() -> None:
    out = normalize_cues("I'll call you back [chuckle] in a minute [cough]")
    assert "[chuckling]" in out
    assert "[cough]" in out
    assert "[coughing]" not in out


def test_ensure_lead_cue_only_when_missing() -> None:
    assert ensure_lead_cue("[curious] yeah") == "[curious] yeah"
    assert ensure_lead_cue("I'll call you back [chuckle] in a minute").startswith("I'll")
    assert ensure_lead_cue("yeah i hear you") == "[clear] yeah i hear you"
    assert ensure_lead_cue("  ") == "  "


def test_pause_alias_vs_moss_duration() -> None:
    assert "[break]" in normalize_cues("[pause] wait")
    stripped = scrub_tts("hello [pause 3.2s] there")
    assert "pause" not in stripped.lower()
    assert "hello" in stripped
    assert "there" in stripped


def test_unclosed_cue_is_junk() -> None:
    assert is_tts_junk("[warm, leftover")


def test_tts_junk_thin_and_narration() -> None:
    assert is_tts_junk("")
    assert is_tts_junk("hi")
    assert is_tts_junk("She smiles and leans closer now")
    assert is_tts_junk("[clear] hi")
    assert is_tts_junk("[clear] She smiles and leans closer now")
    assert not is_tts_junk("[clear] Hello there friend")
    assert not is_tts_junk("你好，很开心认识你")
    assert not is_tts_junk("100")
    assert not is_tts_junk("3.14")
    assert is_tts_junk("42")
    assert is_tts_junk("ok")
    assert not is_tts_junk("Été")
    assert not is_tts_junk("Привет, как дела сегодня")
    assert is_tts_junk("Да")


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
    assert is_asr_hallucination("ok")
    assert not is_asr_hallucination("Été")
    assert not is_asr_hallucination("Привет, как дела сегодня")
    assert not is_asr_hallucination("مرحبا يا صديقي اليوم")
    assert is_asr_hallucination("Да")


def test_asr_emoji_and_mixed() -> None:
    assert is_asr_hallucination("🔥🔥🔥")
    assert not is_asr_hallucination("AB你")


def test_digits_inside_parens_stay_speakable() -> None:
    out = scrub_tts("Call me at (555) 010-1234 please today. The code is (A1).")
    assert "(555)" in out
    assert "010-1234" in out
    assert "(A1)" in out
    aside = scrub_tts("Hello (quietly) there friend")
    assert "quietly" not in aside
    assert "Hello" in aside
    assert "there friend" in aside
    wide = scrub_tts("Please open the door（quietly）today friend.")
    assert "quietly" not in wide
    assert "door" in wide
    assert "today" in wide
    assert "doortoday" not in wide
    assert "（555）" in scrub_tts("Call （555） today friend please.")


def test_a_curly_apostrophe_is_the_same_line() -> None:
    assert same_utterance("I don\u2019t know the answer", "I don't know the answer")
    assert not same_utterance("well go home today", "we'll go home today")
    assert same_utterance("افتح الباب اليوم", "افتح الباب اليوم؟")
    assert same_utterance("hello there friend", "hello there friend،")
    assert same_utterance("hello there friend", "hello there friend।")
    assert is_quit_utterance("bye؟")
    assert is_backchannel("yeah؟")
    assert not same_utterance("open the door", "open the other door")
    assert same_utterance("افتح الباب اليوم", "افتح البـاب اليوم")
    assert same_utterance("مرحبا يا صديقي", "مرحبا يا صديـقي")
    assert not same_utterance("افتح الباب اليوم", "افتح النافذة اليوم")


def test_a_code_closer_after_a_bracket_is_not_spoken() -> None:
    out = scrub_tts("Use the key `a[i][j]` in the code today friend.")
    assert "`" not in out
    assert "a[i][j]" in out
    assert "3 * 4" in scrub_tts("Keep 3 * 4 today friend please.")


def test_a_decomposed_accent_is_the_same_line() -> None:
    composed = "the caf\u00e9 is open today"
    decomposed = "the cafe\u0301 is open today"
    assert same_utterance(decomposed, composed)
    assert not same_utterance("the cafe is open today", composed)


def test_a_newline_entity_keeps_the_words_apart() -> None:
    out = scrub_tts("hello&#10;there today friend.")
    assert "hellothere" not in out
    assert "hello" in out
    assert "there" in out
    tab = scrub_tts("hello&#9;there today friend.")
    assert "hello there" in tab
    assert "hellothere" not in tab
    space = scrub_tts("hello&#32;there today friend.")
    assert "hello there" in space
    assert "hellothere" not in space


def test_emphasis_after_a_comma_is_not_spoken() -> None:
    out = normalize_cues(scrub_tts("*Excited,* the door is open today friend."))
    assert "[excited]" in out
    assert "*" not in out
    assert "the door is open" in out
    kept = scrub_tts("The total is 3 * 4 today friend.")
    assert "3 * 4" in kept


def test_aside_and_bold_stripped() -> None:
    out = scrub_tts("Hello (aside) *bold* world.")
    assert "aside" not in out
    assert "*" not in out
    assert "Hello" in out
    assert "world" in out


def test_fullwidth_lead_punctuation_is_a_cue() -> None:
    out = normalize_cues(scrub_tts("Excited， hello there friend."))
    assert "[excited]" in out
    assert "Excited" not in out
    followed = normalize_cues(scrub_tts("Hello there friend。Excited， hi there friend."))
    assert "[excited]" in followed
    assert "[anxious]" in normalize_cues("Anxious！ hello there friend.")


def test_mood_glued_to_a_stop_is_a_cue() -> None:
    out = normalize_cues(scrub_tts("Done.Excited, hi there friend."))
    assert "[excited]" in out
    assert "Excited" not in out
    assert "3.14" in scrub_tts("It is 3.14 exactly today friend.")
    assert "[happy]" not in normalize_cues(scrub_tts("Dr. Happy birthday today friend."))
    assert "[excited]" not in normalize_cues(scrub_tts("No. Excited, hello there friend."))
    corner = normalize_cues(scrub_tts("Open the door today friend。」 Excited, wait there."))
    assert "[excited]" in corner
    assert "Excited" not in corner
    guillemet = normalize_cues(
        scrub_tts("She said «open the door today friend.» Excited, wait there.")
    )
    assert "[excited]" in guillemet
    assert "Excited" not in guillemet
    assert "[excited]" not in normalize_cues(scrub_tts("Hello」 there today friend please."))


def test_invisible_characters_do_not_hide_a_repeat_or_a_quit() -> None:
    heard = scrub_asr("hello \u200bthere")
    assert same_utterance(heard, "hello there")
    assert is_quit_utterance(scrub_asr("bye\u200b"))
    assert is_asr_hallucination(scrub_asr("Thanks for watching.\u200b"))
    assert not is_quit_utterance(scrub_asr("hello there friend"))


def test_line_separator_and_zero_width_space_do_not_hide_a_mood() -> None:
    line = normalize_cues(scrub_tts("Hello there friend.\u2028Excited, hi there friend."))
    assert "\u2028" not in line
    assert "[excited]" in line
    glued = normalize_cues(scrub_tts("Done.\u200bExcited, hi there friend."))
    assert "[excited]" in glued
    assert "Excited" not in glued
    assert scrub_tts("Hel\u00adlo there friend today.") == "Hello there friend today."


def test_carriage_return_is_a_line_break() -> None:
    out = normalize_cues(scrub_tts("Hello there friend.\r\nExcited, hi there friend."))
    assert "\r" not in out
    assert "[excited]" in out
    blank = normalize_cues(scrub_tts("Line one\r\n\r\nLine two is spoken today friend."))
    assert "\r" not in blank
    assert "Line one" in blank
    assert "Line two" in blank


def test_a_parenthesized_sentence_is_spoken() -> None:
    out = scrub_tts("(I can help with that today friend.)")
    assert "help with that" in out
    assert "(" not in out
    plain = scrub_tts("(Please open the door today)")
    assert "open the door" in plain
    assert "(" not in plain
    aside = scrub_tts("I can help (with that) today friend.")
    assert "with that" not in aside
    assert "I can help" in aside
    assert "today friend" in aside
    short = scrub_tts("(she smiles softly.)")
    assert "smiles" not in short
    unclosed = scrub_tts("(Please open the door today")
    assert "open the door" in unclosed
    assert "(" not in unclosed
    cutoff = scrub_tts("Hello (she smiles")
    assert "smiles" not in cutoff
    wide_reply = scrub_tts("（Please open the door today）")
    assert "open the door" in wide_reply
    assert "（" not in wide_reply
    wide_aside = scrub_tts("你好（悄悄地）今天开门朋友。")
    assert "悄悄地" not in wide_aside
    assert "你好" in wide_aside
    assert "今天" in wide_aside


def test_markdown_table_speaks_the_cells() -> None:
    out = scrub_tts("Results today friend.\n\n| name | score |\n| --- | --- |\n| ada | 10 |\n")
    assert "|" not in out
    assert "---" not in out
    assert "name" in out
    assert "ada" in out
    assert "10" in out
    prose = scrub_tts("Use a | b when you mean either today friend.")
    assert "a | b" in prose


def test_html_and_footnotes_are_not_spoken() -> None:
    out = scrub_tts("See <b>the cat</b> today friend &amp; more words.")
    assert "<" not in out
    assert "amp" not in out
    assert "&" in out
    assert "the cat" in out
    note = scrub_tts("A footnote here[^1] and more words today.")
    assert scrub_tts("Open the door</p>then close it today friend.") == (
        "Open the door then close it today friend."
    )
    assert scrub_tts("hello<hr>there today friend.") == "hello there today friend."
    assert scrub_tts("left<td>right today friend.") == "left right today friend."
    assert scrub_tts("hello<pre>code</pre>there today friend.") == (
        "hello code there today friend."
    )
    assert "5th" in scrub_tts("Meet at 5<sup>th</sup> today friend please.")
    assert "[^1]" not in note
    assert "here" in note
    angled = scrub_tts("Go to [the door](<https://example.com>) today friend.")
    assert angled == "Go to the door today friend."
    opened = scrub_tts("See [the door](https://example.com today friend.")
    assert opened == "See the door today friend."
    bare = scrub_tts("Open the door (https://example.com/v1 today friend.")
    assert bare == "Open the door today friend."
    closed = normalize_cues(
        scrub_tts("See the door (https://example.com). Excited, hi there friend.")
    )
    assert closed == "See the door. [excited] hi there friend."
    titled = scrub_tts('See [the docs](https://example.com "title") today friend.')
    assert titled == "See the docs today friend."
    quoted = scrub_tts("See [the docs](https://example.com 'title') today friend.")
    assert quoted == "See the docs today friend."
    ref = scrub_tts("See [the docs][ref] today friend.\n\n[ref]: https://example.com")
    assert "the docs" in ref
    assert "[ref]" not in ref
    assert "example.com" not in ref
    assert "[sad][whispering]" in scrub_tts("[sad][whispering] hello there friend")
    assert "<3" in scrub_tts("I love this <3 today friend.")
    phoneme = scrub_tts("<|phoneme_start|>HH AH0<|phoneme_end|> today friend.")
    assert "<|phoneme_start|>" in phoneme


def test_script_and_style_are_not_spoken() -> None:
    out = scrub_tts("<script>alert(1)</script> Hello there friend today.")
    assert "alert" not in out
    assert out == "Hello there friend today."
    styled = scrub_tts("Hello <style>body { color: red }</style> there friend today.")
    assert "color" not in styled
    assert "Hello" in styled
    assert "there friend today." in styled
    assert "secret" not in scrub_tts("<script>secret")
    kept = scrub_tts("<details>hidden aside</details> Hello there friend today.")
    assert "hidden aside" in kept
    assert "a < b" in scrub_tts("Math a < b and c > d today friend.")


def test_html_comment_is_not_spoken() -> None:
    out = scrub_tts("Before <!-- hidden note --> after today friend.")
    assert "hidden" not in out
    assert "Before" in out
    assert "after today friend." in out
    assert scrub_tts("Hello <!-- secret").strip() == "Hello"
    fenced = scrub_tts("```\n<!-- not spoken\n```\nHello there friend.")
    assert "not spoken" not in fenced
    assert "Hello there friend." in fenced


def test_image_markdown_keeps_the_alt_text() -> None:
    out = scrub_tts("See ![a cat photo](https://example.com/cat.png) today friend.")
    assert "!" not in out
    assert "example.com" not in out
    assert "a cat photo" in out
    assert "today friend" in out
    local = scrub_tts("See ![a cat photo](notes) today friend.")
    assert "!" not in local
    assert "a cat photo" in local


def test_strikethrough_is_not_spoken() -> None:
    out = scrub_tts("I meant ~~Tuesday~~ Wednesday for the meeting.")
    assert "Tuesday" not in out
    assert "Wednesday" in out
    kept = scrub_tts("Meet ~softly~ on Wednesday please.")
    assert "softly" in kept
    assert "~" not in kept


def test_marks_inside_a_word_stay_speakable() -> None:
    out = scrub_tts("5*5 equals 25 and open fish_audio_suite now")
    assert "5*5" in out
    assert "fish_audio_suite" in out
    bold = scrub_tts("I want **this** one and _italic_ here")
    assert "*" not in bold
    assert "_" not in bold
    assert "this" in bold
    assert "italic" in bold
    assert "2 * 3" in scrub_tts("2 * 3 is six today.")
    listed = scrub_tts("* buy milk today please")
    assert "*" not in listed
    assert "buy milk" in listed


def test_s1_parens_become_cues() -> None:
    out = normalize_cues(scrub_tts("(happy) Hello there friend (break)"))
    assert "[happy]" in out
    assert "[break]" in out
    assert "(" not in out


def test_cue_before_a_parenthetical_stays_a_cue() -> None:
    out = scrub_tts("[happy](softly) Hello there friend")
    assert "[happy]" in out
    assert "softly" not in out
    assert "Hello there friend" in out


def test_a_url_period_does_not_hide_the_next_mood() -> None:
    out = normalize_cues(scrub_tts("See https://example.com. Excited, hi there friend."))
    assert "See." in out
    assert "[excited]" in out
    assert "Excited" not in out
    assert "http" not in out.lower()
    versioned = scrub_tts("See https://example.com/v1.2/file today friend.")
    assert "v1.2" not in versioned
    assert "today friend" in versioned
    after = scrub_tts("打开 https://example.com，然后关门今天朋友。")
    assert "打开" in after
    assert "然后关门" in after
    comma = scrub_tts("Open https://example.com，then close the door today friend.")
    assert "then" in comma
    assert "close the door" in comma
    paren = scrub_tts("Open the door (https://example.com，然后关门) today friend.")
    assert "然后关门" in paren
    assert "today friend" in paren
    labeled = scrub_tts("See [the door](https://example.com，然后关门) today friend.")
    assert "the door" in labeled
    assert "然后关门" in labeled
    aside = scrub_tts("Note (see https://example.com/a) today friend please.")
    assert "today friend please" in aside
    assert "Note" in aside
    assert "example.com" not in aside


def test_a_non_url_link_keeps_the_words() -> None:
    door = scrub_tts("Please open [the door](softly) today friend.")
    assert "the door" in door
    assert "[the door]" not in door
    assert "softly" not in door
    cue = normalize_cues(scrub_tts("[happy](softly) hello there friend."))
    assert "[happy]" in cue
    assert "softly" not in cue
    tone = normalize_cues(scrub_tts("[soft tone](now) hello there friend."))
    assert "[soft tone]" in tone


def test_a_ref_note_is_not_spoken() -> None:
    spoken = scrub_tts("Open the door today friend.<ref>secret note</ref> Then wait please.")
    assert "secret" not in spoken
    assert "friend." in spoken
    assert "Then wait please." in spoken
    assert "friend.secret" not in spoken
    named = scrub_tts('Open the door today friend. <ref name="a">secret note</ref> Then wait.')
    assert "secret" not in named
    assert "Then wait." in named


def test_a_link_does_not_join_or_eat_the_next_word() -> None:
    linked = scrub_tts("See [the docs](https://example.com/a)today friend please.")
    assert "docs" in linked
    assert "today" in linked
    assert "docstoday" not in linked
    image = scrub_tts("See ![the door](https://example.com/a.png)today friend please.")
    assert "door" in image
    assert "doortoday" not in image
    comma = scrub_tts("See https://example.com/a,today friend please.")
    assert "today" in comma
    assert "friend" in comma
    version = scrub_tts("Use https://example.com/v1.2/file today friend please.")
    assert "http" not in version.lower()
    assert "today" in version
    assert "v1" not in version


def test_a_reference_link_keeps_the_words() -> None:
    spoken = scrub_tts("See the [docs][ref] today friend please.")
    assert "docs" in spoken
    assert "[docs]" not in spoken
    assert "ref" not in spoken
    empty = scrub_tts("See the [docs][] today friend please.")
    assert "docs" in empty
    assert "[" not in empty
    cues = normalize_cues(scrub_tts("[happy][whispering] hello there friend."))
    assert "[happy]" in cues
    assert "[whispering]" in cues
    index = scrub_tts("Set a[i][j] today friend please now.")
    assert "a[i][j]" in index
    kept = normalize_cues(scrub_tts("[happy][1] hello there friend."))
    assert "[happy]" in kept
    assert "[1]" not in kept


def test_url_and_heading_strip() -> None:
    out = scrub_tts("# Title\nSee https://example.com/x and [docs](https://x.test) later.")
    assert "http" not in out.lower()
    assert "#" not in out
    assert "Title" in out
    assert "docs" in out
    linked = scrub_tts("Autolink <https://example.com/path> and then more words today friend.")
    assert "<" not in linked
    assert ">" not in linked
    assert "http" not in linked.lower()
    assert "Autolink" in linked
    assert "and then more words" in linked


def test_a_url_mark_does_not_eat_the_next_word() -> None:
    bang = scrub_tts("See https://example.com/a!today friend please.")
    assert bang.split() == ["See", "today", "friend", "please."]
    colon = scrub_tts("See https://example.com/a:today friend please.")
    assert colon.split() == ["See", "today", "friend", "please."]
    semi = scrub_tts("See https://example.com/a;today friend please.")
    assert semi.split() == ["See", "today", "friend", "please."]
    port = scrub_tts("Use https://example.com:8080/file today friend please.")
    assert "8080" not in port
    assert "file" not in port
    assert "today" in port
    query = scrub_tts("See https://example.com/a?x=1 today friend please.")
    assert "x=1" not in query
    assert "today" in query
    version = scrub_tts("Use https://example.com/v1.2/file today friend please.")
    assert "v1" not in version
    assert "today" in version


def test_an_escaped_break_tag_is_not_spoken() -> None:
    out = scrub_tts("Open the door&lt;br&gt;today friend please.")
    assert "br" not in out.split()
    assert out.split() == ["Open", "the", "door", "today", "friend", "please."]
    numeric = scrub_tts("Open the door&#60;br&#62;today friend please.")
    assert numeric.split() == ["Open", "the", "door", "today", "friend", "please."]
    compared = scrub_tts("Math a &lt; b and c &gt; d today friend.")
    assert "a < b" in compared
    escaped = scrub_tts("See &lt;script&gt;secret&lt;/script&gt; today friend please.")
    assert "secret" in escaped
    assert "script" not in escaped.split()


def test_a_spaced_break_tag_is_not_spoken() -> None:
    out = scrub_tts("Open the door</ p>today friend please.")
    assert "<" not in out
    assert out.split() == ["Open", "the", "door", "today", "friend", "please."]
    broken = scrub_tts("Open the door< /br>today friend please.")
    assert "<" not in broken
    assert "doortoday" not in broken
    assert "a < b" in scrub_tts("Math a < b and c > d today friend.")
    assert "[whispering]" in normalize_cues(scrub_tts("< whisper>quietly</ whisper>"))


def test_a_details_block_does_not_join_the_words() -> None:
    out = scrub_tts("<details><summary>Open the door</summary>secret note today friend</details>")
    assert "doorsecret" not in out
    assert "secret" in out
    assert out.split()[:5] == ["Open", "the", "door", "secret", "note"]


def test_a_footnote_definition_is_the_note() -> None:
    out = scrub_tts("Open the door today friend.\n\n[^1]: the secret note is here today\n")
    assert ":" not in out
    assert "secret" in out
    assert out.split()[:5] == ["Open", "the", "door", "today", "friend."]
    inline = scrub_tts("See the door[^note] today friend please.")
    assert "door" in inline
    assert "today" in inline
    assert "[^" not in inline


def test_a_tilde_fence_is_not_spoken() -> None:
    fenced = scrub_tts("Open the door~~~\nprint(1)\n~~~today friend please.")
    assert "print" not in fenced
    assert fenced.split() == ["Open", "the", "door", "today", "friend", "please."]
    kept = scrub_tts("Meet ~softly~ on Wednesday please friend.")
    assert "softly" in kept
    struck = scrub_tts("I meant ~~Tuesday~~ Wednesday for the meeting please.")
    assert "Tuesday" not in struck
    assert "Wednesday" in struck


def test_a_markdown_rule_or_quote_is_not_spoken() -> None:
    rule = scrub_tts("Open the door today friend.\n---\nThen open the window today friend.")
    assert "---" not in rule
    assert rule.split() == [
        "Open",
        "the",
        "door",
        "today",
        "friend.",
        "Then",
        "open",
        "the",
        "window",
        "today",
        "friend.",
    ]
    quoted = scrub_tts("> Open the door today friend please.")
    assert ">" not in quoted
    assert quoted.split()[0] == "Open"
    inline = scrub_tts("The score is 2 > 1 today friend please.")
    assert ">" in inline
    dashes = scrub_tts("Wait---then open the door today friend.")
    assert "---" in dashes


def test_an_image_or_script_does_not_join_the_words() -> None:
    image = scrub_tts("Open the door![the cat](https://example.com/a.png)today friend please.")
    assert "example.com" not in image
    assert image.split() == ["Open", "the", "door", "the", "cat", "today", "friend", "please."]
    script = scrub_tts("Open the door<script>alert(1)</script>today friend please.")
    assert "alert" not in script
    assert script.split() == ["Open", "the", "door", "today", "friend", "please."]
    style = scrub_tts("Open the door<style>body{}</style>today friend please.")
    assert "body" not in style
    assert "doortoday" not in style
    cue = scrub_tts("Open the door[happy](softly) today friend please.")
    assert "[happy]" in cue


def test_a_hidden_span_does_not_join_the_words() -> None:
    thought = scrub_tts("Open the door<think>secret plan</think>today friend please.")
    assert "secret" not in thought
    assert thought.split() == ["Open", "the", "door", "today", "friend", "please."]
    link = scrub_tts("Open the door<https://example.com>today friend please.")
    assert "example.com" not in link
    assert link.split() == ["Open", "the", "door", "today", "friend", "please."]


def test_a_removed_span_does_not_join_the_words() -> None:
    comment = scrub_tts("Open the door<!-- secret -->today friend please.")
    assert "secret" not in comment
    assert comment.split() == ["Open", "the", "door", "today", "friend", "please."]
    fence = scrub_tts("Open the door```print(1)```today friend please.")
    assert "print" not in fence
    assert "today" in fence
    assert "doortoday" not in fence
    assert scrub_tts("hello<b>there") == "hellothere"


def test_a_glued_mark_does_not_join_the_words() -> None:
    bold = scrub_tts("Open the door**today** friend please now.")
    assert bold.split() == ["Open", "the", "door", "today", "friend", "please", "now."]
    strike = scrub_tts("Open the door~~secret~~today friend please.")
    assert "secret" not in strike
    assert "today" in strike
    assert "doortoday" not in strike
    note = scrub_tts("Open the door[^1]today friend please.")
    assert note.split() == ["Open", "the", "door", "today", "friend", "please."]
    assert "5*5" in scrub_tts("Keep 5*5 today friend please.")
    assert "fish_audio" in scrub_tts("Use fish_audio today friend please.")


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


def test_scrub_tts_keeps_fish_control_tokens() -> None:
    speaker = scrub_tts("<|speaker:0|> [happy] Hello there friend")
    assert "<|speaker:0|>" in speaker
    assert "[happy]" in speaker
    phoneme = scrub_tts("<|phoneme_start|>HH AH0 L OW1<|phoneme_end|>")
    assert "<|phoneme_start|>" in phoneme
    assert "<|phoneme_end|>" in phoneme
    assert "HH AH0 L OW1" in phoneme
    asr = scrub_asr("<|speaker:0|> hello there")
    assert "<|" not in asr
    assert "hello there" in asr
    joined = scrub_asr("Open the door<|speaker:2|>today friend please.")
    assert joined.split() == ["Open", "the", "door", "today", "friend", "please."]
    paused = scrub_tts("Open the door<|DELAY:1|>today friend please.")
    assert "doortoday" not in paused
    assert "today" in paused


def test_unclosed_thought_is_not_spoken() -> None:
    out = scrub_tts("Hello there friend. <think>do not say this plan")
    assert "do not say" not in out
    assert "<think>" not in out
    assert "Hello there friend" in out
    closed = scrub_tts("Hello there friend. <think>secret</think> More words follow.")
    assert "secret" not in closed
    assert "More words follow" in closed
    aside = scrub_tts("Hello there friend (do not read this aside")
    assert "aside" not in aside
    assert "Hello there friend" in aside
    number = scrub_tts("Call (555) 010-2000 when you arrive today.")
    assert "(555)" in number
    partial = scrub_tts("Call (555")
    assert "(555" in partial
    cue = scrub_tts("Hello there friend [happy")
    assert "[" not in cue
    assert "Hello there friend" in cue
    assert is_tts_junk(cue) is False
    kept = scrub_tts("[happy] Hello there friend")
    assert "[happy]" in kept


def test_dash_and_plus_bullets_are_not_spoken() -> None:
    dash = scrub_tts("- first item is spoken today friend.")
    assert dash.startswith("first item")
    plus = scrub_tts("+ first item is spoken today friend.")
    assert plus.startswith("first item")
    assert "-" not in scrub_tts("See the note.\n- first item is spoken today friend.")
    assert "2 - 3" in scrub_tts("2 - 3 equals negative one today friend.")
    assert "-5" in scrub_tts("-5 degrees is cold today friend.")
    assert "2 + 3" in scrub_tts("2 + 3 equals five today friend.")
    task = scrub_tts("- [ ] first task is spoken today friend.")
    assert task.startswith("first task")
    assert "[" not in task
    done = scrub_tts("- [x] first task is spoken today friend.")
    assert done.startswith("first task")
    assert "[x]" not in done
    assert "[happy]" in scrub_tts("[happy] Hello there friend.")
    assert "[x]" in scrub_tts("see [x] in the notes today friend.")


def test_code_fence_is_not_spoken() -> None:
    inline = scrub_tts("Here is code ```print(1)``` and then more words today friend.")
    assert "```" not in inline
    assert "print(1)" not in inline
    assert "more words today friend" in inline
    block = scrub_tts(
        "A paragraph.\n\n```\nsecret scratch\n```\nThen the real sentence today friend."
    )
    assert "secret scratch" not in block
    assert "```" not in block
    assert "Then the real sentence" in block
    open_fence = scrub_tts("Hello there friend. ```do not read this code")
    assert "do not read" not in open_fence
    assert "```" not in open_fence
    assert "Hello there friend" in open_fence
    kept = scrub_tts("Use `fish_audio` in the sentence today friend please.")
    assert "fish_audio" in kept
    assert "`" not in kept


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


def test_fish_error_message_is_utf8() -> None:
    body = fish_error_body(502, "bad \ud800 byte")
    assert "\ud800" not in str(body["message"])
    str(body["message"]).encode("utf-8")
    err = FishHttpError(502, "bad \ud800 byte")
    err.message.encode("utf-8")


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
    assert is_quit_utterance("please stop")
    assert is_quit_utterance("Stop, please.")
    assert is_quit_utterance("stop now")
    assert not is_quit_utterance("don't stop now")
    assert not is_quit_utterance("please stop talking about the door")
    assert not is_quit_utterance("I said bye to a friend")


def test_next_tts_cut_skips_abbreviations() -> None:
    buf = "Dr. Smith is here. Next"
    cut = next_tts_cut(buf)
    assert cut > 0
    assert buf[:cut].strip().endswith("here.")
    assert "Dr." in buf[:cut]
    restart = "1. Restart the service now please and thank you extra"
    cut2 = next_tts_cut(restart)
    assert not restart[:cut2].strip().endswith("1.")
    priced = "The total is 3.14. Please pay the invoice today."
    price_cut = next_tts_cut(priced, partial_chars=80)
    assert priced[:price_cut] == "The total is 3.14. "


def test_a_time_or_page_number_ends_the_sentence() -> None:
    clock = normalize_cues(scrub_tts("Meet at 10:30. Excited, hi there friend."))
    assert "[excited]" in clock
    assert "Excited" not in clock
    page = normalize_cues("See page 12. Excited, hi there friend.")
    assert "[excited]" in page
    assert ends_sentence("Meet at 10:30.")
    assert ends_sentence("See page 12.")
    assert not ends_sentence("1. ")


def test_ends_sentence_skips_abbreviations_and_list_numbers() -> None:
    assert not ends_sentence("Dr.")
    assert not ends_sentence("No. ")
    assert not ends_sentence("1. ")
    assert not ends_sentence("A.")
    assert not ends_sentence("Hello. Dr.")
    assert ends_sentence("Hello.")
    assert ends_sentence("3.14.")
    assert ends_sentence("你好。")
    assert not ends_sentence("Hello؟")
    assert ends_sentence("Hello؟ ")
    assert ends_sentence('He said "Done."')
    assert ends_sentence('He said "Done." ')
    assert not ends_sentence('Dr."')
    assert ends_sentence('Hello؟ "')
    assert sentence_closer_hold_at("你好。") == 2
    assert sentence_closer_hold_at("你好。”") == 2
    assert sentence_closer_hold_at("你好。” ") is None
    assert sentence_closer_hold_at("你好。再") is None
    assert sentence_closer_hold_at("Hello. ") is None
    assert sentence_closer_hold_at("Hello!") == 5
    assert sentence_closer_hold_at("Anxious!") is None
    assert sentence_closer_hold_at("Words (aside.)") is None
    assert sentence_closer_hold_at("Done.)") == 4


def test_cjk_period_and_ellipsis_end_the_sentence() -> None:
    buf = "今天天气很好。我们打算下午出去走走顺便买些东西然后回家做饭再休息一会儿才出门见朋友。"
    cut = next_tts_cut(buf, partial_chars=40)
    assert buf[:cut] == "今天天气很好。"
    spaced = "你好。 Excited, goodbye now please."
    spaced_cut = next_tts_cut(spaced, partial_chars=80)
    assert spaced[:spaced_cut] == "你好。 "
    spoken = "Hello… world is ready today please continue."
    ellipsis = next_tts_cut(spoken, partial_chars=80)
    assert spoken[:ellipsis] == "Hello… "


def test_a_long_finished_sentence_still_cuts_early() -> None:
    buf = ("word " * 30).strip() + "."
    assert len(buf) > 40
    cut = next_tts_cut(buf)
    assert 0 < cut < len(buf)
    assert buf[cut - 1] == " "
    short = "Done.Excited, hi there friend."
    assert next_tts_cut(short, partial_chars=80) == short.index("E")


def test_next_tts_cut_forty_chars() -> None:
    buf = "this is a long spoken fragment without any sentence end yet"
    cut = next_tts_cut(buf)
    assert 0 < cut <= 40
    assert buf[cut - 1] == " "
    assert buf[cut:]
    assert next_tts_cut("short") == -1
    assert next_tts_cut("x" * 50) == 40
    glued = "x" * 36 + "[whispering] come closer today friend please"
    cue_cut = next_tts_cut(glued, partial_chars=40)
    assert glued[:cue_cut] == "x" * 36
    assert glued[cue_cut:].startswith("[whispering]")
    noted = "Note [hello. there] and then more words please today friend."
    noted_cut = next_tts_cut(noted, partial_chars=80)
    assert "[hello. there]" in noted[:noted_cut]
    assert noted[:noted_cut].count("[") == noted[:noted_cut].count("]")
    spaced = "x" * 20 + " [hello. there] and then more words please today friend."
    spaced_cut = next_tts_cut(spaced, partial_chars=80)
    assert spaced[spaced_cut:].startswith("[hello. there]")
    assert spaced[:spaced_cut].count("[") == spaced[:spaced_cut].count("]")
    lines = "\n".join(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"])
    line_cut = next_tts_cut(lines, partial_chars=40)
    assert lines[:line_cut].endswith("\n")
    assert lines[line_cut:].startswith("golf")


def test_a_nonbreaking_space_is_a_word_cut() -> None:
    head = "Please open the"
    buf = head + "\u00a0" + ("door" * 20)
    cut = next_tts_cut(buf, partial_chars=40)
    assert buf[:cut] == head + "\u00a0"
    assert buf[cut:].startswith("door")
    ideo = head + "\u3000" + ("door" * 20)
    ideo_cut = next_tts_cut(ideo, partial_chars=40)
    assert ideo[:ideo_cut] == head + "\u3000"
    assert ideo[ideo_cut:].startswith("door")


def test_zero_partial_window_does_not_return_an_empty_piece() -> None:
    buf = "hello there friend"
    assert split_tts_piece(buf, 0, flush_rest=False) is None
    assert split_tts_piece(buf, -5, flush_rest=True) == (buf, "")


def test_split_tts_piece_waits_until_flush() -> None:
    assert split_tts_piece("short", 40, flush_rest=False) is None
    assert split_tts_piece("short", 40, flush_rest=True) == ("short", "")
    split = split_tts_piece("word " * 20, 20, flush_rest=False)
    assert split is not None
    piece, tail = split
    assert len(piece) <= 20
    assert piece.endswith(" ")
    assert tail


def test_retry_pause_stops_on_the_last_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("fish_audio_suite_kit.http_errors.asyncio.sleep", fake_sleep)

    async def run() -> tuple[bool, bool]:
        return await fish_retry_pause(0), await fish_retry_pause(4)

    early, last = asyncio.run(run())
    assert early is False
    assert last is True
    assert slept == [1.0]


def test_transport_error_timeout_and_blank() -> None:
    status, message = fish_transport_error(TimeoutError("late"), timed_out=True)
    assert status == 504
    assert message == "Fish request timed out"
    status, message = fish_transport_error(None, timed_out=False)
    assert status == 502
    assert message == "Fish upstream unreachable"
    status, message = fish_transport_error(ConnectionError("reset"), timed_out=False)
    assert message == "reset"


def test_fish_error_ignores_a_body_status_that_is_not_an_error() -> None:
    success = parse_fish_error(500, {"message": "nope", "status": 200})
    assert success["status"] == 500
    assert success["message"] == "nope"
    assert parse_fish_error(429, {"message": "slow", "status": 0})["status"] == 429
    assert parse_fish_error(500, {"message": "nope", "status": 402})["status"] == 402


def test_fish_validation_array_uses_the_field_messages() -> None:
    body = [
        {"loc": ["body", "text"], "msg": "Field required"},
        {"loc": ["body", "format"], "msg": "unexpected format"},
    ]
    assert parse_fish_error(422, body)["message"] == "Field required; unexpected format"
    mixed = [
        {"loc": ["body", "text"], "msg": "Field required"},
        {"loc": ["body", "format"]},
        "nope",
    ]
    assert parse_fish_error(422, mixed)["message"] == "Field required"
    encoded = parse_fish_error(422, json.dumps(body).encode())
    assert encoded["status"] == 422
    assert encoded["message"] == "Field required; unexpected format"
    assert parse_fish_error(422, b"[1, 2]")["message"] == "[1, 2]"
    wrapped = {"detail": [{"loc": ["body", "input"], "msg": "field required", "type": "missing"}]}
    assert parse_fish_error(422, wrapped)["message"] == "field required"
    named = {"detail": [{"loc": ["body", "voice"], "message": "field required"}]}
    assert parse_fish_error(422, named)["message"] == "field required"
    assert parse_fish_error(400, {"message": ["bad voice", "too long"]})["message"] == (
        "bad voice; too long"
    )


def test_one_validation_object_uses_the_field_message() -> None:
    body = {"detail": {"loc": ["body", "input"], "msg": "field required", "type": "missing"}}
    assert parse_fish_error(422, body)["message"] == "field required"
    hidden = {"message": {"message": None}, "detail": {"msg": "bad voice"}}
    assert parse_fish_error(400, hidden)["message"] == "bad voice"
    nested = {"message": {"message": [{"msg": "field required"}, {"msg": "bad format"}]}}
    assert parse_fish_error(422, nested)["message"] == "field required; bad format"
    words = {"detail": {"message": ["bad voice", "too long"]}}
    assert parse_fish_error(400, words)["message"] == "bad voice; too long"


def test_blank_fish_message_does_not_hide_detail() -> None:
    assert parse_fish_error(400, {"message": "", "detail": "voice not found"})["message"] == (
        "voice not found"
    )
    assert parse_fish_error(400, {"message": "  ", "detail": "voice not found"})["message"] == (
        "voice not found"
    )
    assert parse_fish_error(502, {"message": "", "error": {"message": "upstream"}})["message"] == (
        "upstream"
    )
    kept = parse_fish_error(400, {"message": "bad voice", "detail": "ignored"})
    assert kept["message"] == "bad voice"


def test_fish_error_falls_back_when_the_body_is_empty() -> None:
    assert parse_fish_error(500, "")["message"] == "HTTP 500"
    assert parse_fish_error(500, None)["message"] == "HTTP 500"
    assert parse_fish_error(500, b'{"message": "down"}')["message"] == "down"
    assert parse_fish_error(500, {"message": {"message": None}})["message"] == "error"


def test_blank_caption_is_omitted_and_end_cannot_precede_start() -> None:
    rendered = format_as_srt(
        [
            CaptionCue(0.0, 1.0, "   "),
            CaptionCue(2.0, 1.0, "kept"),
        ]
    )
    assert rendered == "1\n00:00:02,000 --> 00:00:02,001\nkept\n"


def test_mood_lead_with_no_remainder_is_only_the_cue() -> None:
    assert normalize_cues("") == ""
    assert normalize_cues("Excited, ") == "[excited]"
    assert ensure_lead_cue("hello", default="  ") == "[clear] hello"


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


def test_guillemets_and_corner_quotes_stay_speech() -> None:
    for sample in (
        "She smiles. «Hello there friend.»",
        "She smiles. 「Hello there friend.」",
        "She smiles. 『Hello there friend.』",
        "She smiles. ‹Hello there friend.›",
        "She smiles. ›Hello there friend.‹",
        "She smiles. 〈Hello there friend.〉",
        "She smiles. 》Hello there friend.《",
    ):
        quoted = extract_quoted_speech(scrub_tts(sample))
        assert "Hello there friend." in quoted
        assert "smiles" not in quoted
        assert not is_tts_junk(scrub_tts(sample))
    assert is_tts_junk(scrub_tts("She smiles and looks away today."))


def test_a_stage_direction_does_not_silence_the_next_sentence() -> None:
    spoken = "She smiles. It's open today friend."
    assert not is_tts_junk(spoken)
    assert "open today friend" in spoken
    assert not is_tts_junk("She smiles.\nOpen the door today friend.")
    assert not is_tts_junk("She smiles. 'Open the door today friend.'")
    assert is_tts_junk("She smiles and looks away today.")
    assert is_tts_junk("The door looks open today friend.")
    assert is_tts_junk("He looks tired today friend please.")
    assert not is_tts_junk("She said the door is open today friend.")


def test_extract_quoted_keeps_quotes() -> None:
    spoken = extract_quoted_speech(scrub_tts('[warm] "Loud and clear." Stage note.'))
    assert "Loud and clear" in spoken
    assert "Stage note" not in spoken


def test_a_removed_aside_keeps_the_period_on_the_word() -> None:
    spoken = normalize_cues(scrub_tts("Item (a). Excited, open the door today friend."))
    assert spoken.split()[:4] == ["Item.", "[excited]", "open", "the"]
    door = normalize_cues(scrub_tts("See the door (about five). Excited, hi there friend."))
    assert "door." in door
    assert "[excited]" in door
    assert "about" not in door


def test_extract_quoted_open_passthrough_and_empty() -> None:
    open_q = extract_quoted_speech('[warm] "hello there friend')
    assert "hello there friend" in open_q
    assert extract_quoted_speech('He said "x" but wait') == ""
    inches = 'Use a 5" pipe for the drain today.'
    assert extract_quoted_speech(inches) == inches
    mixed = 'The board is 2" wide. She said "hold it steady today."'
    assert extract_quoted_speech(mixed) == '"hold it steady today."'
    assert extract_quoted_speech('She said "room 2" and left the building today.') == '"room 2"'
    assert extract_quoted_speech('She said "100" and left the building today.') == '"100"'
    assert extract_quoted_speech('She said "3.14" and left the building today.') == '"3.14"'
    assert extract_quoted_speech('She said "42" and left the building today.') == ""
    assert extract_quoted_speech("no quotes here at all") == "no quotes here at all"
    assert extract_quoted_speech('[warm] "[clear]"') == '[warm] "[clear]"'
    assert extract_quoted_speech('旁白。"你好朋友" 然后离开。') == '"你好朋友"'
    wrapped = 'She smiles.\n"Hello there\nfriend today."'
    assert extract_quoted_speech(wrapped) == '"Hello there\nfriend today."'
    assert "smiles" not in extract_quoted_speech(wrapped)
    assert "你好朋友" in extract_quoted_speech('"你好朋友')


def test_skip_empty_delta() -> None:
    assert skip_empty_delta("  ")
    assert not skip_empty_delta("hi")


def test_bitrate_snaps_to_documented_values() -> None:
    assert known_mp3_bitrate(64) == 64
    assert known_mp3_bitrate(96) == 128
    assert known_mp3_bitrate(192) == 192
    assert known_opus_bitrate(24000) == 24000
    assert known_opus_bitrate(1) == -1000


def test_clamp_num_keeps_fish_ranges() -> None:
    assert clamp_num(900, 100, chunk_length_hi("https://api.fish.audio"), 200, int) == 300
    assert clamp_num(800, 100, chunk_length_hi("http://127.0.0.1:8080"), 200, int) == 800
    assert clamp_num("nope", 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_num(float("nan"), 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_num(float("inf"), 0.5, 2.0, 1.05, float) == 1.05
    assert clamp_num(float("inf"), 100, 300, 200, int) == 200
    assert clamp_num(9, 0.5, 2.0, 1.05, float) == 2.0
    assert clamp_num(True, 0.0, 1.0, 0.7, float) == 0.7
    assert number_or("16,000", 44100, int) == 16000
    assert number_or("16,000.0", 44100, int) == 16000
    assert number_or("16,5", 44100, int) == 44100
    assert number_or(True, 0.0, float) == 0.0
    assert number_or(False, 3, int) == 3


def test_parse_asr_body_text_or_502() -> None:
    data, text = parse_asr_body({"text": "hello"})
    assert data["text"] == "hello"
    assert text == "hello"
    assert parse_asr_body({})[1] == ""
    assert parse_asr_body({"text": None})[1] == ""
    with pytest.raises(FishHttpError) as missing:
        parse_asr_body(["hello"])
    assert missing.value.status == 502
    with pytest.raises(FishHttpError) as bad_text:
        parse_asr_body({"text": ["hello"]})
    assert bad_text.value.message == "Fish returned a non-object body"


def test_known_model_and_latency() -> None:
    assert known_tts_model(" S2.1-PRO ") == "s2.1-pro"
    assert known_tts_model("MyModel") == "MyModel"
    assert known_tts_model("custom\r\nX-Injected: 1") == "s2.1-pro"
    assert known_tts_model("MyModel\ud800") == "s2.1-pro"
    assert known_latency(" Normal ", "balanced") == "normal"
    assert known_latency("turbo", "balanced") == "balanced"


def test_suite_defaults_and_timing() -> None:
    d = SuiteDefaults()
    assert d.tts_model == "s2.1-pro"
    assert d.tts_partial_chars == 40
    assert d.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "Match the user's length" not in DEFAULT_SYSTEM_PROMPT
    assert "Do not offer a menu of help" in DEFAULT_SYSTEM_PROMPT
    assert "do not tag every sentence" in DEFAULT_SYSTEM_PROMPT
    assert "[excited]" in DEFAULT_SYSTEM_PROMPT
    assert "Good: [curious]" in DEFAULT_SYSTEM_PROMPT
    assert "Do not repeat their sentence" in DEFAULT_SYSTEM_PROMPT
    assert "spoken partner for an adult" in DEFAULT_SYSTEM_PROMPT
    assert "Say only words that should be heard" in DEFAULT_SYSTEM_PROMPT
    assert "speak only as that character" in DEFAULT_SYSTEM_PROMPT
    assert "Do not step outside the scene" in DEFAULT_SYSTEM_PROMPT
    assert "lecture, apologize, or decline" not in DEFAULT_SYSTEM_PROMPT
    assert "yeah i hear you" not in DEFAULT_SYSTEM_PROMPT
    assert "Bad: [happy] on every sentence" in DEFAULT_SYSTEM_PROMPT
    assert "Leave [cough] as [cough]" not in DEFAULT_SYSTEM_PROMPT
    assert "every two sentences" not in DEFAULT_SYSTEM_PROMPT
    line = LatencySnapshot(ttfa=12.4).log_line()
    assert "ttfa=12ms" in line
    assert "asr=" not in line
    heard = LatencySnapshot(asr_ms=40).log_line()
    assert "asr=40ms" in heard
    assert "trace=" not in line
    traced = LatencySnapshot(ttfa=12.4, trace_id="4bf92f3577b34da6a3ce929d0e0e4736").log_line()
    assert "trace=4bf92f3577b34da6a3ce929d0e0e4736" in traced


_SAMPLE_PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_canonical_traceparent() -> None:
    assert canonical_traceparent(_SAMPLE_PARENT) == _SAMPLE_PARENT
    assert canonical_traceparent(_SAMPLE_PARENT.upper()) == _SAMPLE_PARENT
    assert canonical_traceparent("not-a-trace") is None
    assert canonical_traceparent("00-" + "0" * 32 + "-" + "0" * 16 + "-01") is None
    forbidden = "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert canonical_traceparent(forbidden) is None
    assert w3c_trace_headers({"traceparent": forbidden}) == {}


def test_bearer_drops_a_key_that_would_split_the_header() -> None:
    assert bearer("sk-test") == "Bearer sk-test"
    assert bearer("  sk-test  ") == "Bearer sk-test"
    assert bearer("sk-test\r\nX-Injected: 1") == "Bearer "
    assert "\n" not in bearer("sk-test\nX-Injected: 1")
    assert bearer("sk-\ud800") == "Bearer "


def test_w3c_trace_headers_and_mint() -> None:
    headers = w3c_trace_headers({"Traceparent": _SAMPLE_PARENT, "tracestate": "congo=t61rcWkgMzE"})
    assert headers["traceparent"] == _SAMPLE_PARENT
    assert headers["tracestate"] == "congo=t61rcWkgMzE"
    injected = w3c_trace_headers(
        {"traceparent": _SAMPLE_PARENT, "tracestate": "congo=ok\r\nX-Injected: 1"}
    )
    assert injected == {"traceparent": _SAMPLE_PARENT}
    nulled = w3c_trace_headers({"traceparent": _SAMPLE_PARENT, "tracestate": "congo=ok\x00"})
    assert "tracestate" not in nulled
    surrogate = w3c_trace_headers({"traceparent": _SAMPLE_PARENT, "tracestate": "vendor=\ud800"})
    assert surrogate == {"traceparent": _SAMPLE_PARENT}
    assert w3c_trace_headers({}) == {}
    forwarded = ensure_trace_headers({"traceparent": _SAMPLE_PARENT})
    assert forwarded["traceparent"] == _SAMPLE_PARENT
    minted_out = ensure_trace_headers({})
    assert canonical_traceparent(minted_out["traceparent"]) == minted_out["traceparent"]
    minted = make_traceparent()
    assert canonical_traceparent(minted) == minted
    child = make_traceparent(trace_id="4bf92f3577b34da6a3ce929d0e0e4736")
    assert trace_id_of(child) == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert child != minted


def test_fish_error_shape_and_retry_policy() -> None:
    assert should_retry_fish_status(429)
    assert should_retry_fish_status(503)
    assert not should_retry_fish_status(400)
    assert not should_retry_fish_status(401)
    assert not should_retry_fish_status(402)
    assert not should_retry_fish_status(404)
    assert fish_backoff_seconds(0) == 1.0
    assert fish_backoff_seconds(2) == 4.0
    assert parse_fish_error(401, {"message": "Invalid Token", "status": 401}) == {
        "message": "Invalid Token",
        "status": 401,
    }
    assert parse_fish_error(400, b"not json")["message"] == "not json"
    assert parse_fish_error(502, {"error": {"message": "upstream"}})["message"] == "upstream"
    assert fish_error_body(402, "Insufficient credits")["status"] == 402


def test_format_as_srt_and_vtt() -> None:
    cues = [
        CaptionCue(0.0, 0.6, "hello"),
        CaptionCue(0.6, 1.5, "there"),
    ]
    assert format_as_srt(cues) == (
        "1\n00:00:00,000 --> 00:00:00,600\nhello\n\n2\n00:00:00,600 --> 00:00:01,500\nthere\n"
    )
    assert format_as_vtt(cues) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.600\nhello\n\n00:00:00.600 --> 00:00:01.500\nthere\n"
    )
    assert format_as_srt([]) == ""
    assert format_as_vtt([]) == "WEBVTT\n"
    assert format_as_srt([CaptionCue(0.0, 2.0, "  hello  ")]) == (
        "1\n00:00:00,000 --> 00:00:02,000\nhello\n"
    )
    assert format_as_srt([CaptionCue(float("nan"), float("inf"), "hello")]) == (
        "1\n00:00:00,000 --> 00:00:00,001\nhello\n"
    )
    assert format_as_srt([CaptionCue(0.0, 1e308, "hello")]) == (
        "1\n00:00:00,000 --> 00:00:00,001\nhello\n"
    )
    assert format_as_srt([CaptionCue(0.0, 1.0, "hello\n\nthere")]) == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\nthere\n"
    )
    assert format_as_srt([CaptionCue(0.0, 1.0, "hello\r\n\r\nthere")]) == (
        "1\n00:00:00,000 --> 00:00:01,000\nhello\nthere\n"
    )
    assert "hello\n\nthere" not in format_as_vtt([CaptionCue(0.0, 1.0, "hello\n\nthere")])
    spoken = format_as_vtt([CaptionCue(0.0, 1.0, "see A --> B later")])
    assert spoken.count("-->") == 1
    marked = format_as_vtt([CaptionCue(0.0, 1.0, "A & B <00:00:01.000> later")])
    assert "A &amp; B &lt;00:00:01.000&gt; later" in marked
    assert "A & B" not in marked
    srt = format_as_srt([CaptionCue(0.0, 1.0, "A & B <note>")])
    assert "A & B <note>" in srt
    assert "&amp;" not in srt
    zero = format_as_srt([CaptionCue(1.0, 1.0, "hello there")])
    assert "00:00:01,000 --> 00:00:01,001" in zero
    inverted = format_as_vtt([CaptionCue(2.0, 1.0, "hello there")])
    assert "00:00:02.000 --> 00:00:02.001" in inverted
    assert "see A -&gt; B later" in spoken
    indexed = format_as_srt([CaptionCue(0.0, 1.0, "00:00:01,000 --> 00:00:02,000")])
    assert indexed.count("-->") == 1


def test_env_number_keeps_default_when_blank_or_junk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TEST_NUM", "12")
    assert env_int("FISH_TEST_NUM", 3) == 12
    assert env_float("FISH_TEST_NUM", 0.5) == 12.0
    monkeypatch.setenv("FISH_TEST_NUM", "1.5")
    assert env_float("FISH_TEST_NUM", 0.5) == 1.5
    assert env_int("FISH_TEST_NUM", 3) == 3
    monkeypatch.setenv("FISH_TEST_NUM", "nope")
    assert env_int("FISH_TEST_NUM", 3) == 3
    assert env_float("FISH_TEST_NUM", 0.5) == 0.5
    monkeypatch.setenv("FISH_TEST_NUM", "nan")
    assert env_float("FISH_TEST_NUM", 0.5) == 0.5
    monkeypatch.setenv("FISH_TEST_NUM", "inf")
    assert env_float("FISH_TEST_NUM", 0.5) == 0.5
    monkeypatch.setenv("FISH_TEST_NUM", "  ")
    assert env_int("FISH_TEST_NUM", 3) == 3
    assert env_int("FISH_TEST_MISSING", 3) == 3
    monkeypatch.setenv("FISH_TEST_FLAG", " YES ")
    assert env_bool("FISH_TEST_FLAG") is True
    monkeypatch.setenv("FISH_TEST_FLAG", "0")
    assert env_bool("FISH_TEST_FLAG", default=True) is False
    monkeypatch.setenv("FISH_TEST_FLAG", "   ")
    assert env_bool("FISH_TEST_FLAG", default=True) is True
    assert env_bool("FISH_TEST_FLAG_MISSING") is False
    assert env_off("FISH_TEST_OFF_MISSING") is False
    monkeypatch.setenv("FISH_TEST_OFF", " maybe ")
    assert env_off("FISH_TEST_OFF") is False
    monkeypatch.setenv("FISH_TEST_OFF", " OFF ")
    assert env_off("FISH_TEST_OFF") is True
    assert env_base("FISH_TEST_BASE_MISSING", "https://api.fish.audio/") == "https://api.fish.audio"
    assert env_text("FISH_TEST_TEXT_MISSING", "plain") == "plain"
    monkeypatch.setenv("FISH_TEST_TEXT", "  kept  ")
    assert env_text("FISH_TEST_TEXT") == "kept"
    monkeypatch.setenv("FISH_TEST_TOKEN", "  ")
    assert env_token("FISH_TEST_TOKEN", "normal") == "normal"
    monkeypatch.setenv("FISH_TEST_TOKEN", " low ")
    assert env_token("FISH_TEST_TOKEN", "normal") == "low"
    monkeypatch.setenv("FISH_TEST_BASE", "https://example.test/v1/")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    monkeypatch.setenv("FISH_TEST_BASE", "")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://api.fish.audio"
    monkeypatch.setenv("FISH_TEST_BASE", "   ")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio/") == "https://api.fish.audio"
    monkeypatch.setenv("FISH_TEST_BASE", "  https://example.test/v1/  ")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    monkeypatch.setenv("FISH_TEST_BASE", "https://example.test/v1\nbad")
    assert env_base("FISH_TEST_BASE", "https://api.fish.audio") == "https://example.test/v1"
    assert env_bool("FISH_TEST_FLAG_MISSING", default=True) is True
