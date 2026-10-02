from __future__ import annotations

import pytest

from fish_audio_suite_kit import (
    hold_tts,
    is_tts_junk,
    normalize_cues,
    scrub_asr,
    scrub_tts,
)


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


def test_a_code_closer_after_a_bracket_is_not_spoken() -> None:
    out = scrub_tts("Use the key `a[i][j]` in the code today friend.")
    assert "`" not in out
    assert "a[i][j]" in out
    assert "3 * 4" in scrub_tts("Keep 3 * 4 today friend please.")


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
    out = normalize_cues(scrub_tts("*Excited,* the door is open today friend."), lead=True)
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


def test_line_separator_and_zero_width_space_do_not_hide_a_mood() -> None:
    line = normalize_cues(
        scrub_tts("Hello there friend.\u2028Excited, hi there friend."), lead=True
    )
    assert "\u2028" not in line
    assert "[excited]" in line
    glued = normalize_cues(scrub_tts("Done.\u200bExcited, hi there friend."), lead=True)
    assert "[excited]" in glued
    assert "Excited" not in glued
    assert scrub_tts("Hel\u00adlo there friend today.") == "Hello there friend today."


def test_carriage_return_is_a_line_break() -> None:
    out = normalize_cues(scrub_tts("Hello there friend.\r\nExcited, hi there friend."), lead=True)
    assert "\r" not in out
    assert "[excited]" in out
    blank = normalize_cues(scrub_tts("Line one\r\n\r\nLine two is spoken today friend."), lead=True)
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
        scrub_tts("See the door (https://example.com). Excited, hi there friend."), lead=True
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


def test_a_url_period_does_not_hide_the_next_mood() -> None:
    out = normalize_cues(scrub_tts("See https://example.com. Excited, hi there friend."), lead=True)
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
    cue = normalize_cues(scrub_tts("[happy](softly) hello there friend."), lead=True)
    assert "[happy]" in cue
    assert "softly" not in cue
    tone = normalize_cues(scrub_tts("[soft tone](now) hello there friend."), lead=True)
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
    cues = normalize_cues(scrub_tts("[happy][whispering] hello there friend."), lead=True)
    assert "[happy]" in cues
    assert "[whispering]" in cues
    index = scrub_tts("Set a[i][j] today friend please now.")
    assert "a[i][j]" in index
    kept = normalize_cues(scrub_tts("[happy][1] hello there friend."), lead=True)
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
    assert "[whispering]" in normalize_cues(scrub_tts("< whisper>quietly</ whisper>"), lead=True)


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


def test_a_removed_aside_keeps_the_period_on_the_word() -> None:
    spoken = normalize_cues(scrub_tts("Item (a). Excited, open the door today friend."), lead=True)
    assert spoken.split()[:4] == ["Item.", "[excited]", "open", "the"]
    door = normalize_cues(
        scrub_tts("See the door (about five). Excited, hi there friend."), lead=True
    )
    assert "door." in door
    assert "[excited]" in door
    assert "about" not in door


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call (please) me", "Call me"),
        ("Call (please). Then", "Call. Then"),
        ("a (b) (c) d", "a d"),
        ("Call (555) me", "Call (555) me"),
    ],
)
def test_a_removed_aside_leaves_one_space(text: str, expected: str) -> None:
    assert scrub_tts(text) == expected


@pytest.mark.parametrize(
    "hostile",
    [
        "<a" * 10_000,
        "<!--" * 5_000,
        "[^" * 10_000,
        "*" * 20_000,
        "\t" * 20_000,
        "<" + "\t" * 20_000,
        "[ " * 10_000,
        "[" + " " * 20_000,
        "[a b][ " * 3_000,
        "|" + "\t" * 20_000,
        "[x](http://" + "a" * 20_000,
        "(http://" + "a" * 20_000,
        "![ " * 7_000,
        "<think>" * 3_000,
        "<script>" * 2_500,
        "```" * 6_000 + "x",
        "~~~" * 6_000 + "x",
        "(" * 20_000,
        "a < b " * 3_000,
        "&lt;" * 5_000,
        "|a|" + "a|" * 20_000,
        "|a|a" + "\t" * 20_000,
        "_*" * 10_000,
        "a" + "_`" * 10_000 + "b",
        "word " + "\t" * 20_000 + "\n",
        "x" + " \t" * 10_000 + "!",
    ],
    ids=lambda text: f"{text[:10]!r}x{len(text)}",
)
def test_hostile_input_is_scrubbed_in_linear_time(hostile: str) -> None:
    import time

    started = time.perf_counter()
    scrub_tts(hostile)
    hold_tts(hostile, line_start=True, sentence_start=True)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    "hostile",
    ["\t" * 20_000 + ".", " " * 20_000 + "?", "|a|" + "a|" * 20_000, "<" + "\t" * 20_000],
    ids=lambda text: f"{text[:6]!r}x{len(text)}",
)
def test_hostile_input_is_scrubbed_for_asr_in_linear_time(hostile: str) -> None:
    import time

    started = time.perf_counter()
    scrub_asr(hostile)
    assert time.perf_counter() - started < 1.0


def test_an_html_comment_can_end_with_bang_dash_dash_gt() -> None:
    assert _words("before <!-- x --!> after") == ["before", "after"]
    assert _words("before <!-- x --> after") == ["before", "after"]
    assert _words("before <!-- never closes after") == ["before"]


def test_table_rows_become_cells_and_separators_vanish() -> None:
    table = "| Name | Age |\n| --- | :-: |\n| Ann | 30 |"
    assert _words(table) == ["Name", "Age", "Ann", "30"]
    assert _words("a - b") == ["a", "-", "b"]
    assert _words("|x|") == ["|x|"]


def test_emphasis_marks_follow_the_neighbor_rules() -> None:
    assert _words("a *bold* word") == ["a", "bold", "word"]
    assert _words("door**today") == ["door", "today"]
    assert _words("5*5 and fish_audio") == ["5*5", "and", "fish_audio"]
    assert _words("*Excited,* he said") == ["Excited,", "he", "said"]


def test_spaces_before_a_stop_and_before_a_newline_collapse() -> None:
    assert scrub_tts("Hello  \t. Next , yes") == "Hello. Next, yes"
    assert scrub_tts("one \t\ntwo") == "one\ntwo"


def _words(text: str) -> list[str]:
    return scrub_tts(text).split()


def test_an_unclosed_tilde_fence_swallows_the_rest_of_the_reply() -> None:
    assert _words("Here:\n~~~\ncode that never closes") == ["Here:"]
    assert _words("Here:\n```\ncode that never closes") == ["Here:"]
    assert _words("before ~~~code~~~ after") == ["before", "after"]
    assert _words("before ```code``` mid ~~~more~~~ after") == ["before", "mid", "after"]


def test_a_reference_link_label_keeps_its_word_gap() -> None:
    assert _words("Open [the docs][ref]today please") == ["Open", "the", "docs", "today", "please"]
    assert _words("[the docs][ref] first") == ["the", "docs", "first"]


def test_closed_blocks_are_removed_and_an_unclosed_one_drops_the_rest() -> None:
    assert _words("a <!-- x --> b") == ["a", "b"]
    assert _words("a <!-- never closed. gone") == ["a"]
    assert _words("Hi <think>secret</think> there") == ["Hi", "there"]
    assert _words("Hi <think>cut off. gone") == ["Hi"]
    assert _words("x <script>alert(1)</script> y <style>p{}</style> z") == ["x", "y", "z"]
    assert _words("x <SCRIPT>a</script> y") == ["x", "y"]
