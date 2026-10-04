from __future__ import annotations

import asyncio
import threading

from wire_helpers import (
    event_kinds,
    stream_events,
    stream_text,
)

from fish_audio_suite_kit import (
    normalize_cues,
    scrub_tts,
)
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.stream_scrub import delta_events
from fish_audio_suite_voice.wire import (
    FlushEvent,
    TextEvent,
)


def test_split_markup_matches_a_one_shot_scrub() -> None:
    samples = (
        "See ![a cat photo](https://example.com/cat.png) today friend.",
        "See <b>the cat</b> today friend &amp; more words.",
        "See [the docs][ref] today friend.",
        "Meet at 5<sup>th</sup> today friend please.",
        "Results today friend.\n\n| name | score |\n| --- | --- |\n| ada | 10 |\n",
    )
    for sample in samples:
        parts = [sample[index : index + 3] for index in range(0, len(sample), 3)]
        streamed = stream_text(parts, 24)
        assert streamed.split() == normalize_cues(scrub_tts(sample), lead=True).split()


def test_streaming_deltas_keep_a_split_span_intact() -> None:
    thought = stream_text(
        [
            "<thinking>do not say this secret ",
            "plan</thinking> Hello there friend from the office. ",
        ],
        24,
    )
    assert "secret" not in thought
    assert "Hello there friend" in thought
    cutoff = stream_text(
        [
            "Hello there friend from the office. ",
            "<thinking>do not say this secret plan",
        ],
        24,
    )
    assert "secret" not in cutoff
    assert "Hello there friend" in cutoff
    link = stream_text(
        [
            "See [docs](https://example.com/very/long/",
            "path/here) later today friend. ",
        ],
        24,
    )
    assert "http" not in link
    assert "path/here" not in link
    assert "[docs]" not in link
    assert link.split() == ["See", "docs", "later", "today", "friend."]
    pieces = stream_text(
        ["See ", "[docs]", "(", "https://example.com/path/here", ")", " later today friend."],
        24,
    )
    assert pieces.split() == ["See", "docs", "later", "today", "friend."]
    whisper = stream_text(
        ["<whisper>", "come closer please", "</whisper>", " and stay quiet today."],
        24,
    )
    assert "<whisper" not in whisper
    assert "[whispering]" in whisper
    assert "come closer please" in whisper
    assert "stay quiet today" in whisper
    heading = stream_text(["#", " ", "Heading then the real sentence continues today."], 24)
    assert "#" not in heading
    assert "Heading then the real sentence" in heading
    continued = stream_text(["I am ", "excited, ", "hello there friend today please."], 24)
    assert "[excited]" not in continued
    assert "excited," in continued
    fresh = stream_text(["Excited, ", "hello there friend today."], 40)
    assert fresh.split()[:2] == ["[excited]", "hello"]
    split_lead = "Excited, the news is good today friend."
    split_chunks = [split_lead[i : i + 3] for i in range(0, len(split_lead), 3)]
    assert stream_text(split_chunks, 40).split()[:2] == ["[excited]", "the"]
    follow = "Hello there friend. Excited, goodbye now please."
    follow_chunks = [follow[i : i + 3] for i in range(0, len(follow), 3)]
    followed = stream_text(follow_chunks, 40)
    assert "Hello there friend." in followed
    assert "[excited]" in followed
    assert "Excited," not in followed
    mid_lead = "I am excited, really glad today friend."
    mid_chunks = [mid_lead[i : i + 3] for i in range(0, len(mid_lead), 3)]
    heard = stream_text(mid_chunks, 40)
    assert "[excited]" not in heard
    assert "excited," in heard
    titled = "Dr. Happy, the patient arrived today friend."
    titled_chunks = [titled[i : i + 3] for i in range(0, len(titled), 3)]
    titled_heard = stream_text(titled_chunks, 40)
    assert "[happy]" not in titled_heard
    assert "Happy," in titled_heard
    for sample in (
        "Hello؟ Excited, yes today friend.",
        "你好。 Excited, goodbye now please friend.",
        'He said "Done." Excited, the next words today friend.',
        'He said "Done.") Excited, the next words today friend.',
        'She said "Wait." Sad, the news is bad today friend.',
        "CJK 你好。” Excited, after a quote today friend please.",
        "Happy - Hello there friend today please now really yes.",
        "Anxious! Hello there friend today please now really yes.",
        "Words (aside.) and then more words today friend please.",
        "Bullet done.\n* Excited, the item is a mood today friend.",
        "List\n- [x] Excited, the task is done today friend please.",
        "### Excited, the heading is a mood today friend please.",
        "*Sad, no space before the mood today friend please now.",
        "**Excited**, the word is long enough please today.",
        "`Excited`, the word is long enough please today.",
        "Anxious !! Hello there friend today please now yes.",
        "<|ACT smile|> Hello there friend please today now.",
        "<|DELAY 1|> Hello there friend please today now.",
        "[pause 1s]Excited, hello there friend today please.",
        "Task - [X] Excited, the task is done today friend please.",
        "Empty cue [] hello there friend please today really yes.",
        "[sad][whispering] hello there friend please today.",
        "[clear][happy] hello there friend please today now.",
        "a[i][j] stays in the sentence today friend please.",
        "Hello। Excited, yes today friend.",
        "Hello۔ Excited, yes today friend.",
    ):
        one_shot = normalize_cues(scrub_tts(sample), lead=True)
        for size in (1, 2, 3):
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            assert stream_text(chunks, 40).split() == one_shot.split()
    glued = "Hello؟Excited, yes today friend."
    glued_chunks = [glued[i : i + 2] for i in range(0, len(glued), 2)]
    glued_heard = stream_text(glued_chunks, 40)
    assert "[excited]" not in glued_heard
    assert "Excited," in glued_heard
    item = stream_text(["*", " ", "first star item is spoken today friend"], 40)
    assert "*" not in item
    assert "first star item" in item
    dashed = stream_text(["-", " ", "first dash item is spoken today friend"], 40)
    assert "-" not in dashed
    assert "first dash item" in dashed
    task = stream_text(["- [ ] ", "first task is spoken today friend."], 40)
    assert "[" not in task
    assert "first task" in task
    added = stream_text(["2", " ", "-", " ", "3 equals negative one today friend."], 40)
    assert added.split()[:3] == ["2", "-", "3"]
    product = stream_text(["2", " ", "*", " ", "3 equals six today friend please."], 40)
    assert product.split()[:3] == ["2", "*", "3"]
    fenced = stream_text(
        ["Here is code ", "```", "print(1)", "```", " and then more words today friend."],
        40,
    )
    assert "```" not in fenced
    assert "print(1)" not in fenced
    assert "more words today friend" in fenced
    after_sentence = stream_text(list("Hello there friend. * 3 equals six today please."), 12)
    assert after_sentence.split()[:5] == ["Hello", "there", "friend.", "*", "3"]
    hashed = stream_text(list("See the note. # not a heading today friend."), 12)
    assert "#" in hashed
    assert "not a heading" in hashed
    bullet = stream_text(["Hello.\n", "*", " ", "first item is spoken today friend."], 12)
    assert "*" not in bullet
    assert "first item" in bullet
    aside = stream_text(
        [
            "Hello there friend (do not read this ",
            "aside) and more words follow. ",
        ],
        24,
    )
    assert "aside" not in aside
    assert "Hello there friend" in aside
    assert "more words follow" in aside
    cutoff = stream_text(
        [
            "Hello there friend from the office. ",
            "(do not read this aside",
        ],
        24,
    )
    assert "aside" not in cutoff
    assert "Hello there friend" in cutoff


def test_held_span_longer_than_the_window_is_still_spoken() -> None:
    # The open parenthesis is held until the reply ends. A phone number
    # stays; a stage aside is still removed. The first cut used to be the
    # only text that reached Fish.
    kept = "Unclosed (555 and then the rest of the sentence today."
    dropped = "Unclosed (aside and then the rest of the sentence today."
    for sample in (kept, dropped):
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        for size in (1, 2, 3):
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            assert stream_text(chunks, 40).split() == once


def test_a_streamed_url_keeps_the_period_on_the_word() -> None:
    samples = (
        "See the door (https://example.com). Excited, hi there friend.",
        "Open https://example.com, then close the door today friend.",
        "打开 https://example.com，然后关门今天朋友。",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True)
        for size in (1, 2, 3):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            spoken = stream_text(chunks, 24)
            assert spoken.split() == once.split()
    assert "door." in normalize_cues(
        scrub_tts("See the door (https://example.com). Excited, hi there friend.")
    )


def test_a_streamed_entity_stop_still_cues_the_next_mood() -> None:
    samples = (
        "Please open the door today friend&#33; Excited, wait there.",
        "Please open the door today friend&#x21; Excited, wait there.",
        "Please open the door today friend&#33; then close it.",
        "Please open the door today friend&#10;Excited, wait there.",
        "Please open the door today friend&#13;Excited, wait there.",
        "Please open the door today friend&#10;then close it today.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        for size in (1, 2, 4):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert stream_text(chunks, 80).split() == once


def test_a_streamed_line_separator_still_cues_the_next_mood() -> None:
    samples = (
        "Please open the door today friend\rExcited, wait there.",
        "Please open the door today friend\u2028Excited, wait there.",
        "Please open the door today friend\u2029Excited, wait there.",
        "Please open the door today friend\rthen close it today.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        for size in (1, 2, 4):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert stream_text(chunks, 80).split() == once


def test_a_streamed_quote_after_a_stop_still_cues_the_next_mood() -> None:
    samples = (
        "Open the door today friend。」 Excited, wait there.",
        "She said «open the door today friend.» Excited, wait there.",
        "Hello」 there today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        for size in (1, 3, 5):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert stream_text(chunks, 80).split() == once


def test_a_streamed_ref_note_is_not_spoken() -> None:
    sample = "Open the door today friend.<ref>secret note</ref> Then wait please."
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert "secret" not in once
    for size in (1, 3):
        chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
        assert stream_text(chunks, 40).split() == once


def test_a_streamed_url_mark_does_not_eat_the_next_word() -> None:
    samples = (
        "See https://example.com/a!today friend please.",
        "See https://example.com/a;today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert once == ["See", "today", "friend", "please."]
        assert stream_text(list(sample), 40).split() == once


def test_a_streamed_escaped_break_tag_is_not_spoken() -> None:
    sample = "Open the door&lt;br&gt;today friend please."
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert once == ["Open", "the", "door", "today", "friend", "please."]
    assert stream_text(list(sample), 40).split() == once


def test_a_streamed_spaced_break_tag_is_not_spoken() -> None:
    sample = "Open the door</ p>today friend please."
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert "<" not in once
    assert stream_text(list(sample), 40).split() == once


def test_a_streamed_details_block_does_not_join_the_words() -> None:
    sample = "<details><summary>Open the door</summary>secret note today friend</details>"
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert "doorsecret" not in once
    assert stream_text(list(sample), 40).split() == once


def test_a_streamed_footnote_definition_is_the_note() -> None:
    sample = "Open the door today friend.\n\n[^1]: the secret note is here today\n"
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert ":" not in once
    assert "secret" in once
    assert stream_text(list(sample), 40).split() == once


def test_a_streamed_tilde_fence_is_not_spoken() -> None:
    sample = "Open the door~~~\nprint(1)\n~~~today friend please."
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert "print" not in once
    assert stream_text(list(sample), 40).split() == once


def test_a_streamed_rule_or_quote_matches_one_shot() -> None:
    samples = (
        "Open the door today friend.\n---\nThen open the window today friend.",
        "> Open the door today friend please.",
        "The score is 2 > 1 today friend please.",
        "Wait---then open the door today friend.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert stream_text(list(sample), 40).split() == once


def test_a_streamed_image_or_script_does_not_join_the_words() -> None:
    samples = (
        "Open the door![the cat](https://example.com/a.png)today friend please.",
        "Open the door<script>alert(1)</script>today friend please.",
        "Open the door<style>body{}</style>today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert "doortoday" not in once
        assert "doorthe" not in once
        assert stream_text(list(sample), 40).split() == once


def test_a_streamed_hidden_span_does_not_join_the_words() -> None:
    samples = (
        "Open the door<think>secret plan</think>today friend please.",
        "Open the door<https://example.com>today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert "doortoday" not in once
        assert stream_text(list(sample), 40).split() == once


def test_a_streamed_removed_span_does_not_join_the_words() -> None:
    samples = (
        "Open the door<!-- secret -->today friend please.",
        "Open the door```print(1)```today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert "doortoday" not in once
        chunks = list(sample)
        assert stream_text(chunks, 40).split() == once


def test_a_streamed_glued_mark_does_not_join_the_words() -> None:
    samples = (
        "Open the door**today** friend please now.",
        "Open the door~~secret~~today friend please.",
        "Open the door[^1]today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert "doortoday" not in once
        assert "secret" not in once
        chunks = list(sample)
        assert stream_text(chunks, 40).split() == once


def test_a_streamed_speaker_mark_does_not_join_the_words() -> None:
    sample = "Open the door[S1]today friend please."
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    assert "doortoday" not in once
    chunks = list(sample)
    assert stream_text(chunks, 40).split() == once


def test_a_streamed_link_does_not_join_the_next_word() -> None:
    samples = (
        "See [the docs](https://example.com/a)today friend please.",
        "See ![the door](https://example.com/a.png)today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        assert "docstoday" not in once
        assert "doortoday" not in once
        chunks = list(sample)
        assert stream_text(chunks, 40).split() == once


def test_a_streamed_reference_link_keeps_the_words() -> None:
    sample = "See the [docs][ref] today friend please."
    once = normalize_cues(scrub_tts(sample), lead=True).split()
    for size in (1, 3):
        chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
        assert stream_text(chunks, 40).split() == once


def test_a_streamed_fullwidth_aside_does_not_join_the_words() -> None:
    samples = (
        "Please open the door（quietly）today friend.",
        "你好（悄悄地）今天开门朋友。",
        "Call （555） today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        for size in (1, 3):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert stream_text(chunks, 40).split() == once


def test_a_break_tag_keeps_the_word_boundary() -> None:
    samples = (
        "hello<br>there today friend.",
        "hello<br/>there today friend.",
        "Open the door<br>then close it today friend.",
        "Open the door</p>then close it today friend.",
        "hello<hr>there today friend.",
        "hello<p>there today friend.",
        "left<td>right today friend.",
        "hello<pre>code</pre>there today friend.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True).split()
        for size in (1, 2, 3):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert stream_text(chunks, 24).split() == once


def test_three_character_chunks_match_a_one_shot_scrub() -> None:
    # A span split finer than its opener used to be spoken, or the words
    # after it were dropped. The one-shot scrub is the speech we want.
    samples = (
        "<thinking>secret plan</thinking> Hello there friend today.",
        "<whisper>come closer now</whisper> and stay quiet today friend.",
        "Here is code ```print(1)``` and then more words today friend.",
        "### Title\nThe body is spoken today friend please.",
        "Wait for https://example.com/ab today friend please.",
        "Autolink <https://example.com/path> and then more words today friend.",
        "Wiki [[page]] and then more words today friend please.",
        "2 < 3 is still spoken today friend please.",
        "Use `code` in the sentence today friend please now.",
        "**bold words** are spoken today friend please now.",
        "2 * 3 equals six today friend please.",
        "2*3 equals six today friend please now really.",
        "End with `code`.",
        "Code `a[i]` stays in the sentence today friend.",
        "Use the key `a[i][j]` in the code today friend.",
        "A _word_ in the sentence today friend please now.",
        "Hello there.\n* first item is spoken today friend please.",
        "Done.Excited, hi there friend.",
        "Done…Excited, hi there friend.",
        "Hello!Excited, hi there friend.",
        "It is 3.14 exactly today friend.",
        "Dr. Happy birthday today friend.",
        "Before <!-- hidden note --> after today friend.",
        "<!-- secret plan --> Hello there friend today.",
        "<script>alert(1)</script> Hello there friend today.",
        "<style>body { color: red }</style> Hello there friend today.",
        "Hello there friend.\u2028Excited, hi there friend.",
        "Done.\u200bExcited, hi there friend.",
        "Excited， hello there friend.",
        "Hello there friend。Excited， hi there friend.",
        "Meet at 10:30. Excited, hi there friend.",
        "See https://example.com. Excited, hi there friend.",
        "*Excited,* the door is open today friend.",
        "hello&#10;there today friend.",
        "hello&#32;there today friend.",
        "Hello&#160;there today friend.",
        "See the note[^1] today friend.",
        "Use <textarea>secret words</textarea> today friend.",
        'Click <a href="https://example.com">here</a> today friend.',
        "Phoneme <|phoneme|> stays today friend.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample), lead=True)
        for size in (1, 2, 3):
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            heard = stream_text(chunks, 40)
            assert heard.split() == once.split()


def test_streaming_space_token_stays_between_words() -> None:
    tts = FishSpeaker(api_key="k", voice_id="v", partial_chars=20)
    sentence = "The quick brown fox jumps today."
    deltas: list[str] = []
    for word in sentence.split(" "):
        deltas.append(word)
        deltas.append(" ")

    async def collect(pieces: list[str]) -> str:
        events = [event async for event in stream_events(tts, pieces, threading.Event())]
        return "".join(event.text for event in events if isinstance(event, TextEvent))

    assert asyncio.run(collect(deltas)).split() == sentence.split()
    assert asyncio.run(collect(list(sentence))).split() == sentence.split()


def test_streaming_deltas_cut_inside_the_window() -> None:
    tts = FishSpeaker(api_key="k", voice_id="v", partial_chars=20)
    cancel = threading.Event()

    async def collect() -> list[TextEvent | FlushEvent]:
        return [event async for event in stream_events(tts, ["word "] * 30, cancel)]

    events = asyncio.run(collect())
    texts = [event.text for event in events if isinstance(event, TextEvent)]
    assert len(texts) > 1
    assert all(len(piece) <= 20 for piece in texts[:-1])
    assert "".join(texts).split() == ["word"] * 30
    assert isinstance(events[-1], FlushEvent)

    async def cancelled() -> list[TextEvent | FlushEvent]:
        stream = stream_events(tts, ["word "] * 30, cancel)
        first = await anext(stream)
        cancel.set()
        rest = [event async for event in stream]
        return [first, *rest]

    stopped = asyncio.run(cancelled())
    assert len(stopped) == 1
    assert isinstance(stopped[0], TextEvent)


def test_stream_scrub_leaves_a_mood_word_alone_by_default() -> None:
    tts = FishSpeaker(api_key="k", voice_id="v", partial_chars=40)

    async def collect(mood_lead: bool) -> str:
        events = delta_events(
            ["Curious, ", "isn't it today my friend."],
            threading.Event(),
            partial_chars=tts.partial_chars,
            mood_lead=mood_lead,
        )
        return "".join([e.text async for e in events if isinstance(e, TextEvent)])

    assert asyncio.run(collect(False)).split()[:2] == ["Curious,", "isn't"]
    assert asyncio.run(collect(True)).split()[0] == "[curious]"


def test_early_flush_speaks_the_first_piece_then_flushes_again_at_the_end() -> None:
    async def run() -> list[object]:
        tts = FishSpeaker(api_key="k", voice_id="v")
        deltas = ["Hello there my friend. ", "How are you doing today? ", "I hope it is well."]
        return [
            event
            async for event in delta_events(
                deltas, threading.Event(), partial_chars=tts.partial_chars, early_flush=True
            )
        ]

    kinds = event_kinds(asyncio.run(run()))
    assert kinds[0] == "TextEvent"
    assert kinds[1] == "FlushEvent"
    assert kinds[-1] == "FlushEvent"
    assert kinds.count("FlushEvent") == 2


def test_early_flush_with_one_piece_does_not_flush_twice() -> None:
    async def run() -> list[object]:
        return [
            event
            async for event in delta_events(
                ["Hello there my friend."],
                threading.Event(),
                partial_chars=FishSpeaker(api_key="k", voice_id="v").partial_chars,
                early_flush=True,
            )
        ]

    assert event_kinds(asyncio.run(run())) == ["TextEvent", "FlushEvent"]


def test_early_flush_on_an_empty_reply_sends_nothing() -> None:
    async def run() -> list[object]:
        return [
            event
            async for event in delta_events(
                [], threading.Event(), partial_chars=50, early_flush=True
            )
        ]

    assert asyncio.run(run()) == []


def test_early_flush_is_not_sent_after_a_cancel() -> None:
    cancel = threading.Event()

    async def run() -> list[str]:
        kinds: list[str] = []
        async for event in delta_events(
            ["Hello there my friend. ", "How are you doing today?"],
            cancel,
            partial_chars=40,
            early_flush=True,
        ):
            kinds.append(type(event).__name__)
            if type(event).__name__ == "TextEvent":
                cancel.set()
        return kinds

    assert asyncio.run(run()) == ["TextEvent"]


def test_a_mood_word_inside_a_long_sentence_is_not_rewritten_as_a_cue() -> None:
    async def run() -> str:
        # Exactly one full piece goes out, so nothing is left in the ready buffer
        # when the next token starts with a mood word.
        pieces = ["a " * 20, "Excited, people talk. "]
        text = ""
        async for event in delta_events(
            pieces, threading.Event(), partial_chars=40, mood_lead=True
        ):
            text += getattr(event, "text", "") or ""
        return text

    spoken = asyncio.run(run())
    assert "[excited]" not in spoken.lower()
    assert "Excited" in spoken
