from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace

import httpx
import ormsgpack
import pytest
from fishaudio import AsyncFishAudio
from fishaudio.exceptions import RateLimitError, ValidationError, WebSocketError
from fishaudio.types import TTSConfig

from fish_audio_suite_kit import FISH_RETRY_ATTEMPTS, SuiteDefaults, normalize_cues, scrub_tts
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.session import _HeldClient, run_turn
from fish_audio_suite_voice.wire import (
    EventAcc,
    FlushEvent,
    Heard,
    TextEvent,
    TurnRun,
    TurnSpec,
    _pump_ws_audio,
    _spoken_prefix,
    quiet_shutdown,
    text_events,
    turn_failure,
)


class _Sink:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def start(self) -> None:
        return None

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    def finish(self, *, kill: bool = False) -> None:
        return None

    def bytes_played(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)


def _run() -> tuple[TurnRun, _Sink]:
    sink = _Sink()
    spec = TurnSpec(
        api_key="k",
        base_url="https://api.fish.audio",
        voice_id="v",
        model="s2.1-pro",
        audio_format="pcm",
        latency="balanced",
        speed=1.0,
        sample_rate=44100,
        partial_chars=40,
        trace_headers={},
        config=TTSConfig(),
    )
    run = TurnRun(
        spec=spec,
        sink=sink,
        cancel=threading.Event(),
        sent_text="",
        acc=EventAcc(),
        t0=time.perf_counter(),
        audio=Heard(),
    )
    return run, sink


def _events(prepared: str, cancel: threading.Event, partial: int) -> list[TextEvent | FlushEvent]:
    async def collect() -> list[TextEvent | FlushEvent]:
        return [event async for event in text_events(prepared, cancel, partial)]

    return asyncio.run(collect())


def test_text_events_flush_only_after_text() -> None:
    cancel = threading.Event()
    assert _events("", cancel, 40) == []
    assert _events("   ", cancel, 40) == []
    events = _events("word " * 30, cancel, 20)
    texts = [event.text for event in events if isinstance(event, TextEvent)]
    assert len(texts) > 1
    assert all(len(piece) <= 20 for piece in texts[:-1])
    assert isinstance(events[-1], FlushEvent)


def test_text_events_cancel_drops_the_flush() -> None:
    cancel = threading.Event()

    async def collect() -> list[TextEvent | FlushEvent]:
        stream = text_events("word " * 30, cancel, 20)
        first = await anext(stream)
        cancel.set()
        rest = [event async for event in stream]
        return [first, *rest]

    events = asyncio.run(collect())
    assert len(events) == 1
    assert isinstance(events[0], TextEvent)


def _delta_text(deltas: list[str], partial_chars: int) -> str:
    tts = IsolatedFishTts(api_key="k", voice_id="v", partial_chars=partial_chars)

    async def collect() -> str:
        events = [event async for event in tts._delta_events(deltas, threading.Event())]
        return "".join(event.text for event in events if isinstance(event, TextEvent))

    return asyncio.run(collect())


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
        streamed = _delta_text(parts, 24)
        assert streamed.split() == normalize_cues(scrub_tts(sample)).split()


def test_split_strikethrough_is_not_spoken() -> None:
    whole = _delta_text(
        ["I meant ", "~~Tuesday~~", " Wednesday for the meeting."],
        40,
    )
    assert "Tuesday" not in whole
    assert "Wednesday" in whole
    split = _delta_text(
        ["I meant ~~Tues", "day~~ Wednesday for the meeting."],
        40,
    )
    assert "Tuesday" not in split
    assert "Wednesday" in split
    kept = _delta_text(["Meet ", "~softly~", " on Wednesday please."], 40)
    assert "softly" in kept


def test_streaming_deltas_keep_a_split_span_intact() -> None:
    thought = _delta_text(
        [
            "<thinking>do not say this secret ",
            "plan</thinking> Hello there friend from the office. ",
        ],
        24,
    )
    assert "secret" not in thought
    assert "Hello there friend" in thought
    cutoff = _delta_text(
        [
            "Hello there friend from the office. ",
            "<thinking>do not say this secret plan",
        ],
        24,
    )
    assert "secret" not in cutoff
    assert "Hello there friend" in cutoff
    link = _delta_text(
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
    pieces = _delta_text(
        ["See ", "[docs]", "(", "https://example.com/path/here", ")", " later today friend."],
        24,
    )
    assert pieces.split() == ["See", "docs", "later", "today", "friend."]
    whisper = _delta_text(
        ["<whisper>", "come closer please", "</whisper>", " and stay quiet today."],
        24,
    )
    assert "<whisper" not in whisper
    assert "[whispering]" in whisper
    assert "come closer please" in whisper
    assert "stay quiet today" in whisper
    heading = _delta_text(["#", " ", "Heading then the real sentence continues today."], 24)
    assert "#" not in heading
    assert "Heading then the real sentence" in heading
    continued = _delta_text(["I am ", "excited, ", "hello there friend today please."], 24)
    assert "[excited]" not in continued
    assert "excited," in continued
    fresh = _delta_text(["Excited, ", "hello there friend today."], 40)
    assert fresh.split()[:2] == ["[excited]", "hello"]
    split_lead = "Excited, the news is good today friend."
    split_chunks = [split_lead[i : i + 3] for i in range(0, len(split_lead), 3)]
    assert _delta_text(split_chunks, 40).split()[:2] == ["[excited]", "the"]
    follow = "Hello there friend. Excited, goodbye now please."
    follow_chunks = [follow[i : i + 3] for i in range(0, len(follow), 3)]
    followed = _delta_text(follow_chunks, 40)
    assert "Hello there friend." in followed
    assert "[excited]" in followed
    assert "Excited," not in followed
    mid_lead = "I am excited, really glad today friend."
    mid_chunks = [mid_lead[i : i + 3] for i in range(0, len(mid_lead), 3)]
    heard = _delta_text(mid_chunks, 40)
    assert "[excited]" not in heard
    assert "excited," in heard
    titled = "Dr. Happy, the patient arrived today friend."
    titled_chunks = [titled[i : i + 3] for i in range(0, len(titled), 3)]
    titled_heard = _delta_text(titled_chunks, 40)
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
        one_shot = normalize_cues(scrub_tts(sample))
        for size in (1, 2, 3):
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            assert _delta_text(chunks, 40).split() == one_shot.split()
    glued = "Hello؟Excited, yes today friend."
    glued_chunks = [glued[i : i + 2] for i in range(0, len(glued), 2)]
    glued_heard = _delta_text(glued_chunks, 40)
    assert "[excited]" not in glued_heard
    assert "Excited," in glued_heard
    item = _delta_text(["*", " ", "first star item is spoken today friend"], 40)
    assert "*" not in item
    assert "first star item" in item
    dashed = _delta_text(["-", " ", "first dash item is spoken today friend"], 40)
    assert "-" not in dashed
    assert "first dash item" in dashed
    task = _delta_text(["- [ ] ", "first task is spoken today friend."], 40)
    assert "[" not in task
    assert "first task" in task
    added = _delta_text(["2", " ", "-", " ", "3 equals negative one today friend."], 40)
    assert added.split()[:3] == ["2", "-", "3"]
    product = _delta_text(["2", " ", "*", " ", "3 equals six today friend please."], 40)
    assert product.split()[:3] == ["2", "*", "3"]
    fenced = _delta_text(
        ["Here is code ", "```", "print(1)", "```", " and then more words today friend."],
        40,
    )
    assert "```" not in fenced
    assert "print(1)" not in fenced
    assert "more words today friend" in fenced
    after_sentence = _delta_text(list("Hello there friend. * 3 equals six today please."), 12)
    assert after_sentence.split()[:5] == ["Hello", "there", "friend.", "*", "3"]
    hashed = _delta_text(list("See the note. # not a heading today friend."), 12)
    assert "#" in hashed
    assert "not a heading" in hashed
    bullet = _delta_text(["Hello.\n", "*", " ", "first item is spoken today friend."], 12)
    assert "*" not in bullet
    assert "first item" in bullet
    aside = _delta_text(
        [
            "Hello there friend (do not read this ",
            "aside) and more words follow. ",
        ],
        24,
    )
    assert "aside" not in aside
    assert "Hello there friend" in aside
    assert "more words follow" in aside
    cutoff = _delta_text(
        [
            "Hello there friend from the office. ",
            "(do not read this aside",
        ],
        24,
    )
    assert "aside" not in cutoff
    assert "Hello there friend" in cutoff


def test_partial_cut_does_not_split_a_cue() -> None:
    # A 40-character hard cut used to send "[whi" and then "spering]".
    # A period inside the cue used to send "[hello." and then "there]".
    tts = IsolatedFishTts(api_key="k", voice_id="v", partial_chars=40)

    async def pieces_of(sample: str) -> list[str]:
        events = [event async for event in tts._delta_events([sample], threading.Event())]
        return [event.text for event in events if isinstance(event, TextEvent)]

    pieces = asyncio.run(pieces_of("x" * 36 + "[whispering] come closer today friend please."))
    assert pieces
    assert all(piece.count("[") == piece.count("]") for piece in pieces)
    assert any("[whispering]" in piece for piece in pieces)
    noted = asyncio.run(pieces_of("Note [hello. there] and then more words please today friend."))
    assert noted
    assert all(piece.count("[") == piece.count("]") for piece in noted)
    assert any("[hello. there]" in piece for piece in noted)


def test_held_span_longer_than_the_window_is_still_spoken() -> None:
    # The open parenthesis is held until the reply ends. A phone number
    # stays; a stage aside is still removed. The first cut used to be the
    # only text that reached Fish.
    kept = "Unclosed (555 and then the rest of the sentence today."
    dropped = "Unclosed (aside and then the rest of the sentence today."
    for sample in (kept, dropped):
        once = normalize_cues(scrub_tts(sample)).split()
        for size in (1, 2, 3):
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            assert _delta_text(chunks, 40).split() == once


def test_a_streamed_url_keeps_the_period_on_the_word() -> None:
    samples = (
        "See the door (https://example.com). Excited, hi there friend.",
        "Open https://example.com, then close the door today friend.",
        "打开 https://example.com，然后关门今天朋友。",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample))
        for size in (1, 2, 3):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            spoken = _delta_text(chunks, 24)
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
        once = normalize_cues(scrub_tts(sample)).split()
        for size in (1, 2, 4):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert _delta_text(chunks, 80).split() == once


def test_a_streamed_line_separator_still_cues_the_next_mood() -> None:
    samples = (
        "Please open the door today friend\rExcited, wait there.",
        "Please open the door today friend\u2028Excited, wait there.",
        "Please open the door today friend\u2029Excited, wait there.",
        "Please open the door today friend\rthen close it today.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        for size in (1, 2, 4):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert _delta_text(chunks, 80).split() == once


def test_a_streamed_quote_after_a_stop_still_cues_the_next_mood() -> None:
    samples = (
        "Open the door today friend。」 Excited, wait there.",
        "She said «open the door today friend.» Excited, wait there.",
        "Hello」 there today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        for size in (1, 3, 5):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert _delta_text(chunks, 80).split() == once


def test_a_streamed_ref_note_is_not_spoken() -> None:
    sample = "Open the door today friend.<ref>secret note</ref> Then wait please."
    once = normalize_cues(scrub_tts(sample)).split()
    assert "secret" not in once
    for size in (1, 3):
        chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
        assert _delta_text(chunks, 40).split() == once


def test_a_streamed_url_mark_does_not_eat_the_next_word() -> None:
    samples = (
        "See https://example.com/a!today friend please.",
        "See https://example.com/a;today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert once == ["See", "today", "friend", "please."]
        assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_escaped_break_tag_is_not_spoken() -> None:
    sample = "Open the door&lt;br&gt;today friend please."
    once = normalize_cues(scrub_tts(sample)).split()
    assert once == ["Open", "the", "door", "today", "friend", "please."]
    assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_spaced_break_tag_is_not_spoken() -> None:
    sample = "Open the door</ p>today friend please."
    once = normalize_cues(scrub_tts(sample)).split()
    assert "<" not in once
    assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_details_block_does_not_join_the_words() -> None:
    sample = "<details><summary>Open the door</summary>secret note today friend</details>"
    once = normalize_cues(scrub_tts(sample)).split()
    assert "doorsecret" not in once
    assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_footnote_definition_is_the_note() -> None:
    sample = "Open the door today friend.\n\n[^1]: the secret note is here today\n"
    once = normalize_cues(scrub_tts(sample)).split()
    assert ":" not in once
    assert "secret" in once
    assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_tilde_fence_is_not_spoken() -> None:
    sample = "Open the door~~~\nprint(1)\n~~~today friend please."
    once = normalize_cues(scrub_tts(sample)).split()
    assert "print" not in once
    assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_rule_or_quote_matches_one_shot() -> None:
    samples = (
        "Open the door today friend.\n---\nThen open the window today friend.",
        "> Open the door today friend please.",
        "The score is 2 > 1 today friend please.",
        "Wait---then open the door today friend.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_image_or_script_does_not_join_the_words() -> None:
    samples = (
        "Open the door![the cat](https://example.com/a.png)today friend please.",
        "Open the door<script>alert(1)</script>today friend please.",
        "Open the door<style>body{}</style>today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert "doortoday" not in once
        assert "doorthe" not in once
        assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_hidden_span_does_not_join_the_words() -> None:
    samples = (
        "Open the door<think>secret plan</think>today friend please.",
        "Open the door<https://example.com>today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert "doortoday" not in once
        assert _delta_text(list(sample), 40).split() == once


def test_a_streamed_removed_span_does_not_join_the_words() -> None:
    samples = (
        "Open the door<!-- secret -->today friend please.",
        "Open the door```print(1)```today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert "doortoday" not in once
        chunks = list(sample)
        assert _delta_text(chunks, 40).split() == once


def test_a_streamed_glued_mark_does_not_join_the_words() -> None:
    samples = (
        "Open the door**today** friend please now.",
        "Open the door~~secret~~today friend please.",
        "Open the door[^1]today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert "doortoday" not in once
        assert "secret" not in once
        chunks = list(sample)
        assert _delta_text(chunks, 40).split() == once


def test_a_streamed_speaker_mark_does_not_join_the_words() -> None:
    sample = "Open the door[S1]today friend please."
    once = normalize_cues(scrub_tts(sample)).split()
    assert "doortoday" not in once
    chunks = list(sample)
    assert _delta_text(chunks, 40).split() == once


def test_a_streamed_link_does_not_join_the_next_word() -> None:
    samples = (
        "See [the docs](https://example.com/a)today friend please.",
        "See ![the door](https://example.com/a.png)today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        assert "docstoday" not in once
        assert "doortoday" not in once
        chunks = list(sample)
        assert _delta_text(chunks, 40).split() == once


def test_a_streamed_reference_link_keeps_the_words() -> None:
    sample = "See the [docs][ref] today friend please."
    once = normalize_cues(scrub_tts(sample)).split()
    for size in (1, 3):
        chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
        assert _delta_text(chunks, 40).split() == once


def test_a_streamed_fullwidth_aside_does_not_join_the_words() -> None:
    samples = (
        "Please open the door（quietly）today friend.",
        "你好（悄悄地）今天开门朋友。",
        "Call （555） today friend please.",
    )
    for sample in samples:
        once = normalize_cues(scrub_tts(sample)).split()
        for size in (1, 3):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert _delta_text(chunks, 40).split() == once


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
        once = normalize_cues(scrub_tts(sample)).split()
        for size in (1, 2, 3):
            chunks = [sample[index : index + size] for index in range(0, len(sample), size)]
            assert _delta_text(chunks, 24).split() == once


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
        once = normalize_cues(scrub_tts(sample))
        for size in (1, 2, 3):
            chunks = [sample[i : i + size] for i in range(0, len(sample), size)]
            heard = _delta_text(chunks, 40)
            assert heard.split() == once.split()


def test_sdk_limits_are_applied_before_the_socket() -> None:
    tts = IsolatedFishTts(
        api_key="k",
        voice_id="v",
        latency="low",
        chunk_length=800,
        volume=25,
        speed=4,
        temperature=1.5,
        top_p=-0.2,
    )
    spec = tts._spec()
    assert spec.latency == "balanced"
    loud = IsolatedFishTts(api_key="k", voice_id="v", latency=" LOW ")._spec()
    assert loud.latency == "balanced"
    spaced = IsolatedFishTts(api_key="k", voice_id="v", latency=" balanced ")._spec()
    assert spaced.latency == "balanced"
    assert spec.speed == 2.0
    assert spec.config.temperature == 1.0
    assert spec.config.top_p == 0.0
    assert spec.config.chunk_length == 300
    assert spec.config.prosody is not None
    assert spec.config.prosody.volume == 20.0
    quiet = IsolatedFishTts(api_key="k", voice_id="v", volume=-40)
    prosody = quiet._tts_config().prosody
    assert prosody is not None
    assert prosody.volume == -20.0


def test_voice_id_and_model_survive_the_socket_start() -> None:
    spec = IsolatedFishTts(
        api_key="k",
        voice_id="voice-\ud800",
        model="s2.1-pro\nbad",
    )._spec()
    assert "\ud800" not in spec.voice_id
    assert spec.voice_id.startswith("voice-")
    assert spec.model == SuiteDefaults().tts_model
    assert "\n" not in spec.model
    ormsgpack.packb({"reference_id": spec.voice_id})
    kept = IsolatedFishTts(api_key="k", voice_id="v", model="MyModel")._spec()
    assert kept.model == "MyModel"


def test_zero_sample_rate_is_replaced_before_the_socket() -> None:
    for rate in (0, 2**32):
        spec = IsolatedFishTts(api_key="k", voice_id="v", sample_rate=rate)._spec()
        assert spec.sample_rate == SuiteDefaults().sample_rate
        assert spec.config.sample_rate == SuiteDefaults().sample_rate


def test_pcm16_is_a_format_the_socket_accepts() -> None:
    spec = IsolatedFishTts(api_key="k", voice_id="v", audio_format="pcm16")._spec()
    assert spec.audio_format == "pcm"
    assert spec.config.format == "pcm"
    mp3 = IsolatedFishTts(api_key="k", voice_id="v", audio_format="AAC")._spec()
    assert mp3.audio_format == "mp3"
    assert mp3.config.format == "mp3"


def test_streaming_space_token_stays_between_words() -> None:
    tts = IsolatedFishTts(api_key="k", voice_id="v", partial_chars=20)
    sentence = "The quick brown fox jumps today."
    deltas: list[str] = []
    for word in sentence.split(" "):
        deltas.append(word)
        deltas.append(" ")

    async def collect(pieces: list[str]) -> str:
        events = [event async for event in tts._delta_events(pieces, threading.Event())]
        return "".join(event.text for event in events if isinstance(event, TextEvent))

    assert asyncio.run(collect(deltas)).split() == sentence.split()
    assert asyncio.run(collect(list(sentence))).split() == sentence.split()


def test_streaming_deltas_cut_inside_the_window() -> None:
    tts = IsolatedFishTts(api_key="k", voice_id="v", partial_chars=20)
    cancel = threading.Event()

    async def collect() -> list[TextEvent | FlushEvent]:
        return [event async for event in tts._delta_events(["word "] * 30, cancel)]

    events = asyncio.run(collect())
    texts = [event.text for event in events if isinstance(event, TextEvent)]
    assert len(texts) > 1
    assert all(len(piece) <= 20 for piece in texts[:-1])
    assert "".join(texts).split() == ["word"] * 30
    assert isinstance(events[-1], FlushEvent)

    async def cancelled() -> list[TextEvent | FlushEvent]:
        stream = tts._delta_events(["word "] * 30, cancel)
        first = await anext(stream)
        cancel.set()
        rest = [event async for event in stream]
        return [first, *rest]

    stopped = asyncio.run(cancelled())
    assert len(stopped) == 1
    assert isinstance(stopped[0], TextEvent)


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
        == "hello there friend"
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


def test_turn_failure_retries_only_before_audio(capsys: pytest.CaptureFixture[str]) -> None:
    slow = RateLimitError(429, "slow down", None)
    cancel = threading.Event()
    retryable = turn_failure(slow, attempt=0, sent_text="hello", got_audio=False, cancel=cancel)
    assert retryable.retry is True
    heard = turn_failure(slow, attempt=0, sent_text="hello", got_audio=True, cancel=cancel)
    assert heard.retry is False
    assert heard.err_status == 429
    empty = turn_failure(slow, attempt=0, sent_text="", got_audio=False, cancel=cancel)
    assert empty.retry is False
    last = turn_failure(
        slow,
        attempt=FISH_RETRY_ATTEMPTS - 1,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert last.retry is False
    cancel.set()
    barged = turn_failure(slow, attempt=0, sent_text="hello", got_audio=False, cancel=cancel)
    assert barged.retry is False
    assert barged.err_status is None
    assert "no audio" not in capsys.readouterr().err


def test_turn_failure_classifies_socket_validation_and_groups(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cancel = threading.Event()
    socket = turn_failure(
        WebSocketError("dropped"),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert socket.retry is True
    invalid = turn_failure(
        ValidationError("bad voice"),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert invalid.retry is False
    assert invalid.err_status == 400
    grouped = turn_failure(
        BaseExceptionGroup("turn", [RateLimitError(503, "down", None)]),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert grouped.retry is True
    hidden = turn_failure(
        BaseExceptionGroup(
            "turn",
            [asyncio.CancelledError(), RateLimitError(503, "down", None)],
        ),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert hidden.retry is True
    only_cancel = turn_failure(
        BaseExceptionGroup("turn", [asyncio.CancelledError()]),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert only_cancel.retry is False
    refused = turn_failure(
        httpx.ConnectError("refused"),
        attempt=0,
        sent_text="hello",
        got_audio=False,
        cancel=cancel,
    )
    assert refused.retry is True
    assert "status=502" in capsys.readouterr().err


def test_pump_does_not_play_audio_that_arrives_after_cancel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run, sink = _run()
    closed = 0

    async def close_client() -> None:
        nonlocal closed
        closed += 1

    async def chunks():
        yield b"\x01\x02"
        run.cancel.set()
        yield b"\x03\x04"
        yield b""

    async def pump() -> None:
        await _pump_ws_audio(chunks(), run, close_client)

    asyncio.run(pump())
    assert sink.chunks == [b"\x01\x02"]
    assert run.audio.got_audio is True
    assert run.audio.ttfa_ms is not None
    assert closed == 1
    assert "[tts first audio ttfa]" not in capsys.readouterr().out


def test_run_turn_closes_the_sink_when_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"n": 0}

    async def send_turn(*_args: object, **_kwargs: object) -> None:
        called["n"] += 1

    class _Boom(_Sink):
        def __init__(self) -> None:
            super().__init__()
            self.opened = False
            self.finished = False

        def start(self) -> None:
            self.opened = True
            raise RuntimeError("dac")

        def finish(self, *, kill: bool = False) -> None:
            del kill
            self.finished = True

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    run, _sink = _run()
    sink = _Boom()

    async def no_events():
        if False:
            yield ""

    with pytest.raises(RuntimeError, match="dac"):
        asyncio.run(run_turn(run.spec, no_events(), sink, threading.Event(), sent_text="hello"))
    assert sink.opened is True
    assert sink.finished is True
    assert called["n"] == 0


def test_run_turn_closes_the_client_when_finish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = {"n": 0}
    original = _HeldClient.close

    async def send_turn(*_args: object, **_kwargs: object) -> None:
        return None

    async def spy_close(self: _HeldClient) -> None:
        closed["n"] += 1
        await original(self)

    class _FinishBoom(_Sink):
        def finish(self, *, kill: bool = False) -> None:
            del kill
            raise RuntimeError("disk")

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    monkeypatch.setattr("fish_audio_suite_voice.session._HeldClient.close", spy_close)
    run, _sink = _run()

    async def no_events():
        if False:
            yield ""

    with pytest.raises(RuntimeError, match="disk"):
        asyncio.run(
            run_turn(
                run.spec,
                no_events(),
                _FinishBoom(),
                threading.Event(),
                sent_text="hello",
            )
        )
    assert closed["n"] == 1


def test_client_close_after_audio_does_not_forget_the_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AsyncFishAudio.close

    async def send_turn(
        _client: object,
        _events: object,
        run: TurnRun,
        *,
        close_client: object,
    ) -> None:
        del close_client
        run.sink.write(b"\x00\x00" * 16_000)
        run.audio.got_audio = True

    async def boom_close(self: AsyncFishAudio) -> None:
        await original(self)
        raise BaseExceptionGroup("close", [asyncio.CancelledError()])

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    monkeypatch.setattr("fish_audio_suite_voice.session.AsyncFishAudio.close", boom_close)
    tts = IsolatedFishTts(api_key="k", voice_id="v", sample_rate=16_000, audio_format="pcm")

    class Sink:
        def __init__(self) -> None:
            self.n = 0

        def start(self) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            self.n += len(chunk)

        def finish(self, *, kill: bool = False) -> None:
            del kill

        def bytes_played(self) -> int:
            return self.n

    result = tts.speak_isolated("Hello there friend.", Sink())
    assert "Hello" in result.spoken_so_far
    assert result.bytes_played == 32_000
    assert result.got_audio is True


def test_run_turn_stops_when_the_api_key_cannot_be_a_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"n": 0}

    async def send_turn(*_args: object, **_kwargs: object) -> None:
        called["n"] += 1

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    run, sink = _run()
    spec = replace(run.spec, api_key="sk-\nbad")

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(run_turn(spec, no_events(), sink, threading.Event(), sent_text="hello"))
    assert called["n"] == 0
    assert result.error_status == 401
    assert result.got_audio is False


def test_run_turn_retries_before_audio_and_stops_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    calls = {"n": 0}

    async def send_turn(
        client: object,
        events: object,
        run: TurnRun,
        *,
        close_client: object,
    ) -> None:
        del client, events, close_client
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError(429, "slow", None)
        run.audio.got_audio = True
        run.sink.write(b"abcd")

    monkeypatch.setattr("fish_audio_suite_voice.session.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    run, sink = _run()

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(
        run_turn(run.spec, no_events(), sink, threading.Event(), sent_text="hello")
    )
    assert calls["n"] == 2
    assert sum(slept) == pytest.approx(1.0)
    assert result.got_audio is True
    assert result.spoken_so_far == "hello"

    calls["n"] = 0
    slept.clear()

    async def fail_after_audio(
        client: object,
        events: object,
        run: TurnRun,
        *,
        close_client: object,
    ) -> None:
        del client, events, close_client
        calls["n"] += 1
        run.audio.got_audio = True
        raise RateLimitError(429, "slow", None)

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", fail_after_audio)
    run, sink = _run()
    failed = asyncio.run(
        run_turn(run.spec, no_events(), sink, threading.Event(), sent_text="hello")
    )
    assert calls["n"] == 1
    assert slept == []
    assert failed.error_status == 429
    assert failed.got_audio is True

    calls["n"] = 0

    async def drop_after_audio(
        client: object,
        events: object,
        run: TurnRun,
        *,
        close_client: object,
    ) -> None:
        del client, events, close_client
        calls["n"] += 1
        run.audio.got_audio = True
        # About 0.4 s at 44.1 kHz int16, enough for the first word only.
        run.sink.write(b"\x00" * 33_075)
        raise WebSocketError("dropped")

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", drop_after_audio)
    run, sink = _run()
    dropped = asyncio.run(
        run_turn(
            run.spec,
            no_events(),
            sink,
            threading.Event(),
            sent_text="hello there friend",
        )
    )
    assert calls["n"] == 1
    assert dropped.error_status is None
    assert dropped.got_audio is True
    assert dropped.spoken_so_far == "hello"


def test_cancel_scope_after_audio_keeps_only_the_played_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    async def send_turn(
        client: object,
        events: object,
        run: TurnRun,
        *,
        close_client: object,
    ) -> None:
        del client, events, close_client
        calls["n"] += 1
        run.audio.got_audio = True
        run.sink.write(b"\x00" * 33_075)
        raise RuntimeError(
            "Attempted to exit cancel scope in a different task than it was entered in"
        )

    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    run, sink = _run()

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(
        run_turn(
            run.spec,
            no_events(),
            sink,
            threading.Event(),
            sent_text="hello there friend",
        )
    )
    assert calls["n"] == 1
    assert result.got_audio is True
    assert result.cancelled is False
    assert result.spoken_so_far == "hello"


def test_retry_backoff_does_not_replay_after_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = threading.Event()
    calls = {"n": 0}

    async def send_turn(
        client: object,
        events: object,
        run: TurnRun,
        *,
        close_client: object,
    ) -> None:
        del client, events, close_client, run
        calls["n"] += 1
        raise RateLimitError(429, "slow", None)

    async def fake_sleep(_seconds: float) -> None:
        cancel.set()

    monkeypatch.setattr("fish_audio_suite_voice.session.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("fish_audio_suite_voice.session.send_turn", send_turn)
    run, sink = _run()

    async def no_events():
        if False:
            yield ""

    result = asyncio.run(run_turn(run.spec, no_events(), sink, cancel, sent_text="hello"))
    assert calls["n"] == 1
    assert result.cancelled is True
    assert result.error_status is None
    assert result.got_audio is False
    assert result.spoken_so_far == ""


def test_quiet_shutdown_cancels_leftover_tasks() -> None:
    async def scene() -> bool:
        hung = asyncio.create_task(asyncio.sleep(30))
        await quiet_shutdown(asyncio.get_running_loop())
        return hung.cancelled()

    assert asyncio.run(scene()) is True
