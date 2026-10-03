from __future__ import annotations

from fish_audio_suite_kit import (
    CaptionCue,
    format_as_srt,
    format_as_vtt,
)


def test_blank_caption_is_omitted_and_end_cannot_precede_start() -> None:
    rendered = format_as_srt(
        [
            CaptionCue(0.0, 1.0, "   "),
            CaptionCue(2.0, 1.0, "kept"),
        ]
    )
    assert rendered == "1\n00:00:02,000 --> 00:00:02,001\nkept\n"


def test_cues_past_one_hundred_hours_keep_a_later_end() -> None:
    # 359999.9 s is 99:59:59.900 and 360005 s is 100:00:05. As text "100:..." sorts
    # before "99:...", which used to collapse the end onto the start.
    rendered = format_as_srt([CaptionCue(359999.9, 360005.0, "late")])
    assert rendered == "1\n99:59:59,900 --> 100:00:05,000\nlate\n"
    inverted = format_as_srt([CaptionCue(360005.0, 359999.9, "late")])
    assert inverted == "1\n100:00:05,000 --> 100:00:05,001\nlate\n"
    assert "100:00:05.000" in format_as_vtt([CaptionCue(359999.9, 360005.0, "late")])


def test_a_zero_length_or_non_finite_cue_still_gets_one_millisecond() -> None:
    zero = format_as_srt([CaptionCue(1.5, 1.5, "x")])
    assert zero == "1\n00:00:01,500 --> 00:00:01,501\nx\n"
    nan = format_as_srt([CaptionCue(2.0, float("nan"), "x")])
    assert nan == "1\n00:00:02,000 --> 00:00:02,001\nx\n"
    inf = format_as_srt([CaptionCue(float("inf"), float("inf"), "x")])
    assert inf == "1\n00:00:00,000 --> 00:00:00,001\nx\n"


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
