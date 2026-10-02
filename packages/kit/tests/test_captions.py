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
