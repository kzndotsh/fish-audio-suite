"""Turn timed text into SRT and WebVTT subtitles.

Run: uv run python examples/captions.py (offline, no key needed)

A transcript is more useful with timestamps. Video players read two common
subtitle formats:

- SRT, the older one, used by most desktop players.
- WebVTT (`.vtt`), the one web browsers use for the `<track>` element.

Both describe the same thing, a list of "cues": show this text from this time
to that time. The kit builds either file from a list of `CaptionCue` values,
so you never have to format timestamps by hand.

You will see the same three cues printed first as SRT, then as WebVTT.
"""

from fish_audio_suite_kit import CaptionCue, format_as_srt, format_as_vtt

# Each `CaptionCue` is `(start, end, text)`, with times in seconds. In a real
# app these come from a speech-to-text result. Here they are written by hand
# to show three things the formatters handle for you:
CUES = [
    # A plain line at the very start.
    CaptionCue(0.0, 1.8, "Welcome back."),
    # Characters that mean something in WebVTT. `format_as_vtt` escapes `&` and
    # `<` so a browser shows them as text instead of reading them as markup.
    CaptionCue(1.8, 4.25, 'Today: "voices" & <tags>.'),
    # A cue that ends past the one-hour mark, to show the hours field.
    CaptionCue(65.5, 3725.0, "An hour in, the clock keeps counting."),
]


def main() -> None:
    # `format_as_srt` numbers each cue and writes times as `00:00:01,800`
    # (comma before the milliseconds).
    print(format_as_srt(CUES))
    # `format_as_vtt` starts with a `WEBVTT` header and writes times as
    # `00:00:01.800` (dot before the milliseconds).
    print(format_as_vtt(CUES))


if __name__ == "__main__":
    main()
