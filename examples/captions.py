"""Turn timed text into SRT and WebVTT subtitles.

Run: uv run python examples/captions.py (offline, no key needed)
"""

from fish_audio_suite_kit import CaptionCue, format_as_srt, format_as_vtt

CUES = [
    CaptionCue(0.0, 1.8, "Welcome back."),
    CaptionCue(1.8, 4.25, 'Today: "voices" & <tags>.'),
    CaptionCue(65.5, 3725.0, "An hour in, the clock keeps counting."),
]


def main() -> None:
    print(format_as_srt(CUES))
    print(format_as_vtt(CUES))


if __name__ == "__main__":
    main()
