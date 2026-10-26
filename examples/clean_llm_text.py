"""Clean messy LLM output for speech, then cut it into speakable pieces.

Run: uv run python examples/clean_llm_text.py (offline, no key needed)
"""

from fish_audio_suite_kit import next_tts_cut, normalize_cues, scrub_tts

LLM_OUTPUT = (
    "<think>The user wants a greeting.</think>\n"
    "**Hello there!**  [Happy]  Dr. Smith said to read [the guide](https://example.com/docs) first. "
    "It takes five minutes, and then you are ready to start. [calm] Any questions?"
)


def main() -> None:
    cleaned = normalize_cues(scrub_tts(LLM_OUTPUT))
    print(f"cleaned: {cleaned!r}")

    # A model streams a few characters at a time. Send a piece as soon as it can be spoken.
    buf = ""
    for i in range(0, len(cleaned), 7):
        buf += cleaned[i : i + 7]
        cut = next_tts_cut(buf)
        while cut > 0:
            print(f"piece: {buf[:cut]!r}")
            buf = buf[cut:]
            cut = next_tts_cut(buf)
    if buf:
        print(f"piece: {buf!r}")


if __name__ == "__main__":
    main()
