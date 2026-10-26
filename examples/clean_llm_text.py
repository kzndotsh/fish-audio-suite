"""Clean messy LLM output for speech, then cut it into speakable pieces.

Run: uv run python examples/clean_llm_text.py (offline, no key needed)

Language models write for the screen, not for the ear. Their replies carry
markdown (`**bold**`), links, and sometimes hidden "thinking" blocks. A
text-to-speech engine reads all of that out loud unless you remove it first.

This example does two things with the `kit` package, which is pure text
handling with no network calls:

1. Cleans one messy reply so only the words meant to be spoken are left.
2. Pretends the reply arrives a few characters at a time, the way a streaming
   LLM sends it, and shows where it is safe to hand a piece to Fish early.

You will see the cleaned text, then one line per piece that would be sent.
"""

from fish_audio_suite_kit import next_tts_cut, normalize_cues, scrub_tts

# A typical messy reply. It has a `<think>` block the user should never hear,
# markdown bold, a markdown link, an oddly capitalized cue tag, and extra spaces.
# Cue tags in square brackets, such as `[happy]` or `[calm]`, are instructions
# to Fish about tone. They are not spoken, so the cleanup keeps them.
LLM_OUTPUT = (
    "<think>The user wants a greeting.</think>\n"
    "**Hello there!**  [Happy]  Dr. Smith said to read [the guide](https://example.com/docs) first. "
    "It takes five minutes, and then you are ready to start. [calm] Any questions?"
)


def main() -> None:
    # Step 1. Clean the whole reply.
    # `scrub_tts` removes the `<think>` block, the `**` marks, and turns the link
    # into its plain words ("the guide"). `normalize_cues` then rewrites cue tags
    # into the exact form Fish expects, so `[Happy]` becomes `[happy]`. Run them
    # in this order: scrub first, then normalize.
    cleaned = normalize_cues(scrub_tts(LLM_OUTPUT))
    print(f"cleaned: {cleaned!r}")

    # Step 2. Cut a stream into pieces.
    # A real LLM sends its reply in small chunks. Waiting for the whole reply
    # before speaking makes the user wait too. Instead, keep a buffer and ask
    # `next_tts_cut` whether the start of the buffer is ready to send.
    #
    # `next_tts_cut` returns the index where a piece ends, or `-1` for "not yet,
    # keep buffering". It prefers to cut at the end of a sentence, but it knows
    # that `Dr.` is not one. If a sentence runs long, it cuts at a space near
    # 40 characters, so the first audio does not wait for a long sentence.
    buf = ""
    # Feed the cleaned text in 7-character chunks to imitate a streaming model.
    for i in range(0, len(cleaned), 7):
        buf += cleaned[i : i + 7]
        cut = next_tts_cut(buf)
        # One chunk can complete more than one piece, so keep cutting until
        # the buffer has nothing ready.
        while cut > 0:
            print(f"piece: {buf[:cut]!r}")
            buf = buf[cut:]
            cut = next_tts_cut(buf)
    # The stream has ended. Whatever is left in `buf` is the final piece.
    if buf:
        print(f"piece: {buf!r}")


if __name__ == "__main__":
    main()
