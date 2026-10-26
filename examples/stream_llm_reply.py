"""Speak a reply while the LLM is still writing it.

Run: uv run python examples/stream_llm_reply.py (needs FISH_API_KEY and FISH_VOICE_ID)

In a voice assistant, every second of silence feels long. A language model can
take several seconds to write a full reply, so waiting for the whole reply
before speaking makes the assistant feel slow.

``FishSpeaker.speak_stream`` fixes that. You hand it the reply as it is being
written, one small piece at a time, and it starts speaking as soon as the first
sentence is complete. The listener hears the start of the answer while the
model is still writing the end.

This example uses a fake LLM, a generator that yields a fixed reply in small
chunks with short pauses, so it needs no LLM key. To use a real model, pass
your model's token stream in its place. ``speak_stream`` accepts any iterable
or async iterable of strings.

Needs the same ``.env`` values as ``speak_to_file.py``: ``FISH_API_KEY`` and
``FISH_VOICE_ID``. You will see when the first audio arrived, what was spoken,
and the file path.
"""

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from fish_audio_suite_voice import FileSink, FishSpeaker
from fish_audio_suite_voice.envfile import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
REPLY = (
    "[happy] Sure, I can help with that. First, open the settings page. "
    "Then pick a voice from the list. [calm] That is all it takes."
)


def fake_llm() -> Iterator[str]:
    """Yield the reply 6 characters at a time, like a model streaming tokens."""
    for i in range(0, len(REPLY), 6):
        time.sleep(0.02)
        yield REPLY[i : i + 6]


def main() -> None:
    # Load the key and voice from `.env`, as in `speak_to_file.py`.
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    for name, value in (("FISH_API_KEY", api_key), ("FISH_VOICE_ID", voice_id)):
        if not value:
            sys.exit(f"{name} is not set. Put it in {ROOT / '.env'} or export it.")

    out = ROOT / "tmp" / "stream_llm_reply.wav"
    out.parent.mkdir(exist_ok=True)
    started = time.perf_counter()
    speaker = FishSpeaker(api_key=api_key, voice_id=voice_id)
    # `speak_stream` reads the `fake_llm` generator, cleans each piece, and sends
    # a piece to Fish as soon as a sentence is complete. It returns once the
    # whole reply has been spoken.
    #
    # `on_first_audio` is a callback that fires once, the moment the first audio
    # arrives. A real app might use it to stop a "thinking" animation.
    result = speaker.speak_stream(
        fake_llm(),
        FileSink(out),
        on_first_audio=lambda: print(f"audio began {time.perf_counter() - started:.2f}s in"),
    )
    if result.error:
        sys.exit(f"Fish failed: {type(result.error).__name__}: {result.error.message}")
    print(f"first audio: {round(result.tts_first_audio_ms or 0)} ms after the first text")
    # `spoken_so_far` is the text that actually reached the sink. If the user had
    # interrupted, it would be shorter than the full reply, which tells a chat
    # app what the user really heard.
    print(f"spoken: {result.spoken_so_far!r}")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
