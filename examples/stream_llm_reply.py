"""Speak a reply while the LLM is still writing it.

Run: uv run python examples/stream_llm_reply.py (needs FISH_API_KEY and FISH_VOICE_ID)
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
    for i in range(0, len(REPLY), 6):
        time.sleep(0.02)
        yield REPLY[i : i + 6]


def main() -> None:
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
    result = speaker.speak_stream(
        fake_llm(),
        FileSink(out),
        on_first_audio=lambda: print(f"audio began {time.perf_counter() - started:.2f}s in"),
    )
    if result.error:
        sys.exit(f"Fish failed: {type(result.error).__name__}: {result.error.message}")
    print(f"first audio: {round(result.tts_first_audio_ms or 0)} ms after the first text")
    print(f"spoken: {result.spoken_so_far!r}")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
