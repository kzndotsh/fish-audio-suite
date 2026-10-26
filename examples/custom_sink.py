"""Send Fish audio to your own sink: count bytes and draw a level meter.

Run: uv run python examples/custom_sink.py (needs FISH_API_KEY and FISH_VOICE_ID)
"""

import os
import sys
from array import array
from pathlib import Path

from fish_audio_suite_voice import FishSpeaker
from fish_audio_suite_voice.envfile import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class MeterSink:
    """Implements the PlaybackSink protocol without touching a device or a file."""

    output_latency_s = 0.0

    def __init__(self, sample_rate: int) -> None:
        self.total = 0
        # One bar per 100 ms of 16-bit mono PCM.
        self.window_bytes = sample_rate // 10 * 2

    def start(self) -> None:
        self.total = 0

    def write(self, chunk: bytes) -> None:
        self.total += len(chunk)
        for i in range(0, len(chunk) - 1, self.window_bytes):
            window = chunk[i : i + self.window_bytes]
            peak = max(map(abs, array("h", window[: len(window) // 2 * 2])), default=0)
            print(f"{'#' * (peak * 40 // 32768):<40}")

    def finish(self, *, kill: bool = False) -> None:
        print("stopped early" if kill else "done")

    def bytes_played(self) -> int:
        return self.total


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    for name, value in (("FISH_API_KEY", api_key), ("FISH_VOICE_ID", voice_id)):
        if not value:
            sys.exit(f"{name} is not set. Put it in {ROOT / '.env'} or export it.")

    speaker = FishSpeaker(api_key=api_key, voice_id=voice_id)
    result = speaker.speak(
        "[calm] Watch the meter follow my voice.", MeterSink(speaker.sample_rate)
    )
    if result.error:
        sys.exit(f"Fish failed: {type(result.error).__name__}: {result.error.message}")
    print(f"bytes played: {result.bytes_played}")


if __name__ == "__main__":
    main()
