"""Send Fish audio to your own sink: count bytes and draw a level meter.

Run: uv run python examples/custom_sink.py (needs FISH_API_KEY and FISH_VOICE_ID)

`FishSpeaker` does not care where audio goes. It hands every chunk it
receives to a "sink", an object with a few agreed methods. The suite ships
sinks for a file, your speakers, stdout and mpv, but you can write your own to
send audio anywhere: a websocket, a recorder, a visualizer.

A sink only needs these members (the `PlaybackSink` protocol):

- `start()`: called once before the first chunk.
- `write(chunk)`: called for every chunk of audio bytes.
- `finish(kill=False)`: called once at the end. `kill` is `True` if the
  speech was cut off, for example because the user started talking.
- `bytes_played()`: how many bytes the sink accepted.
- `output_latency_s`: how many seconds of audio the sink holds before it is
  heard. Use `0.0` when there is no buffering.

There is no base class to inherit from. Any class with these members works.

This sink plays nothing. It draws a bar of `#` marks for every 100 ms of
audio, taller when the voice is louder, so you can watch the speech go by.

Needs the same `.env` values as `speak_to_file.py`: `FISH_API_KEY` and
`FISH_VOICE_ID`.
"""

import os
import sys
from array import array
from pathlib import Path

from fish_audio_suite_voice import FishSpeaker
from fish_audio_suite_voice.envfile import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class MeterSink:
    """Implements the `PlaybackSink` protocol without touching a device or a file."""

    # This sink shows audio the instant it arrives, so nothing is waiting to be heard.
    output_latency_s = 0.0

    def __init__(self, sample_rate: int) -> None:
        self.total_bytes = 0

        # Bytes received but not yet drawn. See `write` for why this exists.
        self.pending = bytearray()

        # `FishSpeaker` sends raw PCM by default: 16-bit samples (2 bytes each),
        # one channel. At 44,100 samples per second, 100 ms is 4,410 samples,
        # which is 8,820 bytes.
        self.window_bytes = sample_rate // 10 * 2

    def start(self) -> None:
        """Reset the counters. Called once per turn, since a sink can be reused."""
        self.total_bytes = 0
        self.pending.clear()

    def write(self, chunk: bytes) -> None:
        """Count one chunk and draw a bar for each full 100 ms window.

        Chunks arrive in whatever sizes the network delivers. A chunk can end in
        the middle of a 100 ms window, or even in the middle of a 2-byte sample.
        The leftover bytes stay in `pending` and the next chunk is added to them,
        or every sample after an odd-sized chunk would be read one byte off.
        """
        self.total_bytes += len(chunk)
        self.pending += chunk
        while len(self.pending) >= self.window_bytes:
            # `array("h", ...)` reads the bytes as signed 16-bit numbers, from
            # -32768 to 32767. Each number is one sample of the sound wave.
            samples = array("h", self.pending[: self.window_bytes])
            del self.pending[: self.window_bytes]

            # The loudest sample in the window, ignoring whether it is above or
            # below zero, sets the bar height: 0 to 40 `#` marks.
            peak = max(abs(sample) for sample in samples)
            bar_length = peak * 40 // 32768
            print("#" * bar_length)

    def finish(self, *, kill: bool = False) -> None:
        """End the turn. `kill` is `True` when the speech was cut off early."""
        print("stopped early" if kill else "done")

    def bytes_played(self) -> int:
        """Return how many bytes this sink accepted, which the speaker reports back."""
        return self.total_bytes


def main() -> None:
    # Load the key and voice from `.env`, as in `speak_to_file.py`.
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    for name, value in (("FISH_API_KEY", api_key), ("FISH_VOICE_ID", voice_id)):
        if not value:
            sys.exit(f"{name} is not set. Put it in {ROOT / '.env'} or export it.")

    speaker = FishSpeaker(api_key=api_key, voice_id=voice_id)

    # Pass `speaker.sample_rate` so the meter's 100 ms windows match the audio.
    result = speaker.speak(
        "[calm] Watch the meter follow my voice.", MeterSink(speaker.sample_rate)
    )
    if result.error:
        sys.exit(f"Fish failed: {type(result.error).__name__}: {result.error.message}")

    # The speaker reads this number from `MeterSink.bytes_played`, so it equals
    # `MeterSink.total_bytes`.
    print(f"bytes played: {result.bytes_played}")


if __name__ == "__main__":
    main()
