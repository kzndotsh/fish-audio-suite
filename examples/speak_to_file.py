"""Speak one line with Fish and save it as a WAV file.

Run: uv run python examples/speak_to_file.py (needs FISH_API_KEY and FISH_VOICE_ID)

This is the smallest real text-to-speech call in the suite. It sends one line
of text to Fish, receives the audio, and saves it as a WAV file you can open in
any audio player.

Two objects do the work:

- ``FishSpeaker`` holds your settings (key, voice, model) and talks to Fish.
- A "sink" is where the audio goes. ``FileSink`` saves it to a file. Other
  sinks play it on your speakers or pipe it to another program.

Before you run it, put two values in the ``.env`` file at the repository root:

- ``FISH_API_KEY``, your key from the Fish Audio dashboard.
- ``FISH_VOICE_ID``, the id of the voice to use. There is no default voice.

You will see how long Fish took to start sending audio, and where the file is.
"""

import os
import sys
from pathlib import Path

from fish_audio_suite_voice import FileSink, FishSpeaker
from fish_audio_suite_voice.envfile import load_dotenv

# The repository root: this file lives in `<root>/examples/`.
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # `load_dotenv` copies `.env` into `os.environ`. A variable you already
    # exported in your shell wins over the file, so you can override one value
    # for a single run.
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    # Stop early with a clear message instead of failing later inside the call.
    for name, value in (("FISH_API_KEY", api_key), ("FISH_VOICE_ID", voice_id)):
        if not value:
            sys.exit(f"{name} is not set. Put it in {ROOT / '.env'} or export it.")

    # Write into `<root>/tmp/`, which git ignores.
    out = ROOT / "tmp" / "speak_to_file.wav"
    out.parent.mkdir(exist_ok=True)

    # Only `api_key` and `voice_id` are required. `model`, `speed`, `latency`
    # and the rest have sensible defaults you can pass as keyword arguments.
    speaker = FishSpeaker(api_key=api_key, voice_id=voice_id)
    # `speak` blocks until the whole line has been spoken into the sink. The
    # `[happy]` cue tells Fish what tone to use and is not read aloud. `FileSink`
    # collects the audio in memory and writes the WAV file once speech ends.
    result = speaker.speak("[happy] Hello from the Fish Audio suite!", FileSink(out))

    # A Fish failure does not raise. It comes back on the result, so one bad
    # turn cannot crash a long-running voice app. Always check `result.error`.
    # It holds one of the kit error classes shown in `fish_errors.py`.
    if result.error:
        sys.exit(f"Fish failed: {type(result.error).__name__}: {result.error.message}")
    # `tts_first_audio_ms` is how long Fish took to send the first audio, a good
    # measure of how fast the voice feels to a listener.
    print(f"first audio: {round(result.tts_first_audio_ms or 0)} ms")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
