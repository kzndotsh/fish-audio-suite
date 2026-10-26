"""Speak one line with Fish and save it as a WAV file.

Run: uv run python examples/speak_to_file.py (needs FISH_API_KEY and FISH_VOICE_ID)
"""

import os
import sys
from pathlib import Path

from fish_audio_suite_voice import FileSink, FishSpeaker
from fish_audio_suite_voice.envfile import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    for name, value in (("FISH_API_KEY", api_key), ("FISH_VOICE_ID", voice_id)):
        if not value:
            sys.exit(f"{name} is not set. Put it in {ROOT / '.env'} or export it.")

    out = ROOT / "tmp" / "speak_to_file.wav"
    out.parent.mkdir(exist_ok=True)
    speaker = FishSpeaker(api_key=api_key, voice_id=voice_id)
    result = speaker.speak("[happy] Hello from the Fish Audio suite!", FileSink(out))

    # A Fish failure does not raise. It comes back on the result.
    if result.error:
        sys.exit(f"Fish failed: {type(result.error).__name__}: {result.error.message}")
    print(f"first audio: {round(result.tts_first_audio_ms or 0)} ms")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
