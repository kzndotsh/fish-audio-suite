# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1"]
# ///
"""Use the OpenAI Python SDK against the proxy: speech to mp3, then back to text.

Run: uv run examples/openai_sdk_proxy.py (needs the proxy running on 127.0.0.1:8849)
Start the proxy with: uv run fish-audio-suite-proxy
"""

import os
from pathlib import Path

from openai import OpenAI

OUT = Path(__file__).resolve().parent.parent / "tmp" / "openai_sdk_proxy.mp3"


def main() -> None:
    client = OpenAI(base_url="http://127.0.0.1:8849/v1", api_key="sk-local")
    OUT.parent.mkdir(exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=os.environ.get("FISH_VOICE_ID", ""),  # a Fish voice id, not an OpenAI name
        input="[happy] The quick brown fox jumps over the lazy dog.",
        response_format="mp3",
    ) as response:
        response.stream_to_file(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

    with OUT.open("rb") as audio:
        print("heard:", client.audio.transcriptions.create(model="whisper-1", file=audio).text)


if __name__ == "__main__":
    main()
