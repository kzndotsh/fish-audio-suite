# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1"]
# ///
"""Use the OpenAI Python SDK against the proxy: speech to mp3, then back to text.

Run: uv run examples/openai_sdk_proxy.py (needs the proxy running on 127.0.0.1:8849)
Start the proxy with: uv run fish-audio-suite-proxy

Many apps and libraries already know how to talk to OpenAI's audio API. The
proxy in this repo speaks that same API, but sends the work to Fish. Point an
OpenAI client at the proxy and it uses Fish voices without any code changes
beyond the address.

This example uses the official `openai` package to:

1. Turn a sentence into speech and save it as an mp3.
2. Send that mp3 back to be transcribed, to show the round trip works.

The block at the very top of this file (between the `# ///` lines) is inline
script metadata. It tells `uv` to install `openai` just for this run, so
you do not need to add it to the project. That is why the run command is
`uv run examples/...` without `python`.

The Fish key lives on the proxy, not here. Start the proxy with
`FISH_API_KEY` in its environment. You will see the mp3's path and size, then
the transcription.
"""

import os
from pathlib import Path

from openai import OpenAI

MP3_PATH = Path(__file__).resolve().parent.parent / "tmp" / "openai_sdk_proxy.mp3"


def main() -> None:
    # `base_url` sends every request to the local proxy instead of OpenAI.
    # `api_key` can be any value unless the proxy was started with
    # `FISH_PROXY_API_KEYS` set, in which case it must be one of those keys.
    client = OpenAI(base_url="http://127.0.0.1:8849/v1", api_key="sk-local")
    MP3_PATH.parent.mkdir(exist_ok=True)

    # Step 1. Text to speech.
    # `tts-1` is OpenAI's model name. The proxy maps it to the Fish model set by
    # `FISH_TTS_MODEL`. `with_streaming_response` writes the audio to disk as it
    # arrives instead of holding it all in memory first.
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        # `voice` takes a Fish voice id. OpenAI preset names like `nova` are
        # rejected. Left empty, Fish uses its own default voice.
        voice=os.environ.get("FISH_VOICE_ID", ""),
        input="[happy] The quick brown fox jumps over the lazy dog.",
        response_format="mp3",
    ) as response:
        response.stream_to_file(MP3_PATH)

    print(f"wrote {MP3_PATH} ({MP3_PATH.stat().st_size} bytes)")

    # Step 2. Speech to text.
    # `whisper-1` is OpenAI's transcription model name. The proxy maps it to
    # Fish's speech recognition. The `[happy]` cue was a tone instruction, not
    # words, so it does not appear in the transcription.
    with MP3_PATH.open("rb") as audio_file:
        print("heard:", client.audio.transcriptions.create(model="whisper-1", file=audio_file).text)


if __name__ == "__main__":
    main()
