# examples

Short, runnable scripts, each showing one use of the kit, the voice library or the proxy.
Every file is standalone. Run one from the repository root after `uv sync --all-packages`.

The live examples read `FISH_API_KEY` and `FISH_VOICE_ID` from the repository's `.env`
(or the environment) and spend a few hundredths of a cent of Fish credit. They write
their audio to `tmp/`, which git ignores.

| Example | Shows | Needs |
| --- | --- | --- |
| [clean_llm_text.py](clean_llm_text.py) | Scrubbing messy LLM output and cutting a token stream into speakable pieces | nothing |
| [captions.py](captions.py) | Building SRT and WebVTT subtitles from timed cues | nothing |
| [fish_errors.py](fish_errors.py) | Parsing Fish error bodies and catching them by type | nothing |
| [speak_to_file.py](speak_to_file.py) | Speaking one line to a WAV file, and reading a failure off the result | FISH_API_KEY + FISH_VOICE_ID |
| [stream_llm_reply.py](stream_llm_reply.py) | Speaking a reply while the LLM is still writing it | FISH_API_KEY + FISH_VOICE_ID |
| [custom_sink.py](custom_sink.py) | Writing your own `PlaybackSink`, here a byte counter with a level meter | FISH_API_KEY + FISH_VOICE_ID |
| [openai_sdk_proxy.py](openai_sdk_proxy.py) | The OpenAI Python SDK against the proxy: speech to mp3, then back to text | the proxy running |

`openai_sdk_proxy.py` carries its own dependency list, so run it with
`uv run examples/openai_sdk_proxy.py` while `uv run fish-audio-suite-proxy` is up. It
speaks in the proxy's default voice. Export `FISH_VOICE_ID` to use another Fish voice id.
