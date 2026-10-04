# Integrations

Every way to plug fish-audio-suite into something else, from the least code to the most:

| You have | Use | Section |
| --- | --- | --- |
| An app that speaks OpenAI's audio API (Open WebUI, SillyTavern, RisuAI, AIRI, OpenAI SDK code) | The **proxy** | [Apps](#apps-that-speak-openais-audio-api), [SDKs](#openai-sdks-and-plain-http) |
| A Python program with its own LLM and audio | The **voice library** (`FishSpeaker`) | [Python library](#python-the-voice-library) |
| A text pipeline for any TTS or speech-to-text engine | The **kit** | [Text helpers](#python-the-kit-text-helpers) |
| Nothing yet, and you want to talk to an AI | **`fish-voice`** with any LLM | [The voice app](#the-voice-app-with-any-llm) |
| A server to run it on | Docker, Compose, NixOS, a reverse proxy | [INSTALL.md](INSTALL.md#the-proxy) |

You need a Fish Audio API key (`FISH_API_KEY`) and a voice id from fish.audio: the id of a voice model, used as `voice` (OpenAI's name) or `reference_id` (Fish's).

## Apps that speak OpenAI's audio API

Start the proxy, then point the app's OpenAI audio settings at it:

```bash
export FISH_API_KEY=...
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy   # http://127.0.0.1:8849
```

| Setting in the app | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8849/v1` |
| API key | Any non-empty string, or one of your `FISH_PROXY_API_KEYS` if you set them |
| TTS model | `tts-1` (mapped to `FISH_TTS_MODEL`), or a Fish id such as `s2.1-pro` |
| Voice | Your Fish voice id |
| Speech-to-text model | `whisper-1` (mapped to `FISH_ASR_MODEL`) |

The Fish key stays on the proxy. Clients only ever see the proxy's own key.

At a glance (app versions from October 2026):

| App | URL to enter | Speech | Speech-to-text | CORS headers needed |
| --- | --- | --- | --- | --- |
| Open WebUI | `http://127.0.0.1:8849/v1` | Yes | Yes | No, its server calls the proxy |
| SillyTavern | `http://127.0.0.1:8849/v1/audio/speech` | Yes | No | No, its server calls the proxy |
| RisuAI (desktop) | `http://127.0.0.1:8849/v1` | Yes | No | No |
| AIRI | `http://127.0.0.1:8849/v1/` (trailing slash) | Yes | Yes | Yes |

### Open WebUI

In Admin, Settings, Audio, or through environment variables:

```bash
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8849/v1
AUDIO_STT_OPENAI_API_KEY=sk-local
AUDIO_STT_MODEL=whisper-1
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8849/v1
AUDIO_TTS_OPENAI_API_KEY=sk-local
AUDIO_TTS_MODEL=s2.1-pro
AUDIO_TTS_VOICE=your-fish-voice-id
```

Open WebUI rejects an empty key, so give it any string. If Open WebUI runs in Docker, `127.0.0.1` is the container itself. Use `http://host.docker.internal:8849/v1` (with `--add-host=host.docker.internal:host-gateway` on Linux), or put both on one Docker network and use the proxy's container name.

### SillyTavern

Text to speech works through the TTS extension. Speech-to-text does not: the Speech Recognition extension's OpenAI option always calls OpenAI itself.

> [!IMPORTANT]
> SillyTavern wants the full speech endpoint, not the base URL.

1. Open **Extensions**, then **TTS**, and set **Select TTS Provider** to **OpenAI Compatible**. Tick **Enabled**.
2. **Provider Endpoint:** the full endpoint, `http://127.0.0.1:8849/v1/audio/speech`.
3. **Model:** `tts-1`, or a Fish id such as `s2.1-pro`.
4. **Available Voices:** your Fish voice ids, separated by commas.
5. **API Key:** leave it empty, or enter one of your `FISH_PROXY_API_KEYS`.
6. In the voice map below, pick a voice for each character.

SillyTavern's server makes the request, not your browser, so CORS does not matter. The proxy has to be reachable from the machine running SillyTavern, not from the machine with the browser. The **Speed** slider is sent, and the proxy multiplies it by `FISH_SPEED`. The audio format is always MP3.

### RisuAI

Text to speech is set per character. Speech-to-text has no custom URL; its only Whisper use is hardwired to OpenAI.

1. Open the character's settings, then the **TTS** tab, and set **Provider** to **OpenAI**.
2. Tick **Advanced (OpenAI-compatible endpoint)**.
3. **Base URL:** `http://127.0.0.1:8849/v1`. RisuAI adds `/audio/speech` itself.
4. **Model:** `tts-1`, or a Fish id such as `s2.1-pro`.
5. **Voice:** with Advanced ticked this is a text box, so paste your Fish voice id.
6. **Response Format:** leave it on `mp3`. RisuAI plays every response as MP3.
7. **API Key:** leave it empty, or enter one of your `FISH_PROXY_API_KEYS`.

> [!TIP]
> Use the RisuAI desktop app. The web version cannot reach a proxy on your own machine.

Which RisuAI you run decides whether this works:

| Build | Works with a local proxy? | Why |
| --- | --- | --- |
| Desktop app | **Yes, recommended** | It fetches natively, so CORS does not apply |
| Web (risuai.xyz) | No | It refuses `localhost` and `127.0.0.1` ("You are trying local request on web version"), and its hosted relay cannot reach your machine |
| Self-hosted (Node) | Only by a non-loopback address | A `localhost` URL is fetched by the browser and blocked by CORS. Another hostname (a LAN IP or a Docker service name) goes through RisuAI's own server relay, so run the proxy on that address with `FISH_PROXY_API_KEYS` set |

RisuAI never sends `speed`, so `FISH_SPEED` sets the pace. RisuAI also has a built-in "fish-speech" TTS provider that calls Fish directly with your Fish key. Use the proxy instead for its text cleanup and cue handling, or to keep the Fish key off the client.

### AIRI

[Project AIRI](https://github.com/moeru-ai/airi) supports both text to speech and speech-to-text through OpenAI-compatible providers.

1. In **Settings**, then **Providers**, add the **OpenAI Compatible** speech provider:
   - **Base URL:** `http://127.0.0.1:8849/v1/`, with a trailing slash. AIRI rejects the URL without one.
   - **API Key:** any non-empty string, or one of your `FISH_PROXY_API_KEYS`.
   - **Model:** type `tts-1` or `s2.1-pro`. AIRI only offers model ids that contain "tts", so a Fish id has to be typed.
2. In **Settings**, then **Modules**, then **Speech**, pick that provider and enter your Fish voice id as the **Voice Name**.
3. For speech-to-text, add the **OpenAI Compatible** transcription provider under **Providers** (same base URL and key, model `whisper-1`), then select it under **Modules**, then **Hearing**.

> [!WARNING]
> AIRI calls the API from its own window, in both the web app and the desktop app, so the browser's CORS rules apply. The proxy sends no CORS headers, so put a reverse proxy in front that adds them ([below](#apps-that-run-in-the-browser)) and point AIRI at that address instead.

> [!NOTE]
> The AIRI steps come from reading its source code. They have not been tested end to end.

### Other apps

Anything with an "OpenAI" or "OpenAI-compatible" TTS or speech-to-text setting that lets you change the URL works the same way. Check three things in the app's docs:

- **The URL it wants.** Some want the base URL (`http://127.0.0.1:8849/v1`), some the full endpoint (`.../v1/audio/speech`), and some insist on a trailing slash.
- **Whether the voice field is free text.** It has to accept your Fish voice id, not only OpenAI's built-in voice names.
- **Where the request comes from.** A browser page needs CORS headers (see the next section). The app's own server or a desktop app does not.

### Apps that run in the browser

The proxy sends no CORS headers. A web app whose page calls the proxy directly from the browser fails before the request is sent. Firefox shows `NetworkError when attempting to fetch resource`, and Chrome shows `Failed to fetch` with a CORS error in the console. Apps whose own server makes the request (Open WebUI and SillyTavern, for example) are not affected, and neither are native desktop apps such as RisuAI's.

You have three ways around it:

- **Use the app's desktop build**, if it has one. A desktop app is not bound by browser CORS rules.
- **Route the request through the app's own server or proxy setting**, if it offers one.
- **Put the proxy behind a reverse proxy that adds CORS headers.** With Caddy:

```caddy
fish.example.com {
    @preflight method OPTIONS
    header Access-Control-Allow-Origin "https://your-app.example"
    header Access-Control-Allow-Headers "Authorization, Content-Type"
    header Access-Control-Allow-Methods "GET, POST, OPTIONS"
    respond @preflight 204
    reverse_proxy 127.0.0.1:8849
}
```

> [!CAUTION]
> Allow only the origin you use, not `*`, and set `FISH_PROXY_API_KEYS`, because the proxy spends your Fish credits for anyone who can reach it.

For an app on your own machine, a reverse proxy that listens only on loopback can allow any origin. A desktop app loaded from local files may send `Origin: null`, which only `*` matches:

```caddy
http://127.0.0.1:8850 {
    @preflight method OPTIONS
    header Access-Control-Allow-Origin "*"
    header Access-Control-Allow-Headers "Authorization, Content-Type"
    header Access-Control-Allow-Methods "GET, POST, OPTIONS"
    respond @preflight 204
    reverse_proxy 127.0.0.1:8849
}
```

Point the app at `http://127.0.0.1:8850/v1` instead of port 8849.

## OpenAI SDKs and plain HTTP

The proxy implements OpenAI's `/v1/audio/speech`, `/v1/audio/transcriptions` and `/v1/models`, so the official SDKs work with a changed base URL.

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8849/v1", api_key="sk-local")

# Text to speech, streamed to a file as it arrives
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="your-fish-voice-id",
    input="[happy] Hello there!",
    response_format="mp3",
) as response:
    response.stream_to_file("hello.mp3")

# Speech to text
with open("clip.wav", "rb") as audio:
    text = client.audio.transcriptions.create(model="whisper-1", file=audio).text
```

> [!NOTE]
> The SDK's type hints list only OpenAI's voice names, so a type checker may flag the Fish id. It works at runtime.

### JavaScript

```js
import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://127.0.0.1:8849/v1", apiKey: "sk-local" });

const speech = await client.audio.speech.create({
  model: "tts-1",
  voice: "your-fish-voice-id",
  input: "Hello there!",
});
fs.writeFileSync("hello.mp3", Buffer.from(await speech.arrayBuffer()));
```

### curl

```bash
curl -s http://127.0.0.1:8849/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"your-fish-voice-id","input":"Hello there."}' \
  --output hello.mp3

curl -s http://127.0.0.1:8849/v1/audio/transcriptions \
  -F file=@clip.wav -F model=whisper-1 -F response_format=srt
```

### Beyond OpenAI's fields

- **Fish cues work in the text.** `[happy]`, `[whispering]`, `[laughing]` and the like change how the line is read. The proxy normalizes them and strips markdown, `<think>` blocks and URLs.
- **Extra speech fields:** `latency`, `chunk_length`, `temperature`, `top_p`, `sample_rate`, `mp3_bitrate`, `dialogue_only` and more. The full list is in the [proxy README](../packages/proxy/README.md#request-fields).
- **Formats:** `mp3`, `opus`, `wav`, `pcm`, and `pcm16` (24 kHz PCM, OpenAI's name for it). For live playback, `pcm` or `opus` start fastest.
- **Several voices in one request:** send `voice` as a list and mark turns with `<|speaker:0|>`, `<|speaker:1|>` (S2 models).
- **Captions:** `response_format=srt` or `vtt` gives phrase-level subtitles. `verbose_json` adds segments, and word timings with `timestamp_granularities=word`.
- **Tracing:** send a W3C `traceparent` header, and the proxy forwards it to Fish, so a request can be followed through your own tracing system.

## Python: the voice library

Use `FishSpeaker` when your program already has its own LLM and conversation logic, and you only want Fish's live, low-latency speech.

Install it with the kit, from the repository ([INSTALL.md](INSTALL.md#as-a-library)), with the `speakers` extra if you play audio locally.

### Speak a finished reply

```python
from pathlib import Path
from fish_audio_suite_voice import FishSpeaker, FileSink, make_sink

tts = FishSpeaker(api_key=fish_key, voice_id=voice_id, latency="balanced")

tts.speak("[happy] Hello there!", FileSink(Path("hello.wav")))   # to a file
tts.speak("Hello again.", make_sink("sounddevice"))               # to the speakers
```

### Speak while your LLM is still writing

Pass any iterable or async iterable of text pieces. The first sentence plays while the rest is still being generated:

```python
def tokens():
    for chunk in my_llm.stream(messages):   # your LLM client
        yield chunk.text

result = tts.speak_stream(tokens(), make_sink("sounddevice"))
print(result.spoken_so_far)   # what was actually played
```

> [!IMPORTANT]
> An async iterable is read on the speaker's private event loop, so it must not depend on your loop. From your own async code, run the call in a thread instead:

```python
result = await asyncio.to_thread(tts.speak_stream, token_list, sink)
```

### What you get

- **`speak` and `speak_stream` block until the turn ends**, and run Fish's websocket on their own thread and event loop. They are safe under `asyncio.run` or `asyncio.to_thread`.
- **Stopping early:** pass `cancel=threading.Event()` and set it from anywhere, for example when your user starts talking.
- **Knowing when sound starts:** `on_first_audio=callback` runs once, at the first audio chunk.
- **The result is a `TtsResult`:**
  - `spoken_so_far` is the text the listener heard, cut to what was played. Store that in your chat history, not the full reply.
  - `error` is a kit error class you can test with `isinstance`, such as `FishAuthError` or `FishRateLimitError`.
  - `tts_first_audio_ms` is Fish's time to the first audio.
- **Retries:** a Fish 429 or 5xx is retried only before the first audio byte.

### Your own audio output

Pass any object with these methods as the sink. `FileSink`, `StdoutSink`, `SounddeviceSink` and `MpvSink` are built in.

```python
class MySink:
    output_latency_s = 0.0          # seconds of device buffer, for the spoken-prefix estimate
    def start(self) -> None: ...     # called once per turn
    def write(self, chunk: bytes) -> None: ...   # PCM (or encoded audio, per audio_format)
    def finish(self, *, kill: bool = False) -> None: ...   # kill=True: stop at once
    def bytes_played(self) -> int: ...
```

The audio format follows `FishSpeaker(audio_format=..., sample_rate=...)`. The default is 16-bit mono PCM at 44.1 kHz.

### Other building blocks

The package root also exports:
- **`BargeGate`:** interruption detection.
- **`EchoCanceller`:** WebRTC AEC3.
- **`DuplexSession`:** the cancel and quit flags.
- **`ChatBackend`:** a streaming LLM interface.
- **The tune dataclasses.**

Together they are what `fish-voice` is built from, if you want to assemble your own loop.

## Python: the kit text helpers

The kit is a dependency-free library for any speech pipeline, Fish or not:

Install it from the repository ([INSTALL.md](INSTALL.md#as-a-library)).

```python
from fish_audio_suite_kit import normalize_cues, scrub_tts, split_tts_piece

spoken = normalize_cues(scrub_tts(llm_text))    # strip markdown/thoughts/URLs, tidy [cues]
rest = spoken
while rest:
    piece, rest = split_tts_piece(rest, 40, flush_rest=True)
    send_to_tts(piece)                           # sentence-sized pieces
```

| Need | Helpers |
| --- | --- |
| Clean LLM output for any TTS | `scrub_tts`, `normalize_cues`, `strip_cue_tags`, `extract_quoted_speech`, `is_tts_junk` |
| Cut streaming text into speakable pieces | `next_tts_cut`, `split_tts_piece`, `tts_hold_at`, `ends_sentence` |
| Clean speech-to-text output | `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance` |
| Subtitles | `CaptionCue`, `format_as_srt`, `format_as_vtt` |
| Fish errors and retries | `parse_fish_error`, `FishHttpError` and subclasses, `should_retry_fish_status`, `retry_after_s` |
| Tracing | `make_traceparent`, `ensure_trace_headers` |

The [kit README](../packages/kit/README.md) has the full list.

## The voice app with any LLM

`fish-voice` is a complete voice chat: your mic, Fish speech-to-text, an LLM, and Fish's voice back, with interruption. It works with any chat model that speaks OpenAI's chat-completions API.

Install it as described in [INSTALL.md](INSTALL.md#the-voice-app), then set your LLM in `.env`:

| LLM | Settings |
| --- | --- |
| OpenRouter (default) | `OPENROUTER_API_KEY`, `FISH_LLM_MODEL=author/model` |
| Experiential | `FISH_LLM_PROVIDER=experiential`, `EXPLABS_API_KEY`, `FISH_LLM_MODEL_EXPERIENTIAL` |
| Ollama | `FISH_LLM_PROVIDER=custom`, `FISH_LLM_BASE=http://127.0.0.1:11434/v1`, `FISH_LLM_API_KEY=ollama`, `FISH_LLM_MODEL=...` |
| Any other OpenAI-compatible server (LM Studio, vLLM, llama.cpp's server, a hosted API) | `FISH_LLM_PROVIDER=custom`, `FISH_LLM_BASE=<server>/v1`, `FISH_LLM_API_KEY`, `FISH_LLM_MODEL` |

Give it a personality with a character file: `--prompt-file characters/mira.md`, or `FISH_VOICE_SYSTEM_PROMPT_FILE`. The voice rules (cue tags, short spoken replies) are added after it. Model choice and latency tuning are in [PERFORMANCE.md](PERFORMANCE.md).

## Running the proxy as a service

Docker, Docker Compose, the NixOS module, reverse proxies and self-hosted Fish are covered in [INSTALL.md](INSTALL.md#the-proxy).

## Checking an integration

- `curl http://127.0.0.1:8849/health` shows the proxy's settings (never a key).
- `curl http://127.0.0.1:8849/v1/models` lists the model ids it accepts.
- `fish-voice --smoke` checks your Fish key and voice by writing one line to a WAV file.
- If something fails, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
