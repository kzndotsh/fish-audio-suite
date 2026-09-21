# fish-audio-suite

An opinionated toolkit for [Fish Audio](https://fish.audio): TTS, ASR, and live speech. Unofficial — not a Fish Audio product.

Three packages. Use one, or mix them.


| Package | Dist                     | What it does                                                      |
| ------- | ------------------------ | ----------------------------------------------------------------- |
| kit     | `fish-audio-suite-kit`   | Cue tags, TTS/ASR scrubbing, sentence cuts. No network.           |
| proxy   | `fish-audio-suite-proxy` | OpenAI-compatible HTTP on port **8849**.                          |
| voice   | `fish-audio-suite-voice` | One Fish websocket per turn, playback sinks, optional duplex CLI. |


Python 3.12. MIT. CI is GitHub Actions (ruff, basedpyright, pytest), not `nix flake check`.

```bash
uv sync --all-packages --extra cli --group dev
```

Set `FISH_API_KEY` for anything that talks to Fish. Set `FISH_VOICE_ID` for live TTS. Neither has a default.

## kit

Import `fish_audio_suite_kit` in your own Fish HTTP or SDK calls.

```python
from fish_audio_suite_kit import normalize_cues, scrub_tts, next_tts_cut

spoken = normalize_cues(scrub_tts(llm_text))
cut = next_tts_cut(spoken)  # sentence end, or ~40 chars; -1 = keep buffering
```

Also: `extract_quoted_speech`, `is_tts_junk`, `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance`, `SuiteDefaults`, `LatencySnapshot`, `w3c_trace_headers`, `make_traceparent`, `parse_fish_error`, `should_retry_fish_status`, `fish_backoff_seconds`, `format_as_srt`, `format_as_vtt`.

## proxy

Drop-in for clients that speak OpenAI `/v1/audio/speech` and `/v1/audio/transcriptions` (AIRI, OpenWebUI, and similar). Forwards W3C `traceparent` / `tracestate` to Fish TTS and ASR (mints one if the client omitted them). Fish 429/5xx are retried with backoff; other 4xx are returned as `{message, status}`.

```bash
export FISH_API_KEY=…
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
# GET http://127.0.0.1:8849/health
```

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run --rm -p 127.0.0.1:8849:8849 --env-file /path/to/env fish-audio-suite-proxy:latest
```

The env file must contain `FISH_API_KEY` for Fish Cloud. Pass a Fish voice as `voice` (string) or `reference_id` (string, or a list of ids for S2 multi-speaker with `<|speaker:0|>` tags in `input`). Optional speech fields: `seed`, `references` (clips with `audio` + `text`), `use_memory_cache` (`on` / `off`).

Inline `references` audio bytes need Fish **MessagePack** (`application/msgpack`). This proxy forwards JSON only, so JSON `references` work only if the clips are already JSON-safe (for example base64 that Fish accepts). For raw WAV/MP3 bytes, call Fish `POST /v1/tts` with msgpack directly.

Point `FISH_BASE` at a self-hosted [fish-speech](https://github.com/fishaudio/fish-speech) HTTP server (typically `http://127.0.0.1:8080`) to use this proxy as an OpenAI `/v1/audio/speech` adapter. Local fish-speech speaks `POST /v1/tts`, not OpenAI paths. Cloud default remains `https://api.fish.audio`. Cloud `chunk_length` is clamped to 100–300; self-host allows up to 1000.

`response_format=srt` or `vtt` on `/v1/audio/transcriptions` returns those caption files (not JSON). OpenAI clients that send `timestamp_granularities[]` are accepted.

### OpenWebUI

Point Audio at this proxy. OpenWebUI rejects an empty API key. Put a dummy string in the *client*. `FISH_API_KEY` still lives on the proxy.

```bash
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://127.0.0.1:8849/v1
AUDIO_STT_OPENAI_API_KEY=sk-local
AUDIO_STT_MODEL=whisper-1
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://127.0.0.1:8849/v1
AUDIO_TTS_OPENAI_API_KEY=sk-local
AUDIO_TTS_MODEL=s2.1-pro
AUDIO_TTS_VOICE=your-fish-reference-id
```

Same knobs in Admin → Settings → Audio (STT/TTS engine OpenAI, base URL with `/v1`, model `whisper-1` / `s2.1-pro`, voice = Fish id).

## voice

Library first. The `fish-voice` CLI is a talk-back recipe (mic → ASR → LLM → TTS). You do not need the CLI to stream Fish audio.

```python
from pathlib import Path
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
sink = FileSink(Path("turn.wav"))
result = tts.speak_isolated("[clear] Hello there.", sink)
```

Playback sinks: `sounddevice` (default PCM), `file`, `stdout`, optional `mpv`.

```bash
export FISH_API_KEY=… FISH_VOICE_ID=… OPENROUTER_API_KEY=… FISH_LLM_MODEL=…
uv run --extra cli fish-voice --smoke   # writes a wav; no speakers
uv run --extra cli fish-voice           # mic duplex
```

`--smoke` exits 2 if the key or voice id is missing.

## Environment


| Variable                              | Who           | Default                         |
| ------------------------------------- | ------------- | ------------------------------- |
| `FISH_API_KEY`                        | proxy, voice  | none (required)                 |
| `FISH_VOICE_ID`                       | voice         | none (required for TTS)         |
| `FISH_BASE`                           | proxy, voice  | `https://api.fish.audio` (self-host: `http://127.0.0.1:8080`) |
| `FISH_MODEL` / `FISH_TTS_MODEL`       | proxy / voice | `s2.1-pro`                      |
| `FISH_LATENCY`                        | both          | `normal`                        |
| `FISH_SPEED` / `FISH_SPEED_SCALE`     | voice / proxy | `1.05`                          |
| `FISH_CHUNK_LENGTH`                   | both          | `200` (cloud max 300)           |
| `FISH_FORMAT`                         | proxy         | `mp3` (voice live uses `pcm`)   |
| `FISH_ASR_LANGUAGE`                   | proxy, duplex | omit (Fish auto-detects)        |
| `FISH_ASR_STRIP_SPEAKERS`             | proxy         | off                             |
| `FISH_TTS_DIALOGUE_ONLY`              | proxy         | off                             |
| `FISH_VOICE_SPEECH_FRAMES`            | duplex CLI    | `8` (~240 ms min speech)        |
| `FISH_PLAYBACK`                       | voice CLI     | `sounddevice`                   |
| `FISH_LLM_KEY` / `OPENROUTER_API_KEY` | duplex CLI    | none (required for duplex)      |
| `FISH_LLM_MODEL` / `OPENROUTER_MODEL` | duplex CLI    | none (required for duplex)      |
| `FISH_PROXY_HOST` / `FISH_PROXY_PORT` | proxy         | `0.0.0.0` / `8849`              |
| `FISH_PROXY_WORKERS` / `WEB_CONCURRENCY` | proxy      | `1` (OCI: keep 1)               |
| `FISH_PROXY_GRACEFUL_SHUTDOWN`        | proxy         | `120` (seconds)                 |
| `FISH_PROXY_LIMIT_CONCURRENCY`        | proxy         | unset (no 503 cap)              |


## Nix

```nix
inputs.fish-audio-suite.url = "git+file:///home/kaizen/Projects/fish-audio-suite";
# later: github:kzndotsh/fish-audio-suite
```

- `nixosModules.default` — OCI container `127.0.0.1:8849:8849`, `autoStart = false`, `environmentFiles` for `FISH_API_KEY`.
- `nix run .#fish-audio-suite-voice` — PortAudio wrap. mpv optional.

## License

MIT
