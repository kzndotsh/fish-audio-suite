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

Stacked leads stay (`[sad][whispering] …`). Mid-sentence `[chuckle]` becomes `[chuckling]`. `[cough]` is left as `[cough]`.

Also: `extract_quoted_speech`, `is_tts_junk`, `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance`, `ensure_lead_cue`, `SuiteDefaults`, `LatencySnapshot`, `w3c_trace_headers`, `make_traceparent`, `parse_fish_error`, `should_retry_fish_status`, `fish_backoff_seconds`, `format_as_srt`, `format_as_vtt`.

## proxy

Drop-in for clients that speak OpenAI `/v1/audio/speech` and `/v1/audio/transcriptions` (AIRI, OpenWebUI, and similar). Forwards W3C `traceparent` / `tracestate` to Fish TTS and ASR (mints one if the client omitted them). Fish 429/5xx are retried with backoff; other 4xx are returned as-is. Errors use the OpenAI envelope `{error: {code, message, type}}`. Upstream Fish failures add `type: provider_error` and `metadata.provider_name: fish-audio`.

```bash
export FISH_API_KEY=…
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
# GET http://127.0.0.1:8849/health
```

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run --rm -p 127.0.0.1:8849:8849 --env-file /path/to/env fish-audio-suite-proxy:latest
```

The env file must contain `FISH_API_KEY` for Fish Cloud. Pass a Fish voice as `voice` (string) or `reference_id` (string, or a list of ids for S2 multi-speaker with `<|speaker:0|>` tags in `input`). Optional speech fields: `seed`, `use_memory_cache` (`on` / `off`). `references` (clips with base64 `audio` + `text`) and OpenRouter `input_references` are decoded and sent to Fish as MessagePack.

Transcription accepts multipart `file` or JSON `input_audio.data` (base64, optional `data:` URI).

Point `FISH_BASE` at a self-hosted [fish-speech](https://github.com/fishaudio/fish-speech) HTTP server (typically `http://127.0.0.1:8080`) to use this proxy as an OpenAI `/v1/audio/speech` adapter. Local fish-speech speaks `POST /v1/tts`, not OpenAI paths. Cloud default remains `https://api.fish.audio`. Cloud `chunk_length` is clamped to 100–300; self-host allows up to 1000.

`response_format=srt` or `vtt` on `/v1/audio/transcriptions` returns those caption files (not JSON). OpenAI clients that send `timestamp_granularities[]` are accepted. `verbose_json` with `timestamp_granularities=word` includes a `words` array. TTS `response_format=pcm16` is Fish PCM at 24 kHz. Model ids may be `fish-audio/s2.1-pro` or OpenAI/Groq aliases (`tts-1`, `whisper-1`, `gpt-4o-transcribe`); the proxy strips the prefix and maps aliases onto native Fish model headers.

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

Playback sinks: `sounddevice` (default PCM), `file`, `stdout`, optional `mpv`. Mic/speakers need the **PortAudio** C library; `pip`/`uv` do not install it. `--smoke` uses `FileSink` and does not need it.

Duplex `cli` extra includes AEC3 (`pywebrtc-audio`) and the official OpenRouter SDK. Speaker PCM is tapped and subtracted from the mic before VAD. `FISH_VOICE_AEC=0` turns AEC off (bleed delay stays 0.9 s). PipeWire `echo-cancel` is a host trick, not a package dep.

```bash
# Debian/Ubuntu
sudo apt install libportaudio2
# Fedora
sudo dnf install portaudio
# macOS
brew install portaudio
# NixOS (do not use bare `uv run` for duplex)
nix run .#fish-audio-suite-voice
# or this repo's wrapper:
./packages/voice/dev.sh
```

`--smoke` is TTS-only (writes `/tmp/fish-audio-suite-smoke.wav`). Duplex is mic → Fish ASR → OpenRouter → one Fish WS per turn. Settings are env vars (no YAML). Process env wins, then `--env-file`, else `./.env` if present.

```bash
cp .env.example .env   # gitignored; fill keys
./packages/voice/dev.sh --smoke                 # needs FISH_API_KEY + FISH_VOICE_ID
./packages/voice/dev.sh --debug                 # VAD / barge / Fish WS / OpenRouter meta
```

Or export the same vars and `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke`. `--smoke` exits 2 if the key or voice id is missing.

## Environment


| Variable                              | Who           | Default                         |
| ------------------------------------- | ------------- | ------------------------------- |
| `FISH_API_KEY`                        | proxy, voice  | none (required)                 |
| `FISH_VOICE_ID`                       | voice         | none (required for TTS)         |
| `FISH_BASE`                           | proxy, voice  | `https://api.fish.audio` (self-host: `http://127.0.0.1:8080`) |
| `FISH_MODEL` / `FISH_TTS_MODEL`       | proxy / voice | `s2.1-pro`                      |
| `FISH_LATENCY`                        | both          | `normal`                        |
| `FISH_SPEED` / `FISH_SPEED_SCALE`     | voice / proxy | `1`                             |
| `FISH_CHUNK_LENGTH`                   | both          | `200` (cloud max 300)           |
| `FISH_FORMAT`                         | proxy         | `mp3` (voice live uses `pcm`)   |
| `FISH_ASR_LANGUAGE`                   | proxy, duplex | omit (hint only; Fish may still return `zh` on noise) |
| `FISH_ASR_STRIP_SPEAKERS`             | proxy         | off                             |
| `FISH_TTS_DIALOGUE_ONLY`              | proxy         | off                             |
| `FISH_VOICE_SPEECH_FRAMES`            | duplex CLI    | `4` (~120 ms min speech)        |
| `FISH_VOICE_MIN_RMS`                  | duplex CLI    | `200` (floor; may rise in noise) |
| `FISH_VOICE_MIN_VOICED`               | duplex CLI    | `12` (~360 ms VAD-true; drops coughs) |
| `FISH_VOICE_SILENCE_FRAMES`           | duplex CLI    | `40` (~1.2 s quiet ends the turn) |
| `FISH_VOICE_VAD`                      | duplex CLI    | `1` (0–3; higher = pickier)     |
| `FISH_VOICE_COOLDOWN`                 | duplex CLI    | `0.8` (seconds after TTS)       |
| `FISH_VOICE_BLEED_DELAY`              | duplex CLI    | `0.9` (no AEC / AEC extra missing) |
| `FISH_VOICE_AEC`                      | duplex CLI    | on (`0` disables)               |
| `FISH_VOICE_AEC_WET`                  | duplex CLI    | `0.85`                          |
| `FISH_VOICE_AEC_BLEED`                | duplex CLI    | `0.3` when AEC3 is loaded       |
| `FISH_VOICE_BARGE_RMS`                | duplex CLI    | `220` (×2.2 while speakers play only if AEC is off) |
| `FISH_VOICE_DEBUG`                    | duplex CLI    | off (`--debug` or `1`)          |
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
- `nix run .#fish-audio-suite-voice` — PortAudio + Pulse on `LD_LIBRARY_PATH`. Duplex via `uv run` on NixOS will fail with “PortAudio library not found” unless you prefix that path (`./packages/voice/dev.sh` does).

## License

MIT
