# fish-audio-suite-proxy

OpenAI-compatible speech and transcription in front of [Fish Audio](https://fish.audio).

Part of [fish-audio-suite](../../README.md). Unofficial.

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/health` | Process up. Works with `FISH_API_KEY` unset |
| `GET` | `/v1/models` | OpenAI model list |
| `POST` | `/v1/audio/speech` | Audio bytes |
| `POST` | `/v1/audio/transcriptions` | JSON, or an `srt` / `vtt` file |

Default bind is `0.0.0.0:8849`. One worker. Fish 429 and 5xx are retried. Other 4xx are returned as-is.

## Quick start

```bash
export FISH_API_KEY=...
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
curl -s http://127.0.0.1:8849/health
```

```bash
curl -s http://127.0.0.1:8849/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"s2.1-pro","voice":"YOUR_REFERENCE_ID","input":"Hello there."}' \
  --output hello.mp3
```

Speech and transcription return 401 until `FISH_API_KEY` is set.

From the repository root:

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run --rm -p 127.0.0.1:8849:8849 --env-file /path/to/env fish-audio-suite-proxy:latest
```

## Open WebUI

Base URL `http://127.0.0.1:8849/v1`. The key in the client can be any non-empty string. Open WebUI rejects an empty one. The Fish key stays on the proxy.

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

The same fields are under Admin, Settings, Audio.

## How it works

| Client field | Fish |
| --- | --- |
| `model` `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`, `playai-tts` | `s2.1-pro` |
| `model` `whisper-1`, `gpt-4o-transcribe`, Groq whisper slugs | default ASR model |
| `model` `fish-audio/s2.1-pro` | prefix stripped |
| `voice` or `reference_id` | Fish reference id. A list is S2 multi-speaker |
| `response_format=pcm16` | PCM at 24 kHz unless `sample_rate` is set |
| `response_format=srt` or `vtt` | Caption file built from Fish segments |
| `verbose_json` and word timestamps | JSON plus a `words` array |
| Transcription body | Multipart `file`, or JSON `input_audio` (base64, optional `data:` URI) |
| `references`, `input_references` | Decoded and sent as MessagePack |
| `seed`, `use_memory_cache` | Forwarded (`on` / `off`) |
| `traceparent`, `tracestate` | Forwarded. A sampled `traceparent` is minted when the client omits one |

`s2.1-pro-free` and `drama-3-preview` are left alone. Text that is only cues or junk becomes a silent MP3. ASR `language` is omitted unless the request or `FISH_ASR_LANGUAGE` sets it.

Errors use `{error: {code, message, type}}`. A Fish body is `type: provider_error` with `metadata.provider_name: fish-audio`.

`FISH_BASE=http://127.0.0.1:8080` targets self-hosted [fish-speech](https://github.com/fishaudio/fish-speech) (`POST /v1/tts`). Cloud stays `https://api.fish.audio`. Cloud `chunk_length` is clamped to 100–300. Self-host allows up to 1000.

## Settings

| Variable | Default |
| --- | --- |
| `FISH_API_KEY` | none. Speech and transcription return 401 |
| `FISH_BASE` | `https://api.fish.audio` |
| `FISH_MODEL` | `s2.1-pro` |
| `FISH_LATENCY` | `normal` |
| `FISH_SPEED_SCALE` | `1` |
| `FISH_CHUNK_LENGTH` | `200` |
| `FISH_FORMAT` | `mp3` |
| `FISH_ASR_LANGUAGE` | omitted |
| `FISH_ASR_STRIP_SPEAKERS` | off |
| `FISH_TTS_DIALOGUE_ONLY` | off |
| `FISH_PROXY_HOST` / `FISH_PROXY_PORT` | `0.0.0.0` / `8849` |
| `FISH_PROXY_WORKERS` or `WEB_CONCURRENCY` | `1` |
| `FISH_PROXY_GRACEFUL_SHUTDOWN` | `120` seconds |
| `FISH_PROXY_LIMIT_CONCURRENCY` | unset. No 503 cap |

## License

[MIT](../../LICENSE)
