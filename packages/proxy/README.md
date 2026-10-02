# fish-audio-suite-proxy

OpenAI-compatible speech and transcription in front of [Fish Audio](https://fish.audio).

Part of [fish-audio-suite](../../README.md). Unofficial.

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/health` | Process up. Works with `FISH_API_KEY` unset |
| `GET` | `/v1/models` | OpenAI model list |
| `POST` | `/v1/audio/speech` | Audio bytes |
| `POST` | `/v1/audio/transcriptions` | JSON, or an `srt` / `vtt` file |

Default bind is `127.0.0.1:8849`. One worker. Fish 429 and 5xx are retried. Other 4xx are returned as-is.

**Security.** With no `FISH_PROXY_API_KEYS`, the proxy accepts any client. Anyone who can reach the port spends your Fish credits. Keep the default loopback bind, or set `FISH_PROXY_API_KEYS` before you listen on another address. The proxy logs a warning at startup when it does not.

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

Speech and transcription return 503 until `FISH_API_KEY` is set, because the proxy has no key to send.

From the repository root:

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run --rm -p 127.0.0.1:8849:8849 -e FISH_PROXY_HOST=0.0.0.0 --env-file /path/to/env fish-audio-suite-proxy:latest
```

## Open WebUI

Base URL `http://127.0.0.1:8849/v1`. Open WebUI rejects an empty key. Without `FISH_PROXY_API_KEYS` any non-empty string works; with it, use one of those keys. The Fish key stays on the proxy.

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
| `model` `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` | The configured TTS model (`FISH_TTS_MODEL`). Add more with `FISH_TTS_ALIASES` |
| `model` `whisper-1`, `gpt-4o-transcribe`, other ASR names | Default ASR model |
| `model` `fish-audio/s2.1-pro` | Prefix stripped |
| `voice` or `reference_id` | Fish reference id. A list is S2 multi-speaker |
| `response_format` `mp3`, `opus`, `wav`, `pcm`, `pcm16` | The same format. `pcm16` is PCM at 24 kHz unless `sample_rate` is set |
| `response_format` `aac`, `flac`, or anything else | 400. Fish cannot produce it, and other bytes would break the client's decoder |
| `speed` | Multiplied by `FISH_SPEED`, then clamped to 0.5–2.0. OpenAI allows 0.25–4.0 |
| `response_format=srt` or `vtt` | Caption file built from Fish segments |
| Transcription `response_format` other than `json`, `text`, `verbose_json`, `srt`, `vtt` | 400, never a different format |
| `verbose_json` and word timestamps | `segments` always. `words` only when Fish returns word timings |
| Transcription body | Multipart `file`, or JSON `input_audio` (base64, optional `data:` URI) |
| `references`, `input_references` | Decoded and sent as MessagePack. Without clips the body is JSON |
| `seed`, `use_memory_cache` | Forwarded (`on` / `off`) |
| `traceparent`, `tracestate` | Forwarded. A sampled `traceparent` is minted when the client omits one |

`GET /v1/models` lists the ids the proxy knows: the Fish ids, the TTS aliases, `whisper-1`, and the `fish-audio/` prefixed ids. The speech route also forwards any other single-token TTS model id as written, so a newer Fish model works before it is listed.

`s2.1-pro-free` and `drama-3-preview` are left alone. Text that is only cues or junk returns one frame of silence in the requested format. ASR `language` is omitted unless the request or `FISH_ASR_LANGUAGE` sets it.

Errors use `{error: {code, message, type}}`. A Fish body is `type: provider_error` with `metadata.provider_name: fish-audio`. A missing `FISH_API_KEY` is 503. A wrong client key is 401. A body over the cap is 413.

`FISH_BASE=http://127.0.0.1:8080` targets self-hosted [fish-speech](https://github.com/fishaudio/fish-speech) (`POST /v1/tts`). Cloud stays `https://api.fish.audio`. Cloud `chunk_length` is clamped to 100–300. Self-host allows up to 1000.

### Request fields

Beyond the OpenAI fields, a speech request may set `latency`, `chunk_length`, `min_chunk_length`, `format`, `temperature`, `top_p`, `repetition_penalty`, `max_new_tokens`, `normalize`, `normalize_loudness`, `volume`, `sample_rate`, `mp3_bitrate`, `opus_bitrate`, `quality_guard`, and `dialogue_only`. The older `fish_latency`, `fish_format`, `fish_chunk_length`, `fish_min_chunk_length`, and `fish_quality_guard` spellings still work and are deprecated. Use the plain names.

### Text options

`dialogue_only` (or `FISH_TTS_DIALOGUE_ONLY`) keeps only the quoted speech from text that mixes dialogue with narration, such as fiction. It is off by default.

`FISH_MOOD_LEAD` turns a sentence that opens with a mood word (`Excited, ...`) into a cue (`[excited] ...`). `FISH_DROP_NARRATION` treats text that reads as narration as junk and returns silence. Both are off by default, because either can change ordinary sentences.

### Retries

Fish 429 and 5xx are retried until `FISH_PROXY_RETRY_ATTEMPTS` tries have been made in total, counting the first request (the default of 5 allows 4 retries), waiting longer each time with jitter and honoring `Retry-After`. A connection that never opened is retried. A read timeout is not, because Fish may already have made, and billed, the audio. The loop stops at `FISH_PROXY_RETRY_DEADLINE` seconds, or when the client disconnects. The deadline also cuts off a request that is still waiting on Fish, which then returns 504.

## Settings

Read once at startup. `GET /health` shows the values in effect, never a key.

| Variable | Default | Notes |
| --- | --- | --- |
| `FISH_API_KEY` | none | Speech and transcription return 503 until set |
| `FISH_BASE` | `https://api.fish.audio` | Cloud or self-hosted |
| `FISH_TTS_MODEL` | `s2.1-pro` | `FISH_MODEL` still works, deprecated |
| `FISH_TTS_ALIASES` | none | `alias=model,alias=model` on top of the OpenAI names |
| `FISH_ASR_MODEL` | `transcribe-1` | Or `transcribe-1-pro` |
| `FISH_ASR_LANGUAGE` | omitted | |
| `FISH_LATENCY` | `normal` | `low`, `balanced`, `normal` |
| `FISH_SPEED` | `1` | `FISH_SPEED_SCALE` still works, deprecated |
| `FISH_CHUNK_LENGTH` | `200` | |
| `FISH_MIN_CHUNK_LENGTH` | `50` | 0–100 |
| `FISH_FORMAT` | `mp3` | |
| `FISH_MP3_BITRATE` | `128` | 64, 128, 192 |
| `FISH_QUALITY_GUARD` | off | |
| `FISH_ASR_STRIP_SPEAKERS` | off | |
| `FISH_ASR_STRIP_CUES` | off | Drop `[laughter]` style annotations |
| `FISH_TTS_DIALOGUE_ONLY` | off | |
| `FISH_MOOD_LEAD` | off | |
| `FISH_DROP_NARRATION` | off | |
| `FISH_PROXY_API_KEYS` | none | Comma list. Clients send one as a Bearer token |
| `FISH_PROXY_MAX_BODY_BYTES` | `26214400` | 25 MiB. 0 turns the cap off |
| `FISH_PROXY_MAX_INPUT_CHARS` | `4096` | 0 turns the cap off |
| `FISH_PROXY_LOG_TEXT` | off | Log a preview of spoken text |
| `FISH_PROXY_CONNECT_TIMEOUT` | `10` | Seconds |
| `FISH_PROXY_READ_TIMEOUT` | `120` | Seconds. Also the write timeout |
| `FISH_PROXY_POOL_TIMEOUT` | `5` | Seconds |
| `FISH_PROXY_RETRY_ATTEMPTS` | `5` | 1–10 |
| `FISH_PROXY_RETRY_DEADLINE` | `90` | Seconds. 0 means no deadline |
| `FISH_PROXY_HOST` | `127.0.0.1` | |
| `FISH_PROXY_PORT` | `8849` | |
| `FISH_PROXY_WORKERS` or `WEB_CONCURRENCY` | `1` | |
| `FISH_PROXY_KEEP_ALIVE` | `5` | Seconds |
| `FISH_PROXY_GRACEFUL_SHUTDOWN` | `120` | Seconds |
| `FISH_PROXY_LIMIT_CONCURRENCY` | unset | No 503 cap |

## License

[MIT](../../LICENSE)
