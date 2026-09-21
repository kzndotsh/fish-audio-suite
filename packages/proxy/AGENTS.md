# Proxy (`fish-audio-suite-proxy`)

> Scope: `packages/proxy` (inherits root [AGENTS.md](../../AGENTS.md))

OpenAI-shaped Fish HTTP on **8849**: `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`, `/health`.

CLI: `fish-audio-suite-proxy`. Import: `fish_audio_suite_proxy`. Uvicorn uses `uvicorn[standard]` (`loop`/`http` auto → uvloop + httptools). Default **one worker**. Import string `fish_audio_suite_proxy.server:app` so `FISH_PROXY_WORKERS` can spawn processes. `timeout_graceful_shutdown` default 120s (Fish stream budget). Optional `FISH_PROXY_LIMIT_CONCURRENCY` → 503. `ws=none`. Do not set `forwarded-allow-ips=*`.

| Task | Command |
| --- | --- |
| Run | `uv run --package fish-audio-suite-proxy fish-audio-suite-proxy` |
| Tests | `uv run pytest packages/proxy` |
| Image | `docker build -t fish-audio-suite-proxy:latest .` |

Scrub through kit. `FISH_API_KEY` is read in lifespan — importing the app and `GET /health` must work with it unset. `FISH_BASE` may be Fish Cloud or a self-hosted fish-speech `:8080`.

In `server.py`: `whisper-1` → `transcribe-1`. Do not remap `s2.1-pro-free` or `drama-3-preview`. `is_tts_junk` → silent MP3. Forward `seed`, `references`, `use_memory_cache`, `reference_id` (string or list). Fish TTS 4xx/5xx → JSON with the upstream status. Cloud `chunk_length` 100–300; self-host 100–1000. Omit ASR `language` unless the client or `FISH_ASR_LANGUAGE` sets it. Forward valid incoming `traceparent`/`tracestate` onto Fish TTS/ASR; mint a sampled `traceparent` if omitted. Fish errors stay `{message, status}`. Retry 429/5xx (and transport timeouts) with exponential backoff; do not retry other 4xx. Missing `FISH_API_KEY` → 401. Transcription `srt`/`vtt` return those files from Fish segments (kit formatters). Read `timestamp_granularities` and `timestamp_granularities[]`.
