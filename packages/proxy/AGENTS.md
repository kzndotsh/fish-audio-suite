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

OpenAI field translation is split across `fields.py`, `speech.py`, `transcribe.py`, and `errors.py`. `contract.py` re-exports that surface. The FastAPI app, retry loop, and routes stay in `server.py`. Strip `fish-audio/` from model ids. TTS aliases `tts-1` / `tts-1-hd` / `gpt-4o-mini-tts` / `playai-tts` → `s2.1-pro`. ASR aliases (`whisper-1`, `gpt-4o-transcribe`, Groq whisper slugs) → default ASR model. Do not remap `s2.1-pro-free` or `drama-3-preview`. `response_format=pcm16` → Fish `pcm` at 24 kHz unless `sample_rate` is set. `verbose_json` plus `timestamp_granularities=word` adds a `words` array. Transcription accepts multipart `file` or JSON `input_audio`. `is_tts_junk` → silent MP3. Forward `seed`, `use_memory_cache`, `reference_id` (string or list). `references` and `input_references` decode to raw audio and go upstream as MessagePack. Fish TTS 4xx/5xx → OpenAI `{error:{code,message,type}}` with the upstream status; Fish bodies are `provider_error`. Cloud `chunk_length` 100–300; self-host 100–1000. Omit ASR `language` unless the client or `FISH_ASR_LANGUAGE` sets it. Forward valid incoming `traceparent`/`tracestate` onto Fish TTS/ASR; mint a sampled `traceparent` if omitted. Retry 429/5xx (and transport timeouts) with exponential backoff; do not retry other 4xx. Missing `FISH_API_KEY` → 401. Transcription `srt`/`vtt` return those files from Fish segments (kit formatters). Read `timestamp_granularities` and `timestamp_granularities[]`.
