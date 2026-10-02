# Proxy (`fish-audio-suite-proxy`)

> Scope: `packages/proxy` (inherits root [AGENTS.md](../../AGENTS.md))

OpenAI-shaped Fish HTTP on **8849**, bound to `127.0.0.1` unless `FISH_PROXY_HOST` says otherwise: `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`, `/health`. Behavior, fields, and every setting are in [`README.md`](README.md). Keep that table in step with `settings.py`.

CLI: `fish-audio-suite-proxy`. Import: `fish_audio_suite_proxy`. Start uvicorn with the import string `fish_audio_suite_proxy.server:app` so `FISH_PROXY_WORKERS` can spawn processes. `ws=none`. Do not set `forwarded-allow-ips=*`.

| Task | Command |
| --- | --- |
| Run | `uv run --package fish-audio-suite-proxy fish-audio-suite-proxy` |
| Tests | `uv run pytest packages/proxy` |
| Image | `docker build -t fish-audio-suite-proxy:latest .` |

| Module | Owns |
| --- | --- |
| `server.py` | App, lifespan, routes, client-key check |
| `settings.py` | `ProxySettings`, built once in lifespan on `app.state.settings`. No other module reads the env |
| `upstream.py` | `fish_send`: bounded retry, `Retry-After`, deadline, stops on disconnect |
| `limits.py` | Request body cap middleware |
| `models.py` | TTS aliases, ASR id rules, `/v1/models` ids |
| `fields.py` | Format, silence, request-field readers, trace headers |
| `speech.py`, `transcribe.py` | Fish TTS body and ASR upload/response |
| `errors.py` | OpenAI error envelope, `ProxyError` |

Invariants:
- `FISH_API_KEY` is read in lifespan. Importing the app and `GET /health` must work with it unset. A missing key is 503, a wrong client key is 401.
- Scrub and cue logic lives in kit. Do not copy it here.
- Never log request text above DEBUG unless `FISH_PROXY_LOG_TEXT` is set. Never put a key in `/health`.
- Retry only 429, 5xx, and connections that never opened. A read timeout is not repeated.
- Return the format the client asked for, or a 400. Do not swap it.
- A word array comes from Fish word timings only. Segments are not words.
- Forward valid `traceparent` and `tracestate`. Mint a sampled one when absent.
- Fish errors become `{error:{code,message,type}}` with the upstream status. Fish bodies are `provider_error`.
