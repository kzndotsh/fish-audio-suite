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
| `settings.py` | `ProxySettings`, built once in lifespan on `app.state.settings`. No other module reads the env. A renamed variable is read with kit `env_renamed`, which logs the old name; list the pair in the README's rename table |
| `upstream.py` | `fish_send`: bounded retry, `Retry-After` (kit `retry_after_s`), deadline, stops on disconnect, optional per-request read timeout. `FishHttp` is the client Protocol. Transcription passes `FISH_PROXY_ASR_TIMEOUT` as its read timeout and deadline; speech keeps the client-wide ones |
| `limits.py` | Request body cap middleware |
| `models.py` | TTS aliases, ASR id rules, `/v1/models` ids |
| `fields.py` | Format, silence, the `read_*` request-field readers (`read_format`, `read_flag`, `read_choice`, `read_present`, `read_reference_id`), base64 audio (`decode_audio_b64`, `AudioDecodeError`), trace headers |
| `speech.py`, `transcribe.py` | Fish TTS body and ASR upload/response |
| `errors.py` | OpenAI error envelope (`OpenAIErrorBody`), `ProxyError`. `provider_json_error` / `provider_json_from_raw` for a Fish error body, `json_from_call_failure` for a Fish call with no usable answer. `AudioDecodeError` (`fields.py`) is a `ProxyError` with status 400, and `ClipError` (`speech.py`) is the reference-clip kind of it. A transcription error names `input_audio`, never "reference audio" |

Invariants:
- `FISH_API_KEY` is read in lifespan. Importing the app and `GET /health` must work with it unset. A missing key is 503 on any request that would call Fish, and a wrong client key is 401. Text that is only junk returns local silence first, so it never reaches the key check.
- `FISH_PROXY_API_KEYS` set but with no key raises `SettingsError` at startup. Empty or unset means no client auth. Never let a typo turn auth off silently. Secret fields on settings use `repr=False`.
- A transport error message never reaches a client: send the fixed 502 or 504 text and log the detail. Close every multipart form with `async with request.form()`.
- Scrub and cue logic lives in kit. Do not copy it here.
- Every module lists its public names in `__all__`, and a public signature uses only public types (`SpeechControls`, `PackedTts`, `InboundAsr`, `FishHttp`). Formats and models use the kit literals (`AudioFormat`, `AsrFormat`, `FishLatency`); `ClientFormat` adds `pcm16`, which is a request format only. Fish ASR JSON is `AsrBody`, but parse defensively: Fish may send any JSON.
- Never log request text above DEBUG unless `FISH_PROXY_LOG_TEXT` is set. Never put a key in `/health`.
- `/health` is public API. `tts_model`, `tts_format` and `tts_speed` replace `model`, `format` and `speed_scale`; the old keys stay, with the same values, for one minor release. Rename a key the same way.
- Retry only 429, 5xx, and connections that never opened. A read timeout is not repeated.
- Return the format the client asked for, or a 400. Do not swap it. This holds for speech and transcription.
- Fish ASR `segments` are word-level (`text`, `start`, `end` per word) and `duration` is seconds. Fish sends no `words` field. OpenAI `words` are those word segments; OpenAI `segments` and SRT/VTT cues are phrases grouped from them (`caption_cues`), with punctuation taken from the transcript. A `words` array in a body still wins.
- Forward valid `traceparent` and `tracestate`. Mint a sampled one when absent.
- Fish errors become `{error:{code,message,type}}` with the upstream status. Fish bodies are `provider_error`.
