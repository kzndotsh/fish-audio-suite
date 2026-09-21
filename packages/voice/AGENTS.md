# Voice (`fish-audio-suite-voice`)

> Scope: `packages/voice` (inherits root [AGENTS.md](../../AGENTS.md))

`IsolatedFishTts` + sinks + `BargeGate`. `fish-voice` is a duplex recipe on top, not the library.

| File | Owns |
| --- | --- |
| `live.py` | `IsolatedFishTts` — `stream_websocket`, `next_tts_cut`, `speak_isolated` → `asyncio.run`. Fish `TTSConfig` has no `features`; quality-guard stays proxy HTTP. Custom `httpx.AsyncClient` carries `base_url` + W3C headers onto the live WS upgrade. Branch `APIError` / `WebSocketError`; retry 429/5xx only before first audio, replaying `speak()` text. |
| `playback.py` | `SounddeviceSink` (default PCM), `FileSink`, `StdoutSink`, optional `MpvSink` |
| `barge.py` | VAD record + barge (`BLEED_DELAY_S` default 0.9). RMS + webrtcvad; `FISH_VOICE_SPEECH_FRAMES` default 8. Kit BoH/gzip is the ASR post-filter, not a second VAD. |
| `cli.py` | Mic → Fish ASR → LLM → isolated TTS. OpenRouter `:nitro` lives here. No default LLM model. Omit ASR `language` unless `FISH_ASR_LANGUAGE`. `FISH_BASE` from env. One sampled W3C trace id per turn on ASR REST + live TTS. ASR/TTS 401/402/403 exit 2. |

| Task | Command |
| --- | --- |
| Tests | `uv run pytest packages/voice` |
| Smoke | `uv run --extra cli fish-voice --smoke` |

`--smoke` uses `FileSink`; exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing.

Extras `speakers` / `vad` / `cli` are optional. `sounddevice` and `webrtcvad` stay imported inside the functions that need them so `import fish_audio_suite_voice` works without extras.

Do not `aclose()` the websocket iterator (close the client in `finally`). Skip empty deltas; yield `FlushEvent` only if at least one `TextEvent` was sent.
