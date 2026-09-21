# Voice (`fish-audio-suite-voice`)

> Scope: `packages/voice` (inherits root [AGENTS.md](../../AGENTS.md))

`IsolatedFishTts` + sinks + `BargeGate`. `fish-voice` is a duplex recipe on top, not the library.

| File | Owns |
| --- | --- |
| `live.py` | `IsolatedFishTts` — `stream_websocket`, `next_tts_cut`, `speak_isolated` → `asyncio.run` |
| `playback.py` | `SounddeviceSink` (default PCM), `FileSink`, `StdoutSink`, optional `MpvSink` |
| `barge.py` | VAD record + barge (`BLEED_DELAY_S` default 0.9) |
| `cli.py` | Mic → Fish ASR → LLM → isolated TTS. OpenRouter `:nitro` lives here |

| Task | Command |
| --- | --- |
| Tests | `uv run pytest packages/voice` |
| Smoke | `uv run --extra cli fish-voice --smoke` |

`--smoke` uses `FileSink`; exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing.

Extras `speakers` / `vad` / `cli` are optional. `sounddevice` and `webrtcvad` stay imported inside the functions that need them so `import fish_audio_suite_voice` works without extras.

Do not `aclose()` the websocket iterator (close the client in `finally`). Skip empty deltas; yield `FlushEvent` only if at least one `TextEvent` was sent.
