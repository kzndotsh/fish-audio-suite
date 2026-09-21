# Voice (`fish-audio-suite-voice`)

> Scope: `packages/voice` (inherits root [AGENTS.md](../../AGENTS.md))

`IsolatedFishTts` + sinks + `BargeGate`. `fish-voice` is a duplex recipe on top, not the library.

| File | Owns |
| --- | --- |
| `live.py` | `IsolatedFishTts` — `stream_websocket`, `next_tts_cut`, `speak_isolated` on a **private thread + loop** (safe under `asyncio.run` / `to_thread`). Poll Ctrl+C with `asyncio.wait` on `anext`; do not `wait_for` (timeout cancels the WS generator). On barge/Ctrl+C close the **client**; do not `aclose()` the WS iterator. Fish `TTSConfig` has no `features`; quality-guard stays proxy HTTP. Custom `httpx.AsyncClient` carries `base_url` + W3C headers onto the live WS upgrade. Branch `APIError` / `WebSocketError`; retry 429/5xx only before first audio, replaying `speak()` text. Empty Fish audio prints `[tts] no audio voice=…`. |
| `playback.py` | `SounddeviceSink` (default PCM), `FileSink`, `StdoutSink`, optional `MpvSink`. DAC write taps far-end for AEC. |
| `aec.py` | Optional AEC3 (`pywebrtc-audio` extra). 44.1 kHz speaker PCM → 16 kHz; `process(near, far)` when far has energy. `FISH_VOICE_AEC=0` / missing extra / FileSink: mic passthrough. Bleed 0.1 s when loaded else 0.9. PipeWire echo-cancel is host-only. |
| `barge.py` | VAD record + barge. Bleed: `effective_bleed_s` at arm (AEC short vs `FISH_VOICE_BLEED_DELAY` 0.9). RMS + `webrtcvad-wheels`; listen `FISH_VOICE_MIN_RMS` default 200 then `AdaptiveFloor`; `FISH_VOICE_SPEECH_FRAMES` 4 / `FISH_VOICE_MIN_VOICED` 12 / `FISH_VOICE_VAD` 1. Score each frame once (do not replay VAD on the pre-pad ring); start requires VAD. Drop cough/impulse clips before ASR. Kit BoH/gzip is the ASR post-filter, not a second VAD. |
| `cli.py` | Mic → Fish ASR → LLM → `speak_isolated` (full reply, then TTS). OpenRouter `:nitro` lives here. No default LLM model. Omit ASR `language` unless `FISH_ASR_LANGUAGE`. `FISH_BASE` from env. One sampled W3C trace id per turn on ASR REST + live TTS. ASR/TTS 401/402/403 exit 2. Env: process, then `--env-file`, else `./.env`. `--debug` / `FISH_VOICE_DEBUG=1` logs VAD, barge interrupts, Fish WS events, ASR/LLM headers (no secrets). Ctrl+C sets `STOP_RECORD` and the in-flight TTS/LLM cancel; sticky quit (no `STOP_RECORD.clear()` per listen). Second SIGINT is default terminate. |
| `debug.py` | loguru DEBUG sink + Fish WS msgpack tap (audio as byte length). Off unless configured. |

| Task | Command |
| --- | --- |
| Tests | `uv run pytest packages/voice` |
| Smoke | `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke` |
| Local env | `cp .env.example .env` then `./packages/voice/dev.sh --smoke` / `./packages/voice/dev.sh` |

`--smoke` uses `FileSink`; exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing. The duplex loop does not call `speak_deltas`. Do not commit `.env`. `dev.sh` prefixes PortAudio + Pulse on `LD_LIBRARY_PATH` (NixOS); override with `FISH_VOICE_PORTAUDIO_LIB`.

Extras `speakers` / `vad` / `aec` / `cli` are optional (`cli` pulls the first three). `sounddevice`, `webrtcvad`, and `pywebrtc_audio` stay imported inside the functions that need them so `import fish_audio_suite_voice` works without extras. The `vad` extra is **`webrtcvad-wheels`**, not abandoned PyPI `webrtcvad` (that 2017 sdist imports `pkg_resources` and breaks on 3.12). Mic/speakers need system PortAudio; missing lib → `PortAudioMissingError` (CLI exits 2 with install hints). NixOS: flake wrap or `dev.sh`, not bare `uv run`.

Do not `aclose()` the websocket iterator (close the client in `finally`). Skip empty deltas; yield `FlushEvent` only if at least one `TextEvent` was sent. Duplex Ctrl+C must cancel TTS (`TURN.cancel`), not only the mic `STOP_RECORD`.
