# Voice (`fish-audio-suite-voice`)

> Scope: `packages/voice` (inherits root [AGENTS.md](../../AGENTS.md))

`IsolatedFishTts` + sinks + `BargeGate`. `fish-voice` is a duplex recipe on top, not the library.

| File | Owns |
| --- | --- |
| `live.py` | `IsolatedFishTts` speak API. `speak_isolated` runs on a **private thread + loop** (safe under `asyncio.run` / `to_thread`). |
| `session.py` | One Fish turn: retry 429/5xx only before first audio, replaying `speak()` text, on a private loop. |
| `wire.py` | The live websocket. Poll Ctrl+C with `asyncio.wait` on `anext`; do not `wait_for` (timeout cancels the WS generator). On barge/Ctrl+C close the **client**; do not `aclose()` the WS iterator. Custom `httpx.AsyncClient` carries `base_url` + W3C headers onto the live WS upgrade. Branch `APIError` / `WebSocketError`. Empty Fish audio prints `[tts] no audio voice=…`. Fish `TTSConfig` has no `features`; quality-guard stays proxy HTTP. |
| `playback.py` | `SounddeviceSink` writes ~30 ms DAC slices and taps far-end **before** each blocking write so AEC matches playback. `FileSink`, `StdoutSink`, optional `MpvSink`. |
| `aec.py` | Optional AEC3 (`pywebrtc-audio` extra). 44.1 kHz speaker PCM → 16 kHz; `process(near, far)` when far has energy. `FISH_VOICE_AEC=0` / missing extra / FileSink: mic passthrough. Bleed 0.3 s when loaded else 0.9. PipeWire echo-cancel is host-only. |
| `barge.py` | `BargeGate` interrupt. Bleed: `effective_bleed_s` at arm (AEC short vs `FISH_VOICE_BLEED_DELAY` 0.9). Far-end ×2.2 barge floor only when AEC is off; with AEC3 the cleaned mic already dropped speaker bleed. Hits decay after 3 missed frames so a short gap in a phrase does not reset. |
| `listen.py` | One utterance for ASR. Listen start: consecutive VAD+RMS at the newest pre-pad end; 4x peak needs this frame ≥ 50% of peak plus 10 trailing hits. Impulse reject if peak >= 4x floor and voiced hits <= 24. Hold ratio is post-start only. |
| `llm.py` | Duplex chat stream. Both transports share one event consumer. `models.get_async` warns on 404 and does not exit. |
| `transports.py` | OpenRouter SDK (`send_async`, `stream=True`) when the base is `openrouter.ai`; httpx SSE otherwise. `:nitro` + `provider.sort=throughput`. `max_completion_tokens`, one duplex `session_id`, `async with res` on the event stream. `openrouter` stays imported inside the helpers that need it. |
| `cli.py` | Env files, `fish-voice` entry, smoke test. Process env, then `--env-file`, else `./.env`. A non-UTF-8 env file is skipped. Duplex exits 2 when playback is `file`, unknown, or `mpv` is not on PATH. `--debug` / `FISH_VOICE_DEBUG=1`. Second SIGINT is default terminate. |
| `duplex.py` | Mic → Fish ASR → LLM → `speak_isolated` (full reply, then TTS). One sampled W3C trace id per turn on ASR REST + live TTS. ASR/TTS 401/402/403 exit 2. Ctrl+C sets `STOP_RECORD` and the in-flight TTS/LLM cancel; sticky quit (no `STOP_RECORD.clear()` per listen). |
| `asr.py` | Fish ASR over httpx. No default voice. Omit ASR `language` unless `FISH_ASR_LANGUAGE`. Retry 429/5xx. |
| `config.py` | `VoiceCliConfig` from the process environment. `FISH_LLM_BACKEND` is only the ready-line label. |
| `signals.py` | `STOP_RECORD`, `TURN`, `request_quit`. |
| `debug.py` | loguru DEBUG sink + Fish WS msgpack tap (audio as byte length). Off unless configured. |

| Task | Command |
| --- | --- |
| Tests | `uv run pytest packages/voice` |
| Smoke | `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke` |
| Local env | `cp .env.example .env` then `./packages/voice/dev.sh --smoke` / `./packages/voice/dev.sh` |

`--smoke` uses `FileSink`; exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing. The duplex loop does not call `speak_deltas`. Do not commit `.env`. `dev.sh` prefixes PortAudio + Pulse on `LD_LIBRARY_PATH` (NixOS); override with `FISH_VOICE_PORTAUDIO_LIB`.

Extras `speakers` / `vad` / `aec` / `cli` are optional (`cli` pulls the first three plus `openrouter`). `sounddevice`, `webrtcvad`, `pywebrtc_audio`, and `openrouter` stay imported inside the functions that need them so `import fish_audio_suite_voice` works without extras. The `vad` extra is **`webrtcvad-wheels`**, not abandoned PyPI `webrtcvad` (that 2017 sdist imports `pkg_resources` and breaks on 3.12). Mic/speakers need system PortAudio; missing lib → `PortAudioMissingError` (CLI exits 2 with install hints). NixOS: flake wrap or `dev.sh`, not bare `uv run`.

Do not `aclose()` the websocket iterator (close the client in `finally`). Skip empty deltas; yield `FlushEvent` only if at least one `TextEvent` was sent. Duplex Ctrl+C must cancel TTS (`TURN.cancel`), not only the mic `STOP_RECORD`.
