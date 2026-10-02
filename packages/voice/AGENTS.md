# Voice (`fish-audio-suite-voice`)

> Scope: `packages/voice` (inherits root [AGENTS.md](../../AGENTS.md))

`IsolatedFishTts` + sinks + `BargeGate`. `fish-voice` is a duplex recipe on top, not the library.

Library classes take settings and state as arguments. Only `config.cfg()` and `tune.*.from_env()` read the environment, once, with validation (bad value: warn, use default). Do not add `os.environ` reads below that layer.

| File | Owns |
| --- | --- |
| `tune.py` | Frozen `ListenTune`, `BargeTune`, `AecTune`, `LlmTune`, their defaults, and `read_int` / `read_float` / `read_flag`. `LlmTune.from_env` picks the backend and keeps provider keys apart. |
| `config.py` | `VoiceCliConfig`: Fish/TTS fields plus the four tunes, `history_turns`, `mood_lead`, `drop_narration`. |
| `signals.py` | `DuplexSession` (quit `stop`, `TurnSignals`, `EchoCanceller`) and `install_sigint`. No module-level flags. `TurnSignals.fire` is thread-safe for the asyncio event. |
| `live.py` | `IsolatedFishTts` and `IsolatedResult`. `speak_isolated` runs on a **private thread + loop**. SDK coercions (`low` latency, `aac`/`flac`) warn once. |
| `session.py` | One Fish turn: retry 429/5xx only before first audio, replaying `speak()` text. Backoff is kit `fish_backoff_s` through `pause.sleep_unless`. |
| `wire.py` | The live websocket. `_pump_ws_audio` drives the SDK iterator from **one reader task**; the SDK holds an anyio scope inside it, so a task per `anext` raises "cancel scope in a different task". Poll cancel with a timeout on the queue; on barge/Ctrl+C close the **client**, never `aclose()` the iterator. `is_cancel_noise` checks types and the cancel flag first; message text is a logged fallback. |
| `spoken.py` | What the listener heard, for history. PCM: bytes played minus output latency, scaled by speed, cut to a word. Encoded audio cannot be cut, so a cancelled or failed turn records nothing. A finished turn keeps its cues except `[clear]`. |
| `stream_scrub.py` | Streaming scrub behind `speak_deltas` (holds unclosed spans). Duplex does not use it. |
| `pause.py` | `sleep_unless(seconds, cancelled)`, `header_retry_after`. Every retry wait goes through it. |
| `playback.py` | `PlaybackSink`, `make_sink`, `SounddeviceSink` (~30 ms DAC slices; far-end tap **before** each blocking write; records `output_latency_s`), `FileSink`, `StdoutSink`, optional `MpvSink` (mp3 on stdin). |
| `aec.py` | `EchoCanceller` (one per session): far-end tap, lazy AEC3 (`pywebrtc-audio` extra), `align()`, `clean()`, `effective_bleed_s()`. Disabled AEC still marks "playing" so the barge `over` gain works. |
| `floor.py` | `AdaptiveFloor`: quiet-percentile RMS floor shared by listen and barge. |
| `barge.py` | `BargeGate`. Floor follows the room (seed `min_rms`); ×`over` only while the speaker plays with AEC off. VAD mode 1 like listen. Hits decay after 3 misses. A trip keeps 20 frames, clears the tap; next listen starts from that clip with no cooldown. `watch` aligns the far-end ring first. |
| `listen.py` | One utterance for ASR. Start = consecutive VAD+RMS at the newest pre-pad end; an 8x peak needs more hits; impulse reject; a VAD-true frame holds the turn. |
| `asr.py` | Fish ASR over httpx. Takes `cancel` and a session `client` (`asr_client()`). 429/5xx wait with jitter and `Retry-After`. Cues are always stripped for the LLM. |
| `llm.py` | `ChatBackend` protocol, `open_chat_backend(tune)`. One event consumer for both transports. 429 retries once when Retry-After ≤ 15 s. Continuation after a mid-sentence stop is opt-in (`FISH_LLM_CONTINUE`). |
| `transports.py` | OpenRouter SDK (`send_async`, `session_id`) or one pooled httpx SSE client. Referer/title/categories and `:nitro` are OpenRouter-only; `FISH_LLM_NITRO` is off by default. `openrouter` stays imported inside helpers. |
| `envfile.py` | Minimal dotenv loader. Process env wins, then `--env-file`, else `./.env`. Non-UTF-8 files are skipped. |
| `cli.py` | `fish-voice` entry, smoke test (`--out`, else a unique temp file), `run_loop`. Ctrl+C goes to `DuplexSession.request_quit` on the loop; a stray `CancelledError` exits 2, never restarts. |
| `duplex.py` | Mic → Fish ASR → `ChatBackend` → `speak_isolated`. One W3C trace id per turn. ASR/TTS 401/402/403 exit 2. Quit is sticky. Junk and narration filtering use `c.drop_narration`; mood leads use `c.mood_lead`. |
| `debug.py` | `configure_voice_logging` (idempotent; sets a module flag, never `os.environ`). `warn` is the stderr diagnostic. Transcript lines stay on stdout. Fish WS tap only when debug is on. |

| Task | Command |
| --- | --- |
| Tests | `uv run pytest packages/voice` (an autouse `conftest.py` resets the logging flag and the warn-once cache) |
| Smoke | `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke` |
| Local env | `cp .env.example .env` then `./packages/voice/dev.sh --smoke` / `./packages/voice/dev.sh` |

`--smoke` uses `FileSink`; exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing. Do not commit `.env`. `dev.sh` builds PortAudio + Pulse with nix only on NixOS or with `FISH_VOICE_NIX=1`, pinned to this flake's lock; override with `FISH_VOICE_PORTAUDIO_LIB`.

Extras `speakers` / `vad` / `aec` / `cli` are optional. `sounddevice`, `webrtcvad`, `pywebrtc_audio`, and `openrouter` stay imported inside the functions that need them so `import fish_audio_suite_voice` works without extras. The `vad` extra is **`webrtcvad-wheels`**, not PyPI `webrtcvad` (breaks on 3.12). Mic/speakers need system PortAudio; missing lib → `PortAudioMissingError` (CLI exits 2 with hints).

Skip empty deltas; yield `FlushEvent` only after at least one `TextEvent`. `FISH_STREAM_TTS` runs the LLM and TTS together through `_TokenPipe`; `delta_events(early_flush=True)` flushes once after the first piece and again at the end only if more text followed. Fish holds text until a flush, so a single end flush would keep the reply silent. Ctrl+C must cancel TTS (`session.turn`), not only the mic `session.stop`.

Not done: one persistent mic reader shared by listen and barge (each phase still reopens the stream), and `stream_delay_ms` from the real stream latency (the ring is trimmed instead; unverified on hardware).
