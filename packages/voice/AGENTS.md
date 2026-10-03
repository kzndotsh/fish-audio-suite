# Voice (`fish-audio-suite-voice`)

> Scope: `packages/voice` (inherits root [AGENTS.md](../../AGENTS.md))

`IsolatedFishTts` + sinks + `BargeGate`. `fish-voice` is a duplex recipe on top, not the library.

Library classes take settings and state as arguments. Application settings come from `config.load_config()` and `tune.*.from_env()` only, read once with validation (bad value: warn, use default). Do not add settings reads below that layer. Two CLI integration points are exempt because they are not settings: `debug.debug_level()` reads `FISH_VOICE_DEBUG`, and `envfile.py` writes `os.environ` when it loads a dotenv file.

Public surface: every module lists its exports in `__all__`, and the package root exports every type a public signature uses. `tests/test_public_api.py` checks both. Constants are `Final`, sink and backend names are enums or literals, and chat messages are the kit's `ChatMessage`. A new boolean parameter is keyword-only.

| File | Owns |
| --- | --- |
| `tune.py` | Frozen `ListenTune`, `BargeTune`, `AecTune`, `LlmTune`, their defaults, and `read_int` / `read_float` / `read_flag`. `LlmTune.from_env` picks the backend and keeps provider keys apart. |
| `config.py` | `VoiceCliConfig`: Fish/TTS fields plus the four tunes, `history_turns`, `mood_lead`, `drop_narration`. |
| `signals.py` | `DuplexSession` (quit `stop`, `TurnSignals`, `EchoCanceller`) and `install_sigint`. No module-level flags. `TurnSignals.fire` is thread-safe for the asyncio event. |
| `live.py` | `IsolatedFishTts` and `IsolatedResult` (`error` is a kit `FishHttpError`; duplex tells fatal from transient by `isinstance(..., FishAuthError)`, never by status numbers). `speak_isolated(text, sink, *, cancel=None, on_first_audio=None)` runs on a **private thread + loop**; `cancel` is keyword-only. SDK coercions (`low` latency, `aac`/`flac`) warn once. |
| `tts_turn.py` | One Fish turn: retry 429/5xx only before first audio, replaying `speak()` text. Backoff is kit `fish_backoff_s` through `pause.sleep_unless`. |
| `wire.py` | The live websocket. `_pump_ws_audio` drives the SDK iterator from **one reader task** through a one-chunk queue (backpressure, so the reader cannot buffer a long reply); the SDK holds an anyio scope inside it, so a task per `anext` raises "cancel scope in a different task". Poll cancel with a timeout on the queue; a reader that ends with no marker (the stream itself was cancelled) is surfaced, not polled forever; on barge/Ctrl+C close the **client**, never `aclose()` the iterator. `is_cancel_noise` checks types and the cancel flag first; message text is a logged fallback. |
| `spoken.py` | What the listener heard, for history. PCM: bytes played minus output latency, scaled by speed, cut to a word. Encoded audio cannot be cut, so a cancelled or failed turn records nothing. A finished turn keeps its cues except `[clear]`. |
| `stream_scrub.py` | Streaming scrub behind `speak_deltas` (holds unclosed spans), as `_StreamScrubber`; `delta_events` wraps it. Duplex reaches it through `speak_stream_isolated` when `FISH_STREAM_TTS` is on. |
| `pause.py` | `sleep_unless(seconds, cancelled)`. Every retry wait goes through it. `Retry-After` is parsed by the kit's `retry_after_s`. |
| `playback.py` | `PlaybackSink`, `make_sink`, `SounddeviceSink` (~30 ms DAC slices; far-end tap **before** each blocking write; records `output_latency_s`), `FileSink`, `StdoutSink`, optional `MpvSink` (mp3 on stdin). |
| `aec.py` | `EchoCanceller` (one per session): far-end tap, lazy AEC3 (`pywebrtc-audio` extra), `align()`, `clean()`, `effective_bleed_s()`. Disabled AEC still marks "playing" so the barge `over` gain works. |
| `floor.py` | `AdaptiveFloor`: quiet-percentile RMS floor shared by listen and barge. Warm-up is capped by the window size. |
| `barge.py` | `BargeGate`. Floor follows the room (seed `min_rms`) and only learns from frames below the floor itself, not the boosted need; ×`over` only while the speaker plays with AEC off. `aec=None` means no AEC; an explicit `bleed_delay_s` beats the AEC default. VAD mode 1 like listen. Hits decay after 3 misses. A trip keeps 20 frames, clears the tap; next listen starts from that clip with no cooldown. `watch` aligns the far-end ring first. |
| `listen.py` | One utterance for ASR. Start = consecutive VAD+RMS at the newest pre-pad end; an 8x peak needs more hits; impulse reject; a VAD-true frame holds the turn. |
| `asr.py` | Fish ASR over httpx. Takes `cancel` and a session `client` (`asr_client()`). 429/5xx wait with jitter and `Retry-After`. Cues are always stripped for the LLM. |
| `llm.py` | `ChatBackend` protocol, `open_chat_backend(tune)`. One event consumer for both transports. 429 retries once when Retry-After ≤ 15 s. Continuation after a mid-sentence stop is opt-in (`FISH_LLM_CONTINUE`). |
| `transports.py` | OpenRouter SDK (`send_async`, `session_id`) or one pooled httpx SSE client. Referer/title/categories and `:nitro` are OpenRouter-only; `FISH_LLM_NITRO` is off by default. `reasoning_effort` goes on the httpx path only. `openrouter` stays imported inside helpers. |
| `envfile.py` | Minimal dotenv loader. Process env wins, then `--env-file`, else `./.env`. Non-UTF-8 files are skipped. |
| `cli.py` | `fish-voice` entry, smoke test (`--out`, else a unique temp file), `run_loop`. Ctrl+C goes to `DuplexSession.request_quit` on the loop; a stray `CancelledError` exits 2, never restarts. |
| `duplex.py` | The turn loop: hear a line, remember it, answer it, repeat. Mic → Fish ASR → `ChatBackend` → `speak_isolated`, one W3C trace id per turn. ASR/TTS 401/402/403 exit 2. Quit is sticky. Public: `duplex_turns`, `bye`, `EXIT_OK`, `EXIT_FATAL`. |
| `duplex_state.py` | `DuplexContext` (config, TTS client, device, backend, session, ASR client, history, barge prefix) and the exit codes. |
| `hearing.py` | `hear_line`: open the mic, record, Fish ASR, then skip backchannel and hallucinations, detect quit, drop a stale repeat of the last line. Returns a `HeardLine`. |
| `history.py` | `opening_history`, `remember_user`, `trim_history`. Trimming drops an oldest user and assistant pair together; the seed exchange stays pinned. Pure functions on the message list. |
| `reply.py` | `collect_reply`, `speak_reply`, `stream_turn`, `after_speech`, `turn_summary`. Junk and narration filtering use `c.drop_narration`; mood leads use `c.mood_lead`. Streaming TTS (`FISH_STREAM_TTS`) skips the whole-reply junk and `drop_narration` filters, because it never holds the whole reply. An empty streamed reply cancels the Fish turn and speaks nothing. |
| `debug.py` | `configure_voice_logging` (idempotent; sets a module flag, never `os.environ`). `warn` is the stderr diagnostic. Conversation lines (`conversation()`) always go to stdout; on a terminal with debug on they carry the same time columns as the log, piped they stay plain. Fish WS tap only when debug is on. |

| Task | Command |
| --- | --- |
| Tests | `uv run pytest packages/voice` (an autouse `conftest.py` resets the logging flag and the warn-once cache) |
| Smoke | `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke` |
| Local env | `cp .env.example .env` then `./packages/voice/dev.sh --smoke` / `./packages/voice/dev.sh` |

`--smoke` uses `FileSink`; exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing. Do not commit `.env`. `dev.sh` builds PortAudio + Pulse with nix only on NixOS or with `FISH_VOICE_NIX=1`, pinned to this flake's lock; override with `FISH_VOICE_PORTAUDIO_LIB`.

LLM providers live in one table, `LLM_PROVIDERS` in `tune.py` (name, default base, host, key variable, model variable). A provider is always read from the **host of the final base URL**, never from `FISH_LLM_PROVIDER` alone, and each provider reads only its own key variable, so a key cannot go to another host. Add a provider by adding a row, a test in `test_llm_providers.py`, and its README table row. A provider that speaks the OpenAI chat-completions API needs no new backend.

Extras `speakers` / `vad` / `aec` / `cli` are optional. `sounddevice`, `webrtcvad`, `pywebrtc_audio`, and `openrouter` stay imported inside the functions that need them so `import fish_audio_suite_voice` works without extras. The `vad` extra is **`webrtcvad-wheels`**, not PyPI `webrtcvad` (breaks on 3.12). Mic/speakers need system PortAudio; missing lib → `PortAudioMissingError` (CLI exits 2 with hints).

Skip empty deltas; yield `FlushEvent` only after at least one `TextEvent`. `FISH_STREAM_TTS` runs the LLM and TTS together through `_TokenPipe`; `delta_events(early_flush=True)` flushes once when the first sentence ends (judged on all text sent so far, so "Mr" then "." is not one), or after two send windows of text with no sentence end, and again at the end only if more text followed. A flush mid-sentence makes Fish end the fragment as a finished utterance and pause. Fish holds text until a flush, so a single end flush would keep the reply silent. Ctrl+C must cancel TTS (`session.turn`), not only the mic `session.stop`.

Cancellation: a `CancelledError` ends a stream quietly only when `wire.is_own_cancel(flag)` is true (our flag is set and the task has no pending cancel of its own). Anything else, such as `asyncio.timeout` or an outer `task.cancel()`, propagates. Reap a cancelled child with `wire.reap(task)`, never `suppress(BaseException)`, so Ctrl+C and `SystemExit` are not eaten. Secret fields (`api_key`, `fish_api_key`, `LlmTune.key`) are `field(repr=False)`; keep new ones that way and run a repr test. A plain `http` base for a non-loopback host only warns (`warn_if_insecure_base`), since LAN self-hosting is legitimate. Kit's transport errors are fixed text; `debug.with_detail` adds the exception text for the local terminal only.

Not done: one persistent mic reader shared by listen and barge (each phase still reopens the stream), and `stream_delay_ms` from the real stream latency (the ring is trimmed instead; unverified on hardware).
