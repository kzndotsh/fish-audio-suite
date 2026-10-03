# fish-audio-suite-voice

One Fish Audio websocket per turn, playback sinks, and an optional duplex CLI.

Import `IsolatedFishTts` when another app already owns the microphone and the LLM. Use `fish-voice` when you want mic, Fish ASR, an LLM reply, and speakers in one process. The LLM is any OpenAI-compatible chat-completions server, or OpenRouter.

Part of [fish-audio-suite](../../README.md). Unofficial.

## Quick start

```python
from pathlib import Path
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
result = tts.speak_isolated("Hello there.", FileSink(Path("turn.wav")))
```

`speak_isolated` runs the websocket on a private thread and event loop, so it is safe under `asyncio.run` or `to_thread`. Retry of 429 and 5xx happens only before the first audio byte. There is no default `voice_id`.

`speak_isolated(text, sink, *, cancel=None, on_first_audio=None)` takes `cancel` (a `threading.Event`) by keyword. The returned `TtsResult` carries `error`, a kit error such as `FishAuthError` (401, 402, 403), `FishRateLimitError` (429) or `FishUpstreamError` (5xx), when the turn failed with a status. Test it with `isinstance`; `error_status` and `error_message` stay for display. Its timings use the kit's `LatencySnapshot` names: `tts_first_text_ms` (TTS start to the first text sent to Fish) and `tts_first_audio_ms` (TTS start to the first audio from Fish, Fish's time-to-first-audio). `speak_deltas_isolated(deltas, sink, ...)` does the same for a token stream while the model is still writing. The package root exports the types its signatures use: `IsolatedFishTts`, `TtsResult`, `PlaybackSink`, `PlaybackKind`, `make_sink`, `BargeGate`, `EchoCanceller`, `DuplexSession`, `ChatBackend`, `ListenTune`, `BargeTune`, `AecTune`, `LlmSettings` and `PortAudioMissingError`. Each module lists its own exports in `__all__`.

```bash
cp .env.example .env
./packages/voice/dev.sh --smoke    # writes a new temp wav, or pass --out PATH
./packages/voice/dev.sh            # duplex
./packages/voice/dev.sh --debug    # or FISH_VOICE_DEBUG=1; --trace (or 2) adds heartbeats
```

`--smoke` exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing. The same entry point from anywhere the extra is installed:

```bash
uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke
```

## How it works

| Sink | When |
| --- | --- |
| `sounddevice` | Default. Plays PCM through PortAudio in short slices |
| `file` | Writes a WAV or raw PCM when the turn finishes |
| `stdout` | Raw chunks on stdout |
| `mpv` | Optional. Plays mp3 from stdin. Not required for the other sinks |

Duplex is mic, then Fish ASR, then an LLM, then one Fish turn. Settings are environment variables. Process env wins, then `--env-file`, then `./.env`.

`--debug` logs VAD, barge-in, Fish websocket metadata, and LLM metadata on stderr. What was said stays on stdout. Duplex exits 2 when playback is `file`, unknown, or `mpv` is not on `PATH`. It also exits 2 on ASR or TTS 401, 402, or 403.

### PortAudio

Speakers and the microphone need the C library. `uv` and `pip` do not install it.

```bash
# Debian / Ubuntu
sudo apt install libportaudio2
# Fedora
sudo dnf install portaudio
# macOS
brew install portaudio
```

On NixOS use `./packages/voice/dev.sh` or `nix run .#fish-audio-suite-voice`. Both put PortAudio on `LD_LIBRARY_PATH`. A missing library raises `PortAudioMissingError` when the stream opens.

### Echo and barge-in

The `cli` extra includes AEC3. Speaker audio is subtracted from the mic before voice activity detection. `FISH_VOICE_AEC=0` leaves the mic unchanged and keeps the longer bleed delay (0.9 s). With AEC loaded, that delay is 0.3 s.

A barge-in keeps the audio that tripped it and clears the speaker tap. The next listen starts from that clip.

The barge floor follows the room: it is the quiet-percentile of recent mic frames, never below `FISH_VOICE_BARGE_RMS`. If barge-in fires on a cough or a chair, raise `FISH_VOICE_BARGE_FRAMES` (30 ms each) before raising the RMS.

### History after a barge-in

The chat history gets only what you probably heard. For PCM that is a word-aligned estimate from the bytes played, the speed, and the device buffer. Encoded playback (`mpv`) cannot be cut by length, so an interrupted or failed reply is left out of history.

### Cues in replies

With the default system prompt the session starts with one pinned exchange that shows several cues
(`DEFAULT_SEED_EXCHANGE` in the kit). It costs about 40 tokens per request and is never trimmed from
the history. A custom `FISH_VOICE_SYSTEM_PROMPT` gets no seed, so it stays in control of the replies.

### LLM providers

`FISH_LLM_PROVIDER` names the chat provider: `openrouter` (the default) or `experiential`. It
supplies the default base URL, and `FISH_LLM_BASE` overrides it. The provider is always read from
the host of the final base URL, and each provider has its own key variable, so a key never goes
to the wrong host.

| Provider | Default base | Key variable | Model variable |
| --- | --- | --- | --- |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `FISH_LLM_MODEL_OPENROUTER` |
| `experiential` | `https://api.experientiallabs.ai/v1` | `EXPLABS_API_KEY` | `FISH_LLM_MODEL_EXPERIENTIAL` |
| any other host | `FISH_LLM_BASE` | `OPENAI_API_KEY` | `FISH_LLM_MODEL` |

`FISH_LLM_API_KEY` and `FISH_LLM_MODEL` work for every provider. An explicit `FISH_LLM_API_KEY` wins, and
the chosen provider's own model variable comes before `FISH_LLM_MODEL`. Keep both providers in one
`.env` and switch with `FISH_LLM_PROVIDER`.

Experiential speaks the OpenAI chat-completions API, so it runs on the `openai` backend (one pooled
httpx client). `FISH_LLM_BACKEND` picks `openai` or `openrouter` (the SDK) explicitly; unset, it
follows the host. What to know about Experiential, from its published contract:

- **Reasoning.** `FISH_LLM_REASONING_EFFORT` (`none`, `minimal`, `low`, `medium`, `high` or `max`) is
  sent as `reasoning_effort` on the `openai` backend. Reasoning models can be slow to a first word,
  so a low effort suits voice. Which values a model accepts depends on its route, and the value is
  passed through unchanged. Empty omits the field. The OpenRouter SDK backend never sends it.
- **Fields.** Experiential answers 400 to a request field it does not know, so the request carries
  only `model`, `messages`, `stream`, `temperature` and `max_tokens`, plus `reasoning_effort` when set.
- **No `:nitro`.** That suffix is OpenRouter-only (`FISH_LLM_NITRO` never applies to Experiential).
- **Errors.** A spent free allowance is a 429 `insufficient_quota` with no `Retry-After`. It is
  printed with its code and not retried.
- **Debug log.** With `--debug` the response line shows the request id (`x-request-id`), the route
  (`x-gateway-provider`) and the zero-data-retention posture (`x-gateway-zdr`).
- **Privacy.** Platform-funded calls are captured: Experiential stores both the request and the
  model's reply. A free organization cannot turn that off, and only a Pro organization can. Your
  spoken conversation is the prompt, so do not use a free organization for anything private.
- **https only for the automatic key.** `EXPLABS_API_KEY` is picked up only when the base is
  `https`. A plain `http` base would send it unencrypted, so use `FISH_LLM_API_KEY` to override.

OpenRouter-only options: `FISH_LLM_NITRO=1` adds `:nitro` to the model and sorts providers by
`FISH_LLM_PROVIDER_SORT`, and `FISH_LLM_REFERER`, `FISH_LLM_TITLE`, `FISH_LLM_CATEGORIES` set the
attribution (empty disables one). The `openai` backend sends none of those headers.

### Streaming the reply

By default the reply is spoken after the model finishes. Set `FISH_VOICE_STREAM_TTS=1` to speak
while the model is still writing. The Fish socket opens with the request and the first
sentence is flushed at once, so audio starts about when that sentence is done instead of when
the whole reply is. The socket then gets one more flush at the end.

- One extra flush is needed: Fish holds text until a chunk fills or a flush arrives, so a single
  flush at the end keeps a short reply silent until the model is done.
- The reply is scrubbed as it arrives, and the whole-reply junk check is skipped.
- The barge-in gate arms at the first audio chunk.
- If Fish fails before any audio, the finished reply is spoken on the normal path.

### Roleplay helpers

Both are off by default because they also change ordinary English. `FISH_TTS_MOOD_LEAD=1` rewrites a sentence-leading mood word (`Excited, hello`) into a `[cue]`. `FISH_TTS_DROP_NARRATION=1` drops lines that are only stage directions (`She smiles.`).

## Extras

| Extra | Pulls in |
| --- | --- |
| `speakers` | `sounddevice` |
| `vad` | `webrtcvad-wheels` |
| `aec` | `pywebrtc-audio` |
| `cli` | the three above, plus `openrouter` |

`import fish_audio_suite_voice` works without them. The heavy imports happen inside the functions that need them.

## Settings

Every variable is read once at startup into frozen settings objects. A value that is not a number, or is out of range, is ignored with a warning on stderr and the default is used.

Required:

| Variable | Default |
| --- | --- |
| `FISH_API_KEY` | none |
| `FISH_VOICE_ID` | none |
| `FISH_LLM_API_KEY` (fallback: the provider's own key variable, see above) | none. Duplex only |
| `FISH_LLM_MODEL` (the provider's model variable first; `OPENROUTER_MODEL` for OpenRouter and custom hosts) | none. Duplex only |

Speech:

| Variable | Default |
| --- | --- |
| `FISH_BASE` | `https://api.fish.audio` |
| `FISH_TTS_MODEL` | `s2.1-pro` |
| `FISH_LATENCY` | `normal` (or `balanced`) |
| `FISH_SPEED` | `1` |
| `FISH_VOLUME` | `0` (dB) |
| `FISH_TEMPERATURE`, `FISH_TOP_P` | `0.7`, `0.7` |
| `FISH_REPETITION_PENALTY` | `1.2` |
| `FISH_CHUNK_LENGTH` | `200` (cloud max 300) |
| `FISH_MIN_CHUNK_LENGTH` | `50` |
| `FISH_SAMPLE_RATE` | `44100` |
| `FISH_ASR_LANGUAGE` | omitted |
| `FISH_ASR_MODEL` | `transcribe-1` (or `transcribe-1-pro`) |
| `FISH_VOICE_PLAYBACK` | `sounddevice` |
| `FISH_VOICE_DEVICE` | host default. A PortAudio index or name |

LLM:

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_LLM_PROVIDER` | `openrouter` | `openrouter` or `experiential`. Sets the default base, key variable and model variable. See above |
| `FISH_LLM_BACKEND` | from the base URL | `openai` or `openrouter` |
| `FISH_LLM_BASE` (fallback `OPENROUTER_BASE_URL`) | `https://openrouter.ai/api/v1` | API origin |
| `FISH_LLM_REASONING_EFFORT` | unset | `none`, `minimal`, `low`, `medium`, `high` or `max`. Sent on the `openai` backend only |
| `FISH_LLM_TEMPERATURE` | `0.8` | 0 to 2 |
| `FISH_LLM_TIMEOUT` | `120` | Seconds per request |
| `FISH_LLM_MAX_TOKENS` | `1200` | Completion cap |
| `FISH_LLM_NITRO` | off | OpenRouter only. `:nitro` plus provider sort |
| `FISH_LLM_PROVIDER_SORT` | `throughput` | Used with `FISH_LLM_NITRO` |
| `FISH_LLM_REFERER`, `FISH_LLM_TITLE`, `FISH_LLM_CATEGORIES` | this project's | OpenRouter attribution. Empty disables |
| `FISH_LLM_CONTINUE` | off | Send one more request when a reply stops mid-sentence |
| `FISH_VOICE_SYSTEM_PROMPT` | the kit default | System prompt text. Set but blank sends no system prompt |
| `FISH_VOICE_HISTORY_TURNS` | `20` | User and assistant pairs kept |
| `FISH_VOICE_STREAM_TTS` | off | Speak the reply while the model is still writing it. See streaming above |
| `FISH_VOICE_REPEAT_WINDOW` | `1.5` | Seconds. A line equal to the previous one is dropped only if it ends this soon after the mic opens. 0 never drops a repeat |
| `FISH_TTS_MOOD_LEAD`, `FISH_TTS_DROP_NARRATION` | off | See roleplay helpers |

Listen and interrupt. Times are approximate at 30 ms frames.

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_VOICE_SPEECH_FRAMES` | `4` | About 120 ms before speech counts |
| `FISH_VOICE_PRE_PAD_FRAMES` | `20` | Frames kept before the start |
| `FISH_VOICE_MIN_RMS` | `200` | Listen floor seed. It can rise in a loud room |
| `FISH_VOICE_MIN_VOICED_FRAMES` | `12` | About 360 ms of voice. Drops a cough |
| `FISH_VOICE_SILENCE_FRAMES` | `40` | About 1.2 s of quiet ends the turn |
| `FISH_VOICE_VAD` | `1` | WebRTC VAD mode, 0–3. Higher is pickier |
| `FISH_VOICE_COOLDOWN` | `0.8` | Seconds after playback |
| `FISH_VOICE_BLEED_DELAY` | `0.9` | Used when AEC is off or missing |
| `FISH_VOICE_AEC` | on | `0` disables in-process echo cancellation |
| `FISH_VOICE_AEC_WET` | `0.85` | Mix of cleaned mic into the raw mic, 0 to 1 |
| `FISH_VOICE_AEC_BLEED_DELAY` | `0.3` | Bleed delay once AEC3 is loaded |
| `FISH_VOICE_BARGE_FRAMES` | `10` | Loud voiced frames in a row that interrupt (about 300 ms) |
| `FISH_VOICE_BARGE_RMS` | `220` | Barge floor seed. It follows the room, never below this |
| `FISH_VOICE_BARGE_PLAYING_GAIN` | `2.2` | Floor multiplier while the speaker plays and AEC is off. At least 1 |
| `FISH_VOICE_DEBUG` | off | `1` or `--debug` logs events. `2`, `trace` or `--trace` adds mic heartbeats, raw audio events and HTTP lines |

`dev.sh` also reads `FISH_VOICE_ENV_FILE` (env file path), `FISH_VOICE_PORTAUDIO_LIB` (library directories), and `FISH_VOICE_NIX=1` (build PortAudio with nix on a host that is not NixOS).

Commented copies live in [`.env.example`](../../.env.example).

### Renamed in 0.2.0

These variables were renamed. Until the next minor release the old name still works when the new one is unset or blank (except `FISH_VOICE_SYSTEM_PROMPT`, where a blank value means no system prompt), with a `fish-voice: <old> is deprecated; use <new>` warning on stderr. The new name wins when both are set.

| Old | New |
| --- | --- |
| `FISH_PLAYBACK` | `FISH_VOICE_PLAYBACK` |
| `FISH_HISTORY_TURNS` | `FISH_VOICE_HISTORY_TURNS` |
| `FISH_SYSTEM_PROMPT` | `FISH_VOICE_SYSTEM_PROMPT` |
| `FISH_STREAM_TTS` | `FISH_VOICE_STREAM_TTS` |
| `FISH_VOICE_REPEAT_WINDOW_S` | `FISH_VOICE_REPEAT_WINDOW` |
| `FISH_VOICE_PRE_PAD` | `FISH_VOICE_PRE_PAD_FRAMES` |
| `FISH_VOICE_MIN_VOICED` | `FISH_VOICE_MIN_VOICED_FRAMES` |
| `FISH_VOICE_BARGE_OVER` | `FISH_VOICE_BARGE_PLAYING_GAIN` |
| `FISH_VOICE_AEC_BLEED` | `FISH_VOICE_AEC_BLEED_DELAY` |
| `FISH_MOOD_LEAD` | `FISH_TTS_MOOD_LEAD` |
| `FISH_DROP_NARRATION` | `FISH_TTS_DROP_NARRATION` |
| `FISH_LLM_KEY` | `FISH_LLM_API_KEY` |
| `FISH_MODEL` | `FISH_TTS_MODEL` (also read by the proxy, so one `.env` serves both) |
| `FISH_SPEED_SCALE` | `FISH_SPEED` (also read by the proxy) |
| `FISH_VOICE_ENV` (`dev.sh`) | `FISH_VOICE_ENV_FILE` |

Python names renamed in the same release keep their old spelling for one minor release too, with a `DeprecationWarning`:

| Old | New |
| --- | --- |
| `IsolatedResult` | `TtsResult` |
| `IsolatedResult.ttfa_ms`, `.llm_ttfs_ms` | `TtsResult.tts_first_audio_ms`, `.tts_first_text_ms` |
| `IsolatedFishTts.speak_stream_isolated` | `IsolatedFishTts.speak_deltas_isolated` |
| `DuplexSession.stop` | `DuplexSession.quit_requested` |
| `LlmTune`, `.key`, `.openrouter` | `LlmSettings`, `.api_key`, `.uses_openrouter_sdk` |
| `ListenTune.silence_frames_end`, `.speech_frames_start`, `.min_voiced` | `.end_silence_frames`, `.start_speech_frames`, `.min_voiced_frames` |
| `BargeTune.over` | `BargeTune.playing_gain` |
| `AecTune.bleed_s` | `AecTune.bleed_delay_s` |

## License

[MIT](../../LICENSE)
