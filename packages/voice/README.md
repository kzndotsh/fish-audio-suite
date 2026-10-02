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

The `cli` extra includes AEC3. Speaker audio is subtracted from the mic before voice activity detection. `FISH_VOICE_AEC=0` leaves the mic unchanged and keeps the longer bleed delay (0.9 s). With AEC loaded, that delay is 0.3 s. PipeWire `echo-cancel` is a host setup, not a Python dependency.

A barge-in keeps the audio that tripped it and clears the speaker tap. The next listen starts from that clip.

The barge floor follows the room: it is the quiet-percentile of recent mic frames, never below `FISH_VOICE_BARGE_RMS`. If barge-in fires on a cough or a chair, raise `FISH_VOICE_BARGE_FRAMES` (30 ms each) before raising the RMS. When the mic opens after the bleed delay, the far-end reference is trimmed to the audio about to be heard, using the output latency PortAudio reports.

### History after a barge-in

The chat history gets only what you probably heard. For PCM that is a word-aligned estimate from the bytes played, the speed, and the device buffer. Encoded playback (`mpv`) cannot be cut by length, so an interrupted or failed reply is left out of history.

### Cues in replies

A model copies the pattern of its own earlier replies. Without help a conversation settles on one
`[cue]` per reply, whatever the prompt says. With the default system prompt the session therefore
starts with one pinned exchange that shows several cues (`DEFAULT_SEED_EXCHANGE` in the kit). It
costs about 40 tokens per request and is never trimmed from the history. In a test on two models
that took the average from 1.0 to about 2 cues per reply. A custom `FISH_SYSTEM_PROMPT` gets no
seed, so it stays in control of the replies.

### LLM backends

`FISH_LLM_BACKEND` picks `openai` (any chat-completions server over one pooled httpx client) or `openrouter` (the SDK). Unset, it follows the host of `FISH_LLM_BASE`. A key never crosses providers: `OPENROUTER_API_KEY` is only a fallback for the OpenRouter backend and `OPENAI_API_KEY` only for the other. OpenRouter-only options: `FISH_LLM_NITRO=1` adds `:nitro` to the model and sorts providers by `FISH_LLM_PROVIDER_SORT`, and `FISH_LLM_REFERER`, `FISH_LLM_TITLE`, `FISH_LLM_CATEGORIES` set the attribution (empty disables one). The OpenAI backend sends none of those headers.

### Streaming the reply

By default the reply is spoken after the model finishes. Set `FISH_STREAM_TTS=1` to speak
while the model is still writing. The Fish socket opens with the request and the first
sentence is flushed at once, so audio starts about when that sentence is done instead of when
the whole reply is. The socket then gets one more flush at the end.

- One extra flush is needed: Fish holds text until a chunk fills or a flush arrives, so a single
  flush at the end keeps a short reply silent until the model is done.
- The reply is scrubbed as it arrives, and the whole-reply junk check is skipped.
- The barge-in gate arms at the first audio chunk.
- If Fish fails before any audio, the finished reply is spoken on the normal path.
- The `first_audio` timing is measured from the start of ASR, in both modes, so you can compare them.

### Roleplay helpers

Both are off by default because they also change ordinary English. `FISH_MOOD_LEAD=1` rewrites a sentence-leading mood word (`Excited, hello`) into a `[cue]`. `FISH_DROP_NARRATION=1` drops lines that are only stage directions (`She smiles.`).

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
| `FISH_LLM_KEY` (fallback `OPENROUTER_API_KEY` or `OPENAI_API_KEY`, see above) | none. Duplex only |
| `FISH_LLM_MODEL` (fallback `OPENROUTER_MODEL`) | none. Duplex only |

Speech:

| Variable | Default |
| --- | --- |
| `FISH_BASE` | `https://api.fish.audio` |
| `FISH_TTS_MODEL` | `s2.1-pro` |
| `FISH_LATENCY` | `normal` (`balanced` is the other choice for this SDK) |
| `FISH_SPEED` | `1` |
| `FISH_VOLUME` | `0` (dB) |
| `FISH_TEMPERATURE`, `FISH_TOP_P` | `0.7`, `0.7` |
| `FISH_REPETITION_PENALTY` | `1.2` |
| `FISH_CHUNK_LENGTH` | `200` (cloud max 300) |
| `FISH_MIN_CHUNK_LENGTH` | `50` |
| `FISH_SAMPLE_RATE` | `44100` |
| `FISH_ASR_LANGUAGE` | omitted |
| `FISH_ASR_MODEL` | `transcribe-1` (or `transcribe-1-pro`) |
| `FISH_PLAYBACK` | `sounddevice` |
| `FISH_VOICE_DEVICE` | host default. A PortAudio index or name |

LLM:

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_LLM_BACKEND` | from the base URL | `openai` or `openrouter` |
| `FISH_LLM_BASE` (fallback `OPENROUTER_BASE_URL`) | `https://openrouter.ai/api/v1` | API origin |
| `FISH_LLM_TEMPERATURE` | `0.8` | 0 to 2 |
| `FISH_LLM_TIMEOUT` | `120` | Seconds per request |
| `FISH_LLM_MAX_TOKENS` | `1200` | Completion cap |
| `FISH_LLM_NITRO` | off | OpenRouter only. `:nitro` plus provider sort |
| `FISH_LLM_PROVIDER_SORT` | `throughput` | Used with `FISH_LLM_NITRO` |
| `FISH_LLM_REFERER`, `FISH_LLM_TITLE`, `FISH_LLM_CATEGORIES` | this project's | OpenRouter attribution. Empty disables |
| `FISH_LLM_CONTINUE` | off | Send one more request when a reply stops mid-sentence |
| `FISH_SYSTEM_PROMPT` | the kit default | System prompt text |
| `FISH_HISTORY_TURNS` | `20` | User and assistant pairs kept |
| `FISH_STREAM_TTS` | off | Speak the reply while the model is still writing it. See streaming below |
| `FISH_VOICE_REPEAT_WINDOW_S` | `1.5` | A line equal to the previous one is dropped only if it ends this soon after the mic opens. 0 never drops a repeat |
| `FISH_MOOD_LEAD`, `FISH_DROP_NARRATION` | off | See roleplay helpers |

Listen and interrupt. Times are approximate at 30 ms frames.

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_VOICE_SPEECH_FRAMES` | `4` | About 120 ms before speech counts |
| `FISH_VOICE_PRE_PAD` | `20` | Frames kept before the start. At least speech frames + 6 |
| `FISH_VOICE_MIN_RMS` | `200` | Listen floor seed. It can rise in a loud room |
| `FISH_VOICE_MIN_VOICED` | `12` | About 360 ms of voice. Drops a cough |
| `FISH_VOICE_SILENCE_FRAMES` | `40` | About 1.2 s of quiet ends the turn |
| `FISH_VOICE_VAD` | `1` | WebRTC VAD mode, 0–3. Higher is pickier |
| `FISH_VOICE_COOLDOWN` | `0.8` | Seconds after playback |
| `FISH_VOICE_BLEED_DELAY` | `0.9` | Used when AEC is off or missing |
| `FISH_VOICE_AEC` | on | `0` disables in-process echo cancellation |
| `FISH_VOICE_AEC_WET` | `0.85` | Mix of cleaned mic into the raw mic, 0 to 1 |
| `FISH_VOICE_AEC_BLEED` | `0.3` | Bleed delay once AEC3 is loaded |
| `FISH_VOICE_BARGE_FRAMES` | `10` | Loud voiced frames in a row that interrupt (about 300 ms) |
| `FISH_VOICE_BARGE_RMS` | `220` | Barge floor seed. It follows the room, never below this |
| `FISH_VOICE_BARGE_OVER` | `2.2` | Floor multiplier while the speaker plays and AEC is off. At least 1 |
| `FISH_VOICE_DEBUG` | off | `1` or `--debug` logs events. `2`, `trace` or `--trace` adds mic heartbeats, raw audio events and HTTP lines |

`dev.sh` also reads `FISH_VOICE_ENV` (env file path), `FISH_VOICE_PORTAUDIO_LIB` (library directories), and `FISH_VOICE_NIX=1` (build PortAudio with nix on a host that is not NixOS).

Commented copies live in [`.env.example`](../../.env.example).

## License

[MIT](../../LICENSE)
