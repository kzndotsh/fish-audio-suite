# fish-audio-suite-voice

One Fish Audio websocket per turn, playback sinks, and an optional duplex CLI.

Import `IsolatedFishTts` when another app already owns the microphone and the LLM. Use `fish-voice` when you want mic, Fish ASR, an OpenRouter reply, and speakers in one process.

Part of [fish-audio-suite](../../README.md). Unofficial.

## Quick start

```python
from pathlib import Path
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
result = tts.speak_isolated("[clear] Hello there.", FileSink(Path("turn.wav")))
```

`speak_isolated` runs the websocket on a private thread and event loop, so it is safe under `asyncio.run` or `to_thread`. Retry of 429 and 5xx happens only before the first audio byte. There is no default `voice_id`.

```bash
cp .env.example .env
./packages/voice/dev.sh --smoke    # writes /tmp/fish-audio-suite-smoke.wav
./packages/voice/dev.sh            # duplex
./packages/voice/dev.sh --debug    # or FISH_VOICE_DEBUG=1
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

`--debug` logs VAD, barge-in, Fish websocket metadata, and OpenRouter metadata on stderr. What was said stays on stdout. Duplex exits 2 when playback is `file`, unknown, or `mpv` is not on `PATH`. It also exits 2 on ASR or TTS 401, 402, or 403.

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

## Extras

| Extra | Pulls in |
| --- | --- |
| `speakers` | `sounddevice` |
| `vad` | `webrtcvad-wheels` |
| `aec` | `pywebrtc-audio` |
| `cli` | the three above, plus `openrouter` |

`import fish_audio_suite_voice` works without them. The heavy imports happen inside the functions that need them.

## Settings

Required:

| Variable | Default |
| --- | --- |
| `FISH_API_KEY` | none |
| `FISH_VOICE_ID` | none |
| `FISH_LLM_KEY` or `OPENROUTER_API_KEY` | none. Duplex only |
| `FISH_LLM_MODEL` or `OPENROUTER_MODEL` | none. Duplex only |

Speech:

| Variable | Default |
| --- | --- |
| `FISH_BASE` | `https://api.fish.audio` |
| `FISH_TTS_MODEL` | `s2.1-pro` |
| `FISH_LATENCY` | `normal` |
| `FISH_SPEED` | `1` |
| `FISH_CHUNK_LENGTH` | `200` (cloud max 300) |
| `FISH_ASR_LANGUAGE` | omitted |
| `FISH_PLAYBACK` | `sounddevice` |

Listen and interrupt. Times are approximate at 16 kHz frames.

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_VOICE_SPEECH_FRAMES` | `4` | About 120 ms before speech counts |
| `FISH_VOICE_MIN_RMS` | `200` | Noise floor. It can rise in a loud room |
| `FISH_VOICE_MIN_VOICED` | `12` | About 360 ms of voice. Drops a cough |
| `FISH_VOICE_SILENCE_FRAMES` | `40` | About 1.2 s of quiet ends the turn |
| `FISH_VOICE_VAD` | `1` | WebRTC VAD mode, 0–3. Higher is pickier |
| `FISH_VOICE_COOLDOWN` | `0.8` | Seconds after playback |
| `FISH_VOICE_BLEED_DELAY` | `0.9` | Used when AEC is off or missing |
| `FISH_VOICE_AEC` | on | `0` disables in-process echo cancellation |
| `FISH_VOICE_AEC_WET` | `0.85` | Mix of cleaned mic into the raw mic |
| `FISH_VOICE_AEC_BLEED` | `0.3` | Bleed delay once AEC3 is loaded |
| `FISH_VOICE_BARGE_RMS` | `220` | Barge floor. Raised while the speaker plays only if AEC is off |
| `FISH_VOICE_DEBUG` | off | `--debug` or `1` |

Commented copies live in [`.env.example`](../../.env.example).

## License

[MIT](../../LICENSE)
