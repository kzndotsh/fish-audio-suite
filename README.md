# fish-audio-suite

Unofficial community toolkit for apps that need [Fish Audio](https://fish.audio) TTS, ASR, and live speech. **Not affiliated with Fish Audio.**

This is a suite of tools, not a single duplex agent. Pick a recipe:

| Need | Use |
| --- | --- |
| Cue/scrub/cut in your own HTTP or SDK calls | **kit** (`fish-audio-suite-kit`) |
| OpenAI-compat `/v1/audio/speech` and `/v1/audio/transcriptions` (AIRI, OpenWebUI, …) | **proxy** on port **8849** |
| Stream text into one Fish websocket turn, cancel cleanly | **voice** `IsolatedFishTts` |
| Hear it locally | `PlaybackSink`: `sounddevice` (default PCM), `file`, `stdout`, optional `mpv` |
| Full talk-back (mic VAD, barge, LLM, history) | Duplex **CLI recipe** (`fish-voice`). Not required to use the suite. |

Do not vendor Pipecat, LiveKit Agents, or Gemini Live. Stacks that already have those import **kit** (and optionally `IsolatedFishTts`) into *their* pipeline.

## Packages

- `fish-audio-suite-kit` — pure text: `normalize_cues`, `scrub_tts` / `is_tts_junk`, `scrub_asr` / `is_asr_hallucination`, `is_backchannel`, `next_tts_cut`, `SuiteDefaults`, `LatencySnapshot`.
- `fish-audio-suite-proxy` — FastAPI OpenAI-compat HTTP. Reads `FISH_API_KEY` in lifespan, not at import.
- `fish-audio-suite-voice` — `IsolatedFishTts`, `PlaybackSink`, `BargeGate`, plus the `fish-audio-suite-voice` / `fish-voice` CLI.

Python 3.12. uv workspace. MIT. Author: Logan \<kzndotsh\>.

## Contrast

- Official Fish `/compat` — vendor OpenAI-shaped API. This proxy adds kit scrub, AIRI remaps, junk skip, and a local bind you control.
- LiveKit Fish plugin — rooms and WebRTC. This suite is a local toolkit.
- Pipecat `FishAudioTTSService` — frame bus for product bots. We do not depend on `pipecat-ai`.
- voicechat2 — local Whisper/llama/Coqui over WS. We are Fish cloud cascade plus AIRI proxy.
- CosyVoice / MOSS-TTS / SenseVoice / VibeVoice — local synthesizers or ASR. We stay on hosted Fish.
- Athena — local 397B/Orpheus (NCARD). Ideas only; no source copied.
- Rapida voice-ai — GPL telephony orchestration. Not this repo.
- voice-chat-ai — multi-vendor character UI.
- Vakya-AI — AssemblyAI/Gemini/Murf browser demo.
- Asterisk AI voice-agent — PBX AudioSocket pacing.
- ZeeMe / my-ai-companion — Gemini native Live + product UI.
- Hugging Face speech-to-speech / RealtimeTTS — swappable local cascades / engine zoos. We are Fish-hosted cascade plus proxy.

## Setup

```bash
uv sync --all-packages --extra cli --group dev
uv run pytest
```

### Environment

| Variable | Used by | Notes |
| --- | --- | --- |
| `FISH_API_KEY` | proxy, voice | Required for Fish HTTP/WS. No default. |
| `FISH_VOICE_ID` | voice | Required for live TTS / `--smoke`. **No default.** |
| `FISH_BASE` | proxy | Default `https://api.fish.audio` |
| `FISH_MODEL` / `FISH_TTS_MODEL` | proxy / voice | Default `s2.1-pro` |
| `FISH_LATENCY` | both | Default `normal` |
| `FISH_SPEED` / `FISH_SPEED_SCALE` | voice / proxy | Default `1.05` |
| `FISH_CHUNK_LENGTH` | both | Default `200` |
| `FISH_FORMAT` | proxy | Default `mp3` (voice live defaults to `pcm`) |
| `FISH_PLAYBACK` | voice CLI | `sounddevice` \| `file` \| `stdout` \| `mpv` |
| `FISH_LLM_KEY` / `OPENROUTER_API_KEY` | duplex CLI | LLM stream only |
| `FISH_SYSTEM_PROMPT` | duplex CLI | Generic spoken-assistant prompt with `[cue]` tags |

Do not commit `.env`. Do not put voice IDs in public defaults.

### Proxy

```bash
export FISH_API_KEY=…
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
# GET http://127.0.0.1:8849/health
```

Docker:

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run --rm -p 127.0.0.1:8849:8849 --env-file /path/to/ai.env fish-audio-suite-proxy:latest
```

### Voice library

```python
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
sink = FileSink(Path("turn.wav"))
result = tts.speak_isolated("[clear] Hello there.", sink)
```

### Duplex CLI recipe

```bash
export FISH_API_KEY=… FISH_VOICE_ID=… OPENROUTER_API_KEY=…
uv run --extra cli fish-voice --smoke   # file sink, generic hello
uv run --extra cli fish-voice           # mic talk-back
```

`--smoke` fails closed if `FISH_API_KEY` or `FISH_VOICE_ID` is missing.

## Nix / NixOS (document only until you apply)

Flake input:

```nix
inputs.fish-audio-suite.url = "git+file:///home/kaizen/Projects/fish-audio-suite";
# later: github:kzndotsh/fish-audio-suite
```

Module: `inputs.fish-audio-suite.nixosModules.default`.

Draft drop-in for `ai.voice.fish` (do not apply to `~/dotfiles` until asked):

```nix
fish-audio-suite-proxy = {
  image = "fish-audio-suite-proxy:latest";
  ports = [ "127.0.0.1:8849:8849" ];
  environmentFiles = [ "${config.my.secretsDir}/ai.env" ]; # FISH_API_KEY
  autoStart = false;
};
```

`nix run .#fish-audio-suite-voice` wraps PortAudio. mpv is optional.

## License

MIT. Logan \<kzndotsh\>.
