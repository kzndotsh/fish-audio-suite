# fish-audio-suite

Unofficial toolkit for [Fish Audio](https://fish.audio) TTS, ASR, and live speech. Not affiliated with Fish Audio.

Three packages. Use one, or mix them.


| Package | Dist                     | What it does                                                      |
| ------- | ------------------------ | ----------------------------------------------------------------- |
| kit     | `fish-audio-suite-kit`   | Cue tags, TTS/ASR scrubbing, sentence cuts. No network.           |
| proxy   | `fish-audio-suite-proxy` | OpenAI-compatible HTTP on port **8849**.                          |
| voice   | `fish-audio-suite-voice` | One Fish websocket per turn, playback sinks, optional duplex CLI. |


Python 3.12. MIT.

```bash
uv sync --all-packages --extra cli --group dev
```

Set `FISH_API_KEY` for anything that talks to Fish. Set `FISH_VOICE_ID` for live TTS. Neither has a default.

## kit

Import `fish_audio_suite_kit` in your own Fish HTTP or SDK calls.

```python
from fish_audio_suite_kit import normalize_cues, scrub_tts, next_tts_cut

spoken = normalize_cues(scrub_tts(llm_text))
cut = next_tts_cut(spoken)  # sentence end, or ~40 chars; -1 = keep buffering
```

Also: `is_tts_junk`, `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `SuiteDefaults`, `LatencySnapshot`.

## proxy

Drop-in for clients that speak OpenAI `/v1/audio/speech` and `/v1/audio/transcriptions` (AIRI, OpenWebUI, and similar).

```bash
export FISH_API_KEY=…
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
# GET http://127.0.0.1:8849/health
```

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run --rm -p 127.0.0.1:8849:8849 --env-file /path/to/env fish-audio-suite-proxy:latest
```

The env file must contain `FISH_API_KEY`. Pass a Fish reference id as `voice` / `reference_id` on the speech request.

## voice

Library first. The `fish-voice` CLI is a talk-back recipe (mic → ASR → LLM → TTS). You do not need the CLI to stream Fish audio.

```python
from pathlib import Path
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
sink = FileSink(Path("turn.wav"))
result = tts.speak_isolated("[clear] Hello there.", sink)
```

Playback sinks: `sounddevice` (default PCM), `file`, `stdout`, optional `mpv`.

```bash
export FISH_API_KEY=… FISH_VOICE_ID=… OPENROUTER_API_KEY=…
uv run --extra cli fish-voice --smoke   # writes a wav; no speakers
uv run --extra cli fish-voice           # mic duplex
```

`--smoke` exits 2 if the key or voice id is missing.

## Environment


| Variable                              | Who           | Default                         |
| ------------------------------------- | ------------- | ------------------------------- |
| `FISH_API_KEY`                        | proxy, voice  | none (required)                 |
| `FISH_VOICE_ID`                       | voice         | none (required for TTS)         |
| `FISH_BASE`                           | proxy         | `https://api.fish.audio`        |
| `FISH_MODEL` / `FISH_TTS_MODEL`       | proxy / voice | `s2.1-pro`                      |
| `FISH_LATENCY`                        | both          | `normal`                        |
| `FISH_SPEED` / `FISH_SPEED_SCALE`     | voice / proxy | `1.05`                          |
| `FISH_CHUNK_LENGTH`                   | both          | `200`                           |
| `FISH_FORMAT`                         | proxy         | `mp3` (voice live uses `pcm`)   |
| `FISH_PLAYBACK`                       | voice CLI     | `sounddevice`                   |
| `FISH_LLM_KEY` / `OPENROUTER_API_KEY` | duplex CLI    | none                            |
| `FISH_SYSTEM_PROMPT`                  | duplex CLI    | spoken assistant + `[cue]` tags |




## Nix

```nix
inputs.fish-audio-suite.url = "git+file:///home/kaizen/Projects/fish-audio-suite";
# later: github:kzndotsh/fish-audio-suite
```

- `nixosModules.default` — OCI container `127.0.0.1:8849:8849`, `autoStart = false`, `environmentFiles` for `FISH_API_KEY`.
- `nix run .#fish-audio-suite-voice` — PortAudio wrap. mpv optional.

## License

MIT. Logan \<kzndotsh\>.
