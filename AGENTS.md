# AGENTS.md

Unofficial Fish Audio toolkit. Not affiliated with Fish Audio. Dist names stay `fish-audio-suite-{kit,proxy,voice}` only.

## Layout

- `packages/fish-audio-suite-kit` — text only. No sockets.
- `packages/fish-audio-suite-proxy` — FastAPI on **8849**. `FISH_API_KEY` in lifespan.
- `packages/fish-audio-suite-voice` — `live.py` (`IsolatedFishTts`), `playback.py`, `barge.py`, `cli.py` (duplex recipe).

## Invariants

- No default `FISH_VOICE_ID`.
- One Fish `stream_websocket` per turn. Skip empty LLM deltas. One `FlushEvent` at end of turn (not per sentence).
- Isolated TTS on a fresh event loop. Do not merge the LLM token stream onto the Fish socket.
- Default live audio is PCM into `SounddeviceSink`. mpv is optional.
- Assistant history on barge-in is spoken-so-far or omitted. No interrupt markers.
- Ignore ASR while TTS is playing unless the barge gate fired. Skip backchannels and duplicate utterances.
- Latency logs never print utterances or keys.
- Do not vendor Pipecat / Athena / Rapida / CosyVoice / MOSS / SenseVoice / VibeVoice source.
- Do not publish or `gh repo create` unless asked.

## Checks

```bash
uv run pytest
uv build --all
nix flake show
```
