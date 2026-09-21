# fish-audio-suite-voice

Import `IsolatedFishTts`, `PlaybackSink`, and `BargeGate`. The `fish-voice` CLI is a duplex talk-back recipe.

Local: copy `.env.example` to `.env` and run `./packages/voice/dev.sh`. Debug: `./packages/voice/dev.sh --debug` or `FISH_VOICE_DEBUG=1`. NixOS duplex needs that script or `nix run .#fish-audio-suite-voice` (PortAudio is not on `uv`'s `LD_LIBRARY_PATH`). Speaker-bleed AEC is in extra `aec` (also on `cli`); `FISH_VOICE_AEC=0` disables it. PipeWire `echo-cancel` is optional on the host, not a Python dep.

Unofficial. Not affiliated with Fish Audio.
