# Proxy (`fish-audio-suite-proxy`)

> Scope: `packages/proxy` (inherits root [AGENTS.md](../../AGENTS.md))

OpenAI-shaped Fish HTTP on **8849**: `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`, `/health`.

CLI: `fish-audio-suite-proxy`. Import: `fish_audio_suite_proxy`.

| Task | Command |
| --- | --- |
| Run | `uv run --package fish-audio-suite-proxy fish-audio-suite-proxy` |
| Tests | `uv run pytest packages/proxy` |
| Image | `docker build -t fish-audio-suite-proxy:latest .` |

Scrub through kit. `FISH_API_KEY` is read in lifespan — importing the app and `GET /health` must work with it unset.

In `server.py`: remap `drama-3-preview` / `s2.1-pro-free` to the default TTS model, `whisper-1` to `transcribe-1`. `is_tts_junk` → silent MP3, not a Fish round-trip.
