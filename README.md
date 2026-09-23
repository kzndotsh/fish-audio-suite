<div align="center">
    <p>
        <a href="https://github.com/kzndotsh/fish-audio-suite/actions/workflows/ci.yml">
            <img alt="CI" src="https://github.com/kzndotsh/fish-audio-suite/actions/workflows/ci.yml/badge.svg"></a>
        <a href="https://www.python.org/downloads/">
            <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white"></a>
        <a href="https://docs.astral.sh/uv/">
            <img alt="uv" src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?logo=uv&logoColor=white"></a>
        <a href="LICENSE">
            <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
    </p>
    <h1>fish-audio-suite</h1>
    <p><strong>Text, HTTP, and live speech for Fish Audio.</strong></p>
</div>

> [!NOTE]
> Unofficial. Not a Fish Audio product.

---

## Quick start

| You want | Start here |
| --- | --- |
| Open WebUI, AIRI, or any OpenAI audio client | [Proxy](#proxy) |
| One live TTS turn from Python | [Voice](#voice) |
| Cue tags and sentence cuts in your own client | [Kit](#kit) |

```bash
git clone https://github.com/kzndotsh/fish-audio-suite
cd fish-audio-suite
uv sync --all-packages --extra cli
export FISH_API_KEY=...
```

## How it works

```
your app ──► kit ──► text you send to Fish
OpenAI client ──► proxy :8849 ──► Fish HTTP
your app ──► voice ──► Fish websocket ──► speakers
mic ──► fish-voice ──► ASR ──► LLM ──► one websocket turn
```

## Proxy

Binds `0.0.0.0:8849`. `/health` works with no API key. Speech and transcription return 401 until `FISH_API_KEY` is set.

```bash
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
curl -s http://127.0.0.1:8849/health
```

Point the client at `http://127.0.0.1:8849/v1`. The key in the client can be any non-empty string. The Fish key stays on the proxy. `tts-1` and `whisper-1` are mapped onto Fish models.

Docker, Open WebUI, and the field map: [packages/proxy/README.md](packages/proxy/README.md).

## Voice

```python
from pathlib import Path
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
result = tts.speak_isolated("[clear] Hello there.", FileSink(Path("turn.wav")))
```

`speak_isolated` runs the websocket on a private thread, so it is safe under `asyncio.run`. Speakers and the microphone need PortAudio. `uv` does not install it. `--smoke` writes a WAV and skips the device.

```bash
cp .env.example .env
./packages/voice/dev.sh --smoke
```

Sinks, duplex, and listen settings: [packages/voice/README.md](packages/voice/README.md).

## Kit

```python
from fish_audio_suite_kit import normalize_cues, scrub_tts, next_tts_cut

spoken = normalize_cues(scrub_tts(llm_text))
cut = next_tts_cut(spoken)  # sentence end, or about 40 characters; -1 keeps buffering
```

Exports: [packages/kit/README.md](packages/kit/README.md).

## Tech stack

| Component | Technology |
| --- | --- |
| **Packages** | `uv` workspace, three wheels |
| **Kit** | Pure text. No dependencies |
| **Proxy** | FastAPI, uvicorn, httpx |
| **Voice** | `fish-audio-sdk`, loguru, optional PortAudio / OpenRouter |
| **Types** | basedpyright, strict |
| **Lint** | Ruff, NumPy docstrings via pydoclint |
| **Tests** | pytest, branch coverage in CI |

## Project structure

```
├── packages/
│   ├── kit/          # cues, scrubbers, cuts, Fish error shape
│   ├── proxy/        # OpenAI audio HTTP on :8849
│   └── voice/        # live websocket, sinks, fish-voice CLI
├── nix/              # NixOS module
├── Dockerfile        # proxy image
├── flake.nix
├── pyproject.toml    # workspace root, not a fourth package
└── .env.example      # duplex CLI
```

## Commands

```bash
uv sync --all-packages --extra cli --group dev --group test
uv run ruff format packages && uv run ruff check packages
uv run pydoclint --config=pyproject.toml packages
uv run basedpyright
uv run pytest
```

## Settings

| Variable | Used by | Default |
| --- | --- | --- |
| `FISH_API_KEY` | proxy, voice | none |
| `FISH_VOICE_ID` | voice | none |
| `FISH_BASE` | proxy, voice | `https://api.fish.audio` |
| `FISH_MODEL` | proxy | `s2.1-pro` |
| `FISH_TTS_MODEL` | voice | `s2.1-pro` |
| `FISH_PROXY_HOST` / `FISH_PROXY_PORT` | proxy | `0.0.0.0` / `8849` |

Self-hosted [fish-speech](https://github.com/fishaudio/fish-speech) is `FISH_BASE=http://127.0.0.1:8080`. Cloud `chunk_length` stays in 100–300. A self-hosted base allows up to 1000.

Full tables: [proxy](packages/proxy/README.md#settings), [voice](packages/voice/README.md#settings).

## Nix

```nix
inputs.fish-audio-suite.url = "github:kzndotsh/fish-audio-suite";
```

`nixosModules.default` runs the proxy container on `127.0.0.1:8849:8849` with `autoStart = false`. Put `FISH_API_KEY` in `environmentFiles`.

`nix run .#fish-audio-suite-voice` puts PortAudio on `LD_LIBRARY_PATH`. On NixOS, `./packages/voice/dev.sh` does the same. A bare `uv run` of the duplex CLI does not.

## License

[MIT](LICENSE)

Created by [@kzndotsh](https://github.com/kzndotsh)
