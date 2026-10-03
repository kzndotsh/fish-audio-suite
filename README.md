<div align="center">
    <p>
        <a href="https://github.com/kzndotsh/fish-audio-suite/actions/workflows/ci.yml">
            <img alt="CI" src="https://github.com/kzndotsh/fish-audio-suite/actions/workflows/ci.yml/badge.svg"></a>
        <a href="https://www.python.org/downloads/">
            <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white"></a>
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
| Open WebUI or any OpenAI audio client | [Proxy](#proxy) |
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

Binds `127.0.0.1:8849` by default (`FISH_PROXY_HOST` changes it). `/health` works with no API key. Speech and transcription return 503 until `FISH_API_KEY` is set.

```bash
uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
curl -s http://127.0.0.1:8849/health
```

Point the client at `http://127.0.0.1:8849/v1`. The Fish key stays on the proxy. Set `FISH_PROXY_API_KEYS` to require a bearer key from clients; without it any client key is accepted, so keep the proxy on loopback. The Docker image listens on every interface inside the container, so publish it on loopback (`-p 127.0.0.1:8849:8849`) or set `FISH_PROXY_API_KEYS` before you publish it elsewhere. On stop the proxy lets in-flight replies finish for up to `FISH_PROXY_GRACEFUL_SHUTDOWN` seconds (default 120), but Docker kills a container after 10 seconds, so run it with `--stop-timeout 130` (compose: `stop_grace_period: 130s`). `tts-1` and `whisper-1` are mapped onto Fish models.

Docker, Open WebUI, and the field map: [packages/proxy/README.md](packages/proxy/README.md).

## Voice

```python
from pathlib import Path
from fish_audio_suite_voice import IsolatedFishTts, FileSink

tts = IsolatedFishTts(api_key=key, voice_id=voice_id)
result = tts.speak_isolated("Hello there.", FileSink(Path("turn.wav")))
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

## Versioning and the public API

The public API is the set of names in each package's root `__all__`, plus the proxy's HTTP API and
its documented settings. Everything else is internal. While the version is 0.x, a minor release
(0.1 to 0.2) may break the public API and a patch release never does. A name is deprecated, with a
`DeprecationWarning` that names its replacement, for at least one minor release before it is
removed. The GitHub release notes list every change, breaking ones marked with `!` in the pull
request title; there is no `CHANGELOG.md`. Details: [CONTRIBUTING.md](CONTRIBUTING.md).

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
├── Dockerfile        # proxy image (non-root)
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
| `FISH_TTS_MODEL` | proxy, voice | `s2.1-pro` |
| `FISH_ASR_MODEL` | proxy, voice | `transcribe-1` |
| `FISH_LATENCY` | proxy, voice | `normal` |
| `FISH_PROXY_HOST` / `FISH_PROXY_PORT` | proxy | `127.0.0.1` / `8849` |
| `FISH_PROXY_API_KEYS` | proxy | none (any client key accepted) |
| `FISH_LLM_BASE` / `FISH_LLM_KEY` / `FISH_LLM_MODEL` | voice | OpenRouter / none / none |

Self-hosted [fish-speech](https://github.com/fishaudio/fish-speech) is `FISH_BASE=http://127.0.0.1:8080`. Cloud `chunk_length` stays in 100–300. A self-hosted base allows up to 1000.

Full tables: [proxy](packages/proxy/README.md#settings), [voice](packages/voice/README.md#settings).

## Nix

```nix
inputs.fish-audio-suite.url = "github:kzndotsh/fish-audio-suite";
```

`nixosModules.default` runs the proxy as a hardened systemd service on `127.0.0.1:8849` (`services.fish-audio-suite-proxy.enable = true`). Put `FISH_API_KEY` in `environmentFiles`. Options for `host`, `port`, `openFirewall`, `autoStart`, `gracefulShutdownSeconds` (the service waits that plus 10 seconds to stop), and an `oci` backend are in [nix/module.nix](nix/module.nix).

`nix run .#fish-audio-suite-voice` puts PortAudio on the library path. On NixOS, `./packages/voice/dev.sh` does the same. A bare `uv run` of the duplex CLI does not.

## License

[MIT](LICENSE)

Created by [@kzndotsh](https://github.com/kzndotsh)
