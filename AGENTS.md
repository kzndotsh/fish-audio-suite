# AGENTS.md — fish-audio-suite

Unofficial Fish Audio toolkit. Not affiliated with Fish Audio. Dist names stay `fish-audio-suite-{kit,proxy,voice}` only.

Three members under `packages/`. Usage: [`README.md`](README.md).

## Quick reference

| Task | Command |
| --- | --- |
| Sync | `uv sync --all-packages --extra cli --group dev` |
| Test | `uv run pytest` |
| Lint | `uv run ruff check packages` · `uv run ruff format packages` |
| Types | `uv run basedpyright` |
| CI | GitHub Actions `.github/workflows/ci.yml` (ruff, basedpyright, pytest) |
| Wheels | `uv build --all` |
| Proxy | `uv run --package fish-audio-suite-proxy fish-audio-suite-proxy` |
| Voice smoke | `uv run --extra cli fish-voice --smoke` |
| Flake | `nix flake show` |

Python 3.12. **uv** only. Root is virtual (`package = false`).

Load [`.agents/skills/finalize`](.agents/skills/finalize) by name (`/finalize`) when wrapping up a chunk. Do not rely on auto-selection.

## Sub-AGENTS

Read the nested file before editing that tree.

| Tree | Dist / import |
| --- | --- |
| [`nix`](nix/AGENTS.md) | NixOS module |
| [`packages`](packages/AGENTS.md) | Workspace members |
| [`packages/kit`](packages/kit/AGENTS.md) | `fish-audio-suite-kit` / `fish_audio_suite_kit` |
| [`packages/proxy`](packages/proxy/AGENTS.md) | `fish-audio-suite-proxy` / `fish_audio_suite_proxy` |
| [`packages/voice`](packages/voice/AGENTS.md) | `fish-audio-suite-voice` / `fish_audio_suite_voice` |

## Boundaries

Proven from this tree (kit is the only text package; proxy import must work without a key; live.py isolates Fish WS; `FISH_VOICE_ID` has no default).

| Do | Don’t |
| --- | --- |
| Shared cue/scrub/cut/W3C parse/Fish error shape/captions in kit | Copy those regexes into proxy or voice |
| Read `FISH_API_KEY` in lifespan / CLI / `IsolatedFishTts(...)` | `os.environ["FISH_API_KEY"]` at import |
| Empty `FISH_VOICE_ID` unless env sets it | Default voice id |
| One `stream_websocket` per turn; one `FlushEvent` after sent text; TTS via `asyncio.run` | Per-sentence flush; Fish WS on the LLM event loop |
| Barge-in history = `spoken_so_far`, or omit if no audio | Full unplayed LLM reply |
| Three dists only; CLI stays in voice; W3C parse in kit with no OTel | Fourth dist, OpenTelemetry SDK, or a VAD package |

## Gotchas

- Do not `aclose()` the fishaudio websocket iterator. Stop iterating; close the **client**. Empty turn + bare `FlushEvent` is invalid.
- pytest: `--import-mode=importlib` (several `tests/` dirs).
- `nixosModules.default`: `127.0.0.1:8849:8849`, `autoStart = false`. Voice derivation wraps PortAudio on `LD_LIBRARY_PATH`.
