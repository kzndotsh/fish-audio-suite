# AGENTS.md — fish-audio-suite

Unofficial Fish Audio toolkit. Not affiliated with Fish Audio. Dist names stay `fish-audio-suite-{kit,proxy,voice}` only.

Three members under `packages/`. Usage: [`README.md`](README.md).

## Quick reference

| Task | Command |
| --- | --- |
| Sync | `uv sync --all-packages --extra cli --group dev --group test` |
| Test | `uv run pytest` |
| Lint | `uv run ruff check packages` · `uv run ruff format packages` |
| Docstrings | `uv run pydoclint --config=pyproject.toml packages` |
| Types | `uv run basedpyright` |
| CI | `.github/workflows/ci.yml`: Python 3.12/3.13 (ruff, pydoclint, basedpyright, pytest), wheel build + `twine check`, `nix flake check`, Docker build + `/health`. Coverage is CI-only: `--cov` with `--cov-fail-under=71`; `[tool.coverage]` holds branch and sources |
| Supply chain | `dependency-submission.yml` on push to `main` when `uv.lock` changes. `release.yml` publishes wheels on a `v*` tag only when repo var `PYPI_PUBLISH=true` (PyPI trusted publishing). `dependabot.yml` covers actions and uv |
| Wheels | `uv build --all` |
| Proxy | `uv run --package fish-audio-suite-proxy fish-audio-suite-proxy` |
| Voice smoke | `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke` · local: `./packages/voice/dev.sh --smoke` |
| Flake | `nix flake check` · `nix flake show` |

Python 3.12+. **uv** only. Root is virtual (`package = false`). Proxy and voice pin `fish-audio-suite-kit>=0.1,<0.2`; the workspace source overrides it in dev.

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

| Do | Don’t |
| --- | --- |
| Shared cue/scrub/cut/W3C parse/Fish error shape/captions in kit | Copy those regexes into proxy or voice |
| Read `FISH_API_KEY` in lifespan / CLI / `IsolatedFishTts(...)` | `os.environ["FISH_API_KEY"]` at import |
| Empty `FISH_VOICE_ID` unless env sets it | Default voice id |
| One `stream_websocket` per turn; one `FlushEvent` after sent text (streamed turns add one after the first piece); TTS on a private loop (`speak_isolated` / `to_thread`) | Per-sentence flush; Fish WS on the LLM event loop |
| Barge-in history = `spoken_so_far`, or omit if no audio | Full unplayed LLM reply |
| Three dists only; CLI stays in voice; W3C parse in kit with no OTel | Fourth dist, OpenTelemetry SDK, or a VAD package |
| NumPy docstrings on public modules, classes, and functions, in the same change as the signature. First line is imperative. Parameters, Returns, Yields, and Raises match | Docstrings on tests. pydoclint skips one-line summaries and `**/tests/**` |

## Gotchas

- Do not `aclose()` the fishaudio websocket iterator. Stop iterating; close the **client**. Empty turn + bare `FlushEvent` is invalid.
- Ruff `D` runs inside `ruff check`. pydoclint is a separate CI step. `/finalize` updates dirty docstrings and reruns both.
- pytest: `--disable-socket` with `--allow-unix-socket` (asyncio's self-pipe is AF_UNIX). Keep `--cov` out of default addopts. `fail-under` compares the precise percent, so a 71.4 report fails `--cov-fail-under=72`.
- `nixosModules.default` runs the proxy as a `native` systemd service on `127.0.0.1:8849` (`oci` backend optional). Details in [`nix/AGENTS.md`](nix/AGENTS.md).
