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
| Types | `uv run basedpyright` · public API completeness: `uv run python .github/scripts/verifytypes.py` (`just types-public`), floors in `.github/verifytypes-floors.json` |
| Doctests | `uv run pytest` runs the `>>>` examples in kit docstrings (`--doctest-modules`; the root `conftest.py` keeps proxy and voice sources out). A stale example fails the build |
| Gates | `just check` runs lint, docstrings, types and tests with the CI coverage floors. `just --list` shows the rest. `uvx pre-commit install` runs the fast checks per commit |
| CI | `.github/workflows/ci.yml`: Python 3.12/3.13/3.14, each running the shared steps in `.github/actions/gates` (ruff, pydoclint, basedpyright, verifytypes floors, pytest with kit doctests, coverage floors; `release.yml` runs the same action before it publishes), a job on the lowest declared dependency versions (`uv sync --resolution lowest-direct`), zizmor workflow lint, wheel build + `twine check`, `nix flake check`, Docker build + `/health`. Coverage is CI-only: `--cov` with `--cov-fail-under=89`, then per-package floors (kit 94, proxy 94, voice 85) from `coverage report --include`; `[tool.coverage]` holds branch and sources. Keep the `justfile` and `.github/actions/gates/action.yml` floors equal |
| Supply chain | `dependency-submission.yml` on push to `main` when `uv.lock` changes. `audit.yml` runs pip-audit on the exported lock for PRs that touch it and weekly. `release.yml` publishes wheels on a `v*` tag only when repo var `PYPI_PUBLISH=true` (PyPI trusted publishing) and uploads a CycloneDX SBOM artifact. `dependabot.yml` covers actions, uv (runtime and dev groups), docker and nix, with a 7-day cooldown |
| Wheels | `uv build --all` |
| Proxy | `uv run --package fish-audio-suite-proxy fish-audio-suite-proxy` |
| Voice smoke | `uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke` · local: `./packages/voice/dev.sh --smoke` |
| Flake | `nix flake check` · `nix flake show` |

Public API: the names in each package root `__all__` (the proxy's also its HTTP API and documented settings); the rest is internal. **Until 1.0.0 nothing is released** (no tags, no PyPI), so the versions stay at 0.1.0 and there is no changelog. Rename and remove public names freely, and do NOT add deprecation shims, aliases, `DeprecationWarning`s, or renamed-env-var fallbacks. The deprecation policy (one minor release of `DeprecationWarning`, SemVer breaking-change rules) applies only from 1.0.0, and from then GitHub's generated release notes are the changelog (`gh release create vX.Y.Z --verify-tag --generate-notes`). Details in [CONTRIBUTING.md](CONTRIBUTING.md).

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
| One `stream_websocket` per turn; one `FlushEvent` after sent text (streamed turns add one at the end of the first sentence); TTS on a private loop (`speak_isolated` / `to_thread`) | Per-sentence flush; Fish WS on the LLM event loop |
| Barge-in history = `spoken_so_far`, or omit if no audio | Full unplayed LLM reply |
| Three dists only; CLI stays in voice; W3C parse in kit with no OTel | Fourth dist, OpenTelemetry SDK, or a VAD package |
| NumPy docstrings on public modules, classes, and functions, in the same change as the signature. First line is imperative. Parameters, Returns, Yields, and Raises match | Docstrings on tests. pydoclint skips one-line summaries and `**/tests/**` |

## Env var names

| Prefix | Means | Examples |
| --- | --- | --- |
| `FISH_*` | A Fish request default: `FISH_` plus the uppercased Fish wire field | `FISH_LATENCY`, `FISH_CHUNK_LENGTH`, `FISH_SPEED` (`prosody.speed`) |
| `FISH_TTS_*`, `FISH_ASR_*` | An endpoint default where the bare wire name (`model`, `format`) would be ambiguous, and text shaping before TTS | `FISH_TTS_MODEL`, `FISH_TTS_FORMAT`, `FISH_ASR_MODEL`, `FISH_TTS_MOOD_LEAD` |
| `FISH_PROXY_*` | Proxy server behavior | `FISH_PROXY_PORT`, `FISH_PROXY_TTS_ALIASES` |
| `FISH_VOICE_*` | Voice app behavior | `FISH_VOICE_PLAYBACK`, `FISH_VOICE_BARGE_FRAMES` |
| `FISH_LLM_*` | The voice app's chat model | `FISH_LLM_API_KEY`, `FISH_LLM_MODEL` |

`FISH_VOICE_ID` is the one exception: it is the Fish voice and maps to the wire field `reference_id`. Seconds are implied in env names (`FISH_VOICE_COOLDOWN`, `FISH_PROXY_READ_TIMEOUT`); other units are suffixed (`_FRAMES`, `_BYTES`, `_CHARS`). Python fields always carry the unit (`cooldown_s`). `FISH_SPEED` is the default speed; the proxy multiplies a client-sent `speed` by it. Renaming a variable is just a rename: change the code, `.env.example`, the READMEs and `tests/test_docs_env.py` together, and keep no old name.

## Gotchas

- Do not `aclose()` the fishaudio websocket iterator. Stop iterating; close the **client**. Empty turn + bare `FlushEvent` is invalid.
- Ruff `D` runs inside `ruff check`. pydoclint is a separate CI step. `/finalize` updates dirty docstrings and reruns both.
- pytest: `--disable-socket` with `--allow-unix-socket` (asyncio's self-pipe is AF_UNIX). Keep `--cov` out of default addopts. `fail-under` compares the precise percent, so a 71.4 report fails `--cov-fail-under=72`.
- `nixosModules.default` runs the proxy as a `native` systemd service on `127.0.0.1:8849` (`oci` backend optional). Details in [`nix/AGENTS.md`](nix/AGENTS.md).
