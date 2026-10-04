# AGENTS.md — fish-audio-suite

Unofficial Fish Audio toolkit, not affiliated with Fish Audio. Three distributions only:
`fish-audio-suite-{kit,proxy,voice}` under `packages/`. Never add a fourth.

## Read before changing code

- [CONTRIBUTING.md](CONTRIBUTING.md): setup, the checks, docstrings, the public API, env-var names, versioning and releases, commits.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how the packages fit together, and the [invariants](docs/ARCHITECTURE.md#9-invariants), the rules most easily broken, each with its reason.
- The package README (`packages/<kit|proxy|voice>/README.md`) for user-facing behavior and settings.

## Docs to consult

[docs/INDEX.md](docs/INDEX.md) lists every doc. Reach for these by task:

| When you are... | Read |
| --- | --- |
| Changing how modules, threads or packages relate | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), especially the [invariants](docs/ARCHITECTURE.md#9-invariants) |
| Adding or renaming a setting | [CONTRIBUTING.md](CONTRIBUTING.md#environment-variable-names), the package README tables and `.env.example` |
| Touching latency, streaming or playback | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Changing a log line or an error message | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), which quotes them |
| Changing install, Docker or NixOS behavior | [docs/INSTALL.md](docs/INSTALL.md) |
| Changing the proxy's HTTP API or a client-facing behavior | [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) and the [proxy README](packages/proxy/README.md) |
| Meeting an unfamiliar term | [docs/GLOSSARY.md](docs/GLOSSARY.md) |

## Commands

- `uv sync --all-packages --extra cli --group dev --group test`, then `just check` (ruff, pydoclint, basedpyright, public-API types, pytest with the CI coverage floors). `just --list` shows the rest.
- One package: `uv run pytest packages/<kit|proxy|voice>`.
- Do not call the live Fish or LLM APIs, or run `fish-voice --smoke`, unless the user asks. Tests run with the network blocked and must not need a key.

## Rules for agents

- **Until 1.0.0 nothing is released** (no tags, no PyPI, versions stay 0.1.0). Rename or remove public names outright. Do NOT add deprecation shims, aliases, `DeprecationWarning`s or old env-var fallbacks.
- Text, cue, scrub, error and caption logic lives in kit, and proxy and voice import kit only from its root. Never copy kit's regexes.
- Never read `FISH_API_KEY` at import time, and never give a voice id a default.
- Never `aclose()` the Fish websocket iterator; stop iterating and close the client.
- A setting rename changes the code, `.env.example`, the READMEs and `tests/test_docs_env.py` together.
- When a chunk of work is done, load the finalize skill by name (`/finalize`, [`.agents/skills/finalize`](.agents/skills/finalize)). Do not rely on auto-selection.
