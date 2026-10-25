# Gate map

Load this when: deciding which commands to run for dirty paths.
The Python gates live in `.github/actions/gates/action.yml`, which `ci.yml` and
`release.yml` both run, and the `justfile` mirrors them under the same names. If this
file disagrees with them, they win.

Need `uv sync --all-packages --extra cli --group dev --group test` (`just sync`) first if
`.venv` is missing or lock/deps changed.

## Path → commands

Union rows. Fastest first: format/lint → docstrings → types → import contracts → tests → nix.

| Dirty paths | Run |
| --- | --- |
| `packages/**/*.py`, `.github/scripts/**/*.py`, `scripts/*.py`, `pyproject.toml` | `just fmt`, `just lint`, `just docstrings`, `just types` |
| `tests/*.py` (the root tests, which ruff and pydoclint do not cover) | `just types`, `uv run pytest tests` |
| `packages/**/*.py`, `pyproject.toml` (an import added, or a voice module added or moved) | plus `just layers` (`uv run lint-imports --no-cache`). A new voice module goes in the lowest layer of `[tool.importlinter]` it fits |
| a package's root `__init__.py` or any exported name or signature | plus `just types-public`. For kit, `uv run pytest packages/kit/tests/test_api_contract.py`, and `UPDATE_GOLDEN=1` only when the change is intended |
| `packages/kit/**` | plus `uv run pytest packages/kit` |
| `packages/proxy/**` | plus `uv run pytest packages/proxy` |
| `packages/voice/**` | plus `uv run pytest packages/voice` |
| `scripts/**`, `tests/**`, a setting, `.env.example`, a README settings table | plus `uv run pytest tests` (the settings tables, the repo rules, the script tests) |
| more than one member, or `pyproject.toml`, `conftest.py`, the `justfile` or `.github/actions/gates/**` | `just cov` (full suite with the CI coverage floors) |
| `.github/workflows/**`, `.github/actions/**` | plus `just workflows` |
| `packages/*/pyproject.toml`, `uv.lock` | plus `just build`; `just audit` when a dependency changed |
| `flake.nix`, `nix/**` | `nix flake check` (`nix flake show` only lists the outputs; it builds no checks) |
| `Dockerfile` | `docker build -t fish-audio-suite-proxy:ci .` |
| `AGENTS.md`, `docs/**`, `.agents/skills/**`, Markdown | re-read what you changed; `uv run pytest tests` |

`just check` runs every Python gate in CI's order: `lint docstrings types types-public
layers cov`. Run it once at the end when the dirty set touches more than one member.

Do not treat `fish-voice --smoke` as a merge gate. CI also runs the lowest
dependency versions, zizmor, the wheel build, `nix flake check` and the Docker image in
their own jobs.
