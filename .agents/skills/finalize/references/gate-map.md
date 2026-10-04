# Gate map

Load this when: deciding which commands to run for dirty paths.
CI (`.github/workflows/ci.yml`) wins if this file and `CONTRIBUTING.md` diverge.

Need `uv sync --all-packages --extra cli --group dev --group test` first if `.venv` is
missing or lock/deps changed.

## Path → commands

Union rows. Fastest first: format/lint → docstrings → types → tests → nix.

| Dirty paths | Run |
| --- | --- |
| `packages/**/*.py`, `.github/scripts/**/*.py`, `pyproject.toml` | `uv run ruff format packages .github/scripts`, `uv run ruff check packages .github/scripts`, `uv run pydoclint --config=pyproject.toml packages`, `uv run basedpyright` |
| `packages/kit/**` | plus `uv run pytest packages/kit` |
| `packages/proxy/**` | plus `uv run pytest packages/proxy` |
| `packages/voice/**` | plus `uv run pytest packages/voice` |
| more than one member, or `pyproject.toml` / `.github/workflows/ci.yml` / root test config | `uv run pytest` (full `packages/`) |
| `flake.nix`, `nix/**` | `nix flake check` (`nix flake show` only lists the outputs; it builds no checks) |
| `AGENTS.md`, `docs/**`, `.agents/skills/**` | re-read what you changed; `uv run pytest tests` checks the settings tables and the repo rules |

CI Python job always runs ruff check, `ruff format --check packages`,
`pydoclint --config=pyproject.toml packages`, basedpyright, and full pytest
after `uv sync --frozen --all-packages --extra cli --group dev --group test`.

Do not treat `uv build --all` or `fish-voice --smoke` as merge gates. CI also runs
`nix flake check`, the wheel build and the Docker image in their own jobs.
