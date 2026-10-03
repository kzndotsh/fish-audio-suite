# Contributing

Thanks for helping. This is a uv workspace of three packages under `packages/`:
`kit` (shared text and error code), `proxy` (the OpenAI-compatible server) and
`voice` (the live voice client). Read [AGENTS.md](AGENTS.md) first. It lists the
boundaries between the packages, which are the rules most easily broken.

## Set up

```sh
uv sync --all-packages --extra cli --group dev --group test
```

Python 3.12 or newer. Use uv for everything; there is no pip or requirements file.

## Run the checks

`just check` runs every Python gate that CI runs. Without `just`, the commands are:

```sh
uv run ruff check packages
uv run ruff format --check packages
uv run pydoclint --config=pyproject.toml packages
uv run basedpyright
uv run pytest
```

`uv run ruff format packages` rewrites the formatting. `uvx pre-commit install`
runs the fast checks on every commit.

CI also checks that the wheels build, `nix flake check`, the Docker image, the
lowest declared dependency versions, and the GitHub workflows with zizmor. Run
`just workflows` and `just build` if you touch those.

## Tests

Tests live in `packages/<member>/tests/`. They run in random order with every
warning turned into an error and with network sockets disabled, so a test must
not depend on another test or on the real Fish API. Add a test with every
behavior change, and put a regression test beside every bug fix. Do not call the
live Fish API from a test.

## Docstrings

Public modules, classes and functions get NumPy-style docstrings, written in the
same change as the signature. The first line is imperative. `Parameters`,
`Returns`, `Yields` and `Raises` must match the signature, and pydoclint fails
the build when they do not. Tests do not need docstrings.

## Commits and pull requests

Use [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): subject`, with a type of `feat`, `fix`, `docs`, `refactor`, `test`,
`chore`, `ci` or `perf`, and a scope of `kit`, `proxy`, `voice`, `nix` or `ci`
(leave the scope off for cross-cutting work). Keep the subject imperative and
under about 72 characters, and explain why in the body, not which files changed.
Title the pull request the same way. One reviewable concern per commit.

Keep a pull request to one purpose. Do not mix a refactor with a behavior change.
Never commit a key, a voice id or an `.env` file.

## Security

Report vulnerabilities privately. See [SECURITY.md](SECURITY.md).
