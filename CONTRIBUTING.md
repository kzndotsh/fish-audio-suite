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
uv run ruff check packages .github/scripts
uv run ruff format --check packages .github/scripts
uv run pydoclint --config=pyproject.toml packages
uv run basedpyright
uv run python .github/scripts/verifytypes.py
uv run pytest
```

`verifytypes.py` scores how fully typed each package's public API is and fails
under the floors in `.github/verifytypes-floors.json` (`just types-public`).
`pytest` also runs the `>>>` examples in kit's docstrings as doctests.

`just fmt` (or `uv run ruff format packages .github/scripts`) rewrites the formatting. `uvx pre-commit install`
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

Kit docstrings may carry an `Examples` section. It runs as a doctest on every
`pytest` run, so a stale example fails the build. Keep the output exact. Proxy and
voice have no doctests.

## The public API

The public API of a package is the set of names in its root `__all__`
(`fish_audio_suite_kit`, `fish_audio_suite_proxy`, `fish_audio_suite_voice`) and
the behavior their docstrings state. The proxy's surface is also its HTTP API and
its documented `FISH_*` settings. Everything else is internal and can change in
any release: modules and names that start with an underscore, and anything not
exported from the package root. Proxy and voice import kit only through its root.

Changing a public name, signature, return type or documented behavior is an API
change. Kit has a contract test that records its root exports and the signatures
proxy and voice call. When it fails and the change is intended, bump the kit
version, then regenerate and review the file:

```sh
UPDATE_GOLDEN=1 uv run pytest packages/kit/tests/test_api_contract.py
git diff packages/kit/tests/golden/kit_api.json
```

## Versioning and deprecation

Versions follow SemVer, read the 0.x way: a patch release (0.1.0 to 0.1.1) never
breaks the public API, and a minor release (0.1 to 0.2) may. Proxy and voice pin
kit to `>=0.1,<0.2`, so a kit minor bump needs a matching change in both. From
1.0 a breaking change needs a major release.

Remove a public name only after it has been deprecated for at least one minor
release. A deprecated name keeps working and emits `DeprecationWarning`, with
`stacklevel=2` so the warning points at the caller. The message names the
replacement and the version that deprecated the name. A renamed environment variable keeps
its old name the same way and logs a warning that names the new one.

## Releases and release notes

There is no `CHANGELOG.md`. The release notes are the changelog. GitHub's
generated notes list every merged pull request by its title, and titles follow
Conventional Commits, so mark a breaking change with `!` (`feat(kit)!: ...`) and
describe it in the pull request body. To release:

1. Set the same new version in the three `packages/*/pyproject.toml` files and
   commit it. The release workflow fails when the tag and the versions differ.
2. Tag that commit and push the tag, for example `git tag v0.2.0 && git push origin v0.2.0`.
   The workflow tests, builds and publishes when the repo variable `PYPI_PUBLISH` is `true`.
3. Create the GitHub release with its notes:
   `gh release create v0.2.0 --verify-tag --generate-notes`. The workflow does not do
   this itself, so it never holds a token that can write to the repository.

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
