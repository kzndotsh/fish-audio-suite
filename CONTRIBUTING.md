# Contributing

Thanks for helping. This is a uv workspace of three packages under `packages/`:
`kit` (shared text and error code), `proxy` (the OpenAI-compatible server) and
`voice` (the live voice client). Read the [invariants in ARCHITECTURE.md](docs/ARCHITECTURE.md#9-invariants)
first. They are the boundaries between the packages and the rules most easily broken,
each with its reason.

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
`just workflows` and `just build` if you touch those, and `nix flake check` (or
`nix flake show` to list the outputs) if you touch `flake.nix` or `nix/`. Keep the
coverage floors in the `justfile` and `.github/actions/gates/action.yml` equal; a
test fails when they differ.

## Tests

Tests live in `packages/<member>/tests/`. They run in random order with every
warning turned into an error and with network sockets disabled, so a test must
not depend on another test or on the real Fish API. Add a test with every
behavior change, and put a regression test beside every bug fix. Do not call the
live Fish API from a test.

## Environment variable names

A new setting takes its prefix from what it controls:

| Prefix | Means | Examples |
| --- | --- | --- |
| `FISH_*` | A Fish request default: `FISH_` plus the uppercased Fish wire field | `FISH_LATENCY`, `FISH_CHUNK_LENGTH`, `FISH_SPEED` (`prosody.speed`) |
| `FISH_TTS_*`, `FISH_ASR_*` | An endpoint default where the bare wire name (`model`, `format`) would be ambiguous, and text shaping before TTS | `FISH_TTS_MODEL`, `FISH_ASR_MODEL`, `FISH_TTS_MOOD_LEAD` |
| `FISH_PROXY_*` | Proxy server behavior | `FISH_PROXY_PORT`, `FISH_PROXY_TTS_ALIASES` |
| `FISH_VOICE_*` | Voice app behavior | `FISH_VOICE_PLAYBACK`, `FISH_VOICE_BARGE_FRAMES` |
| `FISH_LLM_*` | The voice app's chat model | `FISH_LLM_API_KEY`, `FISH_LLM_MODEL` |

`FISH_VOICE_ID` is the one exception: it is the Fish voice and maps to the wire field
`reference_id`. Seconds are implied in env names (`FISH_VOICE_COOLDOWN`,
`FISH_PROXY_READ_TIMEOUT`); other units are suffixed (`_FRAMES`, `_BYTES`, `_CHARS`).
Python fields always carry the unit (`cooldown_s`). `FISH_SPEED` is the default speed,
and the proxy multiplies a client-sent `speed` by it. `tests/test_docs_env.py` checks
that every variable the code reads appears in the README tables and `.env.example`, and
the reverse.

## Adding an LLM provider

Providers live in one table, `LLM_PROVIDERS` in `packages/voice/src/fish_audio_suite_voice/llm_tune.py`:
a name, a default base, a host, a key variable and a model variable. To add one:

1. Add a row to the table.
2. Add a test in `packages/voice/tests/test_llm_providers.py`.
3. Add its row to the provider table in the voice README.

A provider that speaks the OpenAI chat-completions API needs no new backend. The
provider is always read from the host of the final base URL, never from
`FISH_LLM_PROVIDER` alone, and each provider reads only its own key variable, so a
key cannot go to another host.

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
proxy and voice call. When it fails and the change is intended, regenerate and review the file (the
version stays 0.1.0 until 1.0.0):

```sh
UPDATE_GOLDEN=1 uv run pytest packages/kit/tests/test_api_contract.py
git diff packages/kit/tests/golden/kit_api.json
```

## Versioning and deprecation

Nothing has been released: there are no tags and nothing is on PyPI, and nothing
will be until 1.0.0. The three packages stay at 0.1.0 until then, and there is no
changelog until a release. Until 1.0.0, rename and remove public names freely.
Do NOT add deprecation shims, aliases, `DeprecationWarning`s, or renamed
environment-variable fallbacks. A rename changes the code, `.env.example`, the
READMEs and the docs-env test together, and keeps no old name.

From 1.0.0 the versions follow SemVer. A patch release never breaks the public
API, a minor release adds to it, and a breaking change needs a major release.
Remove a public name only after it has been deprecated for at least one minor
release. A deprecated name keeps working and emits `DeprecationWarning`, with
`stacklevel=2` so the warning points at the caller, and the message names the
replacement and the version that deprecated it. A renamed environment variable
keeps its old name the same way and logs a warning that names the new one.
Proxy and voice pin kit to `>=0.1,<0.2` today; the pin follows the kit release. When
a kit change breaks proxy or voice, bump that range in the same change, and release
the three packages together.

## Releases and release notes

Nothing is published before 1.0.0, so these steps apply from then. There is no `CHANGELOG.md`. The release notes are the
changelog. GitHub's generated notes list every merged pull request by its title, and titles follow
Conventional Commits, so mark a breaking change with `!` (`feat(kit)!: ...`) and
describe it in the pull request body. To release:

1. Set the same new version in the three `packages/*/pyproject.toml` files and
   commit it. The release workflow fails when the tag and the versions differ.
2. Tag that commit and push the tag, for example `git tag v1.0.0 && git push origin v1.0.0`.
   The workflow tests, builds and publishes when the repo variable `PYPI_PUBLISH` is `true`.
3. Create the GitHub release with its notes:
   `gh release create v1.0.0 --verify-tag --generate-notes`. The workflow does not do
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
