# The gates CI runs, under the same names, so a local run matches CI.
# `just` lists the recipes. `just check` runs every Python gate.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Install every workspace member with the dev and test groups.
sync:
    uv sync --all-packages --extra cli --group dev --group test

# Ruff check and the format check, as CI runs them.
lint:
    uv run ruff check packages .github/scripts scripts
    uv run ruff format --check packages .github/scripts scripts

# Rewrite the code in the project's format.
fmt:
    uv run ruff format packages .github/scripts scripts

# Docstrings that match their signatures. Ruff's D rules already run in `lint`.
docstrings:
    uv run pydoclint --config=pyproject.toml packages

# Strict type check.
types:
    uv run basedpyright

# How fully typed each package's public API is, against the floors in .github/verifytypes-floors.json.
types-public:
    uv run python .github/scripts/verifytypes.py

# Tests without coverage, plus the kit doctests. Extra arguments go to pytest, e.g. `just test -k barge`.
test *args:
    uv run pytest {{ args }}

# Tests and kit doctests with branch coverage and the CI floors. Keep the numbers in step with .github/actions/gates.
cov:
    uv run pytest --cov --cov-report=term-missing --cov-fail-under=89
    uv run coverage report --include="packages/kit/*" --fail-under=94 --skip-covered
    uv run coverage report --include="packages/proxy/*" --fail-under=94 --skip-covered
    uv run coverage report --include="packages/voice/*" --fail-under=85 --skip-covered

# Every Python gate CI runs, in CI's order (the steps of .github/actions/gates).
check: lint docstrings types types-public cov

# Lint the GitHub workflows. Set GH_TOKEN to include the online checks.
workflows:
    uvx zizmor==1.30.1 --no-progress .github

# Check the locked dependencies for known vulnerabilities.
audit:
    uv export --frozen --all-packages --no-emit-workspace --extra cli --group dev --group test --output-file "${TMPDIR:-/tmp}/fish-audit-requirements.txt"
    uvx pip-audit==2.10.1 --requirement "${TMPDIR:-/tmp}/fish-audit-requirements.txt" --disable-pip --progress-spinner off

# Build the three wheels and check their metadata.
build:
    uv build --all --out-dir dist
    uvx twine check --strict dist/*
