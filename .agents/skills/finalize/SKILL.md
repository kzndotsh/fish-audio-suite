---
name: finalize
description: >-
  Use when finishing a chunk of work, before considering a task done, or
  when the user types /finalize, "wrap this up", "close the loop", or
  "run the post-change checks". Hygiene, mapped docs, tests, CI gates,
  then a conventional commit plan for approval. Do NOT trigger mid-implementation
  or when the user names one specific command (e.g. "run pytest").
metadata:
  owner: fish-audio-suite
  sources: AGENTS.md, .github/workflows/ci.yml, pyproject.toml, README.md
---

# Finalize

Closes a work chunk so CI will not surprise, then drafts grouped commits
and **stops for approval**.

## Outcomes

- **Clean** — close-out done; gates pass; commit plan posted; nothing committed.
- **Changed** — files were edited to close the chunk; list them.
- **Committed** — only after explicit approval; list SHAs.
- **Blocked** — gate, missing tests, or a product call; report verbatim and stop.

## Edit scope

May edit README / nested `AGENTS.md`, tests for the dirty behavior, and
files a failing gate points at. May `git commit` only after the user
approves the posted plan. Does not push or change CI unless asked.

## Instructions

1. List dirty paths (`git status --porcelain`; add `git diff --name-only` if needed).
2. Load [close-out.md](references/close-out.md) and apply hygiene + test-gap on those paths.
3. Docs: patch root [`README.md`](../../../README.md) and the dirty package README if a public claim, env var, or command changed. Verify against the diff. No new docs unless asked.
4. Touch a nested `AGENTS.md` only if Commands, Boundaries, or Gotchas actually changed. Read that tree's `AGENTS.md` before editing it.
5. Load [gate-map.md](references/gate-map.md), union commands for the dirty set, run fastest first. On failure: read the error, fix, rerun that gate. Same gate still failing after one fix → **Blocked**.
6. Do not run `fish-voice --smoke` or live Fish/OpenRouter calls unless the user asked (needs `FISH_API_KEY` / `FISH_VOICE_ID`).
7. Load [commit-plan.md](references/commit-plan.md), review **all** uncommitted work, draft the grouping, post it, and **stop**.
8. On explicit approval only, execute that plan (or the user's edited version).

## Gotchas

- CI is GitHub Actions (ruff, basedpyright, pytest), not `nix flake check`.
- Format with `uv run ruff format packages` so `ruff format --check` in CI passes.
- Pytest: `--import-mode=importlib`; prefer `uv run pytest packages/<member>` over the whole suite.
- Dist names stay `fish-audio-suite-{kit,proxy,voice}` only.
- Do not start this while implementation is still in progress.
- Do not run a repo-wide docs pass; stay on the dirty set.
