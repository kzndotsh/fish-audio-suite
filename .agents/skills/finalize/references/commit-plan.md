# Commit plan

Load this when: `/finalize` gates are clean and uncommitted work
remains — draft grouped conventional commits, then stop for approval.

## Review

Read all uncommitted work before grouping: `git status --porcelain`,
`git diff` (unstaged + staged), `git diff --cached`, and
`git log -15 --format='%s'` for this repo's subject style.

Include untracked files. Exclude secrets (`.env*`, credentials, keys,
`FISH_API_KEY`, `FISH_VOICE_ID`).

## Grouping (more commits, not fewer)

Each commit is one reviewable concern. Prefer splitting over a blob.

Keep together: a behavior + its tests; README/`AGENTS.md` with the code
that changed the claim.

Split: shared kit primitive vs proxy/voice consumer; `fix` vs `feat`;
docs-only vs product; CI/Nix vs Python.

Assign **whole files** to a commit. Do not `git add -p` or `git add -i`.
If one file mixes two concerns, put it with the later consumer and note
that in the plan, or ask.

Order: dependencies first (kit → proxy/voice → docs-only leftovers).

## Message

Conventional: `type(scope): subject`

Types: `feat` `fix` `docs` `refactor` `test` `chore` `ci` `perf`.
Scopes used here: `kit`, `proxy`, `voice`, `nix`, `ci`. Omit scope for
cross-cutting work. Match `git log` when unsure.

Subject: imperative, lowercase after the colon, no trailing period,
~72 chars. Body (when needed): why, not a file list. No emojis.

Do not invent `CHANGELOG.md`.

## Approval gate

Post the plan as a numbered list: files, proposed message (subject +
body). **Stop. Do not run `git commit`.**

Commit only on an explicit yes in a later turn (`commit that`, `lgtm`,
`approved`, edits to the plan). Re-read status before executing — if
the tree changed, re-draft and stop again.

## Execute (after approval only)

User git protocol: `git status`, `git diff`, `git log -15 --format='%s'`
in parallel, then for each approved commit: `git add` those paths,
`git commit` with HEREDOC (`-m "$(cat <<'EOF' … EOF)"`). No `--no-verify`,
`--amend` (unless the user's amend rules are all met), or push.
After the last commit, `git status` and report SHAs.
