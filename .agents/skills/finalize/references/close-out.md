# Close-out checks

Load this when: running `/finalize` after a work chunk — hygiene, test
gaps, and scoped doc accuracy. Not a full-repo audit.

## Hygiene (dirty files only)

- No leftover debug prints, commented-out blocks, or unused imports ruff
  would catch after format + check.
- No new `Any`, type ignores, or `cast` to silence basedpyright. Use
  `Type | None`, not `Optional[Type]`. No inline imports, except lazy imports of
  optional extras in voice (`sounddevice`, `webrtcvad`, `pysilero_vad`, `pywebrtc_audio`,
  `openrouter`, `websockets` and `certifi` for Deepgram, and the `tui` module for `textual`).
- No file over 1600 lines; split rather than land bloat.
- A new voice module goes in the lowest layer of `[tool.importlinter]` in `pyproject.toml`
  that it fits, and `just layers` passes.
- Stay in the requested scope — no drive-by refactors.
- Do not add a `CHANGELOG.md`; this repo does not keep one.
- After gates pass, draft commits via [commit-plan.md](commit-plan.md);
  do not commit in the close-out turn.

## Docstrings (dirty Python only)

NumPy style. Ruff `D` and pydoclint both gate CI. When a public module,
class, or function in the diff changes behavior, signature, or return:

- Update the summary so the first line is imperative and matches what the
  code does now.
- Parameters, Returns, Yields, and Raises must name the same arguments,
  types, and failures as the signature. Drop a section that no longer applies.
- Notes only for non-obvious behavior. Do not restate the signature.
- A one-line summary is enough when pydoclint's short-docstring skip applies.
  Tests stay excluded.

Then double-check: `just lint` and `just docstrings`. A mismatch is a
gate failure, not a follow-up.

Kit-only text helpers stay in kit. Do not copy cue/scrub/cut/W3C/Fish-error
regexes into proxy or voice. Do not read `FISH_API_KEY` at import.
`FISH_VOICE_ID` has no default.

## Tests

Behavior change with no covering test → add one, or report **Blocked**.

Tests live in `packages/<member>/tests/` (one `test_<area>.py` per area, such as
`test_cues.py`, `test_proxy_speech.py` or `test_playback.py`, plus new files if a module needs its own). Grep for
the export name. Prefer `uv run pytest packages/<member>`.

New tests: one behavior per test, assert contracts not call sequences.
If a gate fails: code bug → fix code; obsolete assertion → update the
test; unclear product → **Blocked**.

Do not require live Fish (`--smoke`) for close-out.

## Docs (touched surfaces only)

If the diff changes a public API, CLI, env var, port, or Nix output,
update the matching page: root `README.md`, `packages/<member>/README.md`,
`docs/ARCHITECTURE.md` only when a module moved or an invariant changed, and `CONTRIBUTING.md` only when a workflow rule changed.
A new or renamed setting also changes `.env.example`, and `tests/test_docs_env.py` checks
the tables. A changed log line or error message changes `docs/TROUBLESHOOTING.md`, which
quotes them. A changed script changes `scripts/README.md`.
Do not invent a docs site or crawl every markdown file.
