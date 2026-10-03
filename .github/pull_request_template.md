<!-- Title in Conventional Commits form, e.g. `fix(proxy): reject an empty bearer token`. -->

## What and why

<!-- What changed, and the problem it solves. Link an issue if there is one. -->

## How it was checked

<!-- Commands you ran, a test you added, or a manual run. A bug fix needs a regression test. -->

## Checklist

- [ ] `just check` passes (ruff, pydoclint, basedpyright, pytest with the coverage floors)
- [ ] New behavior or a bug fix has a test
- [ ] Public signatures have matching NumPy docstrings
- [ ] README or AGENTS.md updated if a setting, command or boundary changed
- [ ] No keys, voice ids or `.env` files are in the diff
- [ ] Anything that changes behavior for existing users is listed under "What and why"
