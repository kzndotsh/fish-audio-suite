# `packages/`

> Inherits root [AGENTS.md](../AGENTS.md)

uv workspace (`members = ["packages/*"]`). Each child is a hatchling src layout. Proxy and voice depend on kit through `[tool.uv.sources]` `workspace = true`. Do not document a standalone `pip install` of proxy or voice. Kit has to come from this workspace.

| Dir | Dist / import |
| --- | --- |
| [`kit`](kit/AGENTS.md) | `fish-audio-suite-kit` / `fish_audio_suite_kit` |
| [`proxy`](proxy/AGENTS.md) | `fish-audio-suite-proxy` / `fish_audio_suite_proxy` |
| [`voice`](voice/AGENTS.md) | `fish-audio-suite-voice` / `fish_audio_suite_voice` |

Tests live in `packages/<member>/tests/`. Commands and repo-wide boundaries stay in the root file.
