# `packages/`

> Inherits root [AGENTS.md](../AGENTS.md)

uv workspace (`members = ["packages/*"]`). Each child is a hatchling src layout. Proxy and voice depend on kit as `fish-audio-suite-kit>=0.1,<0.2` and resolve it from the workspace in dev (`[tool.uv.sources]`). Bump that range with a kit breaking release; the three dists publish together.

| Dir | Dist / import |
| --- | --- |
| [`kit`](kit/AGENTS.md) | `fish-audio-suite-kit` / `fish_audio_suite_kit` |
| [`proxy`](proxy/AGENTS.md) | `fish-audio-suite-proxy` / `fish_audio_suite_proxy` |
| [`voice`](voice/AGENTS.md) | `fish-audio-suite-voice` / `fish_audio_suite_voice` |

Tests live in `packages/<member>/tests/`. Commands and repo-wide boundaries stay in the root file.
