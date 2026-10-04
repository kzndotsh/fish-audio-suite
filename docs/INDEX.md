# Documentation index

Every doc in this repository, what is in it, and who it is for. Start with the row that matches what you are trying to do.

## I want to...

| Goal | Go to |
| --- | --- |
| Install it, or run the proxy as a service | [INSTALL.md](INSTALL.md) |
| Connect it to Open WebUI, SillyTavern, RisuAI, AIRI, an OpenAI SDK, or my own Python code | [INTEGRATIONS.md](INTEGRATIONS.md) |
| Make replies come sooner, or pick a faster model | [PERFORMANCE.md](PERFORMANCE.md) |
| Fix something that is not working | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Understand how the packages fit together, or change the code | [ARCHITECTURE.md](ARCHITECTURE.md), then [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Look up a term such as barge-in, cue or flush | [GLOSSARY.md](GLOSSARY.md) |

## The docs

| Doc | Audience | What it covers |
| --- | --- | --- |
| [INSTALL.md](INSTALL.md) | Users and operators | What you need, then installing `fish-voice`, the proxy and the libraries from a clone, with `uv tool`, Docker, Compose or NixOS. Reverse proxies, self-hosted Fish, updating and uninstalling |
| [INTEGRATIONS.md](INTEGRATIONS.md) | Users and developers | Per-app setup for Open WebUI, SillyTavern, RisuAI and AIRI, with a table of which need CORS headers. The OpenAI SDKs and curl. The `FishSpeaker` library and the kit text helpers. Running `fish-voice` with any LLM. A CORS helper for browser apps |
| [PERFORMANCE.md](PERFORMANCE.md) | Users and developers | Where a turn's time goes, the recommended low-latency settings and what each one trades away, what the code already does, how to choose an LLM, running Ollama, how to read the debug log, and the known limits |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Users | Symptoms grouped by area (audio, Fish, listening, barge-in, playback, the LLM, the proxy), each with the log line to look for and the fix |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers and agents | The project layout, system and per-turn diagrams, each package's components, data stores, deployment, security, testing, and the **invariants**: the rules most easily broken, each with its reason |
| [GLOSSARY.md](GLOSSARY.md) | Everyone | Definitions of the project's terms, in one alphabetical table |

## Elsewhere in the repository

| File | What it covers |
| --- | --- |
| [README.md](../README.md) | What the project is, the three packages, and a quick start |
| [packages/kit/README.md](../packages/kit/README.md) | The text helpers: every function group and its settings |
| [packages/proxy/README.md](../packages/proxy/README.md) | The OpenAI-compatible server: the request field map, text options, retries and every setting |
| [packages/voice/README.md](../packages/voice/README.md) | The library and `fish-voice`: LLM providers, characters, streaming, echo and barge-in, and every setting |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Setup, the checks, docstrings, the public API, env-var names, versioning, releases and commits |
| [SECURITY.md](../SECURITY.md) | Supported versions and how to report a vulnerability privately |
| [AGENTS.md](../AGENTS.md) | The short list of rules for coding agents, with pointers to the docs above |
| [.env.example](../.env.example) | Every setting, with its default |

## Keeping the docs true

- **The settings tables are checked.** `tests/test_docs_env.py` fails when a variable the code reads is missing from the READMEs or `.env.example`, or the other way round, and when a documented default differs from the code.
- **The repo rules are checked.** `tests/test_repo_rules.py` covers the coverage floors, kit's purity, kit imports and where the voice package reads the environment.
- **Everything else is reviewed by hand.** When a module moves or an invariant changes, update [ARCHITECTURE.md](ARCHITECTURE.md). When a log line or an error message changes, update [TROUBLESHOOTING.md](TROUBLESHOOTING.md), which quotes them.
