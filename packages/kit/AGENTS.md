# Kit (`fish-audio-suite-kit`)

> Scope: `packages/kit` (inherits root [AGENTS.md](../../AGENTS.md))

Pure text. `pyproject.toml` has no network or audio deps.

Import: `fish_audio_suite_kit`. Tests: `uv run pytest packages/kit`.

## Public API

| Name | Role |
| --- | --- |
| `normalize_cues` | Lowercase `[Tags]`; mood-lead → cue; aliases (`laugh`/`laughs` → `laughing`, `whisper`/`whispers` → `whispering`, `pause` → `break`) |
| `scrub_tts` / `is_tts_junk` | Strip markdown, thoughts, `[pause Xs]`, `[S1]`; keep Fish `[cue]` |
| `extract_quoted_speech` | Keep quoted dialogue (optional `[cue]`); drop stage notes |
| `scrub_asr` / `is_asr_hallucination` | Strip speakers, timestamps, `<|…|>`; drop CJK / nospeech / thanks-for-watching |
| `is_backchannel` / `is_quit_utterance` | `yeah` / `uh huh` / … ; `bye` / `quit` / … |
| `next_tts_cut` | Flush index or `-1`. Sentence end (skip `Dr.` / `1.`); else ~40 chars |
| `SuiteDefaults` / `LatencySnapshot` | Shared knobs; millisecond timings, no utterance field |

Do not add `httpx` / FastAPI / `fishaudio` / sounddevice here.
