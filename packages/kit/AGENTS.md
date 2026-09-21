# Kit (`fish-audio-suite-kit`)

> Scope: `packages/kit` (inherits root [AGENTS.md](../../AGENTS.md))

Pure text. `pyproject.toml` has no network or audio deps.

Import: `fish_audio_suite_kit`. Tests: `uv run pytest packages/kit`.

## Public API

| Name | Role |
| --- | --- |
| `normalize_cues` | Lowercase `[Tags]`; keep stacked leads; S1 `(happy)` → `[happy]`; mood-lead → cue; aliases (`laugh`/`sigh`/`chuckle`/`whisper`/`pause` → `laughing`/`sighing`/`chuckling`/`whispering`/`break`) |
| `scrub_tts` / `is_tts_junk` | Strip markdown, thoughts, `[pause Xs]`, `[S1]`; keep Fish `[cue]`, S1 parens rewritten to brackets, speaker/phoneme `<|…|>` tokens |
| `extract_quoted_speech` | Keep quoted dialogue (optional `[cue]`); drop stage notes |
| `scrub_asr` / `is_asr_hallucination` | Optional speaker strip, timestamps, `<|…|>`; drop nospeech / thanks-for-watching (incl. 谢谢观看); keep real CJK |
| `is_backchannel` / `is_quit_utterance` | `yeah` / `uh huh` / … ; `bye` / `quit` / … |
| `next_tts_cut` | Flush index or `-1`. Sentence end (skip `Dr.` / `1.`); else ~40 chars |
| `SuiteDefaults` / `LatencySnapshot` | Shared knobs; millisecond timings, no utterance field |

Do not add `httpx` / FastAPI / `fishaudio` / sounddevice here.
