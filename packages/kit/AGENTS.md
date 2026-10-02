# Kit (`fish-audio-suite-kit`)

> Scope: `packages/kit` (inherits root [AGENTS.md](../../AGENTS.md))

Pure text. No network, audio, or OpenTelemetry. Do not add `httpx` / FastAPI / `fishaudio` / sounddevice.

Import: `fish_audio_suite_kit`. Tests: `uv run pytest packages/kit`. `tests/` has one file per area. `test_scrub_tts.py` covers `scrub_markdown`, and `_charsets` and `text_filters` are exercised through the other files.

## Layout

| Module | Owns |
| --- | --- |
| `_charsets` | Sentence stops, closers, CJK ranges, thought-tag names, `plain_breaks`, `utf8_text`. Add a shared character set here, never a second copy |
| `cues` | `[cue]` normalize, S1 parens, aliases, `strip_cue_tags`, mood leads |
| `scrub_markdown` | `scrub_tts`: thoughts, markdown, HTML, URLs, asides |
| `stream_holds` | `hold_tts`: one list of hold checks. A hold mirrors a `scrub_markdown` rule |
| `dialogue` | `extract_quoted_speech`, `is_tts_junk`, narration, the letter floor |
| `cues` | `normalize_cues`, `ensure_lead_cue`, `strip_cue_tags`, the S1 cue names |
| `cuts` | `next_tts_cut`, `split_tts_piece`, `ends_sentence` |
| `asr_text` | `scrub_asr`, `is_asr_hallucination`, backchannel and quit gates, watermarks |
| `defaults` | `SuiteDefaults`, clamps, env readers, `chunk_length_hi` |
| `http_errors` | Fish error shape, retry and backoff helpers |
| `captions` | `CaptionCue`, `format_as_srt`, `format_as_vtt`. No network |
| `trace_context` | W3C `traceparent` parse and make, `ensure_trace_headers`. No OpenTelemetry |
| `_charsets` | Private. Stops, closers, CJK ranges, thought-tag names, shared regexes |
| `text_filters` | Re-export facade only |

## Invariants

- Roleplay behavior is opt-in and off by default: `normalize_cues(lead=True)` rewrites a sentence-leading mood word, and `hold_tts(lead=True)` holds for it. `is_tts_junk(drop_narration=True)` drops unquoted stage directions. Nothing in the default path may match ordinary English.
- ASR and TTS gates drop only noise. Floor is `min_letters=2` with a `short_words` allowlist. `is_quit_utterance` and `is_backchannel` take caller `phrases`. The defaults hold explicit goodbyes and listener noise only.
- `SuiteDefaults.system_prompt` is voice formatting and cue use only. No persona, scene, or refusal wording.
- `scrub_tts` with no extra arguments is the whole string. `before` / `after` are one neighbor character and keep an edge space. Neighbor state travels as an argument, never a context variable.
- A sentence stop, closer, CJK range, or thought-tag name is defined once in `_charsets`.
- Retry is 429 and 5xx only, five attempts. `fish_backoff_s` adds jitter and honors `Retry-After`. `fish_sleep_before_retry` returns True when the caller should try again. `fish_retry_pause` is the legacy stop-flag form.
- `FishHttpError.unreachable()` / `.timed_out()` / `.non_json()` / `.non_object()` build the 502 and 504 errors. The `fish_unreachable()` style tuple helpers stay for callers.
- `chunk_length_hi` decides cloud from the parsed hostname. `self_hosted=` overrides. Env is read by callers, never at import.
- `scrub_asr` keeps `[cue]` annotations unless `strip_cues=True`. Digit-only brackets always stay.
- A regex over model text is bounded or anchored: `_LABEL`, `_TAG` and `_ASIDE` in `scrub_markdown`, a run-start lookbehind for space runs, and `_erase_spans` for open-to-close blocks. An unbounded scan from every opener is quadratic. `test_scrub_tts.py` times hostile 20k-character inputs.
- Public API is `__init__.py`. Another package must not import a private module.
