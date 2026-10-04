# Kit (`fish-audio-suite-kit`)

> Scope: `packages/kit` (inherits root [AGENTS.md](../../AGENTS.md))

Pure text. No network, audio, or OpenTelemetry. Do not add `httpx` / FastAPI / `fishaudio` / sounddevice.

Import: `fish_audio_suite_kit`. Tests: `uv run pytest packages/kit`. `tests/` has one file per area. `test_scrub_tts.py` covers `scrub_tts`, and `_charsets` and `text_filters` are exercised through the other files.

## Layout

| Module | Owns |
| --- | --- |
| `_charsets` | Sentence stops, closers, CJK ranges, thought-tag names, `plain_breaks`, `utf8_text`. Add a shared character set here, never a second copy |
| `cues` | `normalize_cues`, `ensure_lead_cue`, `strip_cue_tags`, S1 parens, aliases, mood leads |
| `scrub_tts` | `scrub_tts`: strips thoughts, markdown, HTML, URLs and asides from text bound for TTS |
| `stream_holds` | `tts_hold_at`: one list of hold checks. A hold mirrors a `scrub_tts` rule |
| `dialogue` | `extract_quoted_speech`, `is_tts_junk`, narration, the letter floor |
| `cuts` | `next_tts_cut`, `split_tts_piece`, `ends_sentence` |
| `asr_text` | `scrub_asr`, `is_asr_hallucination`, `asr_language_hint`, backchannel and quit gates, watermarks |
| `defaults` | `SuiteDefaults`, the default system prompt, the Fish model, latency and format name tables, `chunk_length_hi`, `is_insecure_fish_base`, `strip_base` |
| `env` | `env_int` / `env_float` / `env_bool` / `env_text` / `env_token` / `env_base`, `parse_number`, `clamp_number`. A bad value falls back to the default |
| `timing` | `LatencySnapshot`, `elapsed_ms`, `MS_PER_S` |
| `http_errors` | `FishHttpError` and its subclasses, `FishErrorBody`, retry and backoff helpers, `retry_after_s` |
| `literals` | `FishLatency`, `AudioFormat`, `AsrFormat`, `TtsModel`, `ChatRole`, `ChatMessage`. Types only |
| `payloads` | `AsrBody`, `AsrSegment`, `AsrWord`, `OpenAIErrorDetail`, `OpenAIErrorBody`. Types only |
| `_version` | Private. `read_version` for `__version__` |
| `captions` | `CaptionCue`, `format_as_srt`, `format_as_vtt`. No network |
| `trace_context` | W3C `traceparent` parse and make, `ensure_trace_headers`. No OpenTelemetry |
| `text_filters` | Re-export facade only |

## Invariants

- Roleplay behavior is opt-in and off by default: `normalize_cues(lead=True)` rewrites a sentence-leading mood word, and `tts_hold_at(lead=True)` holds for it. `is_tts_junk(drop_narration=True)` drops unquoted stage directions. Nothing in the default path may match ordinary English.
- ASR and TTS gates drop only noise. Floor is `min_letters=2` with a `short_words` allowlist. `is_quit_utterance` and `is_backchannel` take caller `phrases`. The defaults hold explicit goodbyes and listener noise only.
- `SuiteDefaults.system_prompt` is voice formatting and cue use only. No persona, scene, or refusal wording.
- `scrub_tts` with no extra arguments is the whole string. `before` / `after` are one neighbor character and keep an edge space. Neighbor state travels as an argument, never a context variable.
- A sentence stop, closer, CJK range, or thought-tag name is defined once in `_charsets`.
- Retry is 429 and 5xx only, five attempts. `fish_backoff_s` adds jitter and honors `Retry-After`. `fish_sleep_before_retry` returns True when the caller should try again.
- Errors are typed: `FishAudioSuiteError` is the base, `FishHttpError.from_status(...)` returns `FishAuthError` (401/402/403), `FishRateLimitError` (429), `FishTimeoutError` (504), `FishUpstreamError` (other 5xx) or a plain `FishHttpError`. `.for_unreachable()` / `.for_timeout()` / `.for_non_json()` / `.for_non_object()` build the 502 and 504 ones. Callers catch by class or read `.retryable` and `.retry_after`; they do not compare status numbers.
- No deprecated names: nothing is released before 1.0.0, so a rename replaces the old name outright (no wrapper, alias, `__getattr__`, or renamed-env-var reader). Tests exercise only the current names, and the suite runs with warnings as errors.
- `Retry-After` is parsed in one place, `retry_after_s`. Other packages must not write their own.
- Closed value sets live in `literals` and a table lookup narrows them (`known_latency`, `known_audio_format`), so a helper never needs a `cast`. `normalize_tts_model` stays `str` because another model id passes through; `catalog_tts_model` narrows to `TtsModel`.
- `LatencySnapshot` stays positional-compatible: a new field goes last and it is not keyword-only.
- Every module has an `__all__`, and `tests/test_typed_api.py` checks that each name resolves. `__init__.py` is the curated API, and `tests/test_api_contract.py` with `golden/kit_api.json` fails on any change to it or to a signature proxy and voice use. Regenerate with `UPDATE_GOLDEN=1` only for an intended change.
- A public function gets a full NumPy docstring. Pure ones get an `Examples` section, and `pytest --doctest-modules packages/kit/src` runs them.
- `describe_transport_error` never returns exception text; callers log `exc` themselves. `captions` compares integer milliseconds, not formatted clocks. `env_int` and `env_float` take ASCII decimals only.
- `chunk_length_hi` decides cloud from the parsed hostname. `self_hosted=` overrides. Env is read by callers, never at import.
- `scrub_asr` keeps `[cue]` annotations unless `strip_cues=True`. Digit-only brackets always stay.
- A regex over model text is bounded or anchored: `_LABEL`, `_TAG` and `_ASIDE` in `scrub_tts`, a run-start lookbehind for space runs, and `_erase_spans` for open-to-close blocks. An unbounded scan from every opener is quadratic. `test_scrub_tts.py` times hostile 20k-character inputs.
- Public API is `__init__.py`. Another package must not import a private module.
