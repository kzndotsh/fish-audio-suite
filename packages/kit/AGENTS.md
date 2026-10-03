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
| `stream_holds` | `tts_hold_at`: one list of hold checks. A hold mirrors a `scrub_markdown` rule |
| `dialogue` | `extract_quoted_speech`, `is_tts_junk`, narration, the letter floor |
| `cues` | `normalize_cues`, `ensure_lead_cue`, `strip_cue_tags`, the S1 cue names |
| `cuts` | `next_tts_cut`, `split_tts_piece`, `ends_sentence` |
| `asr_text` | `scrub_asr`, `is_asr_hallucination`, backchannel and quit gates, watermarks |
| `defaults` | `SuiteDefaults`, `LatencySnapshot`, clamps, env readers (`env_renamed` for a renamed variable), `chunk_length_hi`, `is_insecure_fish_base` |
| `http_errors` | `FishHttpError` and its subclasses, `FishErrorBody`, retry and backoff helpers, `retry_after_s` |
| `literals` | `FishLatency`, `AudioFormat`, `AsrFormat`, `TtsModel`, `ChatRole`, `ChatMessage`. Types only |
| `payloads` | `AsrBody`, `AsrSegment`, `AsrWord`, `OpenAIErrorDetail`, `OpenAIErrorBody`. Types only |
| `_deprecation` | `deprecated(replacement, since)` for a function or method, `deprecated_fields(since, old="new")` for renamed dataclass fields. Both exported from the root |
| `_version` | Private. `read_version` for `__version__` |
| `captions` | `CaptionCue`, `format_as_srt`, `format_as_vtt`. No network |
| `trace_context` | W3C `traceparent` parse and make, `ensure_trace_headers`. No OpenTelemetry |
| `_charsets` | Private. Stops, closers, CJK ranges, thought-tag names, shared regexes |
| `text_filters` | Re-export facade only |

## Invariants

- Roleplay behavior is opt-in and off by default: `normalize_cues(lead=True)` rewrites a sentence-leading mood word, and `tts_hold_at(lead=True)` holds for it. `is_tts_junk(drop_narration=True)` drops unquoted stage directions. Nothing in the default path may match ordinary English.
- ASR and TTS gates drop only noise. Floor is `min_letters=2` with a `short_words` allowlist. `is_quit_utterance` and `is_backchannel` take caller `phrases`. The defaults hold explicit goodbyes and listener noise only.
- `SuiteDefaults.system_prompt` is voice formatting and cue use only. No persona, scene, or refusal wording.
- `scrub_tts` with no extra arguments is the whole string. `before` / `after` are one neighbor character and keep an edge space. Neighbor state travels as an argument, never a context variable.
- A sentence stop, closer, CJK range, or thought-tag name is defined once in `_charsets`.
- Retry is 429 and 5xx only, five attempts. `fish_backoff_s` adds jitter and honors `Retry-After`. `fish_sleep_before_retry` returns True when the caller should try again.
- Errors are typed: `FishAudioSuiteError` is the base, `FishHttpError.from_status(...)` returns `FishAuthError` (401/402/403), `FishRateLimitError` (429), `FishTimeoutError` (504), `FishUpstreamError` (other 5xx) or a plain `FishHttpError`. `.for_unreachable()` / `.for_timeout()` / `.for_non_json()` / `.for_non_object()` build the 502 and 504 ones. Callers catch by class or read `.retryable` and `.retry_after`; they do not compare status numbers.
- Deprecated names (`fish_retry_pause`, `fish_backoff_seconds`, the `fish_unreachable()` style tuple helpers, `w3c_trace_headers`, and the 0.2.0 renames such as `hold_tts`, `clamp_num`, `retry_after_seconds` and `FishHttpError.unreachable()`) must not be used inside the kit or by the other packages. A renamed function or method keeps its old name wrapped with `deprecated(new, since)`. A renamed constant or class goes in `_DEPRECATED_ALIASES` in `__init__.py`, which the root `__getattr__` resolves with a warning; it stays out of `__all__`. A renamed dataclass field uses `deprecated_fields` (old keyword and attribute). A renamed env var is read through `env_renamed`, and the caller passes its own `warn`. The suite runs with warnings as errors, so a stray call fails a test. A test of a deprecated name uses `pytest.warns`.
- `Retry-After` is parsed in one place, `retry_after_s`. Other packages must not write their own.
- Closed value sets live in `literals` and a table lookup narrows them (`known_latency`, `known_audio_format`), so a helper never needs a `cast`. `normalize_tts_model` stays `str` because another model id passes through; `catalog_tts_model` narrows to `TtsModel`.
- `LatencySnapshot` stays positional-compatible: a new field goes last and it is not keyword-only.
- Every module has an `__all__`, and `tests/test_typed_api.py` checks that each name resolves. `__init__.py` is the curated API, and `tests/test_api_contract.py` with `golden/kit_api.json` fails on any change to it or to a signature proxy and voice use. Regenerate with `UPDATE_GOLDEN=1` only for an intended change.
- A public function gets a full NumPy docstring. Pure ones get an `Examples` section, and `pytest --doctest-modules packages/kit/src` runs them.
- `describe_transport_error` never returns exception text; callers log `exc` themselves. `captions` compares integer milliseconds, not formatted clocks. `env_int` and `env_float` take ASCII decimals only.
- `chunk_length_hi` decides cloud from the parsed hostname. `self_hosted=` overrides. Env is read by callers, never at import.
- `scrub_asr` keeps `[cue]` annotations unless `strip_cues=True`. Digit-only brackets always stay.
- A regex over model text is bounded or anchored: `_LABEL`, `_TAG` and `_ASIDE` in `scrub_markdown`, a run-start lookbehind for space runs, and `_erase_spans` for open-to-close blocks. An unbounded scan from every opener is quadratic. `test_scrub_tts.py` times hostile 20k-character inputs.
- Public API is `__init__.py`. Another package must not import a private module.
