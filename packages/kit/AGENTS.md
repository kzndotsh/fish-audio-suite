# Kit (`fish-audio-suite-kit`)

> Scope: `packages/kit` (inherits root [AGENTS.md](../../AGENTS.md))

Pure text. `pyproject.toml` has no network or audio deps.

Import: `fish_audio_suite_kit`. Tests: `uv run pytest packages/kit`.

## Public API

| Name | Role |
| --- | --- |
| `normalize_cues` | Lowercase `[Tags]`; keep stacked leads; S1 `(happy)` → `[happy]`; mood-lead → cue; aliases (`laugh`/`sigh`/`chuckle`/`whisper`/`pause` → `laughing`/`sighing`/`chuckling`/`whispering`/`break`). Inline `[chuckle]` aliases too. Leave `[cough]` |
| `ensure_lead_cue` | If a reply has no `[cue]` at all, prepend `[clear]`. Does not invent mid-reply tags |
| `scrub_tts` / `is_tts_junk` | Strip markdown, thoughts, `[pause Xs]`, `[S1]`; keep Fish `[cue]`, S1 parens rewritten to brackets, speaker/phoneme `<|…|>` tokens |
| `extract_quoted_speech` | Keep quoted dialogue (optional `[cue]`); drop stage notes |
| `scrub_asr` / `is_asr_hallucination` | Optional speaker strip, timestamps, `<|…|>`; drop nospeech / YouTube-caption BoH (thanks-for-watching, Amara, 谢谢观看) and gzip-repetition; drop 1-char CJK-only; keep real CJK sentences |
| `is_backchannel` / `is_quit_utterance` | `yeah` / `uh huh` / `嗯` / … ; `bye` / `quit` / … |
| `next_tts_cut` / `split_tts_piece` | Flush index or `-1`. Sentence end (skip `Dr.` / `1.`); else ~40 chars. `split_tts_piece` returns that piece and the tail, or `None` while a stream should keep buffering |
| `SuiteDefaults` / `LatencySnapshot` / `elapsed_ms` / `MS_PER_S` / `known_tts_model` / `known_latency` / `clamp_num` / `chunk_length_hi` | Shared knobs; millisecond timings (`MS_PER_S` is 1000), no utterance field; optional `trace_id` on the log line. `known_tts_model` lowercases a catalog id and leaves any other id as written. `known_latency` accepts `low`/`balanced`/`normal` and keeps the default otherwise. `clamp_num` keeps speed at 0.5–2, temperature / top_p / early_stop at 0–1, `chunk_length` at 100–300 on `api.fish.audio` (1000 self-hosted), and `min_chunk_length` at 0–100. `known_mp3_bitrate` keeps 64 and 192 and snaps anything else to 128. `known_opus_bitrate` keeps the documented set and snaps anything else to -1000 |
| `env_base` / `strip_base` / `env_text` / `env_token` / `env_bool` / `env_off` / `env_int` / `env_float` / `number_or` | Process env. `strip_base` and `env_base` remove surrounding space and trailing slashes. `env_text` strips space and keeps a blank value. `env_token` strips space and keeps the default when blank. A missing key keeps the default. Blank flags and blank or non-numeric numbers keep the default. Flags are true only for `1`/`true`/`yes`/`on`. `env_off` is true only for `0`/`false`/`no`/`off`. `number_or` parses a number and keeps the default on junk |
| `w3c_trace_headers` / `ensure_trace_headers` / `make_traceparent` | Parse, forward-or-mint, and mint W3C `traceparent` (+ optional `tracestate`). No OpenTelemetry |
| `FISH_TTS_PATH` / `FISH_ASR_PATH` / `parse_fish_error` / `should_retry_fish_status` / `fish_attempt_exhausted` / `fish_backoff_seconds` / `fish_retry_pause` / `fish_transport_error` / `fish_request_error` / `bearer` | Fish `/v1/tts` and `/v1/asr`. `{message, status}` body; retry 429 and 5xx only; the fifth attempt is the last; `fish_retry_pause` sleeps `2 ** attempt` until that last try; timeout is 504, other transport failure is 502; `fish_request_error` applies that rule when the exception is the caller's timeout type; a retry loop that ends with no error uses that 502. `fish_non_json` and `fish_non_object` are that same 502 when the body is not JSON or not a JSON object. `parse_asr_body` returns that object and its text, and raises the non-object 502 when the body or the text field has the wrong shape. `Authorization` value is `Bearer` plus the key |
| `CaptionCue` / `format_as_srt` / `format_as_vtt` | Timed ASR cues → SubRip / WebVTT. No network |

Do not add `httpx` / FastAPI / `fishaudio` / sounddevice here.
