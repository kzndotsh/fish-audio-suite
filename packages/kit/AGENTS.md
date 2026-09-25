# Kit (`fish-audio-suite-kit`)

> Scope: `packages/kit` (inherits root [AGENTS.md](../../AGENTS.md))

Pure text. `pyproject.toml` has no network or audio deps.

Import: `fish_audio_suite_kit`. Tests: `uv run pytest packages/kit`.

## Public API

| Name | Role |
| --- | --- |
| `normalize_cues` | Lowercase `[Tags]`; keep stacked leads; S1 `(happy)` → `[happy]`; a Fish emotion at the start of a sentence (`Anxious, …`) becomes that cue. whispering, shouting, and screaming do too. Sound-effect words do not. Aliases (`laugh`/`sigh`/`chuckle`/`whisper`/`pause` → `laughing`/`sighing`/`chuckling`/`whispering`/`break`). Inline `[chuckle]` aliases too. Leave `[cough]` |
| `mood_lead_hold_at` | Index of an unfinished sentence-lead word (`Exc` before `ited,`). None once the cue is complete, the word has ended, or the text is mid-sentence |
| `hold_tts` | Index where a stream must keep buffering an unfinished span. `len(text)` when the chunk is stable. `before` is scrubbed text already accepted on the line |
| `ensure_lead_cue` | If a reply has no `[cue]` at all, prepend `[clear]`. Does not invent mid-reply tags |
| `scrub_tts` / `is_tts_junk` | Strip markdown, thoughts, `[pause Xs]`, `[S1]`; keep Fish `[cue]`, S1 parens rewritten to brackets, speaker/phoneme `<|…|>` tokens. No extra arguments is the whole string. `line_start=False` keeps a mid-line bullet. `continued=True` drops a leading parenthesis aside. `before` and `after` are one neighbor character and keep an edge space |
| `extract_quoted_speech` | Keep quoted dialogue (optional `[cue]`); drop stage notes |
| `scrub_asr` / `is_asr_hallucination` / `is_caption_watermark` | Optional speaker strip, timestamps, `<|…|>`; drop nospeech / YouTube-caption BoH (thanks-for-watching, Amara, 谢谢观看) and gzip-repetition; drop 1-char CJK-only; keep real CJK sentences. `is_caption_watermark` is only the known caption, so a short `ok` segment stays |
| `is_backchannel` / `is_quit_utterance` / `same_utterance` | `yeah` / `uh huh` / `嗯` / … ; `bye` / `quit` / … ; `hello.` repeats `hello` |
| `next_tts_cut` / `split_tts_piece` / `ends_sentence` / `sentence_closer_hold_at` | Flush index or `-1`. Sentence end (skip `Dr.` / `1.`); else ~40 chars. `split_tts_piece` returns that piece and the tail, or `None` while a stream should keep buffering. `ends_sentence` is false for `Dr.`, `No.`, a one-letter initial, and `1.`. A closing quote after the stop still counts. `sentence_closer_hold_at` keeps that stop buffered until the next character shows whether a closer follows. |
| `SuiteDefaults` / `LatencySnapshot` / `elapsed_ms` / `MS_PER_S` / `known_tts_model` / `known_latency` / `clamp_num` / `chunk_length_hi` | Shared knobs; millisecond timings (`MS_PER_S` is 1000), no utterance field; optional `trace_id` on the log line. `known_tts_model` lowercases a catalog id. Any other id is kept when it is a single token; whitespace or a control character becomes `s2.1-pro`. `known_latency` accepts `low`/`balanced`/`normal` and keeps the default otherwise. `clamp_num` keeps speed at 0.5–2, temperature / top_p / early_stop at 0–1, `chunk_length` at 100–300 on `api.fish.audio` (1000 self-hosted), and `min_chunk_length` at 0–100. `known_mp3_bitrate` keeps 64 and 192 and snaps anything else to 128. `known_opus_bitrate` keeps -1000, 24000, 32000, 48000, and 64000; anything else snaps to -1000 |
| `DEFAULT_SYSTEM_PROMPT` / `FISH_TTS_MODEL_IDS` / `FISH_LATENCIES` / bound constants | One system prompt. Catalog ids: `s2.1-pro`, `s2.1-pro-free`, `s2-pro`, `s1`, `drama-3-preview`. Bounds: `CHUNK_LENGTH_LO` 100, `CLOUD_CHUNK_HI` 300, `SELF_HOST_CHUNK_HI` 1000, `MIN_CHUNK_LO` 0, `MIN_CHUNK_HI` 100, `TTS_SPEED_LO` 0.5, `TTS_SPEED_HI` 2, `UNIT_LO` 0, `UNIT_HI` 1 |
| `skip_empty_delta` / `utf8_text` | Whitespace-only pieces are not `TextEvent`s. `utf8_text` round-trips with `errors="replace"` |
| `canonical_traceparent` / `trace_id_of` | Normalize a header, or pull its trace id. None when the header is invalid or the trace or span id is all zeros |
| `env_base` / `strip_base` / `env_text` / `env_token` / `env_bool` / `env_off` / `env_int` / `env_float` / `number_or` | Process env. `strip_base` and `env_base` remove surrounding space and trailing slashes. `env_text` strips space and keeps a blank value. `env_token` strips space and keeps the default when blank. A missing key keeps the default. Blank flags and blank or non-numeric numbers keep the default. Flags are true only for `1`/`true`/`yes`/`on`. `env_off` is true only for `0`/`false`/`no`/`off`. `number_or` parses a number and keeps the default on junk |
| `w3c_trace_headers` / `ensure_trace_headers` / `make_traceparent` | Parse, forward-or-mint, and mint W3C `traceparent` (+ optional `tracestate`). No OpenTelemetry |
| `FISH_TTS_PATH` / `FISH_ASR_PATH` / `FISH_RETRY_ATTEMPTS` / `FishHttpError` / `parse_fish_error` / `should_retry_fish_status` / `fish_attempt_exhausted` / `fish_backoff_seconds` / `fish_retry_pause` / `fish_transport_error` / `fish_request_error` / `fish_error_body` / `fish_unreachable` / `fish_non_json` / `fish_non_object` / `parse_asr_body` / `bearer` | Fish `/v1/tts` and `/v1/asr`. `{message, status}` body. Retry 429 and 5xx only. Five attempts, so index 4 is the last. `fish_retry_pause` sleeps `2 ** attempt` and returns True when the caller must stop. Timeout is 504. Other transport failure, a non-JSON body, a non-object body, and a retry loop that ends with no error are 502. `parse_asr_body` returns the object and its text, and raises that 502 when the body or the text field has the wrong shape. `bearer` always prefixes `Bearer ` |
| `CaptionCue` / `format_as_srt` / `format_as_vtt` | Timed ASR cues → SubRip / WebVTT. No network |

Do not add `httpx` / FastAPI / `fishaudio` / sounddevice here.
