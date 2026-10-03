# fish-audio-suite-kit

Text helpers for [Fish Audio](https://fish.audio). Cue tags, TTS and ASR scrubbing, sentence cuts, shared defaults, W3C trace headers, and the Fish error shape.

No HTTP client, no audio, no API key. Part of [fish-audio-suite](../../README.md). Unofficial.

## Quick start

From a clone of the suite:

```bash
uv sync --all-packages
```

```python
from fish_audio_suite_kit import normalize_cues, scrub_tts, next_tts_cut

spoken = normalize_cues(scrub_tts(llm_text))
cut = next_tts_cut(spoken)
```

`cut` is the end index of the next piece, or `-1` while the buffer should keep growing. A sentence end wins. `Dr.`, a one-letter initial, and `1.` do not. Otherwise the cut is the last space before about 40 characters.

Python 3.12. The package has no runtime dependencies.

## How it works

Stacked leads stay stacked (`[sad][whispering] …`). A mid-sentence `[chuckle]` becomes `[chuckling]`. `[cough]` stays `[cough]`.

Two roleplay helpers are opt-in. `normalize_cues(text, lead=True)` turns a sentence-leading mood word into a cue (`Happy, hello` → `[happy] hello`), and `is_tts_junk(text, drop_narration=True)` drops unquoted stage directions. Both match ordinary English, so they are off by default. `tts_hold_at(..., lead=True)` is the streaming counterpart of `lead`.

| Group | Names | What they do |
| --- | --- | --- |
| Cues | `normalize_cues`, `ensure_lead_cue`, `strip_cue_tags` | Lowercase tags, rewrite S1 `(happy)` to `[happy]`, alias `laugh` / `sigh` / `whisper` / `pause`. `ensure_lead_cue` prepends a cue only when the caller passes a name and the reply has none. `strip_cue_tags` removes known cues and keeps what is spoken |
| TTS text | `scrub_tts`, `is_tts_junk`, `extract_quoted_speech` | Strip markdown, thoughts, and stage directions. Keep Fish cue tags. `is_tts_junk` means return silence instead of calling Fish |
| ASR text | `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance`, `is_same_utterance` | Drop timestamps, speaker labels, nospeech, and caption boilerplate. Short answers (`no`, `ok`, `hi`) are kept. Quit and backchannel phrases are overridable with `phrases=`. `is_same_utterance` folds case and punctuation to spot a repeat |
| Cuts | `next_tts_cut`, `split_tts_piece`, `tts_hold_at`, `is_empty_delta` | Flush index, or `(piece, tail)` for a stream. `tts_hold_at` is where a streamed buffer must keep an unfinished span, and `is_empty_delta` is true for a piece with nothing to say |
| Defaults | `SuiteDefaults`, `clamp_number`, `parse_number`, `normalize_tts_model`, `catalog_tts_model`, `known_latency`, `known_audio_format`, `known_asr_format`, `chunk_length_hi` | Speed stays in 0.5–2. Cloud `chunk_length` stays in 100–300. Any other base allows up to 1000. Cloud is matched on the parsed hostname, and `self_hosted=` overrides. The bounds are `TTS_SPEED_LO`/`_HI`, `CHUNK_LENGTH_LO`, `CHUNK_LENGTH_CLOUD_HI`, `CHUNK_LENGTH_SELF_HOSTED_HI`, `MIN_CHUNK_LENGTH_LO`/`_HI` and `UNIT_INTERVAL_LO`/`_HI` (temperature, top_p) |
| Env | `env_text`, `env_bool`, `env_off`, `env_int`, `env_float`, `env_base`, `is_insecure_fish_base` | Blank values keep the default. Flags are true only for `1` / `true` / `yes` / `on`. Numbers must be plain ASCII decimals. Anything else keeps the default. `is_insecure_fish_base` is true for an `http` base on a non-loopback host |
| Trace | `make_traceparent`, `ensure_trace_headers` | W3C `traceparent` only. No OpenTelemetry SDK. |
| HTTP shape | `parse_fish_error`, `parse_asr_body`, `retry_after_s`, `describe_transport_error`, `describe_request_error`, `should_retry_fish_status`, `fish_backoff_s`, `fish_sleep_before_retry`, `bearer` | Retry 429 and 5xx. Backoff is exponential with jitter and honors `Retry-After`. `retry_after_s` is the one parser for that header and drops negative, non-finite and absurd values. `describe_transport_error` maps a timeout or connection failure to a status and a fixed message, never the exception text. `bearer` always prefixes `Bearer ` |
| Errors | `FishHttpError` and `FishAuthError`, `FishRateLimitError`, `FishUpstreamError`, `FishTimeoutError`, all under `FishAudioSuiteError` | `FishHttpError.from_status(status, message)` returns the matching class, so callers catch by type instead of comparing status numbers. `for_unreachable()`, `for_timeout()`, `for_non_json()` and `for_non_object()` build the fixed 502 and 504 errors. `.retryable` is true for 429 and 5xx, and `.retry_after` carries Fish's hint |
| Types | `FishLatency`, `AudioFormat`, `AsrFormat`, `TtsModel`, `ChatRole`, `ChatMessage`, `AsrBody`, `AsrSegment`, `OpenAIErrorBody`, `OpenAIErrorDetail`, `FishErrorBody` | Closed value sets and the JSON shapes the kit reads and builds, so callers type against them instead of `str` and `dict[str, Any]` |
| Captions | `CaptionCue`, `format_as_srt`, `format_as_vtt` | Timed text to SubRip or WebVTT |
| Timing | `LatencySnapshot`, `elapsed_ms` | One turn's times in milliseconds: `asr_ms`, `llm_first_token_ms`, `tts_first_text_ms`, `tts_first_audio_ms`, `voice_to_voice_ms`, `first_audio_ms`, and `trace_id`. `log_line()` prints them without utterance text |

Docstrings on those functions are the contract, and the ones with an Examples section are run as doctests (`pytest --doctest-modules packages/kit/src`).

`kit.__version__` is the installed version.

## License

[MIT](../../LICENSE)
