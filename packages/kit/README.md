# fish-audio-suite-kit

Text helpers for [Fish Audio](https://fish.audio). Cue tags, TTS and ASR scrubbing, sentence cuts, shared defaults, W3C trace headers, and the Fish error shape.

No HTTP client, no audio, no API key. Part of [fish-audio-suite](../../README.md). Unofficial.

## Quick start

From a clone of the suite:

```bash
uv sync --all-packages
```

```python
from fish_audio_suite_kit import next_tts_cut, normalize_cues, scrub_tts

llm_text = "**Oh** really? (happy) That's [laugh] great. <think>hmm</think> Dr. Smith agrees."

spoken = normalize_cues(scrub_tts(llm_text))
# 'Oh really? [happy] That's [laughing] great. Dr. Smith agrees.'

cut = next_tts_cut(spoken)  # 11, so the first piece is 'Oh really? '
```

Run `scrub_tts` first: it removes markdown, `<think>` blocks and URLs and keeps Fish `[cue]` tags. `normalize_cues` then tidies the cues (`(happy)` becomes `[happy]`, `[laugh]` becomes `[laughing]`).

`cut` is the end index of the next piece, or `-1` while the buffer should keep growing. A sentence end wins. `Dr.`, a one-letter initial, and `1.` do not. Otherwise the cut is the last space before about 40 characters.

To split a whole reply into pieces, call `split_tts_piece(rest, 40, flush_rest=True)` in a loop. It returns `(piece, rest)`, and `flush_rest=True` sends the final fragment even with no sentence end. While a reply is still streaming, pass `flush_rest=False` so an unfinished tail waits for more text (it then returns None until a cut is ready), and use `tts_hold_at` to keep half-finished spans (an open `**`, a URL) out of the send. The voice package's `FishSpeaker.speak_stream` does all of this for you.

Python 3.12 or newer. The package has no runtime dependencies.

## How it works

Stacked leads stay stacked (`[sad][whispering] …` becomes `[sad] [whispering] …`). A mid-sentence `[chuckle]` becomes `[chuckling]`. `[cough]` stays `[cough]`.

Two roleplay helpers are opt-in. `normalize_cues(text, lead=True)` turns a sentence-leading mood word into a cue (`Happy, hello` → `[happy] hello`), and `is_tts_junk(text, drop_narration=True)` drops unquoted stage directions. Both match ordinary English, so they are off by default. `tts_hold_at(..., lead=True)` and `mood_lead_hold_at` are the streaming counterparts of `lead`.

| Group | Names | What they do |
| --- | --- | --- |
| Cues | `normalize_cues`, `official_cue`, `ensure_lead_cue`, `strip_cue_tags`, `split_cues`, `last_emotion`, `MoodCarry` | Lowercase tags, rewrite S1 `(happy)` to `[happy]`, alias `laugh` / `sigh` / `whisper` / `pause`. `official_cue` maps one cue's text to the Fish cue it stands for, or None. `ensure_lead_cue` prepends a cue only when the caller passes a name and the reply has none. `strip_cue_tags` removes known cues and keeps what is spoken. `split_cues` splits text into its cues and the words around them, so a display can style cues without a markup parser. `MoodCarry` puts the last emotion cue in front of a streamed piece that starts a sentence without one, and `last_emotion` finds that cue |
| TTS text | `scrub_tts`, `is_tts_junk`, `extract_quoted_speech`, `utf8_text` | Strip markdown, thoughts, and stage directions. Keep Fish cue tags. `is_tts_junk` means return silence instead of calling Fish. `extract_quoted_speech` keeps only the quoted dialogue. `utf8_text` replaces characters UTF-8 cannot encode |
| ASR text | `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance`, `is_same_utterance`, `asr_tone`, `asr_language_hint`, `is_caption_watermark`, `without_watermark_segments` | Drop timestamps, speaker labels, nospeech, and caption boilerplate. Short answers (`no`, `ok`, `hi`, `DEFAULT_SHORT_WORDS`) are kept. Quit and backchannel phrases (`DEFAULT_QUIT_PHRASES`, `DEFAULT_BACKCHANNELS`) are overridable with `phrases=`. `is_same_utterance` folds case and punctuation to spot a repeat. `asr_tone` lists the tone annotations `transcribe-1-pro` left, such as `laughter`. `asr_language_hint` reduces a tag such as `en-US` to the code Fish accepts. `is_caption_watermark` and `without_watermark_segments` drop the caption lines ASR invents for silence |
| Cuts | `next_tts_cut`, `split_tts_piece`, `ends_sentence`, `tts_hold_at`, `mood_lead_hold_at`, `sentence_closer_hold_at`, `is_empty_delta` | Flush index, or `(piece, tail)` for a stream. `ends_sentence` is true on a real sentence boundary. The `*_hold_at` functions return where a streamed buffer must keep an unfinished span (markup, a sentence-lead mood word, a closing quote that may follow a stop), and `is_empty_delta` is true for a piece with nothing to say |
| Turns | `judge_turn`, `TurnJudgement`, `TurnReason` | Whether a transcript looks like a finished turn, and why. A trailing comma, `and`, `the` or `um` is unfinished. A full stop or a one- or two-word answer is finished. English words only; other languages are judged by punctuation |
| Defaults | `SuiteDefaults`, `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_SEED_EXCHANGE`, `FISH_TTS_MODEL_IDS`, `FISH_LATENCIES`, `clamp_number`, `parse_number`, `normalize_tts_model`, `catalog_tts_model`, `known_latency`, `known_audio_format`, `known_asr_format`, `known_mp3_bitrate`, `known_opus_bitrate`, `chunk_length_hi`, `strip_base` | Speed stays in 0.5–2. MP3 bitrates other than 64, 128 and 192 snap to 128. Cloud `chunk_length` stays in 100–300. Any other base allows up to 1000. Cloud is matched on the parsed hostname, and `self_hosted=` overrides. The bounds are `TTS_SPEED_LO`/`_HI`, `CHUNK_LENGTH_LO`, `CHUNK_LENGTH_CLOUD_HI`, `CHUNK_LENGTH_SELF_HOSTED_HI`, `MIN_CHUNK_LENGTH_LO`/`_HI` and `UNIT_INTERVAL_LO`/`_HI` (temperature, top_p) |
| Env | `env_text`, `env_token`, `env_bool`, `env_off`, `env_int`, `env_float`, `env_base`, `is_insecure_fish_base` | Blank values keep the default. `env_token` reads a single-word setting. Flags are true only for `1` / `true` / `yes` / `on`. Numbers must be plain ASCII decimals. Anything else keeps the default. `is_insecure_fish_base` is true for an `http` base on a non-loopback host |
| Trace | `make_traceparent`, `ensure_trace_headers`, `canonical_traceparent`, `trace_id_of` | W3C `traceparent` only. No OpenTelemetry SDK. `canonical_traceparent` validates a header, and `trace_id_of` pulls out its trace id |
| HTTP shape | `FISH_TTS_PATH`, `FISH_ASR_PATH`, `FISH_RETRY_ATTEMPTS`, `parse_fish_error`, `parse_asr_body`, `fish_error_body`, `retry_after_s`, `describe_transport_error`, `describe_request_error`, `should_retry_fish_status`, `fish_attempt_exhausted`, `fish_backoff_s`, `fish_sleep_before_retry`, `bearer` | Retry 429 and 5xx, up to `FISH_RETRY_ATTEMPTS` (5) tries in all. Backoff is exponential with jitter and honors `Retry-After`. `retry_after_s` is the one parser for that header and drops negative, non-finite and absurd values. `describe_transport_error` maps a timeout or connection failure to a status and a fixed message, never the exception text. `bearer` always prefixes `Bearer ` |
| Errors | `FishHttpError` and `FishAuthError`, `FishRateLimitError`, `FishUpstreamError`, `FishTimeoutError`, all under `FishAudioSuiteError` | `FishHttpError.from_status(status, message)` returns the matching class, so callers catch by type instead of comparing status numbers. `for_unreachable()`, `for_timeout()`, `for_non_json()` and `for_non_object()` build the fixed 502 and 504 errors. `.retryable` is true for 429 and 5xx, and `.retry_after` carries Fish's hint |
| Types | `FishLatency`, `AudioFormat`, `AsrFormat`, `TtsModel`, `ChatRole`, `ChatMessage`, `AsrBody`, `AsrSegment`, `AsrWord`, `OpenAIErrorBody`, `OpenAIErrorDetail`, `FishErrorBody` | Closed value sets and the JSON shapes the kit reads and builds, so callers type against them instead of `str` and `dict[str, Any]` |
| Captions | `CaptionCue`, `format_as_srt`, `format_as_vtt` | Timed text to SubRip or WebVTT |
| Timing | `LatencySnapshot`, `elapsed_ms`, `MS_PER_S` | One turn's times in milliseconds: `asr_ms`, `llm_first_token_ms`, `tts_first_text_ms`, `tts_first_audio_ms`, `voice_to_voice_ms`, `first_audio_ms`, `end_wait_ms` (the silence waited out after the last word), and `trace_id`. `since_stop_ms` adds the last two. `log_line()` prints them without utterance text |

Docstrings on those functions are the contract, and the ones with an Examples section are run as doctests (`pytest --doctest-modules packages/kit/src`).

`kit.__version__` is the installed version.

## License

[MIT](../../LICENSE)
