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

`CueStyle` runs both, in that order, in one call: `CueStyle().shape(llm_text)` gives the same `spoken`. Build one style per model with `CueStyle.for_model(model, official=..., lead=...)` and shape every reply with it. Its `mode` is `official` (only Fish's own cues, the rest mapped or removed), `free` (every cue as written) or `directed` (a model that `takes_directions`, such as `drama-3`: directions and paired tags go through as written). A stream passes its neighbours to `shape` (`line_start`, `continued`, `before`, `after`, `sentence_start`) and holds text back with `style.hold_at`, so the shaper and the holds read the same `lead`. `carries_mood` says whether `MoodCarry` applies. The proxy and the voice app shape text only this way.

`cut` is the end index of the next piece, or `-1` while the buffer should keep growing. A sentence end wins. `Dr.`, a one-letter initial, and `1.` do not. Otherwise the cut is the last space before about 40 characters.

To split a whole reply into pieces, call `split_tts_piece(rest, 40, flush_rest=True)` in a loop. It returns `(piece, rest)`, and `flush_rest=True` sends the final fragment even with no sentence end. While a reply is still streaming, pass `flush_rest=False` so an unfinished tail waits for more text (it then returns None until a cut is ready), and use `tts_hold_at` to keep half-finished spans (an open `**`, a URL) out of the send. The voice package's `FishSpeaker.speak_stream` does all of this for you.

Python 3.12 or newer. The package has no runtime dependencies.

## How it works

Stacked leads stay stacked (`[sad][whispering] …` becomes `[sad] [whispering] …`). A mid-sentence `[chuckle]` becomes `[chuckling]`. `[cough]` stays `[cough]`.

Two roleplay helpers are opt-in. `normalize_cues(text, lead=True)` turns a sentence-leading mood word into a cue (`Happy, hello` → `[happy] hello`), and `is_tts_junk(text, drop_narration=True)` drops unquoted stage directions. Both match ordinary English, so they are off by default. `tts_hold_at(..., lead=True)` is the streaming counterpart of `lead`.

| Group | Names | What they do |
| --- | --- | --- |
| Shaping | `CueStyle`, `CueMode` | One frozen value for how cues are shaped (`official`, `free` or `directed`, plus `lead`), built with `CueStyle.for_model`. `shape` scrubs, keeps only quoted speech with `dialogue_only=True`, then normalizes cues. `hold_at` is `tts_hold_at` with the style's `lead` |
| Cues | `normalize_cues`, `official_cue`, `ensure_lead_cue`, `strip_cue_tags`, `split_cues`, `cue_pattern`, `last_emotion`, `MoodCarry` | Lowercase tags, rewrite S1 `(happy)` to `[happy]`, alias `laugh` / `sigh` / `whisper` / `pause` / `sob` / `groan` and the rest (every alias is a known tag, so `(sob)` and `[groan]` are treated like the tags they stand for). With `directed=True`, for a model that `takes_directions` (`drama-3`), `normalize_cues` leaves every direction as written and `scrub_tts` keeps paired tags such as `<emphasis>`. `official_cue` maps one cue's text to the Fish cue it stands for, or None. `ensure_lead_cue` prepends a cue only when the caller passes a name and the reply has none. `strip_cue_tags` removes known cues and keeps what is spoken. `split_cues` splits text into its cues and the words around them, so a display can style cues without a markup parser. `cue_pattern(max_chars, min_chars=1, capture=False, flat=False)` is the regex source for one cue, bounded, for a caller that must find cue spans itself (`flat=True` also refuses a `[` inside). Never write a cue regex of your own. `MoodCarry` puts the last emotion cue in front of a streamed piece that starts a sentence without one, and `last_emotion` finds that cue |
| TTS text | `scrub_tts`, `is_tts_junk`, `extract_quoted_speech`, `utf8_text`, `cut_at_control`, `is_header_safe` | Strip markdown, thoughts, and stage directions. Keep Fish cue tags. `is_tts_junk` means return silence instead of calling Fish: a cue with no words, or fewer than two letters unless it is a short answer (`DEFAULT_SHORT_WORDS`). `extract_quoted_speech` keeps only the quoted dialogue. `utf8_text` replaces characters UTF-8 cannot encode. `cut_at_control` keeps a value to its first line, and `is_header_safe` says whether it can go in an HTTP header as is (`token=True` also refuses a space) |
| ASR text | `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance`, `is_same_utterance`, `asr_tone`, `asr_language_hint`, `is_caption_watermark`, `without_watermark_segments` | Drop timestamps, speaker labels, nospeech, and caption boilerplate. Short answers (`no`, `ok`, `hi`, `DEFAULT_SHORT_WORDS`) are kept; `is_asr_hallucination` and `is_tts_junk` share that floor, except that ASR counts one CJK character beside a letter as speech. Quit and backchannel phrases (`DEFAULT_QUIT_PHRASES`, `DEFAULT_BACKCHANNELS`) are overridable with `phrases=`. `is_same_utterance` folds case and punctuation to spot a repeat. `asr_tone` lists the tone annotations `transcribe-1-pro` left, such as `laughter`. `asr_language_hint` reduces a tag such as `en-US` to the code Fish accepts. `is_caption_watermark` and `without_watermark_segments` drop the caption lines ASR invents for silence |
| Cuts | `next_tts_cut`, `split_tts_piece`, `sentence_cuts`, `ends_sentence`, `tts_hold_at` | Flush index, or `(piece, tail)` for a stream. `sentence_cuts` yields the end of each finished sentence, with no length window. `ends_sentence` is true on a real sentence boundary. `tts_hold_at` returns where a streamed buffer must keep an unfinished span (markup, a sentence-lead mood word, a closing quote that may follow a stop) |
| Turns | `judge_turn`, `TurnJudgement`, `TurnReason` | Whether a transcript looks like a finished turn, and why. A trailing comma, `and`, `the` or `um` is unfinished. A full stop or a one- or two-word answer is finished. English words only; other languages are judged by punctuation |
| Defaults | `SuiteDefaults`, `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_SEED_EXCHANGE`, `FISH_TTS_MODEL_IDS`, `normalize_tts_model`, `takes_directions`, `known_latency`, `known_mp3_bitrate`, `known_opus_bitrate`, `chunk_length_hi`, `strip_base`, `clamp_number`, `parse_number`, `parse_flag` | Speed stays in 0.5–2. MP3 bitrates other than 64, 128 and 192 snap to 128. Cloud `chunk_length` stays in 100–300. Any other base allows up to 1000. Cloud is matched on the parsed hostname. The bounds are `TTS_SPEED_LO`/`_HI`, `CHUNK_LENGTH_LO`, `CHUNK_LENGTH_CLOUD_HI`, `MIN_CHUNK_LENGTH_LO`/`_HI` and `UNIT_INTERVAL_LO`/`_HI` (temperature, top_p). The prompt's mood, sound and pause lists come from the cue tables. `parse_flag` reads `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off` (only `true`/`1`/`false`/`0` with `strict=True`) and returns None for anything else |
| Env | `env_text`, `env_token`, `env_bool`, `env_int`, `env_float`, `env_base`, `is_insecure_fish_base` | Blank values keep the default. `env_token` reads a single-word setting. Flags are true only for `1` / `true` / `yes` / `on`. Numbers must be plain ASCII decimals. Anything else keeps the default. `is_insecure_fish_base` is true for an `http` base on a non-loopback host |
| Trace | `make_traceparent`, `ensure_trace_headers`, `trace_id_of` | W3C `traceparent` only. No OpenTelemetry SDK. `ensure_trace_headers` forwards a valid header or mints one, and `trace_id_of` pulls out its trace id |
| HTTP shape | `FISH_TTS_PATH`, `FISH_ASR_PATH`, `FISH_RETRY_ATTEMPTS`, `parse_fish_error`, `parse_asr_body`, `retry_after_s`, `fish_request_id`, `describe_request_error`, `should_retry_fish_status`, `fish_attempt_exhausted`, `fish_backoff_s`, `bearer` | Retry 429 and 5xx, up to `FISH_RETRY_ATTEMPTS` (5) tries in all. Backoff is exponential with jitter and honors `Retry-After`. `retry_after_s` is the one parser for that header and drops negative, non-finite and absurd values. `fish_request_id` reads Fish's request id from the body or `x-request-id` as one short line. `describe_request_error` maps a timeout or connection failure to a status and a fixed message, never the exception text. `bearer` always prefixes `Bearer ` |
| Errors | `FishHttpError` and its subclasses `FishAuthError`, `FishRateLimitError`, `FishUpstreamError`, `FishTimeoutError` | `FishHttpError(status, message)` returns the matching class, so callers catch by type instead of comparing status numbers. `for_unreachable()`, `for_timeout()`, `for_non_json()` and `for_non_object()` build the fixed 502 and 504 errors. `.retryable` is true for 429 and 5xx, and `.retry_after` carries Fish's hint |
| Types | `FishLatency`, `AudioFormat`, `AsrFormat`, `TtsModel`, `ChatRole`, `ChatMessage`, `AsrBody`, `AsrSegment`, `AsrWord`, `OpenAIErrorBody`, `OpenAIErrorDetail`, `FishErrorBody` | Closed value sets and the JSON shapes the kit reads and builds, so callers type against them instead of `str` and `dict[str, Any]` |
| Captions | `CaptionCue`, `format_as_srt`, `format_as_vtt` | Timed text to SubRip or WebVTT |
| Timing | `elapsed_ms`, `MS_PER_S` | Milliseconds since a `time.perf_counter` reading. The record of a voice turn's times, `LatencySnapshot`, lives in voice (`fish_audio_suite_voice.latency`), which is its only user |

Docstrings on those functions are the contract, and the ones with an Examples section are run as doctests (`pytest --doctest-modules packages/kit/src`).

`kit.__version__` is the installed version.

## License

[MIT](../../LICENSE)
