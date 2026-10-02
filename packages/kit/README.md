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

Two roleplay helpers are opt-in. `normalize_cues(text, lead=True)` turns a sentence-leading mood word into a cue (`Happy, hello` → `[happy] hello`), and `is_tts_junk(text, drop_narration=True)` drops unquoted stage directions. Both match ordinary English, so they are off by default. `hold_tts(..., lead=True)` is the streaming counterpart of `lead`.

| Group | Names | What they do |
| --- | --- | --- |
| Cues | `normalize_cues`, `ensure_lead_cue`, `strip_cue_tags` | Lowercase tags, rewrite S1 `(happy)` to `[happy]`, alias `laugh` / `sigh` / `whisper` / `pause`. `ensure_lead_cue` prepends a cue only when the caller passes a name and the reply has none. `strip_cue_tags` removes known cues and keeps what is spoken |
| TTS text | `scrub_tts`, `is_tts_junk`, `extract_quoted_speech` | Strip markdown, thoughts, and stage directions. Keep Fish cue tags. `is_tts_junk` means return silence instead of calling Fish |
| ASR text | `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance` | Drop timestamps, speaker labels, nospeech, and caption boilerplate. Short answers (`no`, `ok`, `hi`) are kept. Quit and backchannel phrases are overridable with `phrases=` |
| Cuts | `next_tts_cut`, `split_tts_piece` | Flush index, or `(piece, tail)` for a stream |
| Defaults | `SuiteDefaults`, `clamp_num`, `known_tts_model`, `known_latency`, `chunk_length_hi` | Speed stays in 0.5–2. Cloud `chunk_length` stays in 100–300. Any other base allows up to 1000. Cloud is matched on the parsed hostname, and `self_hosted=` overrides |
| Env | `env_text`, `env_bool`, `env_off`, `env_int`, `env_float`, `env_base`, `is_insecure_fish_base` | Blank values keep the default. Flags are true only for `1` / `true` / `yes` / `on`. Numbers must be plain ASCII decimals, so `"１０"` and `"1_000"` keep the default. `is_insecure_fish_base` is true for an `http` base on a non-loopback host |
| Trace | `make_traceparent`, `w3c_trace_headers`, `ensure_trace_headers` | W3C `traceparent` only. No OpenTelemetry SDK |
| HTTP shape | `parse_fish_error`, `should_retry_fish_status`, `fish_backoff_s`, `fish_sleep_before_retry`, `bearer` | Retry 429 and 5xx. Backoff is exponential with jitter and honors `Retry-After`. `bearer` always prefixes `Bearer ` |
| Captions | `CaptionCue`, `format_as_srt`, `format_as_vtt` | Timed phrases to SubRip or WebVTT |

Docstrings on those functions are the contract.

## License

[MIT](../../LICENSE)
