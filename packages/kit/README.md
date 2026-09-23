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

Stacked leads stay stacked (`[sad][whispering] …`). A mid-sentence `[chuckle]` becomes `[chuckling]`. `[cough]` stays `[cough]`. A mood word becomes a cue only when it leads the sentence (`Happy, hello`).

| Group | Names | What they do |
| --- | --- | --- |
| Cues | `normalize_cues`, `ensure_lead_cue` | Lowercase tags, rewrite S1 `(happy)` to `[happy]`, alias `laugh` / `sigh` / `whisper` / `pause`. `ensure_lead_cue` prepends `[clear]` only when the reply has no cue at all |
| TTS text | `scrub_tts`, `is_tts_junk`, `extract_quoted_speech` | Strip markdown, thoughts, and stage directions. Keep Fish cue tags. `is_tts_junk` means return silence instead of calling Fish |
| ASR text | `scrub_asr`, `is_asr_hallucination`, `is_backchannel`, `is_quit_utterance` | Drop timestamps, speaker labels, nospeech, and caption boilerplate |
| Cuts | `next_tts_cut`, `split_tts_piece` | Flush index, or `(piece, tail)` for a stream |
| Defaults | `SuiteDefaults`, `clamp_num`, `known_tts_model`, `known_latency` | Speed stays in 0.5–2. Cloud `chunk_length` stays in 100–300. Any other base allows up to 1000 |
| Env | `env_text`, `env_bool`, `env_off`, `env_int`, `env_float`, `env_base` | Blank values keep the default. Flags are true only for `1` / `true` / `yes` / `on` |
| Trace | `make_traceparent`, `w3c_trace_headers`, `ensure_trace_headers` | W3C `traceparent` only. No OpenTelemetry SDK |
| HTTP shape | `parse_fish_error`, `should_retry_fish_status`, `fish_backoff_seconds`, `bearer` | Retry 429 and 5xx. `bearer` always prefixes `Bearer ` |
| Captions | `CaptionCue`, `format_as_srt`, `format_as_vtt` | Timed phrases to SubRip or WebVTT |

Docstrings on those functions are the contract.

## License

[MIT](../../LICENSE)
