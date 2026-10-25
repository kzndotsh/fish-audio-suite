# scripts

Small command-line tools for working on the project. They are not part of the
packages and are not published.

## eval_models.py

Sends the same prompts to several chat models and compares how they answer, through
the same backend code `fish-voice` uses, with the same system prompt and opening
example. For each reply it records the time to the first token and the total time,
and checks how well it suits a voice: an opening cue, no trailing cue, a short
length, no markdown and no emoji. The table also shows what each reply cost and which
provider served it, as OpenRouter reports them (other servers report no cost, and the
column shows a dash).

It calls live APIs and spends credits, a few hundredths of a cent per request on a
small model.

**Set the models and prompts** near the top of the file. Edit the `MODELS` and `PROMPTS`
lists, or point `PROMPTS_FILE` at a file (a path from the repository root), which is
then used instead of `PROMPTS`. They run when you pass nothing:

```bash
uv run python scripts/eval_models.py
```

A prompts file looks like this: each prompt can run over several lines, and a line of
three or more dashes starts the next one.

```text
Hey, can you hear me? Tell me something fun.
---
I had a rough day. Can you cheer me up?
Keep it short.
```

**Or pass the prompts for one run.** A flag always wins over the prompt settings at the top. The models always come from the `MODELS` list:

```bash
uv run python scripts/eval_models.py -p "Hey, can you hear me? Tell me something fun."

uv run python scripts/eval_models.py --prompts-file prompts.txt --runs 5 --warmup --out report.md
```

| Flag | Meaning |
| --- | --- |
| `-p`, `--prompt` | A user line. Repeatable. Replaces `PROMPTS` |
| `--prompts-file` | Prompts separated by a line of dashes (`---`). Replaces `PROMPTS` |
| `--system-file` | A character file. The voice rules follow it, or go where `{{default_prompt}}` is in the file, as with `fish-voice --prompt-file` |
| `-n`, `--runs` | Repeats of each prompt on each model (default 1) |
| `--warmup` | One unmeasured request per model first, so connection setup is not timed |
| `--max-tokens`, `--temperature`, `--reasoning-effort`, `--provider-sort` | Override the matching `FISH_LLM_*` setting |
| `--out` | Write a report: `.md` for Markdown (the summary, then every reply), `.json` for the raw runs |
| `--quiet` | Do not print each reply as it arrives, only the summary table |
| `--dry-run` | List what would run and make no requests |

Settings come from the environment and `--env-file` (default `.env`), the same
variables `fish-voice` reads: the provider's key (such as `OPENROUTER_API_KEY`) or
`FISH_LLM_API_KEY`, and the `FISH_LLM_*` settings. Requests run one at
a time, so they do not compete for a rate limit.

## eval_voices.py

Speaks the same text with several Fish voices and TTS models, through `FishSpeaker`
(the code `fish-voice` speaks with), and saves each clip to `tmp/` so you can play them
one after another. The folder is ignored by git. It also prints the time to the first
audio, the total time and the length of each clip.

Each file is named with the run's timestamp, the voice id and the model, so a run's
clips sort together:

```text
tmp/20261004-123456_0123456789abcdef0123456789abcdef_s2.1-pro.wav
```

It calls the live Fish API and spends credits, a few hundredths of a cent for a short line.

**Set what to run** near the top of the file:

| Setting | Meaning |
| --- | --- |
| `VOICES` | Fish voice ids. A `# note` after an id is just for you. Empty uses `FISH_VOICE_ID` from `.env` |
| `TTS_MODELS` | Fish TTS models to try with each voice (default `s2.1-pro`) |
| `TEXT` | What to say. Fish `[cue]` tags work in it |
| `OUT_DIR` | Where clips go, from the repository root (default `tmp`) |

```bash
uv run python scripts/eval_voices.py                       # the settings at the top
uv run python scripts/eval_voices.py -t "[calm] Take a slow breath in, and out."
uv run python scripts/eval_voices.py --find narrator --language en   # list voices to copy ids from
uv run python scripts/eval_voices.py --dry-run             # list the files, make no requests
```

| Flag | Meaning |
| --- | --- |
| `-t`, `--text` / `--text-file` | What to say. Replaces `TEXT` |
| `--latency` | Fish latency mode (default `normal`) |
| `--sample-rate` | Sample rate of the clips (default 44100) |
| `--out-dir` | Write the clips elsewhere |
| `--env-file` | Settings file (default `.env`) |
| `--find WORDS` | List public voices whose title matches, most used first, then stop. `--language` keeps one language, such as `en`, and `--count` sets how many (default 15) |
| `--dry-run` | List the files that would be written and make no requests |

The text goes through the voice app's own speech cleanup in `FishSpeaker.speak` (`scrub_tts`,
then `normalize_cues`), with cues Fish does not support mapped or dropped as `fish-voice` does
by default (`FISH_TTS_CUES=official`). Your key is `FISH_API_KEY`, from `.env` or the environment.

## mic_levels.py

Records your mic for a few seconds and prints its level about every 90 ms, then a
summary: the noise floor, the average level while you talk, the loudest moment and how
far a plosive or bump sits above your speech. It is for tuning how the waveform in
`fish-voice --tui` scales a real mic. It calls no API, sends nothing and saves no audio.
It needs the `speakers` extra. It reads the raw mic: `fish-voice` adds automatic gain and
noise suppression, so a quiet voice reads louder in the app than here.

```bash
uv run python scripts/mic_levels.py --seconds 12   # the default length
uv run python scripts/mic_levels.py --device 3     # a PortAudio device number
```

Stay quiet for the first two seconds, then talk as you would to the assistant. On NixOS,
set `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH` first.

## vad_compare.py

Plays one recording through each voice detector and the real barge-in gate, and shows
where each would have interrupted the bot. For every engine and strictness it prints a
timeline of the frames called speech, a caret under each interruption and a one-line
summary. Typing, coughs and chairs should leave the timeline empty and trip nothing. It
calls no API and sends nothing.

```bash
uv run python scripts/vad_compare.py --record 30 take.wav   # record, then compare
uv run python scripts/vad_compare.py take.wav --engines webrtc,silero --strictness 1,2
uv run python scripts/vad_compare.py take.wav --process
```

| Flag | Meaning |
| --- | --- |
| `wav` | A 16 kHz mono 16-bit WAV. `--record` writes one in that format |
| `--record SECONDS` | Record the mic this long and save it to `wav` first, so every detector hears the same take. Needs the `speakers` extra |
| `--device` | A PortAudio input device number, for `--record` |
| `--engines` | Comma-separated engines (default `webrtc,silero`). A plugin is named as `package.module:factory`. An engine that cannot load is reported as skipped |
| `--strictness` | Comma-separated levels from 0 to 3 (default `1`), as in `FISH_VOICE_VAD` |
| `--process` | Run the take through the app's mic cleanup first (automatic gain and noise suppression, set by `FISH_VOICE_AGC` and `FISH_VOICE_NOISE_SUPPRESSION`), so the detectors hear what `fish-voice` hears. Needs the `aec` extra |

The WAV is the raw mic. A useful take is ten seconds of typing, ten of talking, then
typing while you talk. The gate runs as if the bot were silent, with no echo
cancellation and the default barge-in settings (`FISH_VOICE_BARGE_*` are not read). Silero needs the `silero` extra. On NixOS, set
`LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH` first.

## deepgram_check.py

Streams your mic to Deepgram's Flux the way `FISH_VOICE_STT=deepgram` does, and prints each
turn: the words as they arrive, the final text, and the time from your last speech to the
final text. Use it to try Deepgram, and to see what streaming saves, before turning it on.
It calls the live Deepgram API and spends a little credit, but no Fish or language-model
calls. It needs `DEEPGRAM_API_KEY` (in `.env` or the environment) and the `speakers`,
`vad` and `deepgram` extras. It sends the raw mic, without the app's automatic gain and noise
suppression.

```bash
uv run python scripts/deepgram_check.py --turns 3
uv run python scripts/deepgram_check.py --device 3   # a PortAudio device number
```

Say a sentence, then stop. Ctrl+C ends it. On NixOS, set
`LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH` first.
