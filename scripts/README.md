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

Settings come from the environment and `--env-file` (default `.env`): the same
`OPENROUTER_API_KEY`, `FISH_LLM_*` variables `fish-voice` reads. Requests run one at
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
| `--find WORDS` | List public voices whose title matches, most used first, then stop. Use `--language` and `--count` to narrow it |
| `--dry-run` | List the files that would be written and make no requests |

The text goes through the same cleanup the voice app applies (`scrub_tts`, then
`normalize_cues`) before it is sent to Fish. Your key comes from `.env` (`FISH_API_KEY`).
