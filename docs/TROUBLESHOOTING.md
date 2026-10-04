# Troubleshooting

Find your symptom, check the log line it names, and apply the fix.

> [!TIP]
> Most answers need the debug log. Run `./packages/voice/dev.sh --debug` before anything else.

## First steps

1. **Turn on the debug log.** Run `./packages/voice/dev.sh --debug`, or `fish-voice --debug`. Use `--trace` to also see mic levels every 20 frames, raw websocket events and HTTP lines. Logs go to stderr, and the conversation goes to stdout.
2. **Check Fish on its own.** `fish-voice --smoke` synthesizes one line into a WAV file. If that fails, the problem is the key, the voice or the network, not your audio setup.
3. **Check what was loaded.** The first line printed shows which env file was read (`env: /path/.env`). The `fish-voice ready | ...` line shows the TTS model, voice, latency mode, playback and LLM in use.

## Audio devices

### No sound, or `PortAudioMissingError` at start

PortAudio is a C library that `uv` and `pip` do not install.

```bash
sudo apt install libportaudio2   # Debian, Ubuntu
sudo dnf install portaudio       # Fedora
brew install portaudio           # macOS
```

On NixOS, run `./packages/voice/dev.sh` or `nix run .#fish-audio-suite-voice`. Both put PortAudio on the library path.

### Sound comes out of the wrong device, or the mic hears nothing

List the devices, then set `FISH_VOICE_DEVICE` to an index or a name:

```bash
uv run --package fish-audio-suite-voice --extra cli python -m sounddevice
```

In `--trace`, `listen mic ... peak_rms=` sits around 50–100 in a quiet room, and speech reaches several hundred. A value of exactly `0` on every line means no signal at all: the mic is muted (many USB mics have a mute button) or muted in the OS mixer.

### `BLOCKER: ...` and exit code 2

Something required is missing. The line names it: `FISH_API_KEY`, `FISH_VOICE_ID`, the LLM key or model, or a playback mode that cannot work in duplex (`file`, or `mpv` not on your PATH). Set it in `.env` and run again.

## Fish

### Exit code 2 after `[asr] 401`, `402` or `403`, or `[tts] 401`, `402` or `403`

Fish refused the key. 401 means a wrong key, 402 an empty balance and 403 a key without access. Check `FISH_API_KEY` and your Fish account.

> [!NOTE]
> These stop the program on purpose, because no later turn can succeed.

### `[tts silent] voice=... model=...` or `[tts] no audio`

Fish accepted the request but sent no audio. Usually `FISH_VOICE_ID` is wrong, private, or not available on the chosen model. Try the voice on fish.audio, or switch `FISH_TTS_MODEL`.

### `[tts] retry status=429 attempt=...` or `5xx`

Fish is rate-limiting you or is having trouble. Retries happen only before the first audio byte, so you never hear a sentence twice. If it keeps happening, slow down or check Fish's status page.

### `asr done ... lang=fi` (or `tl`) on English speech

Fish labelled a short clip, often one starting with "Uh", as another language. The transcript is still correct English, so you can ignore it. Setting `FISH_ASR_LANGUAGE=en` stops the guessing.

## Listening and turn-taking

### Short answers ("yes", "no") are ignored

Look for `listen reject too_little_voice frames=... voiced_hits=7`. A one-word answer has about 200–250 ms of voice, and the default needs 360 ms (`FISH_VOICE_MIN_VOICED_FRAMES=12`). Set it to `7`. If coughs and desk taps then start turns, use `8` or `9`.

### The bot cuts in while I pause mid-sentence

Your pause was longer than the end-of-speech silence, `FISH_VOICE_SILENCE_FRAMES` (frames are 30 ms; the default 40 is 1.2 s). Raise it. `30` (0.9 s) is a good middle ground if you have lowered it for speed.

### My quick reply right after the bot finishes is lost

The mic stays shut for `FISH_VOICE_COOLDOWN` seconds (default 0.8) after a reply, and anything said then is not recorded. Set `0.3`. If that makes the bot hear itself (see below), use `0.4` or `0.5`.

### A turn starts with no one talking, right after the bot finishes

In the log, `listen speech_start` comes within a few hundred ms of `listen open`, and the transcript is empty, junk, or the bot's own last words. The speaker's echo reached the mic before the room went quiet. Raise `FISH_VOICE_COOLDOWN`, lower the speaker volume, or move the mic away from the speaker.

### `asr.skip backchannel` or `asr.skip hallucination`

These are deliberate. A transcript that is only a hesitation ("um", "hmm") is skipped, as are lines speech-to-text invents in silence, such as "Thanks for watching". Said over the bot, "yeah" and "mm-hmm" are skipped as well, because they mean "go on". On a turn of its own, "yeah" is answered.

## Barge-in

### Coughs, chairs or keyboard noise interrupt the bot

The log shows `barge interrupt` after hits from short, loud noises. Raise `FISH_VOICE_BARGE_FRAMES` (default 10, about 300 ms) before raising `FISH_VOICE_BARGE_RMS`. A noise that interrupts but holds no words is treated as a false alarm: the reply resumes where it was cut.

### The bot interrupts itself

`barge hit` lines appear while no one is talking, with `far=True`. Echo cancellation is missing the speaker's sound. Check that `aec on AEC3` appears at start. `aec skip extra pywebrtc-audio not installed` means it is missing: install the `cli` extra, which includes it. Also check that `FISH_VOICE_AEC` is not `0`. Lower the volume, or raise `FISH_VOICE_BARGE_FRAMES`. Values below about 8 let echo through.

### Talking over the bot does nothing

You were not loud enough for long enough. Check `barge mic rms=... need=...` in `--trace`: your voice has to beat `need` for `FISH_VOICE_BARGE_FRAMES` frames in a row. Lower `FISH_VOICE_BARGE_FRAMES` a little, or speak up. Without echo cancellation, barge-in also waits out the speaker bleed (`FISH_VOICE_BLEED_DELAY`, 0.9 s).

## Playback

### Clicks, gaps or stutters

Look for `tts underrun after N kB played`. The speaker ran out of audio between chunks. Playback runs on its own thread, so this should be rare. If it shows up, a very slow model is the usual cause: Fish is waiting for text. Use streaming (`FISH_VOICE_STREAM_TTS=1`) and a faster model.

Not every audio system reports underruns: on PipeWire the line can be missing even when the speaker briefly ran dry. If you hear a blip at sentence ends with no underrun line, compare the `tts audio` arrival times with how long each chunk plays (bytes ÷ 88,200 per second at 44.1 kHz). A chunk that arrives just as the previous one ends is the cause. `FISH_LATENCY=balanced` makes Fish deliver sooner.

### A sigh, laugh or odd noise after the last sentence

The model ended its reply with a cue such as `[sigh]`. The default prompt tells it not to, but some roleplay models still do. Check the `llm ▸` line. If it keeps happening, add "Never end a reply with a cue" to your character file.

### The bot talks about its cues ("you asked me to whisper")

A small model read an earlier `[whispering]` cue in the chat history as something you asked for, and explained or obeyed it. The default prompt says cues are silent stage directions that must never be mentioned. A character file gets the same rules appended, but a small model can still slip. Use a larger model, or repeat the rule in your character file.

## The LLM

The warning line reads `[llm] HTTP <status> from <provider>, model=<model>: <message>`, with the provider's own message unwrapped.

### `HTTP 429` (rate limited)

- **"temporarily rate-limited upstream":** the model's shared provider pool is busy. It is retried once after about 1 s. Single-provider models such as `thedrummer/cydonia-24b-v4.1` hit this most. Pick a model served by several providers.
- **A quota or credit message** (`insufficient_quota` on Experiential): your account is out of credit. This is not retried; top up.

### `HTTP 403` before the model even runs

Some providers filter requests before the model sees them (xAI does). The model is not refusing; the provider is. Switch to another model or provider.

On Experiential, `403 model_not_granted` means the model id is wrong or not on your plan. A leftover OpenRouter suffix such as `:nitro` causes it too; `FISH_LLM_NITRO` only applies to OpenRouter.

### The reply is a polite deflection or a refusal

That is the model's own content policy, and settings will not change it. [PERFORMANCE.md](PERFORMANCE.md#choosing-an-llm) lists roleplay-tuned models.

### `[llm] empty reply (model=... finish=...)`

The model returned no text.
- **`finish=length`:** a thinking model may have spent the whole `FISH_LLM_MAX_TOKENS` budget thinking. A provider's usage page shows it as nearly all reasoning tokens (for example `1,500 · 1,492 reasoning`) and an "incomplete" status. Set `FISH_LLM_REASONING_EFFORT=low`, or `none` where the route accepts it. Raising `FISH_LLM_MAX_TOKENS` alone rarely helps, and a long roleplay prompt can still draw heavy reasoning at `low`, so a model without hidden reasoning is the reliable fix.
- **Local models:** some community builds ship an empty chat template and answer nothing. Use another build.

### `[llm] unknown model ...`

OpenRouter does not know the model id. Check the spelling on openrouter.ai, including the `author/` prefix.

### `unsupported_parameter` naming the reasoning effort

`The effort parameter value 'none' is not supported by this model route. Supported values: 'low', 'high', 'max'.` Not every route accepts every `FISH_LLM_REASONING_EFFORT` value. Use one the message lists, usually `low`.

### Replies take 30 seconds or more, or fail with a 502 and an HTML page

The model is served by one provider and that provider is overloaded or down. Replies arrive late, all at once, or end in `HTTP 502` with the provider's error page in the body. `fish-voice` waits up to `FISH_LLM_TIMEOUT` seconds (default 120) for each reply, so a hung model looks like the bot not hearing you. Switch to a model with several providers. A test request with `curl` shows whether the route itself is slow:

```bash
time curl -s https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" -H 'content-type: application/json' \
  -d '{"model":"author/model","messages":[{"role":"user","content":"Say hi."}]}'
```

### The first reply is very slow, then fine

- **Local models:** the first request loads the model into memory (15–20 s for a mid-sized model). Set `OLLAMA_KEEP_ALIVE=-1` on the Ollama server to keep it loaded.
- **Cloud models:** the first turn opens new connections. Later turns reuse them for 2 minutes.

### Replies are sometimes fast and sometimes slow

Compare `llm first token` across turns, and the `provider=` on each `llm stream_end` line. A spread from 0.4 s to 2.6 s on one provider is that provider's load. See [PERFORMANCE.md](PERFORMANCE.md#choosing-an-llm) for picking a steadier model.

### `fish-voice: ... is not used because the base is not https`

A provider's own key is never sent over plain `http`, where anyone on the network could read it. Use an `https` base, or set `FISH_LLM_API_KEY` to send a key anyway, for example to a local server.

## The proxy

| Response | Meaning | Fix |
| --- | --- | --- |
| `503 proxy has no FISH_API_KEY configured` | The proxy started without a Fish key | Set `FISH_API_KEY` in its environment or env file, then restart |
| `401 Invalid API key` | `FISH_PROXY_API_KEYS` is set, and the client sent no key or a wrong one | Send one of those keys as `Authorization: Bearer ...` |
| `400 unsupported response_format ...` | Fish cannot produce that format | Use `mp3`, `opus`, `wav`, `pcm` or `pcm16` |
| `400 input is longer than N characters` | The text is over `FISH_PROXY_MAX_INPUT_CHARS` | Send shorter text, or raise the cap |
| `413 request body exceeds N bytes` | The body is over `FISH_PROXY_MAX_BODY_BYTES` (25 MiB) | Raise it for long recordings; an hour of 128 kbps MP3 is about 55 MiB |
| `400 invalid JSON body` | Not JSON, not UTF-8, or nested too deeply | Fix the client |
| `502` or `504`, "Fish did not answer before the retry deadline" | Fish was down, slow or unreachable | Check Fish's status; for long transcriptions, raise `FISH_PROXY_ASR_TIMEOUT` |

**Roleplay narration is read aloud** (`*walks in*`, "She smiles."): set `FISH_TTS_DIALOGUE_ONLY=1` to speak only the quoted dialogue. `FISH_TTS_DROP_NARRATION=1` also turns a line that is only narration into silence. Both are off by default because they can change ordinary sentences. `/health` shows the values in effect.

**A browser app fails with `NetworkError when attempting to fetch resource`** (Firefox) or `Failed to fetch` (Chrome): the page called an API directly from the browser, and that server sent no CORS headers. This happens with the proxy, and with LLM providers too. Use the app's desktop build or its server-side proxy setting, or add CORS headers in a reverse proxy. See [INTEGRATIONS.md](INTEGRATIONS.md#apps-that-run-in-the-browser).

**The proxy will not start and names `FISH_PROXY_API_KEYS`:** the variable is set but holds no key. This is on purpose, so a typo cannot turn authentication off. Remove the variable, or give it a key.

**Occasional 502 errors behind nginx or a load balancer:** the proxy closed an idle connection that the balancer still meant to reuse. Set `FISH_PROXY_KEEP_ALIVE` higher than the balancer's idle timeout, for example `65`.

**The Docker container takes 10 seconds to stop and cuts off speech:** run it with `--stop-timeout 130` (in compose, `stop_grace_period: 130s`). In-flight replies get up to `FISH_PROXY_GRACEFUL_SHUTDOWN` seconds (default 120) to finish.

## Still stuck

Open an issue with:
- the `fish-voice ready` line, or the proxy's start-up lines
- the `--debug` log around the problem, with keys removed (the log never prints keys, but check anyway)
- your OS and how you run it (`dev.sh`, `uv run`, Docker or Nix)

> [!CAUTION]
> For security problems, follow [SECURITY.md](../SECURITY.md) instead of opening a public issue.
