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

Fish labelled a short clip, often one starting with "Uh", as another language. The transcript is still correct English, so you can ignore it. Fish detects the language whether or not you send a hint, and with `transcribe-1-pro` the hint does not change the transcript, so `FISH_ASR_LANGUAGE=en` may not change the label.

## Listening and turn-taking

### Short answers ("yes", "no") are ignored

Look for `listen reject too_little_voice frames=... voiced_hits=7`. A one-word answer has about 200–250 ms of voice, and the default needs 360 ms (`FISH_VOICE_MIN_VOICED_FRAMES=12`). Set it to `7`. If coughs and desk taps then start turns, use `8` or `9`. With `FISH_VOICE_STT=deepgram` the same setting decides how much voice there must be before anything is sent to Deepgram. A barge-in clip counts in full, so a short interruption still gets through.

### Soft speech is missed, or talking over the bot does nothing, because the mic is quiet

Look for `listen reject too_little_voice`, short clips that come back as `asr skip hallucination (0 chars)`, and `barge hit` lines that sit just above the floor (about 220 to 330 rms) and fall back before they reach `FISH_VOICE_BARGE_FRAMES`. The default gates are `FISH_VOICE_MIN_RMS=200` and `FISH_VOICE_BARGE_RMS=220`, about -44 dBFS. They assume your voice is well above that and the room is well below it.

Measure before changing anything. `scripts/mic_levels.py` calls no API and saves no audio, and its summary gives your noise floor and speaking level in dB below full scale. Stay quiet for the first two seconds and then talk for the rest, or the "speech" figure is only room noise. Conversational speech usually averages around -26 to -20 dBFS. In one setup (a Yeti X) speech averaged -37.7 dBFS and soft words dipped under the gates. A gain increase of about 8 dB brought it to -31.4 and fixed the missed interruptions.

Things to consider:

- **Order of fixes.** Move closer to the mic, then raise the mic's own gain, then the system input volume. Software gain amplifies the room noise by the same amount, so it cannot improve speech against noise. Moving closer is the only step that does.
- **Do not overshoot.** Keep the noise floor below about -47 dBFS (an rms of roughly 140). Near the 200 rms gate the room itself starts to look like speech. In the same setup the floor rose from -55.4 to -48.8 dBFS with the extra gain.
- **Volume scales are cubic.** In `pavucontrol` and `wpctl`, 60% is about -13 dB and 44% is about -21 dB (dB = 60 x log10 of the fraction). A linear setting such as WirePlumber's `default-source-volume` is the cube of that percentage, so 0.22 matches 60%.
- **Other apps share the input.** A chat or call app on the same default source hears the change. On PipeWire the volume is remembered per device, so a value set in a config file applies only when nothing is saved.
- **Echo cancellation takes some of your voice** while the speakers play, so your level during a reply is lower than in `mic_levels.py`. Headphones remove that. The measurement above is of the raw mic, not of what is left after cancellation.
- **Lowering the gates instead** (for example `FISH_VOICE_MIN_RMS` near 120 and `FISH_VOICE_BARGE_RMS` near 150) is untested and relies on a quiet room, so prefer fixing the level first.

### Replies are too quiet, too loud or distorted with loudness normalisation

`FISH_VOICE_NORMALIZE_LOUDNESS=1` moves each reply toward `FISH_VOICE_LOUDNESS_TARGET_DB` by at most 12 dB, so a very quiet voice stays a little quiet, and a limiter bends peaks rather than clipping them. Lower the target (`-22`) if a boosted voice sounds harsh, or turn the setting off to hear the voice as Fish sends it. The level is measured on the speech so far, so it can settle a second into a reply.

### The bot cuts in while I pause mid-sentence

Your pause was longer than the end-of-speech silence, `FISH_VOICE_SILENCE_FRAMES` (frames are 30 ms; the default 40 is 1.2 s). Raise it. `30` (0.9 s) is a good middle ground if you have lowered it for speed. With Fish ASR you can also set `FISH_VOICE_TURN_CHECK=1`, which keeps listening when your words stop on "and", "but" or a comma.

### The turn check waits when I had finished, or answers too soon

Run with `--debug` and look for `turn.extend <reason> clip=1 '...'`. The reason says what the words looked like: `open_punctuation` (a comma, a colon or an ellipsis), `dangling_word` ("and", "but", "because", "the", "um", a preposition) or `number_list`. Fish ASR sometimes ends a finished sentence on a comma, and the check then waits for a continuation, at most `FISH_VOICE_TURN_EXTEND_FRAMES` less `FISH_VOICE_TURN_SILENCE_FRAMES` of quiet (about 0.84 s). No `turn.extend` line means the words looked finished and the short `FISH_VOICE_TURN_SILENCE_FRAMES` ended the clip: raise it (`18` is about 0.54 s) if it still cuts you off. The check reads English only, and does nothing with `FISH_VOICE_STT=deepgram`.

### My quick reply right after the bot finishes is lost

The mic stays shut for `FISH_VOICE_COOLDOWN` seconds (default 0.8) after a reply, and anything said then is not recorded. Set `0.3`. If that makes the bot hear itself (see below), use `0.4` or `0.5`.

### A turn starts with no one talking, right after the bot finishes

In the log, `listen speech_start` comes within a few hundred ms of `listen open`, and the transcript is empty, junk, or the bot's own last words. The speaker's echo reached the mic before the room went quiet. Raise `FISH_VOICE_COOLDOWN`, lower the speaker volume, or move the mic away from the speaker. With `--debug`, the `listen cooldown` line after each reply shows how close the echo came: `speech_like` near `start_needs` means it could start a turn.

### `asr.skip backchannel` or `asr.skip hallucination`

These are deliberate. A transcript that is only a hesitation ("um", "hmm") is skipped, as are lines speech-to-text invents in silence, such as "Thanks for watching". Said over the bot, "yeah" and "mm-hmm" are skipped as well, because they mean "go on". On a turn of its own, "yeah" is answered.

## Streaming speech recognition (Deepgram)

### `BLOCKER: FISH_VOICE_STT=deepgram needs ...`

Set `DEEPGRAM_API_KEY`, and install the `deepgram` extra (`uv sync --extra deepgram`, or `pip install 'fish-audio-suite-voice[deepgram]'`). The Nix package already includes it.

### `[stt] could not reach Deepgram: ... CERTIFICATE_VERIFY_FAILED`

The WebSocket could not verify Deepgram's certificate. The connection uses `certifi`'s certificates, so this means `certifi` is missing or a proxy is rewriting TLS. That turn is heard with Fish ASR instead, from the speech already captured.

### `stt.empty Flux was sent N kB (X s of audio) and found no speech`

Flux heard the audio and recognised no words, so it never started a turn. The usual cause is a quiet mic: speech around -50 to -40 dBFS (an rms of roughly 100 to 300) was missed in testing. Raise the input volume or the mic's gain, and set `FISH_VOICE_STT_SAVE_DIR=tmp/stt` to save what was sent and listen to it. A `stt.noise` line instead means the local gate dropped a knock or a cough before anything was sent. A `stt.gave_up` line means Flux returned no words within 1.2 s of streaming starting, so the turn was dropped as noise early, as with repeated taps.

### Turns end late, with `stt end trigger=manual`

Flux was not sure enough that you had finished (its `confidence` was below `FISH_VOICE_EOT_THRESHOLD`), so the silence limit (`FISH_VOICE_SILENCE_FRAMES`) ended the turn. Lower `FISH_VOICE_EOT_THRESHOLD` (0.5 at the least) to let Flux end it sooner.

## Barge-in

### Coughs, chairs, sniffs or keyboard noise interrupt the bot

The log shows `barge interrupt` after hits from short, loud noises. Raise `FISH_VOICE_BARGE_FRAMES` (default 10, about 300 ms) before raising `FISH_VOICE_BARGE_RMS`. A noise that interrupts but holds no words is treated as a false alarm: the reply resumes where it was cut.

Hits do not have to be in a row. The count falls by one only after 3 missed frames, so a run of small sounds such as sniffs, breathing or typing can add up to an interruption. The default voice detector (`FISH_VOICE_VAD_ENGINE=webrtc`) is lenient enough to call those sounds voiced.

These help:

- `FISH_VOICE_VAD_ENGINE=silero` (the `silero` extra) uses a neural detector that tells speech from clicks and noise. Check it first with `scripts/vad_compare.py`: record some typing and some talking, and it shows where each detector would have interrupted.
- `FISH_VOICE_VAD=2` makes the detector pickier. Listening and barge-in both use it, so check that your first words are not clipped and that soft speech still arrives. If they are, go back to `1`.
- `FISH_VOICE_BARGE_RMS` set above the level of those sounds and below your voice. Compare the `barge hit ... rms=` lines for a sniff and for a real interruption first. In that setup sniffs read 245 to 650 and real interruptions 740 or more, but your mic will differ.

### The bot interrupts itself

`barge hit` lines appear while no one is talking, with `far=True`. Echo cancellation is missing the speaker's sound. Check that `aec on AEC3` appears at start. `aec skip extra pywebrtc-audio not installed` means it is missing: install the `cli` extra, which includes it. Also check that `FISH_VOICE_AEC` is not `0`. Lower the volume, or raise `FISH_VOICE_BARGE_FRAMES`. Values below about 8 let echo through.

### Talking over the bot does nothing

You were not loud enough for long enough. Check `barge mic rms=... need=...` in `--trace`: your voice has to beat `need` for `FISH_VOICE_BARGE_FRAMES` frames in a row. Lower `FISH_VOICE_BARGE_FRAMES` a little, or speak up. Without echo cancellation, barge-in also waits out the speaker bleed (`FISH_VOICE_BLEED_DELAY`, 0.9 s).

## Playback

### Clicks, gaps or stutters

Look for `tts underrun after N kB played`. The speaker ran out of audio between chunks. Playback runs on its own thread, so this should be rare. If it shows up, a very slow model is the usual cause: Fish is waiting for text. Use streaming (`FISH_VOICE_STREAM_TTS=1`) and a faster model.

Not every audio system reports underruns: on PipeWire the line can be missing even when the speaker briefly ran dry. If you hear a blip at sentence ends with no underrun line, compare the `tts audio` arrival times with how long each chunk plays (bytes ÷ 88,200 per second at 44.1 kHz). A chunk that arrives just as the previous one ends is the cause. `FISH_LATENCY=balanced` makes Fish deliver sooner.

In `--tui` the screen shares the process with playback. It asks for a bigger sound-card buffer (`FISH_VOICE_OUTPUT_LATENCY`, `high` there by default), lets the playback thread in sooner, and collects garbage less often. An underrun line now also shows `chunk_wait` (the audio arrived late), `in_python` (the thread was busy before the write) and `write` (the call itself waited, usually for Python's lock).

### A faint click at the start or end of a sentence

Fish can begin a sentence on a loud sample right after silence, and stop one while it is
still loud, and the step between sound and nothing is heard as a tick. `fish-voice` fades
the first and last 4 ms of each sound next to silence, which removes it. The fade is on
by default (`FISH_VOICE_FADE_MS=4`). If you still hear clicks, try `8`. If words seem to
start or end too softly, lower it, and `0` turns the fade off. The fade only changes the
few milliseconds at a sentence edge, never the length of the audio, and it applies to the
`pcm` format only. It holds back about 10 ms of audio to see what follows.

### A voice hums or says "mm" between sentences

Look at the `tts say ...` lines in a `--debug` log for cues that are descriptions, such as `[smiling]`, `[smiling wider]` or `[soft chuckle]`. Some voices act those out as a sound instead of changing tone. In one test the same text sounded cleanest with plain cues (`[happy]`, `[curious]`), worse with none, and worst with the descriptions. Keep `FISH_TTS_CUES=official` (the default), which maps those cues to the nearest official one. A sentence the model wrote without a cue now gets the previous mood automatically, since a piece with no cue can start cold. If the sound is still there with only official cues, it is probably the voice: try the same text with another voice.

### A sigh, laugh or odd noise after the last sentence

The model ended its reply with a cue such as `[sigh]`. The default prompt tells it not to, but some roleplay models still do. Check the `llm ▸` line. If it keeps happening, add "Never end a reply with a cue" to your character file.

### The bot talks about its cues ("you asked me to whisper")

A small model read an earlier `[whispering]` cue in the chat history as something you asked for, and explained or obeyed it. The default prompt says cues are silent stage directions that must never be mentioned. A character file gets the same rules appended, but a small model can still slip. Use a larger model, or repeat the rule in your character file.

## The LLM

The warning line reads `[llm] HTTP <status> from <provider>, model=<model>: <message>`, with the provider's own message unwrapped.

### `[llm] stopping: ...` and exit code 2

The model cannot answer any turn until a setting changes, so the app stops instead of going back to listening in silence. The line says what to fix:

- `unknown model ...` or `no provider serves model ...`: OpenRouter has no such model, or keeps its page but no provider hosts it any more. A model can still be listed while its endpoints list is empty. Choose another `FISH_LLM_MODEL` that shows at least one provider. The app checks this at start on the OpenRouter backend, and on any backend when a reply gets HTTP 404.
- `the key was refused`: HTTP 401. Check `FISH_LLM_API_KEY` or the provider's key variable.
- `the account is out of credits`: HTTP 402. Add credits.

A 400, 403, 429 or 5xx ends only that reply, since the next request can succeed.

### `HTTP 429` (rate limited)

- **"temporarily rate-limited upstream":** the model's shared provider pool is busy. It is retried once after about 1 s. Single-provider models such as `thedrummer/cydonia-24b-v4.1` hit this most. Pick a model served by several providers.
- **A quota or credit message** (`insufficient_quota` on Experiential): your account is out of credit. This is not retried; top up.

### `HTTP 403` before the model even runs

Some providers filter requests before the model sees them (xAI does). The model is not refusing; the provider is. Switch to another model or provider.

On Experiential, `403 model_not_granted` means the model id is wrong or not on your plan. A leftover OpenRouter suffix such as `:nitro` causes it too; `FISH_LLM_NITRO` only applies to OpenRouter.

### The reply is a polite deflection or a refusal

That is the model's own content policy, and settings will not change it. [PERFORMANCE.md](PERFORMANCE.md#choosing-an-llm) lists roleplay-tuned models.

### `HTTP 400 ... provider rejected the request` and the bot says nothing

`W llm HTTP 400, model=...: provider rejected the request; verify the request fields against the model alias capabilities`, then `tts stream empty reply, nothing to speak`. The wording points at the request fields, but a hosted model can send the same 400 when its safety filter refuses the prompt text. In one setup every field and message shape returned 200 for the model with a plain test prompt, and only the real character prompt was rejected.

To tell them apart, send the same fields with a plain prompt, or the default prompt (`fish-voice` without `--prompt-file`). A 200 there means the content was refused, and no setting will change that. Use a model that accepts the prompt, such as a roleplay-tuned one ([PERFORMANCE.md](PERFORMANCE.md#choosing-an-llm)).

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

### The reply stops in the middle of a sentence

The last `tts say` line is a fragment such as `'[calm] Just'`, and `llm stream_end finish=stop` shows a small `chunks=` count (2 or 3, where other servers show 20 to 50). `finish=stop` means the server ended the reply, not a length limit (that is `finish=length`).

To see who cut it, run with `--trace` (or `FISH_VOICE_LOG_DIR`, which keeps the lines in a file). Each chunk the server sent is logged as `llm.event {...}`, and a stream that closed without the end marker logs `llm.event stream closed without [DONE]`.

- The last `llm.event` has `finish_reason: "stop"` and the text on screen also stops there: the model or the server ended the reply. Nothing here cut it.
- The events hold more text than was spoken: the cleanup removed it, for example an unclosed `(`, `*` or `[`.
- The stream closed without `[DONE]`, or a `llm.event` line is missing words that arrived: the connection ended early.

Compare `out=` with the characters spoken. About 3.5 to 4 characters per token is normal for English. A much lower ratio means text went missing after the model wrote it. `FISH_LLM_CONTINUE=1` sends one more request when a reply stops mid-sentence with `finish=stop`.

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
- the `--debug` log around the problem, with keys removed (the log never prints keys, but check anyway). To capture a whole run without watching the screen, set `FISH_VOICE_LOG_DIR=logs` (or pass `--log-dir logs`): each run writes a complete log there, at the most detailed level. It also holds what you said and what the model answered, so read it before sharing
- your OS and how you run it (`dev.sh`, `uv run`, Docker or Nix)

> [!CAUTION]
> For security problems, follow [SECURITY.md](../SECURITY.md) instead of opening a public issue.
