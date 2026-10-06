# Performance

How fast `fish-voice` answers, where the time goes, what the code already does about it, and the settings that trade a little quality for speed. The proxy has its own section at the end.

## Where a turn's time goes

A turn starts when you stop talking and ends when you hear the first word of the reply. With good settings it takes about 1.1 to 1.6 seconds. Here is one real turn with the settings recommended below:

| Step | Time | What decides it |
| --- | --- | --- |
| Silence that ends your turn | 0.75 s | `FISH_VOICE_SILENCE_FRAMES` |
| Fish speech-to-text | 0.15–0.3 s | Fish, clip length, a warm connection |
| LLM first token | 0.3–0.7 s | The model and the provider serving it |
| First sentence written | about 0.2 s | The model's tokens per second |
| Fish text-to-speech first audio | about 0.4 s | `FISH_LATENCY` |

The Fish websocket opens while the LLM is still thinking, so its setup time (about 0.3 s) is hidden. The LLM is usually the step that varies most from one turn to the next.

With `FISH_VOICE_STT=deepgram` the first two rows change. Your speech is recognised while you talk, so there is no clip to upload, and Deepgram Flux decides when you have finished instead of waiting out the full silence. If Flux is not sure by the silence limit, the app asks it to end the turn. In our logs the `asr` step took 0.6 to 1.1 s, and the slower turns were the ones the silence limit ended. `FISH_VOICE_EAGER_EOT_THRESHOLD` goes further and starts the model on Flux's early guess, which saves roughly another 150 to 250 ms for about 50 to 70 percent more model calls.

## Recommended settings for low latency

> [!TIP]
> Put these in `.env`. Each one is a trade-off, described in the sections below.

```bash
FISH_VOICE_STREAM_TTS=1          # speak while the model is still writing
FISH_LATENCY=balanced            # Fish's recommended mode for voice agents
FISH_VOICE_SILENCE_FRAMES=25     # end your turn after 0.75 s of silence (default 1.2 s)
FISH_VOICE_COOLDOWN=0.3          # reopen the mic 0.3 s after a reply (default 0.8 s)
FISH_VOICE_MIN_VOICED_FRAMES=7   # let a one-word "yes" through (default needs 360 ms)
FISH_LLM_PROVIDER_SORT=latency   # the default: route to the provider with the fastest first token
```

Then pick a fast model (see [Choosing an LLM](#choosing-an-llm)).

## What the code already does

You get these without changing any settings.

### Speaking

- **Streamed replies.** With `FISH_VOICE_STREAM_TTS=1`, tokens go to Fish as they arrive. The first sentence is flushed as soon as it ends, so you hear it while the model is still writing the rest. The flush waits for a real sentence end ("Dr." is not one), because an earlier flush makes Fish say half a sentence as if it were finished.
- **The TTS socket opens alongside the LLM.** The Fish websocket connects while the LLM is producing its first token, so its handshake costs nothing.
- **Playback runs on its own thread.** Writing audio to the speaker blocks for as long as the chunk plays. It runs on a worker thread, so the TTS loop keeps sending text to Fish and receiving audio during playback. Before this change, each chunk arrived only 20 to 40 ms before the previous one ran out, which could be heard as a click at sentence ends.
- **One chunk of read-ahead.** The audio reader stays one chunk ahead of the speaker and no further, so a long reply does not pile up in memory, and a barge-in stops quickly.
- **Nothing polls.** Tokens reach the TTS thread through a wake-up rather than a 5 ms check. <kbd>Ctrl</kbd>+<kbd>C</kbd> and barge-in wake the TTS loop at once rather than on a 250 ms timer.

### Listening

- **Echo cancellation.** WebRTC AEC3 removes the bot's own voice from the mic, so the barge-in gate can listen while it talks.
- **Barge-in resume.** If a noise interrupts a reply and Fish hears no words in it (a cough, a tap, a lone "mm-hmm"), the reply picks up where it was cut, with the mood cue that was active.
- **Short answers count.** A lone "yeah", "yep" or "uh-huh" on its own turn is an answer and gets a reply. Said over the bot, it means "go on" and is ignored. A hesitation ("um", "hmm") is always ignored.

### Network

- **Warm connections.** The ASR and LLM clients keep idle connections open for 120 s, so a new turn skips the TLS handshake. The httpx default of 5 s is shorter than the gap between turns, so every turn used to reconnect. On a live run this cut ASR from about 0.3–0.7 s to about 0.2 s, and the LLM's first token from about 1.0 s to about 0.5 s.
- **Latency routing.** On OpenRouter, every request sends `provider.sort: "latency"`, so a model served by several providers goes to the one with the fastest first token.
- **Retries only where they help.** A rate-limited (429) LLM request is retried once after about 1 s, but not when the account's quota is exhausted. Fish retries 429 and 5xx errors only before the first audio byte, so a retry never replays words you already heard.

### Text processing

All text helpers run in linear time on hostile input. Two were quadratic and are fixed:

- **Unclosed quotes.** Extracting dialogue from 20,000 characters of unclosed quotes took 9.8 s. It now takes about 0.1 s.
- **Held spans.** While streaming, a long `<think>` block or an unclosed `(` re-ran every hold check over the whole buffer on every token. 8,000 held characters took 1.4 to 2.1 s; they now take 12 to 146 ms.

In normal use, scrubbing costs about 50 µs per token. Per-frame mic work (VAD, AEC, level tracking) stays under 1 ms per 30 ms frame. CPU is not a bottleneck.

## Settings and their trade-offs

### Fish text-to-speech

| Setting | Default | Faster | Trade-off |
| --- | --- | --- | --- |
| `FISH_LATENCY` | `normal` | `balanced` | Fish quotes about 300 ms against about 500 ms to the first audio; in practice it was 400 ms against 650 to 800 ms. The voice can sound a little less polished |
| `FISH_CHUNK_LENGTH` | `200` | `100`–`150` | Smaller chunks start sooner. Fish allows 100 to 300 on the cloud. Prosody across long sentences can suffer |
| `FISH_MIN_CHUNK_LENGTH` | `50` | lower | Allowed range 0 to 100 |
| `FISH_SAMPLE_RATE` | `44100` | `24000` | About 45% less audio over the wire. Fish documents no difference in time to first audio |

> [!NOTE]
> `FISH_LATENCY=low` is a real, faster mode: Fish's compatible APIs default to it ("streaming-optimized") and say to set `normal` for maximum quality. The Python SDK (1.3.0) rejects it, so the app uses `balanced` and warns. Nothing here has tried it, and Fish gives no figure for it. See the future considerations in [ARCHITECTURE](ARCHITECTURE.md#10-future-considerations).

### Fish speech-to-text

| Setting | Default | Faster | Trade-off |
| --- | --- | --- | --- |
| `FISH_ASR_MODEL` | `transcribe-1-pro` | `transcribe-1` | Fish describes the non-pro model as faster for short recordings. Pro adds speaker labels and emotion cues, which a one-person conversation does not use. Worth an A/B test |
| `FISH_ASR_LANGUAGE` | detected | `en` (or yours) | With `transcribe-1-pro` it changes nothing you hear: Fish detects the language either way, and the hint is only reported when detection fails. A short "Uh, …" can still be labelled Finnish or Tagalog while the text comes back right |

Fish's speech-to-text has no streaming mode; each clip is sent once after you stop talking.

### Fish cost and limits

As Fish documents them in October 2026; check Fish's pricing page for today's numbers.

- **Text to speech:** $15 per million UTF-8 bytes, so a 1,200-character reply costs about 2 cents. Cue tags count as text, and a CJK character is 3 bytes. `FISH_TTS_MODEL=s2.1-pro-free` is the same model at no cost for testing, under fair use and with no latency guarantee.
- **Speech to text:** $0.36 per audio hour for `transcribe-1` and `transcribe-1-pro` alike. The whole clip is billed, silence included, rounded up to the second. With `FISH_VOICE_STT=deepgram` Fish ASR is not used, and Deepgram bills separately.
- **Concurrency:** 5 requests on the starter tier (under $100 paid), 15 from $100 and 50 from $1,000. The limit counts every request on the account, a text-to-speech socket holds its slot for the whole reply, and a 429 carries no `Retry-After` header. One chat is nowhere near it; several sessions or a busy proxy could be.

### Turn-taking

| Setting | Default | Faster | Trade-off |
| --- | --- | --- | --- |
| `FISH_VOICE_SILENCE_FRAMES` | `40` (1.2 s) | `25` (0.75 s) | The single biggest fixed cost per turn. Production voice agents use 0.4 to 0.5 s, but they pair it with a turn-detection model. With silence alone, a pause mid-sentence longer than this ends your turn. `30` (0.9 s) is a middle ground |
| `FISH_VOICE_COOLDOWN` | `0.8` | `0.3` | How long the mic stays shut after a reply. Anything said in this gap is lost, so a quick "yes" can vanish. Too short, and echo from loud speakers in an echoey room can open a junk turn |
| `FISH_VOICE_MIN_VOICED_FRAMES` | `12` (360 ms) | `7` (210 ms) | A one-word answer is about 200 to 250 ms of voice. Lower lets "yes" and "no" through, but some coughs and desk taps reach speech-to-text too, where they usually come back empty |
| `FISH_VOICE_SPEECH_FRAMES` | `4` (120 ms) | | How long a sound must last before it starts a turn |
| `FISH_VOICE_STT` | `fish` | `deepgram` | Streams your speech to Deepgram Flux while you talk, which ends the turn by the model's own judgment instead of a fixed silence. Needs the `deepgram` extra and a key. Quiet speech was missed more often than with Fish ASR in testing, and near-homophones can differ |
| `FISH_VOICE_EOT_THRESHOLD` | `0.7` | `0.6` | How sure Flux must be that you have finished. Lower ends turns sooner but may cut in on a long pause. Turns that end at `trigger=manual` waited for the silence timer instead |
| `FISH_VOICE_EAGER_EOT_THRESHOLD` | `0` (off) | `0.5` | Starts the reply on Flux's early guess, held back until it is confirmed. Faster, and costs 50 to 70 percent more model calls |

### Barge-in

| Setting | Default | Notes |
| --- | --- | --- |
| `FISH_VOICE_BARGE_FRAMES` | `10` (300 ms) | Loud voiced frames in a row needed to interrupt. If coughs or chair noises cut replies off, raise this before raising the volume floor. Below 8, speaker echo can interrupt the bot |
| `FISH_VOICE_BARGE_RMS` | `220` | The loudness floor. It follows the room and never drops below this |
| `FISH_VOICE_AEC` | on | Leave it on. Without echo cancellation, barge-in has to wait out the speaker's bleed (`FISH_VOICE_BLEED_DELAY`, 0.9 s) |

### LLM

| Setting | Default | Notes |
| --- | --- | --- |
| `FISH_LLM_PROVIDER_SORT` | `latency` | `throughput` or `price` instead, or `off` for OpenRouter's own price-weighted routing. It matters only for a model served by more than one provider |
| `FISH_LLM_NITRO` | off | Adds `:nitro` (OpenRouter's fast-provider tier) |
| `FISH_LLM_REASONING_EFFORT` | unset | For a thinking model, `none` or `low` avoids seconds of hidden thinking before the first word |
| `FISH_LLM_MAX_TOKENS` | `1200` | Lower keeps replies short, so turns end sooner |
| `FISH_VOICE_HISTORY_TURNS` | `20` | Fewer turns mean a shorter prompt, which helps the first token slightly |

## Choosing an LLM

The model and its provider decide most of the time between your words and the bot's. Two numbers matter:

- **Time to first token**, including its worst case. A model with a 0.7 s median but a 2 s 90th percentile feels slow every few turns.
- **Provider count.** With several providers, latency routing can pick the fastest. A model with one provider gets whatever that provider's shared pool gives that minute.

OpenRouter publishes live stats per provider. To see them for a model, using your key:

```bash
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models/sao10k/l3.1-euryale-70b/endpoints \
  | jq '.data.endpoints[] | {provider_name, latency_last_30m, throughput_last_30m}'
```

A snapshot of roleplay-capable models (October 2026, last 30 minutes, fastest provider):

> [!NOTE]
> These numbers move from hour to hour. Run the `curl` above before you pick.

| Model | Providers | First token p50 / p90 | Notes |
| --- | --- | --- | --- |
| `sao10k/l3.1-euryale-70b` | 2 | 220 / 951 ms | Roleplay-tuned 70B. Measured live at 0.3 to 0.7 s |
| `mistralai/mistral-nemo` | 6 | 323 / 429 ms | Very steady, almost free |
| `mistralai/mistral-small-3.2-24b-instruct` | 4 | 290 / 424 ms | The base model Cydonia is tuned from |
| `sao10k/l3-lunaris-8b` | 3 | 140 / 295 ms | Fastest, but small |
| `thedrummer/cydonia-24b-v4.1` | 1 | 691 / 2049 ms | Single provider; measured live anywhere from 0.4 to 2.6 s |

With `--debug`, the `llm stream_end` line shows which provider served each reply (`provider=DeepInfra`).

### Local models (Ollama)

A local model removes network variance. Once loaded, a 9B model reaches its first word in under 0.2 s on a good GPU. Tips:

- **The first request loads the model**, which can take 15 to 20 s. Set `OLLAMA_KEEP_ALIVE=-1` (or a long time such as `24h`) on the Ollama server so it stays in memory between sessions.
- **Turn thinking off** on thinking models, which otherwise think silently before every reply: `FISH_LLM_REASONING_EFFORT=none`.
- **Check the chat template.** Some community GGUF builds ship an empty one, and the model then answers nonsense or nothing.

```bash
FISH_LLM_PROVIDER=custom
FISH_LLM_BASE=http://127.0.0.1:11434/v1
FISH_LLM_API_KEY=ollama
FISH_LLM_MODEL=<your model>
FISH_LLM_REASONING_EFFORT=none
```

## Reading the debug log

Run `./packages/voice/dev.sh --debug` (or `--trace` for more). Each turn ends with a summary:

```
turn summary first audio 1.18s · asr 0.15s · llm first token 0.46s · tts first audio 1.01s · total 8.82s
```

- **first audio:** from the end of your speech to the first sound, not counting the end-of-speech silence. This is the wait you feel.
- **asr:** Fish speech-to-text.
- **llm first token:** the model's first token.
- **tts first audio:** from opening the Fish socket to the first audio, so it includes waiting for the LLM. Fish's own delay is the gap between `tts_first_text` and `tts_first_audio` on the `turn timing` line.
- **total:** until the reply finished playing.

Lines worth watching:

| Line | Meaning |
| --- | --- |
| `tts underrun after N kB played (chunk_wait X ms, in_python Y ms, write Z ms)` | The speaker ran dry between chunks, heard as a click or gap. A large `chunk_wait` means the audio arrived late (Fish was slow to deliver it). A large `in_python` means the playback thread itself was stalled before the write. A `write` far above the length of a slice (about 30 ms) means the call itself waited, usually to get Python's lock back from a busy thread. Should not happen with current code |
| `barge hit 3/10 ...` then `barge decay` | Something loud nearly interrupted the bot. Frequent near-misses with no one talking mean echo is leaking |
| `stt end trigger=model confidence=0.8` | Flux ended the turn itself. `trigger=manual` means the silence limit ended it, usually because Flux was not sure |
| `stt.empty ...` / `stt.gave_up ...` / `stt.noise ...` | A streamed turn with no words in it, dropped on purpose. The text says how much audio was sent, if any |
| `llm.speculate hit` / `miss` / `dropped` | With the early reply on: the reply written ahead of time was used, replaced because the words changed, or thrown away |
| `listen reject too_little_voice` | A sound too short to be a turn. If real words show up here, lower `FISH_VOICE_MIN_VOICED_FRAMES` |
| `listen speech_start` right after `listen open`, with no one talking | Echo after the reply. Raise `FISH_VOICE_COOLDOWN` |
| `asr.skip backchannel` / `hallucination` | A clip was dropped after transcription, on purpose |
| `llm stream_end ... provider=X` | Which provider served the reply |

## Proxy

The OpenAI-compatible proxy is already tuned for its job:

- **Streamed audio passes through as it arrives.** Speech responses are no longer held back until 4 KB has built up. With small upstream chunks, the time to the first byte fell from 354 ms to 2 ms.
- **One shared, warm connection pool** to Fish: 100 connections, all kept alive for 30 s.
- **uvloop and httptools** are used automatically (installed with `uvicorn[standard]`).
- **The body-limit middleware is pure ASGI**, so it adds almost nothing per request and never buffers a streaming response.
- **Hostile input is bounded.** Request text is capped (`FISH_PROXY_MAX_INPUT_CHARS`), bodies are capped (`FISH_PROXY_MAX_BODY_BYTES`), quote extraction is linear, and deeply nested JSON is a 400, not a crash.

Settings worth knowing:

| Setting | Default | When to change it |
| --- | --- | --- |
| `FISH_LATENCY` | `normal` | `balanced` for apps that play speech live |
| `FISH_TTS_FORMAT` | `mp3` | `pcm` or `opus` for low-latency players |
| `FISH_PROXY_KEEP_ALIVE` | `5` s | Behind nginx or a cloud load balancer that keeps idle connections for 60 s, set `65`. A shorter backend keep-alive causes occasional 502 errors |
| `FISH_PROXY_LIMIT_CONCURRENCY` | unset | Set (for example `100`) to reject overload with a quick 503 instead of queueing |
| `FISH_PROXY_WORKERS` | `1` | One worker handles a lot: a speech request costs about 0.5 ms of CPU. Add workers only for many concurrent hour-long transcriptions |
| `FISH_PROXY_ASR_TIMEOUT` | `900` s | Long recordings on `transcribe-1-pro` take minutes |

Transcribing an hour-long file spends about 0.2 s of CPU building the captions, which briefly delays other requests on the same worker.

## Known limits

These are known and left alone for now:

- **One Fish websocket per turn.** Fish's protocol allows several utterances on one socket, but the setup cost is already hidden behind the LLM, so reuse would gain little.
- **The mic closes and reopens around each reply.** Speech in the cooldown gap, and about 50 to 300 ms after a barge-in, is not recorded. A mic that stays open would fix both.
- **Silence-only turn-taking on the Fish ASR path.** Deepgram Flux decides turns itself (see above). The Fish ASR path still waits out a fixed silence; a turn-detection model (Pipecat's Smart Turn runs in under 100 ms on CPU) would allow a 0.3 to 0.5 s wait there without cutting off mid-sentence pauses.
- **Trailing silence goes to speech-to-text.** Each clip ends with the full end-of-speech silence (a third of a short clip). Trimming it to about 0.25 s would cut the upload and reduce invented words. The gain is mostly accuracy, and about 20 to 50 ms of time.
- **Very long held spans while streaming.** An unclosed `(`, code fence or URL in a 30,000-character reply still costs about 1.5 s in total, spread across the stream. Real replies stay well under that.
