# fish-audio-suite-voice

One Fish Audio websocket per turn, playback sinks, and an optional duplex CLI.

Import `FishSpeaker` when another app already owns the microphone and the LLM. Use `fish-voice` when you want mic, Fish ASR, an LLM reply, and speakers in one process. The LLM is any OpenAI-compatible chat-completions server, or OpenRouter.

Part of [fish-audio-suite](../../README.md). Unofficial.

## Quick start

```python
from pathlib import Path
from fish_audio_suite_voice import FishSpeaker, FileSink

tts = FishSpeaker(api_key=key, voice_id=voice_id)
result = tts.speak("Hello there.", FileSink(Path("turn.wav")))
```

`speak` runs the websocket on a private thread and event loop, so it is safe under `asyncio.run` or `to_thread`. Retry of 429 and 5xx happens only before the first audio byte. There is no default `voice_id`.

`speak(text, sink, *, cancel=None, on_first_audio=None)` takes `cancel` (a `threading.Event`) by keyword. The returned `TtsResult` carries `error`, a kit error such as `FishAuthError` (401, 402, 403), `FishRateLimitError` (429) or `FishUpstreamError` (5xx), when the turn failed with a status. Test it with `isinstance`; `error_status` and `error_message` stay for display. Its timings use the kit's `LatencySnapshot` names: `tts_first_text_ms` (TTS start to the first text sent to Fish) and `tts_first_audio_ms` (TTS start to the first audio from Fish, Fish's time-to-first-audio). `speak_stream(deltas, sink, ...)` does the same for a token stream while the model is still writing. The package root exports the types its signatures use: `FishSpeaker`, `TtsResult`, `PlaybackSink`, `PlaybackKind`, `make_sink`, `BargeGate`, `EchoCanceller`, `DuplexSession`, `ChatBackend`, `ListenTune`, `BargeTune`, `AecTune`, `LlmTune` and `PortAudioMissingError`. Each module lists its own exports in `__all__`.

```bash
cp .env.example .env
./packages/voice/dev.sh --smoke    # writes a new temp wav, or pass --out PATH
./packages/voice/dev.sh            # duplex
./packages/voice/dev.sh --debug    # or FISH_VOICE_DEBUG=1; --trace (or 2) adds heartbeats
./packages/voice/dev.sh --prompt-file characters/mira.md   # a character, see below
```

`--smoke` exits 2 if `FISH_API_KEY` or `FISH_VOICE_ID` is missing. The same entry point from anywhere the extra is installed:

```bash
uv run --package fish-audio-suite-voice --extra cli fish-voice --smoke
```

## How it works

| Sink | When |
| --- | --- |
| `sounddevice` | Default. Plays PCM through PortAudio in short slices |
| `file` | Writes a WAV or raw PCM when the turn finishes |
| `stdout` | Raw chunks on stdout |
| `mpv` | Optional. Plays mp3 from stdin. Not required for the other sinks |

Duplex is mic, then Fish ASR (or Deepgram Flux, see below), then an LLM, then one Fish turn. Settings are environment variables. Process env wins, then `--env-file`, then `./.env`.

`--debug` logs VAD, barge-in, Fish websocket metadata, and LLM metadata on stderr. What was said stays on stdout. To keep a whole run, set `FISH_VOICE_LOG_DIR=logs` (or pass `--log-dir logs`): each run writes a complete log there, with what you said and what the model answered, to a file only you can read. Duplex exits 2 when playback is `file`, unknown, or `mpv` is not on `PATH`. It also exits 2 on ASR or TTS 401, 402, or 403.

### PortAudio

Speakers and the microphone need the C library. `uv` and `pip` do not install it.

```bash
# Debian / Ubuntu
sudo apt install libportaudio2
# Fedora
sudo dnf install portaudio
# macOS
brew install portaudio
```

On NixOS use `./packages/voice/dev.sh` or `nix run .#fish-audio-suite-voice`. Both put PortAudio on `LD_LIBRARY_PATH`. A missing library raises `PortAudioMissingError` when the stream opens.

### Echo and barge-in

The `cli` extra includes WebRTC's audio processor, which cleans every mic frame before voice activity detection and speech recognition, as a call app does:

- a high-pass filter removes rumble and breath below about 100 Hz,
- automatic gain (`FISH_VOICE_AGC`) raises a quiet voice to a steady level,
- noise suppression (`FISH_VOICE_NOISE_SUPPRESSION`) removes steady background noise such as a fan,
- and while the bot speaks, AEC3 subtracts its voice from the mic. `FISH_VOICE_AEC=0` turns only that off and keeps the longer bleed delay (0.9 s). With AEC loaded, that delay is 0.3 s.

Noise suppression does little against key clicks. `scripts/vad_compare.py --process` shows what the detectors make of a recording after this cleanup.

A barge-in keeps the audio that tripped it and clears the speaker tap. The next listen starts from that clip.

The barge floor follows the room: it is the quiet-percentile of recent mic frames, never below `FISH_VOICE_BARGE_RMS`. If barge-in fires on a cough or a chair, raise `FISH_VOICE_BARGE_FRAMES` (30 ms each) before raising the RMS. If sniffs, breathing, typing or a bumped mic interrupt it, try `FISH_VOICE_VAD_ENGINE=silero`, or `FISH_VOICE_VAD=2`, which counts fewer of those sounds as speech. It also affects normal listening, so check that soft speech and the first word of a sentence still come through. See [Troubleshooting](../../docs/TROUBLESHOOTING.md#coughs-chairs-sniffs-or-keyboard-noise-interrupt-the-bot).

If talking over the bot does nothing, your voice may be reaching the gate quietly after echo cancellation. `scripts/mic_levels.py` prints your noise floor and speaking level, and raising the mic's own gain (not only software volume) helps more than lowering the thresholds. See [Troubleshooting](../../docs/TROUBLESHOOTING.md#soft-speech-is-missed-or-talking-over-the-bot-does-nothing-because-the-mic-is-quiet) for the numbers and the order to fix things in.

### History after a barge-in

The chat history gets only what you probably heard. For PCM that is a word-aligned estimate from the bytes played, the speed, and the device buffer. Encoded playback (`mpv`) cannot be cut by length, so an interrupted or failed reply is left out of history.

If the interrupt turns out not to be speech (Fish hears no words, a backchannel such as "mm-hmm", or a clip too short to keep), the rest of the reply is spoken again from where it was cut, and joins the same history message. A real line drops the rest. Only text the model had already written is resumed.

On a turn of its own, a lone "yeah", "yep" or "uh-huh" is an answer and gets a reply. Only a hesitation such as "um" or "hmm" is skipped there.

### Cues in replies

With the default system prompt the session starts with one pinned exchange that shows several cues
(`DEFAULT_SEED_EXCHANGE` in the kit). It costs about 40 tokens per request and is never trimmed from
the history. A custom `FISH_VOICE_SYSTEM_PROMPT` gets no seed, so it stays in control of the replies.

How Fish's S2 models read cues, from Fish's guides: a cue is square brackets around plain language, so `[warm and happy]` and `[laughing nervously]` work as well as the official names. Layered cues work (`[sad][whispering]`, at most three per sentence is advised), a sentence's emotion cue works best at its start, and sound cues can go anywhere. Follow a sound with the words of it (`[laughing] Ha ha`), use cues sparingly, and do not stack conflicting emotions. The kit keeps stacks whole, rewrites the older S1 `(parenthesis)` and alias spellings (`[laugh]` becomes `[laughing]`), and never speaks a cue. The S1 model uses parentheses and a fixed list, and the app does not write them back for it, so `FISH_TTS_MODEL=s1` is not supported for cues.

Free-form cues are risky on some voices: a description of a face or a voice (`[smiling]`, `[soft chuckle]`, `[echoing voice]`) can be acted out as a hum or a small sound, which sounds like an "mhm" between sentences. The default prompt therefore asks for one word from a list of about 20 moods, and `FISH_TTS_CUES=official` (the default) is the safety net: any cue Fish does not list becomes the nearest official one (`smiling` to `happy`, `gentle` to `calm`, `soft chuckle` to `chuckling`) or is removed (`in a storytelling voice`). `FISH_TTS_CUES=free` sends cues as written. Whatever the mode, a sentence the model wrote without a cue is sent with the mood the last sentence had, because each piece goes to Fish on its own and a sentence with no cue starts cold; a piece that only continues a sentence is left alone.

### Characters and system prompts

Put a character or scene in a file and point the session at it:

```bash
./packages/voice/dev.sh --prompt-file characters/mira.md     # or FISH_VOICE_SYSTEM_PROMPT_FILE=...
```

The file is UTF-8 text, up to 64 KiB. Its text comes first, then a blank line, then the default voice
rules (spoken replies, `[cue]` tags), and the opening cue example is pinned as usual. So a file only
has to describe who is talking. If the file is missing, empty, too large or not UTF-8, fish-voice warns
and uses the default prompt. The flag beats the variable, and the file beats `FISH_VOICE_SYSTEM_PROMPT`.

**Choose where the voice rules go.** Put `{{default_prompt}}` in the file and the default voice rules
are inserted right there, instead of after the file. Use it to put the rules first and the character
after them, or between two parts of the character:

```text
{{default_prompt}}

You are Mira, a dry-witted ship's engineer. Stay in character.
```

The slot may have spaces inside the braces (`{{ default_prompt }}`), and every slot in the file is
replaced. A file without the slot keeps the old behavior: the rules come after it. A file that holds
only the slot is the default prompt.

To write the whole prompt yourself, set `FISH_VOICE_SYSTEM_PROMPT` instead. It is used as written, with
no voice rules and no cue example, so ask for `[emotion]` cue tags in it if you want an expressive voice.

### LLM providers

`FISH_LLM_PROVIDER` names the chat provider: `openrouter` (the default) or `experiential`. It
supplies the default base URL, and `FISH_LLM_BASE` overrides it. The provider is always read from
the host of the final base URL, and each provider has its own key variable, so a key never goes
to the wrong host.

| Provider | Default base | Key variable | Model variable |
| --- | --- | --- | --- |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `FISH_LLM_MODEL_OPENROUTER` |
| `experiential` | `https://api.experientiallabs.ai/v1` | `EXPLABS_API_KEY` | `FISH_LLM_MODEL_EXPERIENTIAL` |
| any other host | `FISH_LLM_BASE` | `OPENAI_API_KEY` | `FISH_LLM_MODEL` |

`FISH_LLM_API_KEY` and `FISH_LLM_MODEL` work for every provider. An explicit `FISH_LLM_API_KEY` wins, and
the chosen provider's own model variable comes before `FISH_LLM_MODEL`. Keep both providers in one
`.env` and switch with `FISH_LLM_PROVIDER`.

The backend follows the base URL's host: OpenRouter uses its SDK, and every other server (Experiential,
Ollama, any OpenAI-compatible host) uses one pooled httpx client that speaks the OpenAI
chat-completions API. What to know about Experiential, from its published contract:

- **Reasoning.** `FISH_LLM_REASONING_EFFORT` (`none`, `minimal`, `low`, `medium`, `high` or `max`) is
  sent as `reasoning_effort` to OpenAI-compatible servers and as `reasoning.effort` to OpenRouter.
  Reasoning models can be slow to a first word, so a low effort suits voice. Which values a model
  accepts depends on its route, and the value is passed through unchanged. Empty omits the field.
- **Fields.** Experiential answers 400 to a request field it does not know, so the request carries
  only `model`, `messages`, `stream`, `temperature` and `max_tokens`, plus `reasoning_effort` when set.
- **No `:nitro`.** That suffix is OpenRouter-only (`FISH_LLM_NITRO` never applies to Experiential).
- **Errors.** A spent free allowance is a 429 `insufficient_quota` with no `Retry-After`. It is
  printed with its code and not retried.
- **Debug log.** With `--debug` the response line shows the request id (`x-request-id`), the route
  (`x-gateway-provider`) and the zero-data-retention posture (`x-gateway-zdr`).
- **Privacy.** Platform-funded calls are captured: Experiential stores both the request and the
  model's reply. A free organization cannot turn that off, and only a Pro organization can. Your
  spoken conversation is the prompt, so do not use a free organization for anything private.
- **https only for the automatic key.** `EXPLABS_API_KEY` is picked up only when the base is
  `https`. A plain `http` base would send it unencrypted, so use `FISH_LLM_API_KEY` to override.

OpenRouter-only options: `FISH_LLM_PROVIDER_SORT` picks the provider by `latency` (the default, so
the first word comes sooner), `throughput` or `price`, and `off` leaves it to OpenRouter. It only
changes anything for a model that more than one provider serves. `FISH_LLM_SERVICE_TIER` asks for
a processing tier: `priority` costs more for steadier speed, `ultrafast` (select OpenAI models
only) costs several times the standard price and falls back to `priority`, then `default`, where
it is not offered, and `flex` is cheaper and slower. With `--debug`, `llm stream_end` shows the
tier that served the reply. `FISH_LLM_NITRO=1` adds `:nitro`
to the model, and `FISH_LLM_REFERER`, `FISH_LLM_TITLE`, `FISH_LLM_CATEGORIES` set the attribution
(empty disables one). Other servers get none of these.

### Streaming the reply

By default the reply is spoken after the model finishes. Set `FISH_VOICE_STREAM_TTS=1` to speak
while the model is still writing. The Fish socket opens with the request and the first
sentence is flushed at once, so audio starts about when that sentence is done instead of when
the whole reply is. The socket then gets one more flush at the end.

- One extra flush is needed: Fish holds text until a chunk fills or a flush arrives, so a single
  flush at the end keeps a short reply silent until the model is done.
- The reply is scrubbed as it arrives, and the whole-reply junk check is skipped.
- The barge-in gate arms at the first audio chunk.
- If Fish fails before any audio, the finished reply is spoken on the normal path.

### Local models (Ollama)

Any OpenAI-compatible local server works. For Ollama:

```bash
FISH_LLM_PROVIDER=custom
FISH_LLM_BASE=http://127.0.0.1:11434/v1
FISH_LLM_API_KEY=ollama          # any value, Ollama has no key
FISH_LLM_MODEL=qwen3.5:9b        # the exact name from `ollama list`
FISH_LLM_REASONING_EFFORT=none
```

- **Thinking models** (Qwen 3.5, Gemma 4 and their abliterated or "heretic" finetunes) write their
  answer into a hidden reasoning field first, so a short reply comes back empty. `none` turns that off,
  and Ollama 0.34 honored it. Without it the first word can take many seconds.
- **The first request after idle loads the model**, which took about 17 s for a 9B model. Keep it
  loaded with `OLLAMA_KEEP_ALIVE=-1` in Ollama's own environment.
- **A model imported from a bare GGUF can have no chat template.** It then ignores the conversation
  and writes unrelated text. `ollama show --modelfile <name>` should show a `TEMPLATE` for the model's
  family, not just `{{ .Prompt }}`.
- Once a model is loaded, the first word arrived in under 0.1 s on this machine. That is the
  network-free baseline the cloud providers cannot reach.

### Roleplay helpers

Both are off by default because they also change ordinary English. `FISH_TTS_MOOD_LEAD=1` rewrites a sentence-leading mood word (`Excited, hello`) into a `[cue]`. `FISH_TTS_DROP_NARRATION=1` drops lines that are only stage directions (`She smiles.`).

### The full-screen app

```bash
uv run --package fish-audio-suite-voice --extra cli --extra tui fish-voice --tui
```

`--tui` runs the same session in a full-screen terminal app (built on [Textual](https://textual.textualize.io)). It shows the conversation as it streams, with each `[cue]` styled and never read as markup, the state (listening, thinking, speaking), a live waveform drawn in dots (bars grow from the middle and scroll left; green and brighter the louder you are, grey when quiet; a cursor blinks at the right edge while it listens, and an amber wave scrolls by while the model thinks; while the reply plays the strip shows its level in the accent colour and its title changes from `mic` to `speaker`, for the default PortAudio playback only). The mic and the speaker are scaled the same way, from the last few seconds of the signal itself (its noise floor and its speech level), so they look alike whatever their volume, a quiet room stays flat and one loud bump does not shrink the speech after it, a pulsing "you" line while you talk that turns into your words once they are transcribed, the last turn as a bar showing where the wait before you heard audio went (asr, llm, tts), and the log. You can also type a line instead of speaking, mute the mic, and stop a reply that is playing.

| Key | Does |
| --- | --- |
| Enter | Send the typed line |
| Escape | Stop the reply (only while the model is thinking or the reply is playing) |
| F2 | Mute or unmute the mic. Muted, the mic stays closed: no listening between turns and no barge-in watch during a reply. Typing or Esc still interrupts. Unmuting during a reply applies from the next reply |
| Ctrl+Q | Quit |

While it runs, the app asks the sound card for a bigger playback buffer (`FISH_VOICE_OUTPUT_LATENCY`, `high` in `--tui` by default), lets the playback thread in sooner, and collects garbage less often, because drawing the screen shares a process with playback and otherwise makes it click. Ctrl+C is not quit here, because the app uses it to copy. Without the `tui` extra, `fish-voice --tui` says how to install it and exits 2. The panels shrink with the terminal: under 72 columns the timings and the log go, under 22 rows the timings go, and under 50x14 it asks for a bigger window (typing and Ctrl+Q still work). With no terminal on stdin or stdout, `--tui` falls back to the plain loop. `NO_COLOR` is honoured, and `TEXTUAL_ANIMATIONS=none` stops the pulsing "you" line, the listening cursor and the thinking wave from moving. A modern terminal works best; macOS's Terminal.app is limited to 256 colors, so use iTerm2, Kitty, WezTerm or Ghostty there.

### Streaming speech recognition (Deepgram)

By default `fish-voice` waits for a pause, then sends the whole utterance to Fish ASR. With
`FISH_VOICE_STT=deepgram` it streams your speech to [Deepgram's Flux](https://developers.deepgram.com/docs/flux/quickstart)
while you talk instead. Flux says for itself when you have finished, so the reply starts
sooner: there is no fixed silence to wait out and no upload and transcribe step afterwards. The
words also appear in the full-screen app as you say them.

```bash
uv sync --extra deepgram          # or pip install 'fish-audio-suite-voice[deepgram]'
export DEEPGRAM_API_KEY=...
FISH_VOICE_STT=deepgram fish-voice
```

What happens, and what leaves your machine:

- Nothing is sent while the mic waits for you. The same voice detector as before decides when
  speech starts, and then the first frames are held until there is as much voiced audio as the
  batch path asks for (`FISH_VOICE_MIN_VOICED_FRAMES`), so a knock or a cough never leaves your
  machine. Only then is the connection opened and audio sent, beginning with the held pre-roll so
  the first word is not clipped. A barge-in sends its saved clip first. (Deepgram
  closes a connection that sits without audio, so none is held open while you are quiet.)
- Every connection asks Deepgram not to keep the audio or the transcript (`mip_opt_out=true`).
  `FISH_VOICE_DEEPGRAM_REGION` picks a regional endpoint that processes it in the EU, Australia or
  India.
- If Flux has returned no words 1.2 s after streaming starts, the turn is dropped as noise
  straight away instead of being sent until you stop (a run of taps or knocks, for instance).
- `FISH_VOICE_SILENCE_FRAMES` is still a limit: if Flux has not ended the turn by then, it is
  asked to.
- If Deepgram cannot be reached when you start speaking, that turn is heard with Fish ASR from
  the speech already captured, with a warning, so nothing you said is lost. A missing key,
  a refused key or a missing `deepgram` extra stops the app at the start instead.
- `FISH_VOICE_EAGER_EOT_THRESHOLD` starts the model's reply early. Nothing is shown, spoken or
  added to the history until Flux confirms the same words; if you carried on, or the words
  differ, the early reply is thrown away and asked for again.
- Fish's own realtime transcription (the OpenAI-compatible socket) is not an alternative: it returns the transcript once, after a `commit`, with no words while you speak, and it refuses server-side voice detection. Deepgram Flux is the streaming option.
- A transcript is judged exactly as a Fish one is: quit words, stray "mm-hmm" over a reply, and
  stale copies of the last line are dropped.
- Only the PortAudio microphone path streams. Typed lines, mute and interrupt work as before.

### Sounding more human (opt in)

Three settings, each off by default, so nothing changes until you turn one on.

**End-of-turn check** (`FISH_VOICE_TURN_CHECK=1`, Fish speech recognition only). The Fish path waits
`FISH_VOICE_SILENCE_FRAMES` (1.2 s) of quiet before every reply, because a shorter wait cuts you off
mid-thought. With the check on, the first wait is short (`FISH_VOICE_TURN_SILENCE_FRAMES`, 12 frames,
about 0.36 s) and the words decide. A clip that ends on a comma, an ellipsis, "and", "but",
"because", "the", "um" or a preposition, or in the middle of a list of numbers, is not taken as the end of the turn.
A clip that ends with a full stop, a question mark, an exclamation mark or a short answer is.

- While the clip is transcribed, the mic is already listening again. If the turn was over, that
  recording is closed before the reply starts. If it was not, what you say next is transcribed too and
  joined to the first part, and the whole line is judged and answered as one. The first part is never
  dropped, even when nothing more comes.
- You wait at most as long as before. A continuation ends on `FISH_VOICE_TURN_EXTEND_FRAMES` of
  quiet (40, the old limit), and the wait for it to start is that minus the short silence. At most
  `FISH_VOICE_TURN_EXTENSIONS` (1, or 2) continuations are joined to one turn.
- Quit words, mute, a typed line and Ctrl+C work while it waits. Mute closes the mic and drops the
  turn, and a muted mic is never reopened for a continuation.
- The judgement reads English. Text in another language is judged by its punctuation alone.
  Deepgram has its own turn detection, so the check does nothing with `FISH_VOICE_STT=deepgram`.
- The log (`--debug`) shows `turn.extend` with the reason each time it waits for more.

**Tone as context** (`FISH_VOICE_HEAR_TONE=1`, experimental, Fish speech recognition only).
`transcribe-1-pro` marks how speech sounds in its text (`[laughter]`, an emotion word), and the app
removes those marks before the model sees your words. With this on, the marks are kept
beside the clean text, and the model is told how you sounded for that turn only, in a
note after your line: `(Note on how they sounded: laughter, tired. This is not something they said.
Do not repeat or mention it, and never write it as a [cue] or in brackets.)`

- The note is added to the request for that turn and never saved. History and the screen keep your plain words, and the next
  request has no memory of it.
- It is not in the speech path. The note is in round brackets, which the text scrubber
  removes, so a model that echoes it halfway through a reply is not heard saying it.
  A reply made of nothing but the note would be spoken, so this is a safeguard and not a guarantee.
- Only a few plain words from Fish are passed on (at most three marks, none with digits or symbols).
- It is unproven how well it works on real speech. Try it and judge the replies.

**Loudness** (`FISH_VOICE_NORMALIZE_LOUDNESS=1`). Fish voices differ by up to 19 dB in level, and the
realtime socket cannot ask Fish to even them out. The app measures the speech in each reply as it
arrives and brings it to `FISH_VOICE_LOUDNESS_TARGET_DB` (-18 dBFS while speaking), then limits any peak above that so a boost never clips.

- The level is set from the first chunk of the reply and drifts slowly after that, so a loud word does
  not make the next quiet one pump. Pauses are not counted as speech and are not boosted.
- The change is at most 12 dB up or down. A very quiet voice ends up quieter than the target.
- It runs before the sound goes to the speaker and to the echo canceller, so the canceller sees
  what the speaker plays. It does not touch your system volume and adds no delay.
- It applies to PCM playback (the default speakers). mpv plays mp3 and is left alone.

## Extras

| Extra | Pulls in |
| --- | --- |
| `speakers` | `sounddevice` |
| `vad` | `webrtcvad-wheels` |
| `aec` | `pywebrtc-audio` |
| `cli` | the three above, plus `openrouter` |
| `silero` | `pysilero-vad`, for `FISH_VOICE_VAD_ENGINE=silero` |
| `tui` | `textual`, for `fish-voice --tui` |
| `deepgram` | `websockets`, for `FISH_VOICE_STT=deepgram` |

`import fish_audio_suite_voice` works without them. The heavy imports happen inside the functions that need them.

## Voice detectors

One detector decides which 30 ms frames hold speech, both for starting and ending your turn and for barge-in. `FISH_VOICE_VAD_ENGINE` picks it:

- `webrtc` (the default) is WebRTC's VAD. It is small and fast, but it calls many short sounds speech, such as key clicks and steady fan noise.
- `silero` is Silero VAD v6, a small neural model run in C++ by `pysilero-vad`. It tells speech from clicks and noise much better, and costs about half a millisecond per frame. Install the `silero` extra.
- `package.module:factory` loads your own. `factory(strictness)` gets `FISH_VOICE_VAD` (0 to 3) and returns an object with `is_speech(frame: bytes, sample_rate: int) -> bool`, which is called with each 30 ms frame of 16 kHz mono 16-bit PCM in order. `webrtcvad.Vad` has this shape. A new detector is opened for every listen and every barge-in watch, so it can keep state between frames.

For Silero, `FISH_VOICE_VAD` sets the speech probability a frame must reach: 0.35, 0.5, 0.65 or 0.8. If the chosen detector cannot load, `fish-voice` stops at start-up and says why.

To compare detectors on your own room and keyboard, record once and play it through each: `uv run python scripts/vad_compare.py --record 30 take.wav`. See [scripts/README.md](../../scripts/README.md#vad_comparepy).

## Settings

Every variable is read once at startup into frozen settings objects. A value that is not a number, or is out of range, is ignored with a warning on stderr and the default is used.

Required:

| Variable | Default |
| --- | --- |
| `FISH_API_KEY` | none |
| `FISH_VOICE_ID` | none |
| `FISH_LLM_API_KEY` (fallback: the provider's own key variable, see above) | none. Duplex only |
| `FISH_LLM_MODEL` (the provider's model variable first; `OPENROUTER_MODEL` for OpenRouter and custom hosts) | none. Duplex only |

Speech:

| Variable | Default |
| --- | --- |
| `FISH_BASE` | `https://api.fish.audio` |
| `FISH_TTS_MODEL` | `s2.1-pro` (or `s2.1-pro-free`, the same model free for testing under fair use, with no latency guarantee) |
| `FISH_LATENCY` | `normal` (or `balanced`) |
| `FISH_SPEED` | `1` |
| `FISH_VOLUME` | `0` (dB) |
| `FISH_TEMPERATURE`, `FISH_TOP_P` | `0.7`, `0.7` |
| `FISH_REPETITION_PENALTY` | `1.2` |
| `FISH_CHUNK_LENGTH` | `200` (cloud max 300) |
| `FISH_MIN_CHUNK_LENGTH` | `50` |
| `FISH_SAMPLE_RATE` | `44100` |
| `FISH_ASR_LANGUAGE` | omitted |
| `FISH_ASR_MODEL` | `transcribe-1-pro` (or `transcribe-1`) |
| `FISH_VOICE_PLAYBACK` | `sounddevice` |
| `FISH_VOICE_DEVICE` | host default. A PortAudio index or name |
| `FISH_VOICE_OUTPUT_LATENCY` | `auto`, or `low` / `high`. A bigger playback buffer (`high`) stops clicks when the machine is busy, and starts the sound a little later. `auto` is `high` in `--tui` and `low` otherwise |

LLM:

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_LLM_PROVIDER` | `openrouter` | `openrouter` or `experiential`. Sets the default base, key variable and model variable. See above |
| `FISH_LLM_BASE` (fallback `OPENROUTER_BASE_URL`) | `https://openrouter.ai/api/v1` | API origin |
| `FISH_LLM_REASONING_EFFORT` | unset | `none`, `minimal`, `low`, `medium`, `high` or `max`. Which values work depends on the model's route |
| `FISH_LLM_TEMPERATURE` | `0.8` | 0 to 2 |
| `FISH_LLM_TIMEOUT` | `120` | Seconds per request |
| `FISH_LLM_MAX_TOKENS` | `1200` | Completion cap |
| `FISH_LLM_NITRO` | off | OpenRouter only. Adds `:nitro` to the model |
| `FISH_LLM_PROVIDER_SORT` | `latency` | OpenRouter only. `latency`, `throughput`, `price` or `off` |
| `FISH_LLM_SERVICE_TIER` | unset | OpenRouter only. `default`, `flex`, `priority` or `ultrafast`. Unset lets OpenRouter pick. Higher tiers cost more |
| `FISH_LLM_REFERER`, `FISH_LLM_TITLE`, `FISH_LLM_CATEGORIES` | this project's | OpenRouter attribution. Empty disables |
| `FISH_LLM_CONTINUE` | off | Send one more request when a reply stops mid-sentence |
| `FISH_VOICE_SYSTEM_PROMPT` | the kit default | System prompt text, used as written. Set but blank sends no system prompt |
| `FISH_VOICE_SYSTEM_PROMPT_FILE` | none | A character file. The voice rules follow it, or go where `{{default_prompt}}` is in the file. Beats `FISH_VOICE_SYSTEM_PROMPT`. See characters above |
| `FISH_VOICE_HISTORY_TURNS` | `20` | User and assistant pairs kept. Over it, the oldest quarter goes at once, so the start of the prompt stays the same for several turns and a provider's prompt cache keeps working |
| `FISH_VOICE_STREAM_TTS` | off | Speak the reply while the model is still writing it. See streaming above |
| `FISH_VOICE_REPEAT_WINDOW` | `1.5` | Seconds. A line equal to the previous one is dropped only if it ends this soon after the mic opens. 0 never drops a repeat |
| `FISH_TTS_MOOD_LEAD`, `FISH_TTS_DROP_NARRATION` | off | See roleplay helpers |
| `FISH_TTS_CUES` | `official` | `official` keeps only Fish's own cues and maps the rest to the nearest one (`[smiling]` becomes `[happy]`) or drops them. `free` sends every cue as written. See cues in replies |

Speech recognition. See [Streaming speech recognition](#streaming-speech-recognition-deepgram).

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_VOICE_STT` | `fish` | `fish` sends a clip to Fish ASR when you stop. `deepgram` streams your speech to Deepgram Flux |
| `DEEPGRAM_API_KEY` | none | Your Deepgram key. Required when `FISH_VOICE_STT=deepgram` |
| `FISH_VOICE_DEEPGRAM_MODEL` | `flux-general-en` | `flux-general-multi` for ten languages |
| `FISH_VOICE_DEEPGRAM_REGION` | `global` | `eu`, `au` or `in` to have Deepgram process the audio inside the European Union, Australia or India. A regional request fails instead of going elsewhere |
| `FISH_VOICE_EOT_THRESHOLD` | `0.7` | How sure Flux must be that you have finished, 0.5 to 1. Higher waits a little longer and cuts in less often |
| `FISH_VOICE_EAGER_EOT_THRESHOLD` | `0` (off) | 0.3 to 0.9, and no higher than `FISH_VOICE_EOT_THRESHOLD`. Flux says when a turn is probably over a little before it is sure, and the reply starts being written then, held back until Flux confirms it. Saves about 150 to 250 ms, and costs 50 to 70 percent more model calls. See the streaming section |
| `FISH_VOICE_STT_SAVE_DIR` | unset | A folder. Each streamed turn's audio, as Deepgram was sent it, is saved there as a 16 kHz WAV, to find out why it heard nothing. Only for debugging: it writes your voice to disk |

Sounding more human. All off by default. See [Sounding more human](#sounding-more-human-opt-in).

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_VOICE_TURN_CHECK` | off | End the first clip on a short silence and let the words decide whether the turn is over. Fish ASR only |
| `FISH_VOICE_TURN_SILENCE_FRAMES` | `12` | About 0.36 s of quiet that ends the first clip while the check is on. Never more than `FISH_VOICE_SILENCE_FRAMES` |
| `FISH_VOICE_TURN_EXTEND_FRAMES` | `40` | Quiet frames that end a continuation, about 1.2 s. The wait for it to begin is this minus `FISH_VOICE_TURN_SILENCE_FRAMES` |
| `FISH_VOICE_TURN_EXTENSIONS` | `1` | Continuations one turn may have, 1 or 2 |
| `FISH_VOICE_HEAR_TONE` | off | Experimental. Tell the model how you sounded (`laughter`, `tired`) for that turn only, from the marks Fish ASR puts in its text |
| `FISH_VOICE_NORMALIZE_LOUDNESS` | off | Bring each reply to a steady level, with a soft limiter, before it plays |
| `FISH_VOICE_LOUDNESS_TARGET_DB` | `-18` | The level, in dB below full scale, that speech is brought to. Read only with `FISH_VOICE_NORMALIZE_LOUDNESS`. -40 to -6 |

Listen and interrupt. Times are approximate at 30 ms frames.

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_VOICE_SPEECH_FRAMES` | `4` | About 120 ms before speech counts |
| `FISH_VOICE_PRE_PAD_FRAMES` | `20` | Frames kept before the start |
| `FISH_VOICE_MIN_RMS` | `200` | Listen floor seed. It can rise in a loud room |
| `FISH_VOICE_MIN_VOICED_FRAMES` | `12` | About 360 ms of voice. Drops a cough |
| `FISH_VOICE_SILENCE_FRAMES` | `40` | About 1.2 s of quiet ends the turn |
| `FISH_VOICE_VAD` | `1` | Voice detector strictness, 0–3. Higher is pickier. Listening and barge-in both use it |
| `FISH_VOICE_VAD_ENGINE` | `webrtc` | The voice detector: `webrtc`, `silero` (the `silero` extra) or a plugin as `package.module:factory`. See [Voice detectors](#voice-detectors) |
| `FISH_VOICE_COOLDOWN` | `0.8` | Seconds after playback |
| `FISH_VOICE_FADE_MS` | `4` | Milliseconds of fade at each sentence start and end (a sound next to silence), so it does not click. 0 turns it off. Up to 50 |
| `FISH_VOICE_BLEED_DELAY` | `0.9` | Used when AEC is off or missing |
| `FISH_VOICE_AEC` | on | `0` disables in-process echo cancellation |
| `FISH_VOICE_AEC_WET` | `0.85` | Mix of cleaned mic into the raw mic, 0 to 1 |
| `FISH_VOICE_AEC_BLEED_DELAY` | `0.3` | Bleed delay once AEC3 is loaded |
| `FISH_VOICE_AGC` | on | Raise a quiet voice to a steady level before anything listens. `0` turns it off |
| `FISH_VOICE_NOISE_SUPPRESSION` | `18` | Decibels of steady background noise (fans, hum) removed from the mic: `0` (off), `6`, `12`, `18` or `21` |
| `FISH_VOICE_BARGE_FRAMES` | `10` | Loud voiced frames in a row that interrupt (about 300 ms) |
| `FISH_VOICE_BARGE_RMS` | `220` | Barge floor seed. It follows the room, never below this |
| `FISH_VOICE_BARGE_PLAYING_GAIN` | `2.2` | Floor multiplier while the speaker plays and AEC is off. At least 1 |
| `FISH_VOICE_DEBUG` | off | `1` or `--debug` logs events. `2`, `trace` or `--trace` adds mic heartbeats, raw audio events, HTTP lines and each raw chat event (`llm.event`) |
| `FISH_VOICE_LOG_DIR` | unset | A folder. Each run writes a complete log to a new `fish-voice-<time>.log` there, at the most detailed level (events, mic heartbeats, raw traffic), whatever the screen shows. It also holds what you said and what the model answered, and is readable only by you. `--log-dir DIR` does the same |

`dev.sh` also reads `FISH_VOICE_ENV_FILE` (env file path), `FISH_VOICE_PORTAUDIO_LIB` (library directories), and `FISH_VOICE_NIX=1` (build PortAudio with nix on a host that is not NixOS).

Commented copies live in [`.env.example`](../../.env.example).

## License

[MIT](../../LICENSE)
