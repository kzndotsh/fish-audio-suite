# Glossary

Terms used across the fish-audio-suite docs and code.

| Term | Meaning |
| --- | --- |
| **AEC / AEC3** | Acoustic echo cancellation. WebRTC's AEC3 removes the bot's own voice from the microphone signal, using what was sent to the speakers as the reference ("far end") |
| **AGC** | Automatic gain control. Part of the mic cleanup: it raises a quiet voice to a steady level (`FISH_VOICE_AGC`, on by default) |
| **ASR** | Automatic speech recognition: Fish's speech-to-text, which hears one finished clip. `FISH_VOICE_STT=deepgram` streams to Deepgram Flux instead |
| **Backchannel** | A listener noise such as "mm-hmm" or "yeah". Over a reply it means "go on" and is ignored; on its own turn "yeah" is an answer |
| **Barge-in** | Talking over the bot. Enough loud voiced frames in a row cancel the reply and start a new listen |
| **Barge floor** | The loudness a frame must pass to count toward a barge-in: the quiet percentile of recent mic frames, never below `FISH_VOICE_BARGE_RMS` |
| **Bleed delay** | Time after playback starts before barge-in listens, so the speaker's own sound is not taken for a user: 0.3 s with AEC loaded (`FISH_VOICE_AEC_BLEED_DELAY`), 0.9 s without (`FISH_VOICE_BLEED_DELAY`) |
| **Carry-over** | What a barge-in leaves for the next turn (`CarryOver`): the clip that tripped it, which the next listen starts from instead of waiting out the cooldown, and the unheard rest of the cut reply, spoken again if the interrupt turns out to be noise |
| **Chunk length** | Fish's `chunk_length`: how much text Fish gathers before it starts synthesizing. Smaller starts sooner. The cloud allows 100 to 300 |
| **Cooldown** | Seconds after a reply ends before a new turn can start (`FISH_VOICE_COOLDOWN`). The mic stays open, and the `listen.cooldown` debug line reports what it heard |
| **CORS** | Cross-origin resource sharing. A browser blocks a page from calling another server unless that server sends the right headers. The proxy sends none, so browser apps need a reverse proxy that adds them |
| **Cue** | A bracketed stage direction for Fish TTS, such as `[happy]` or `[laughing]`. Never spoken as words |
| **Duplex** | The two-way loop in `fish-voice`: listen and speak, with interruption |
| **Eager end of turn** | Flux's early guess that you have probably finished, sent a little before it is sure (`FISH_VOICE_EAGER_EOT_THRESHOLD`). The model's reply starts being written then, held back until Flux confirms the same words |
| **Early flush** | A `FlushEvent` sent after the first sentence of a streamed reply, so Fish starts speaking before the model finishes |
| **End-of-speech silence** | How long the mic must hear quiet before your turn ends (`FISH_VOICE_SILENCE_FRAMES`, 30 ms per frame) |
| **End-of-turn check** | The Fish-path option (`FISH_VOICE_TURN_CHECK`) that ends the first clip on a short silence and reads the words to decide whether the turn is over. If it stops on "and", "because" or a comma, the mic keeps listening and the next clip is joined on |
| **End of turn (Flux)** | Deepgram Flux's own decision that you have finished speaking, with a confidence from 0 to 1. It ends the turn at `FISH_VOICE_EOT_THRESHOLD`; if it has not by the silence limit, the app asks it to (`trigger=manual`) |
| **Far end** | The audio sent to the speakers. Echo cancellation subtracts it from what the mic hears |
| **Flux** | Deepgram's streaming speech recognition model with turn detection built in (`flux-general-en`, `flux-general-multi`) |
| **Flush** | Tells Fish to synthesize the text it holds. Fish waits for a flush (or a full chunk) before speaking |
| **Held span** | Streamed text kept back because it is unfinished: an open `**`, `<think>`, `(` or URL that a scrub rule would remove once closed |
| **Import contracts** | The rules in `pyproject.toml` (`[tool.importlinter]`) for which package and voice module may import which: the proxy and voice never import each other, kit imports neither, and voice modules import only the layers below their own. `just layers` checks them |
| **Interim words** | The words Flux has so far in a turn, sent about four times a second. The TUI shows them as you speak |
| **Kit** | `fish-audio-suite-kit`, the shared library |
| **Latency mode** | Fish's `latency` setting: `normal` (the most stable and best-sounding output, with slightly higher latency) or `balanced` (about 300 ms to first audio, Fish's API default and the one recommended for conversation). The SDK has no `low` |
| **Lead cue** | A cue at the start of a reply or sentence that sets its mood |
| **Loudness normalisation** | Bringing each reply to a steady level (`FISH_VOICE_NORMALIZE_LOUDNESS`), measured on the speech as it streams and applied before the speaker and the echo canceller, with a soft limiter so a boost never clips |
| **Mic cleanup** | WebRTC's audio processor run on every mic frame before voice detection and speech recognition: a high-pass filter, automatic gain, noise suppression, and AEC3 while the speaker plays. Needs the `aec` extra, which `cli` includes |
| **Noise suppression** | Part of the mic cleanup: it removes steady background noise such as a fan (`FISH_VOICE_NOISE_SUPPRESSION`, in dB: 0 for off, 6, 12, 18 or 21; default 18). It does little against key clicks |
| **OpenAI-compatible** | A server that accepts OpenAI's request formats, such as `/v1/chat/completions` or `/v1/audio/speech`, so OpenAI clients can talk to it by changing the base URL |
| **p50 / p90** | The median, and the value 90% of requests stay under. For an LLM's first token, p90 is the slow turn you notice |
| **Private loop** | The separate asyncio event loop and thread a Fish websocket turn runs on |
| **Provider sort** | OpenRouter's `provider.sort`: which of a model's providers to prefer. The voice app sends `latency` by default |
| **Reference id / voice id** | The id of a Fish voice model. Fish calls it `reference_id`; OpenAI clients send it as `voice`; the voice app reads `FISH_VOICE_ID` |
| **Session mic** | The one mic stream (`SessionMic`) that stays open for the whole session. A reader thread cleans every frame and keeps the last two seconds, so a listen that starts late, such as the one after a barge-in, misses no words. Mute releases it |
| **Sink** | Where audio goes: `SounddeviceSink` (speakers), `FileSink`, `StdoutSink`, `MpvSink` |
| **Soft limiter** | Bends peaks above a knee toward a ceiling below full scale instead of cutting them flat, so a boosted reply does not clip |
| **Speculative reply** | A reply written ahead of time for a turn Flux thinks is probably over. Invisible and silent until Flux confirms the same words; otherwise thrown away |
| **Spoken so far** | The part of a reply that was actually heard, estimated from bytes played. Only this goes into history after a barge-in |
| **SSE** | Server-Sent Events: the streaming format OpenAI-compatible chat servers use to send tokens as they are generated |
| **Tone hint** | A short note, in round brackets and for one request only, telling the model how the user sounded (`laughter`, `tired`), from the marks Fish ASR puts in its text (`FISH_VOICE_HEAR_TONE`, experimental). Never saved, never spoken |
| **TTFA / first token** | Time to first audio from Fish; time to the first token from the LLM |
| **Turn** | One user line and the bot's reply |
| **Underrun** | The speaker running out of audio, heard as a click or a gap. Logged as `tts underrun`, with `chunk_wait`, `in_python` and `write` timings that say whether the audio came late or the playback thread stalled |
| **VAD** | Voice activity detection: deciding whether a 30 ms frame contains speech, for starting and ending a turn and for barge-in |
| **VAD engine** | The detector `FISH_VOICE_VAD_ENGINE` picks: `webrtc` (the default), `silero` (the `silero` extra), or a `package.module:factory` plugin that returns an object with `is_speech(frame, sample_rate)`. `fish-voice` stops at start-up when it cannot load |
| **VAD strictness** | `FISH_VOICE_VAD`, from 0 to 3: how sure the engine must be before a frame counts as speech. Higher ignores more noise and may miss soft speech. For Silero it sets the probability threshold |
