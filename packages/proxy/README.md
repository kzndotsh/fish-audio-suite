# fish-audio-suite-proxy

OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions` on port 8849.

`response_format=srt` or `vtt` returns those caption files from Fish segments. `timestamp_granularities` and `timestamp_granularities[]` are accepted. JSON `input_audio` is accepted. `pcm16` and `fish-audio/` model slugs are mapped onto native Fish. Errors use the OpenAI `{error:…}` envelope.

Unofficial. Not affiliated with Fish Audio.
