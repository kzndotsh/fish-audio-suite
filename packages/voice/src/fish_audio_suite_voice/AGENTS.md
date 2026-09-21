# `fish_audio_suite_voice`

> Inherits [`../../AGENTS.md`](../../AGENTS.md)

| File | Owns |
| --- | --- |
| [`__init__.py`](__init__.py) | `IsolatedFishTts`, sinks, `BargeGate` |
| [`live.py`](live.py) | Isolated Fish `stream_websocket` |
| [`playback.py`](playback.py) | `PlaybackSink` implementations |
| [`barge.py`](barge.py) | Mic VAD + barge gate |
| [`cli.py`](cli.py) | Duplex recipe (`fish-voice`) |
