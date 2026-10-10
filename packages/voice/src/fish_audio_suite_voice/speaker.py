"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Any, Final

from fishaudio.types import LatencyMode, Prosody, TTSConfig

from fish_audio_suite_kit import (
    CHUNK_LENGTH_CLOUD_HI,
    CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_INTERVAL_HI,
    UNIT_INTERVAL_LO,
    AudioFormat,
    FishLatency,
    SuiteDefaults,
    clamp_number,
    known_mp3_bitrate,
    normalize_cues,
    normalize_tts_model,
    scrub_tts,
    strip_base,
    utf8_text,
)
from fish_audio_suite_voice.cancel import is_cancel_noise
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.stream_scrub import delta_events
from fish_audio_suite_voice.tts_turn import TurnSpec, run_isolated, run_turn, text_events
from fish_audio_suite_voice.tune import DEFAULT_FADE_MS
from fish_audio_suite_voice.wire import TtsResult

__all__ = [
    "FishSpeaker",
]

_STOCK: Final = SuiteDefaults()
# The installed SDK Prosody model rejects anything outside this range.
_SDK_VOLUME_LO: Final = -20.0
_SDK_VOLUME_HI: Final = 20.0


_SDK_FORMATS: Final[dict[str, AudioFormat]] = {
    "wav": "wav",
    "pcm": "pcm",
    "mp3": "mp3",
    "opus": "opus",
}


@cache
def _warn_coerced(kind: str, given: str, used: str) -> None:
    # Once per distinct pair. A silent swap hid a wrong sink format.
    warn(f"fish-voice: {kind} {given!r} is not supported by the Fish SDK, using {used!r}")


def _sdk_format(fmt: str) -> AudioFormat:
    key = fmt.strip().lower()
    # pcm16 is this suite's raw PCM name. aac and flac are the proxy's MP3
    # aliases. The SDK rejects every other spelling before the socket opens.
    if key == "pcm16":
        key = "pcm"
    elif key in {"aac", "flac"}:
        _warn_coerced("audio format", key, "mp3")
        key = "mp3"
    chosen = _SDK_FORMATS.get(key)
    if chosen is None:
        _warn_coerced("audio format", key, "pcm")
        return "pcm"
    return chosen


def _sdk_latency(latency: str) -> LatencyMode:
    """Map a Fish latency mode onto one the installed SDK accepts.

    ``low`` becomes ``balanced`` with a one-time warning only because
    fish-audio-sdk 1.3.0's ``LatencyMode`` lacks ``"low"``, and building the
    config with it raises before any audio is sent. The Fish docs list ``low``
    as a valid mode, so keep it in the kit's ``FishLatency`` and drop this
    coercion once the SDK accepts it. Any other unknown value becomes ``normal``.
    """
    key = latency.strip().lower()
    if key == "low":
        _warn_coerced("latency", key, "balanced")
        return "balanced"
    if key == "balanced":
        return "balanced"
    if key != "normal":
        _warn_coerced("latency", key, "normal")
    return "normal"


def _sdk_chunk_length(chunk_length: int) -> int:
    # Self-hosted HTTP allows 1000. The SDK model rejects anything above 300.
    return min(CHUNK_LENGTH_CLOUD_HI, max(CHUNK_LENGTH_LO, chunk_length))


def _sdk_sample_rate(sample_rate: int) -> int:
    # The SDK accepts 0 and rates past the WAV header. The writer then raises
    # and the collected audio is lost.
    if sample_rate < 1 or sample_rate > 2**32 - 1:
        return _STOCK.sample_rate
    return sample_rate


def _sdk_speed(speed: float) -> float:
    return clamp_number(speed, TTS_SPEED_LO, TTS_SPEED_HI, _STOCK.speed, float)


def _sdk_volume(volume: float) -> float:
    return min(_SDK_VOLUME_HI, max(_SDK_VOLUME_LO, volume))


def _spoken(text: str, *, lead: bool = True, official: bool = False) -> str:
    return normalize_cues(scrub_tts(text), lead=lead, official=official)


def _quiet_result(cancelled: bool) -> TtsResult:
    return TtsResult(
        spoken_so_far="",
        bytes_played=0,
        got_audio=False,
        cancelled=cancelled,
        tts_first_audio_ms=None,
        tts_first_text_ms=None,
    )


@dataclass(kw_only=True, eq=False)
class FishSpeaker:
    """Speak text through Fish, one websocket per turn on a private loop.

    The turn never shares the caller's event loop, so the LLM stream is not
    held up. Do not merge the LLM socket onto this websocket.
    """

    api_key: str = field(repr=False)
    voice_id: str
    model: str = _STOCK.tts_model
    latency: FishLatency = _STOCK.latency
    speed: float = _STOCK.speed
    audio_format: str = "pcm"
    sample_rate: int = _STOCK.sample_rate
    temperature: float = _STOCK.temperature
    top_p: float = _STOCK.top_p
    repetition_penalty: float = _STOCK.repetition_penalty
    chunk_length: int = _STOCK.chunk_length
    min_chunk_length: int = _STOCK.min_chunk_length
    volume: float = _STOCK.volume
    mood_lead: bool = False
    official_cues: bool = False
    partial_chars: int = _STOCK.tts_partial_chars
    fade_ms: float = DEFAULT_FADE_MS
    base_url: str = _STOCK.fish_base
    trace_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Strip the Fish base URL and copy the trace header map."""
        self.base_url = strip_base(self.base_url)
        self.trace_headers = dict(self.trace_headers)

    def speak(
        self,
        text: str,
        sink: PlaybackSink,
        *,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        """Run one Fish websocket on a private thread and event loop.

        Parameters
        ----------
        text : str
            Full reply. Scrubbed and given a lead cue inside ``speak``.
        sink : PlaybackSink
            Where PCM or encoded audio is written.
        cancel : threading.Event or None, optional
            Set to stop the turn. A new event is created when omitted.
        on_first_audio : Callable or None, optional
            Called once on the websocket's thread at the first audio chunk.

        Returns
        -------
        TtsResult
            How much was spoken. ``spoken_so_far`` is the played prefix, not
            the full unplayed reply.

        Raises
        ------
        PortAudioMissingError
            ``sounddevice`` could not load libportaudio when the sink opened.
        OSError
            The sink could not open or write, for example ``mpv`` is not on
            PATH or the disk is full.

        Notes
        -----
        Any other error the sink raises is re-raised too, so a sink that fails
        to open never looks like a silent turn. A Fish failure is not raised: it
        comes back in ``TtsResult``. Safe to call from ``asyncio.run`` or
        ``to_thread``. The Fish websocket
        must not share the LLM's event loop. On cancel, close the httpx client.
        Do not ``aclose()`` the websocket iterator.
        """
        stop = cancel or threading.Event()
        return self._run_on_thread(
            lambda: self._speak_here(text, sink, stop, on_first_audio=on_first_audio), stop
        )

    def speak_stream(
        self,
        deltas: Iterable[str] | AsyncIterable[str],
        sink: PlaybackSink,
        *,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        """Speak a token stream on a private thread while the model is still writing.

        Parameters
        ----------
        deltas : Iterable or AsyncIterable of str
            Model tokens. An async iterable is read on the private loop, so it
            must not depend on the caller's loop.
        sink : PlaybackSink
            Where PCM or encoded audio is written.
        cancel : threading.Event or None, optional
            Set to stop the turn. A new event is created when omitted.
        on_first_audio : Callable or None, optional
            Called once on the websocket's thread at the first audio chunk.

        Returns
        -------
        TtsResult
            How much was spoken. Tokens arrive as a stream, so there is no
            full text to replay: a failed turn is not retried here, and the
            caller can speak the finished reply instead.

        Raises
        ------
        PortAudioMissingError
            ``sounddevice`` could not load libportaudio when the sink opened.
        OSError
            The sink could not open or write, as for ``speak``.

        Notes
        -----
        Flushes once after the first sentence and once at the end. Fish holds
        text until a chunk fills or a flush arrives, so one flush at the end
        would keep the reply silent until the model finished.
        """
        stop = cancel or threading.Event()
        return self._run_on_thread(
            lambda: self._speak_stream_here(
                deltas, sink, stop, early_flush=True, on_first_audio=on_first_audio
            ),
            stop,
        )

    def _run_on_thread(
        self,
        make: Callable[[], Coroutine[Any, Any, TtsResult]],
        cancel: threading.Event,
    ) -> TtsResult:
        result: TtsResult | None = None
        # thread.join does not re-raise. A sink that fails to open would
        # otherwise look like a silent turn.
        error: Exception | None = None

        def worker() -> None:
            nonlocal result, error
            try:
                result = run_isolated(make())
            except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError, GeneratorExit) as e:
                if not is_cancel_noise(e, cancelled=cancel.is_set()):
                    warn(f"[tts] {e}")
                result = _quiet_result(cancel.is_set())
            except Exception as exc:  # noqa: BLE001 - re-raised on the caller thread after join
                error = exc

        thread = threading.Thread(target=worker, name="fish-tts", daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        if result is None:
            return _quiet_result(cancel.is_set())
        return result

    async def _speak_here(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        """Speak one full string on the caller's loop.

        Parameters
        ----------
        text : str
            Reply text. Cues are normalized before the websocket opens.
        sink : PlaybackSink
            Playback target.
        cancel : threading.Event
            Stops the turn. Also used as the retry boundary: 429 and 5xx
            replay only before the first audio byte.
        on_first_audio : Callable or None, optional
            Called once at the first audio chunk.

        Returns
        -------
        TtsResult
            Played audio and the spoken prefix.

        Notes
        -----
        ``speak`` calls this on a private loop, which keeps the caller's loop
        free for the LLM.
        """
        prepared = _spoken(text, lead=self.mood_lead, official=self.official_cues)
        # The full reply is known here, so use its length as the partial-cut
        # window. This prevents comma cuts from splitting short orphan fragments
        # like "honey?" off the end of a sentence — fragments that drama-3-preview
        # renders with inconsistent prosody. Sentence-boundary cuts still fire.
        # partial_chars flows into TurnSpec so send_turn's replay path uses the
        # same window (it rebuilds text_events from sent_text + spec.partial_chars).
        partial = len(prepared) or self.partial_chars
        return await run_turn(
            self._spec(partial_chars=partial),
            text_events(prepared, cancel, partial),
            sink,
            cancel,
            sent_text=prepared,
            on_first_audio=on_first_audio,
        )

    async def _speak_stream_here(
        self,
        deltas: Iterable[str] | AsyncIterable[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        *,
        early_flush: bool = False,
        on_first_audio: Callable[[], None] | None = None,
    ) -> TtsResult:
        """Stream model deltas, cutting them into Fish text events.

        Parameters
        ----------
        deltas : Iterable or AsyncIterable of str
            Token stream. Empty pieces are skipped.
        sink : PlaybackSink
            Playback target.
        cancel : threading.Event
            Stops the turn.
        early_flush : bool, optional
            Flush once after the first piece so audio starts before the model
            finishes. Default False.
        on_first_audio : Callable or None, optional
            Called once at the first audio chunk.

        Returns
        -------
        TtsResult
            Played audio. The text arrived as deltas, so there is no full text
            to replay and a failed turn is not retried.

        Notes
        -----
        ``speak_stream`` calls this on a private loop. Duplex uses it when
        streaming is on. Otherwise it waits for the full reply and uses
        ``speak``, which can replay on 429 or 5xx. A thought,
        parenthesis, bracket, or URL stays buffered until it closes, so a
        cut cannot speak the inside of a span the closer would remove.
        """
        return await run_turn(
            self._spec(),
            delta_events(
                deltas,
                cancel,
                partial_chars=self.partial_chars,
                mood_lead=self.mood_lead,
                official=self.official_cues,
                early_flush=early_flush,
            ),
            sink,
            cancel,
            sent_text="",
            on_first_audio=on_first_audio,
        )

    def _spec(self, *, partial_chars: int | None = None) -> TurnSpec:
        return TurnSpec(
            api_key=self.api_key,
            base_url=self.base_url,
            # The id is MessagePacked into the start event. A surrogate makes
            # that pack fail and the socket never opens. The model is a header.
            voice_id=utf8_text(self.voice_id),
            model=normalize_tts_model(self.model),
            audio_format=_sdk_format(self.audio_format),
            latency=_sdk_latency(self.latency),
            speed=_sdk_speed(self.speed),
            sample_rate=_sdk_sample_rate(self.sample_rate),
            partial_chars=partial_chars if partial_chars is not None else self.partial_chars,
            trace_headers=MappingProxyType(dict(self.trace_headers)),
            config=self._tts_config(),
            fade_ms=self.fade_ms,
        )

    def _tts_config(self) -> TTSConfig:
        return TTSConfig(
            format=_sdk_format(self.audio_format),
            mp3_bitrate=known_mp3_bitrate(_STOCK.mp3_bitrate),
            sample_rate=_sdk_sample_rate(self.sample_rate),
            latency=_sdk_latency(self.latency),
            normalize=_STOCK.normalize,
            chunk_length=_sdk_chunk_length(self.chunk_length),
            min_chunk_length=self.min_chunk_length,
            temperature=clamp_number(
                self.temperature, UNIT_INTERVAL_LO, UNIT_INTERVAL_HI, _STOCK.temperature, float
            ),
            top_p=clamp_number(self.top_p, UNIT_INTERVAL_LO, UNIT_INTERVAL_HI, _STOCK.top_p, float),
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=_STOCK.max_new_tokens,
            condition_on_previous_chunks=_STOCK.condition_on_previous_chunks,
            early_stop_threshold=_STOCK.early_stop_threshold,
            prosody=Prosody(speed=_sdk_speed(self.speed), volume=_sdk_volume(self.volume)),
        )
