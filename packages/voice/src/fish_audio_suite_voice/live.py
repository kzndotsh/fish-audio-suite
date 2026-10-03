"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from functools import cache
from typing import Any, cast

from fishaudio.types import AudioFormat, LatencyMode, Prosody, TTSConfig

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    CLOUD_CHUNK_HI,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_HI,
    UNIT_LO,
    SuiteDefaults,
    clamp_num,
    known_mp3_bitrate,
    known_tts_model,
    normalize_cues,
    scrub_tts,
    strip_base,
    utf8_text,
)
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.session import (
    IsolatedResult,
    TurnSpec,
    is_cancel_noise,
    run_isolated,
    run_turn,
    text_events,
)
from fish_audio_suite_voice.stream_scrub import delta_events

_STOCK = SuiteDefaults()
# The installed SDK Prosody model rejects anything outside this range.
_SDK_VOLUME_LO = -20.0
_SDK_VOLUME_HI = 20.0


_SDK_FORMATS = frozenset({"wav", "pcm", "mp3", "opus"})


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
    if key not in _SDK_FORMATS:
        _warn_coerced("audio format", key, "pcm")
        key = "pcm"
    return cast(AudioFormat, key)


def _sdk_latency(latency: str) -> LatencyMode:
    # low is a Fish HTTP mode. This SDK only accepts normal and balanced, and
    # building the config with low raises before any audio is sent.
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
    return min(CLOUD_CHUNK_HI, max(CHUNK_LENGTH_LO, chunk_length))


def _sdk_sample_rate(sample_rate: int) -> int:
    # The SDK accepts 0 and rates past the WAV header. The writer then raises
    # and the collected audio is lost.
    if sample_rate < 1 or sample_rate > 2**32 - 1:
        return _STOCK.sample_rate
    return sample_rate


def _sdk_speed(speed: float) -> float:
    return clamp_num(speed, TTS_SPEED_LO, TTS_SPEED_HI, _STOCK.speed, float)


def _sdk_volume(volume: float) -> float:
    return min(_SDK_VOLUME_HI, max(_SDK_VOLUME_LO, volume))


def _spoken(text: str, *, lead: bool = True) -> str:
    return normalize_cues(scrub_tts(text), lead=lead)


def _quiet_result(cancelled: bool) -> IsolatedResult:
    return IsolatedResult(
        spoken_so_far="",
        bytes_played=0,
        got_audio=False,
        cancelled=cancelled,
        ttfa_ms=None,
        llm_ttfs_ms=None,
    )


@dataclass(kw_only=True, eq=False)
class IsolatedFishTts:
    """Fish `stream_websocket` on a fresh loop. Do not merge the LLM socket onto this WS."""

    api_key: str = field(repr=False)
    voice_id: str
    model: str = _STOCK.tts_model
    latency: str = _STOCK.latency
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
    partial_chars: int = _STOCK.tts_partial_chars
    base_url: str = _STOCK.fish_base
    trace_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Strip the Fish base URL and copy the trace header map."""
        self.base_url = strip_base(self.base_url)
        self.trace_headers = dict(self.trace_headers)

    def speak_isolated(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> IsolatedResult:
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
        IsolatedResult
            How much was spoken. ``spoken_so_far`` is the played prefix, not
            the full unplayed reply.

        Raises
        ------
        Exception
            Re-raised from the private thread. A sink that fails to open,
            including ``PortAudioMissingError``, must not look like silence.

        Notes
        -----
        Safe to call from ``asyncio.run`` or ``to_thread``. The Fish websocket
        must not share the LLM's event loop. On cancel, close the httpx client.
        Do not ``aclose()`` the websocket iterator.
        """
        cancel = cancel or threading.Event()
        return self._run_on_thread(
            lambda: self.speak(text, sink, cancel, on_first_audio=on_first_audio), cancel
        )

    def speak_stream_isolated(
        self,
        deltas: Iterable[str] | AsyncIterable[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
    ) -> IsolatedResult:
        """Speak a token stream on a private thread while the model is still writing.

        Parameters
        ----------
        deltas : Iterable or AsyncIterable of str
            Model tokens. An async iterable is read on the private loop, so it
            must not depend on the caller's loop.
        sink : PlaybackSink
            Where PCM or encoded audio is written.
        cancel : threading.Event
            Set to stop the turn.
        on_first_audio : Callable or None, optional
            Called once on the websocket's thread at the first audio chunk.

        Returns
        -------
        IsolatedResult
            How much was spoken. ``sent_text`` is empty, so a failed turn is not
            replayed here. The caller can speak the finished reply instead.

        Raises
        ------
        Exception
            Re-raised from the private thread, as ``speak_isolated`` does.

        Notes
        -----
        Flushes once after the first sentence and once at the end. Fish holds
        text until a chunk fills or a flush arrives, so one flush at the end
        would keep the reply silent until the model finished.
        """
        return self._run_on_thread(
            lambda: self.speak_deltas(
                deltas, sink, cancel, early_flush=True, on_first_audio=on_first_audio
            ),
            cancel,
        )

    def _run_on_thread(
        self,
        make: Callable[[], Coroutine[Any, Any, IsolatedResult]],
        cancel: threading.Event,
    ) -> IsolatedResult:
        result: IsolatedResult | None = None
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

    async def speak(
        self,
        text: str,
        sink: PlaybackSink,
        cancel: threading.Event,
        on_first_audio: Callable[[], None] | None = None,
    ) -> IsolatedResult:
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
        IsolatedResult
            Played audio and the spoken prefix.

        Notes
        -----
        Prefer ``speak_isolated`` when the caller is already inside an event
        loop that must stay free for the LLM. Duplex does that.
        """
        prepared = _spoken(text, lead=self.mood_lead)
        return await run_turn(
            self._spec(),
            text_events(prepared, cancel, self.partial_chars),
            sink,
            cancel,
            sent_text=prepared,
            on_first_audio=on_first_audio,
        )

    async def speak_deltas(
        self,
        deltas: Iterable[str] | AsyncIterable[str],
        sink: PlaybackSink,
        cancel: threading.Event,
        *,
        early_flush: bool = False,
        on_first_audio: Callable[[], None] | None = None,
    ) -> IsolatedResult:
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
        IsolatedResult
            Played audio. ``sent_text`` stays empty because the text arrived
            as deltas, so a retry cannot replay the turn.

        Notes
        -----
        Duplex reaches this through ``speak_stream_isolated`` when streaming is
        on. Otherwise it waits for the full reply and uses ``speak_isolated``,
        which can replay on 429 or 5xx. A thought,
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
                early_flush=early_flush,
            ),
            sink,
            cancel,
            sent_text="",
            on_first_audio=on_first_audio,
        )

    def _spec(self) -> TurnSpec:
        return TurnSpec(
            api_key=self.api_key,
            base_url=self.base_url,
            # The id is MessagePacked into the start event. A surrogate makes
            # that pack fail and the socket never opens. The model is a header.
            voice_id=utf8_text(self.voice_id),
            model=known_tts_model(self.model),
            audio_format=_sdk_format(self.audio_format),
            latency=_sdk_latency(self.latency),
            speed=_sdk_speed(self.speed),
            sample_rate=_sdk_sample_rate(self.sample_rate),
            partial_chars=self.partial_chars,
            trace_headers=self.trace_headers,
            config=self._tts_config(),
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
            temperature=clamp_num(self.temperature, UNIT_LO, UNIT_HI, _STOCK.temperature, float),
            top_p=clamp_num(self.top_p, UNIT_LO, UNIT_HI, _STOCK.top_p, float),
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=_STOCK.max_new_tokens,
            condition_on_previous_chunks=_STOCK.condition_on_previous_chunks,
            early_stop_threshold=_STOCK.early_stop_threshold,
            prosody=Prosody(speed=_sdk_speed(self.speed), volume=_sdk_volume(self.volume)),
        )


__all__ = ["IsolatedFishTts", "IsolatedResult", "is_cancel_noise"]
