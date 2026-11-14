"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Any, Final

from fishaudio.types import LatencyMode, Prosody, TTSConfig

from fish_audio_suite_kit import (
    AudioFormat,
    CueStyle,
    SuiteDefaults,
    known_mp3_bitrate,
    normalize_tts_model,
    strip_base,
    utf8_text,
)
from fish_audio_suite_voice.logutil.debug import warn
from fish_audio_suite_voice.session.cancel import is_cancel_noise
from fish_audio_suite_voice.tts.playback import PlaybackSink, sink_format
from fish_audio_suite_voice.tts.text import WholeText, delta_events
from fish_audio_suite_voice.tts.tune import TtsTune
from fish_audio_suite_voice.tts.turn import run_isolated, run_turn
from fish_audio_suite_voice.tts.websocket import TtsResult, TurnSpec

__all__ = [
    "FishSpeaker",
]

_STOCK: Final = SuiteDefaults()
# The installed SDK Prosody model rejects anything outside this range.
_SDK_VOLUME_LO: Final = -20.0
_SDK_VOLUME_HI: Final = 20.0


@cache
def _warn_coerced(kind: str, given: str, used: str) -> None:
    # Once per distinct pair. A silent swap hid a wrong setting.
    warn(f"fish-voice: {kind} {given!r} is not supported by the Fish SDK, using {used!r}")


def _sdk_latency(latency: str) -> LatencyMode:
    """Map a Fish latency mode onto one the installed SDK accepts.

    ``low`` becomes ``balanced`` with a one-time warning only because
    fish-audio-sdk 1.4.0's ``LatencyMode`` lacks ``"low"``, and building the
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


def _sdk_volume(volume: float) -> float:
    return min(_SDK_VOLUME_HI, max(_SDK_VOLUME_LO, volume))


def _sdk_config(tune: TtsTune, audio_format: AudioFormat) -> TTSConfig:
    """Build the SDK config Fish is sent. Only the SDK's own limits are applied here."""
    return TTSConfig(
        format=audio_format,
        mp3_bitrate=known_mp3_bitrate(_STOCK.mp3_bitrate),
        sample_rate=tune.sample_rate,
        latency=_sdk_latency(tune.latency),
        normalize=_STOCK.normalize,
        chunk_length=tune.chunk_length,
        min_chunk_length=tune.min_chunk_length,
        temperature=tune.temperature,
        top_p=tune.top_p,
        repetition_penalty=tune.repetition_penalty,
        max_new_tokens=_STOCK.max_new_tokens,
        condition_on_previous_chunks=_STOCK.condition_on_previous_chunks,
        early_stop_threshold=_STOCK.early_stop_threshold,
        prosody=Prosody(speed=tune.speed, volume=_sdk_volume(tune.volume)),
    )


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

    Attributes
    ----------
    api_key : str
        Fish key. Read from wherever the caller keeps it, never at import.
    voice_id : str
        The Fish voice. It has no default.
    model : str
        Fish TTS model, sent as the ``model`` header.
    tune : TtsTune
        Latency, speed, sampling, chunking, volume, the cut window, and the fade and level
        applied before the sink. ``TtsTune.from_env()`` reads them from the environment.
    cue_style : CueStyle or None
        How cues are shaped before Fish reads them. When it is None, the style is
        ``CueStyle.for_model(model, official=False, lead=False)``: every cue as written, or
        directions as written for a directed model.
    base_url : str
        Fish origin. A trailing slash is dropped.

    Notes
    -----
    The audio format is the sink's: Fish is asked for what ``sink.audio_format`` names, and
    for PCM when the sink does not say.
    """

    api_key: str = field(repr=False)
    voice_id: str
    model: str = _STOCK.tts_model
    tune: TtsTune = field(default_factory=TtsTune)
    cue_style: CueStyle | None = None
    base_url: str = _STOCK.fish_base

    def __post_init__(self) -> None:
        """Strip the Fish base URL."""
        self.base_url = strip_base(self.base_url)

    def _style(self) -> CueStyle:
        if self.cue_style is not None:
            return self.cue_style
        return CueStyle.for_model(self.model, official=False, lead=False)

    def speak(
        self,
        text: str,
        sink: PlaybackSink,
        *,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> TtsResult:
        """Run one Fish websocket on a private thread and event loop.

        Parameters
        ----------
        text : str
            Full reply. It is shaped with the speaker's cue style (``CueStyle.shape``)
            inside ``speak``.
        sink : PlaybackSink
            Where PCM or encoded audio is written, in the format the sink declares.
        cancel : threading.Event or None, optional
            Set to stop the turn. A new event is created when omitted.
        on_first_audio : Callable or None, optional
            Called once on the websocket's thread at the first audio chunk.
        trace_headers : Mapping or None, optional
            Sent on the websocket upgrade, such as this turn's ``traceparent``. A valid
            ``traceparent`` is forwarded and a missing one is minted.

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
        to open or write never looks like a silent turn. Only an error raised
        while the turn is being cancelled is dropped. A Fish failure is not
        raised: it comes back in ``TtsResult``. Safe to call from ``asyncio.run``
        or ``to_thread``. The Fish websocket must not share the LLM's event loop.
        On cancel, close the httpx client. Do not ``aclose()`` the websocket
        iterator.
        """
        return self.speak_shaped(
            self._style().shape(text),
            sink,
            cancel=cancel,
            on_first_audio=on_first_audio,
            trace_headers=trace_headers,
        )

    def speak_shaped(
        self,
        text: str,
        sink: PlaybackSink,
        *,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> TtsResult:
        """Speak a whole reply that is already shaped, as ``speak`` does after shaping it.

        Parameters
        ----------
        text : str
            The reply as Fish should read it, shaped once with the speaker's cue style.
        sink : PlaybackSink
            Where PCM or encoded audio is written, in the format the sink declares.
        cancel : threading.Event or None, optional
            Set to stop the turn. A new event is created when omitted.
        on_first_audio : Callable or None, optional
            Called once on the websocket's thread at the first audio chunk.
        trace_headers : Mapping or None, optional
            Sent on the websocket upgrade, as for ``speak``.

        Returns
        -------
        TtsResult
            How much was spoken, as for ``speak``.

        Notes
        -----
        A caller that checks or edits the shaped text before speaking it (the duplex
        loop gives a reply a lead cue and drops junk) uses this, so the text is not
        shaped a second time. It raises what ``speak`` raises.
        """
        stop = cancel or threading.Event()
        # The whole reply is known, so its length is the partial-cut window. A
        # comma then never splits a short tail such as "honey?" off a sentence,
        # which drama-3 speaks with uneven prosody. Sentence ends still
        # cut. The window rides with the text, so a replay cuts the same way.
        source = WholeText(text, len(text) or self.tune.partial_chars, self._style())
        return self._run_on_thread(
            lambda: run_turn(
                self._spec(sink, trace_headers), source, sink, stop, on_first_audio=on_first_audio
            ),
            stop,
        )

    def speak_stream(
        self,
        deltas: Iterable[str] | AsyncIterable[str],
        sink: PlaybackSink,
        *,
        cancel: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> TtsResult:
        """Speak a token stream on a private thread while the model is still writing.

        Parameters
        ----------
        deltas : Iterable or AsyncIterable of str
            Model tokens. An async iterable is read on the private loop, so it
            must not depend on the caller's loop.
        sink : PlaybackSink
            Where PCM or encoded audio is written, in the format the sink declares.
        cancel : threading.Event or None, optional
            Set to stop the turn. A new event is created when omitted.
        on_first_audio : Callable or None, optional
            Called once on the websocket's thread at the first audio chunk.
        trace_headers : Mapping or None, optional
            Sent on the websocket upgrade, as for ``speak``.

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
        would keep the reply silent until the model finished. A thought,
        parenthesis, bracket, or URL stays buffered until it closes, so a cut
        cannot speak the inside of a span the closer would remove.
        """
        stop = cancel or threading.Event()

        def turn() -> Coroutine[Any, Any, TtsResult]:
            events = delta_events(
                deltas, stop, partial_chars=self.tune.partial_chars, style=self._style()
            )
            return run_turn(
                self._spec(sink, trace_headers), events, sink, stop, on_first_audio=on_first_audio
            )

        return self._run_on_thread(turn, stop)

    def _run_on_thread(
        self,
        make: Callable[[], Coroutine[Any, Any, TtsResult]],
        cancel: threading.Event,
    ) -> TtsResult:
        result: TtsResult | None = None
        # thread.join does not re-raise. A sink that fails to open would
        # otherwise look like a silent turn.
        error: BaseException | None = None

        def worker() -> None:
            nonlocal result, error
            try:
                result = run_isolated(make())
            except (asyncio.CancelledError, GeneratorExit):
                result = _quiet_result(cancel.is_set())
            except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001 - re-raised on the caller thread after join
                # Tearing down a cancelled turn can raise from anyio or the SDK. That is
                # not a failure. Anything else, such as a sink that raised, is re-raised
                # on the caller's thread after join.
                if is_cancel_noise(exc, cancelled=cancel.is_set()):
                    result = _quiet_result(cancel.is_set())
                else:
                    error = exc

        thread = threading.Thread(target=worker, name="fish-tts", daemon=True)
        thread.start()
        thread.join()
        if error is not None:
            raise error
        if result is None:
            return _quiet_result(cancel.is_set())
        return result

    def _spec(self, sink: PlaybackSink, trace_headers: Mapping[str, str] | None) -> TurnSpec:
        tune = self.tune
        return TurnSpec(
            api_key=self.api_key,
            base_url=self.base_url,
            # The id is MessagePacked into the start event. A surrogate makes
            # that pack fail and the socket never opens. The model is a header.
            voice_id=utf8_text(self.voice_id),
            model=normalize_tts_model(self.model),
            trace_headers=MappingProxyType(dict(trace_headers or {})),
            config=_sdk_config(tune, sink_format(sink)),
            fade_ms=tune.fade_ms,
            loudness_db=tune.loudness_db,
        )
