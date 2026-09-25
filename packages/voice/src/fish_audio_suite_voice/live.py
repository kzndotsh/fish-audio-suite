"""One Fish TTS websocket per turn, isolated from the caller's event loop."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from fishaudio import TextEvent
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
    ends_sentence,
    hold_tts,
    known_mp3_bitrate,
    known_tts_model,
    normalize_cues,
    scrub_tts,
    skip_empty_delta,
    split_tts_piece,
    strip_base,
    utf8_text,
)
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.playback import PlaybackSink
from fish_audio_suite_voice.session import (
    IsolatedResult,
    TurnSpec,
    as_async,
    is_cancel_noise,
    run_isolated,
    run_turn,
    text_events,
)
from fish_audio_suite_voice.wire import flush_if_sent

_STOCK = SuiteDefaults()
# The installed SDK Prosody model rejects anything outside this range.
_SDK_VOLUME_LO = -20.0
_SDK_VOLUME_HI = 20.0


_SDK_FORMATS = frozenset({"wav", "pcm", "mp3", "opus"})


def _sdk_format(fmt: str) -> AudioFormat:
    key = fmt.strip().lower()
    # pcm16 is this suite's raw PCM name. aac and flac are the proxy's MP3
    # aliases. The SDK rejects every other spelling before the socket opens.
    if key == "pcm16":
        key = "pcm"
    elif key in {"aac", "flac"}:
        key = "mp3"
    if key not in _SDK_FORMATS:
        key = "pcm"
    return cast(AudioFormat, key)


def _sdk_latency(latency: str) -> LatencyMode:
    # low is a Fish HTTP mode. This SDK only accepts normal and balanced, and
    # building the config with low raises before any audio is sent.
    # "LOW" used to miss both comparisons and go out as normal.
    key = latency.strip().lower()
    if key == "low":
        return "balanced"
    if key == "balanced":
        return "balanced"
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


def _hold_at(text: str, *, line_start: bool, sentence_start: bool, before: str = "") -> int:
    return hold_tts(
        text,
        line_start=line_start,
        sentence_start=sentence_start,
        before=before,
    )


def _at_line_start(ready: str) -> bool:
    text = ready.rstrip(" \t")
    return not text or text.endswith("\n")


def _stable_prefix(
    text: str, *, line_start: bool, sentence_start: bool, before: str = ""
) -> tuple[str, str]:
    cut = _hold_at(text, line_start=line_start, sentence_start=sentence_start, before=before)
    return text[:cut], text[cut:]


# A closer split from its word ("words" then "** ") is not an operator.
# " * " still is: the mark does not start the chunk.
_ORPHAN_CLOSER_RE = re.compile(r"^[*_`~]+(?=\s)")


def _fold_stream_breaks(text: str) -> str:
    # A carriage return or a Unicode line separator is a newline only after
    # scrub. Until then the next mood looks mid-sentence and is spoken.
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )


def _glue_sentence_stop(ready: str, piece: str) -> str:
    # "(https://example.com)" is removed after "door " was already buffered.
    # The period arrives next, and Fish says "door" and then "dot".
    if (
        piece
        and piece[0] in ".!?…。！？,;:，；："
        and ready[-1:].isspace()
        and ready[-1:] not in "\n\r"
    ):
        ready = ready[:-1]
    return ready + piece


def _drop_orphan_closer(stable: str, ready: str) -> str:
    if not ready or not ready[-1].isalnum():
        return stable
    # "~~~" then a newline is a code fence, not a leftover closer.
    # Stripping it spoke the code.
    if stable.startswith(("~~~", "```")):
        return stable
    return _ORPHAN_CLOSER_RE.sub("", stable)


def _continues_sentence(ready: str, incoming: str = "") -> bool:
    if not ready.rstrip():
        return False
    # A new line is its own sentence, even when the previous line has no stop.
    # "List\nExcited," is a cue. The newline was already sent, so the next
    # token would otherwise look mid-sentence and the mood would be spoken.
    if ready.rstrip(" \t").endswith("\n"):
        return False
    if ends_sentence(ready):
        return False
    # "Hello؟" + " Excited" — the space is still in this token, not in ready.
    # "Dr." + " Happy" is not a new sentence.
    return not (incoming[:1].isspace() and ends_sentence(ready + incoming[:1]))


def _scrub_chunk(
    text: str,
    *,
    lead: bool = True,
    line_start: bool = True,
    before: str = "",
    after: str = "",
    continued: bool = False,
) -> str:
    """Scrub one stable stream chunk with the text already accepted around it."""
    if not text:
        return ""
    # A chunk edge is the middle of the reply. Whole-string scrub strips
    # that space, and cue rewrite strips it again. The newline is how the
    # next chunk knows it starts a line.
    lead_space = text[0] == " "
    trail_space = text[-1] == " "
    body = text.lstrip(" \t")
    lead_breaks = min(len(body) - len(body.lstrip("\n")), 2)
    ended_line = text.rstrip(" \t").endswith("\n")
    cleaned = scrub_tts(
        text,
        line_start=line_start,
        continued=continued,
        before=before[-1:],
        after=after[:1],
    )
    spoken = normalize_cues(cleaned, lead=lead)
    if (lead_space or cleaned[:1] == " ") and spoken[:1] != " ":
        spoken = f" {spoken}"
    if (trail_space or cleaned[-1:] == " ") and spoken[-1:] != " ":
        spoken = f"{spoken} "
    if lead_breaks and not spoken.startswith("\n"):
        spoken = ("\n" * lead_breaks) + spoken.lstrip(" ")
    elif ended_line and not spoken.endswith("\n"):
        spoken = f"{spoken.rstrip(' ')}\n"
    return spoken


def _quiet_result(cancelled: bool) -> IsolatedResult:
    return IsolatedResult(
        spoken_so_far="",
        bytes_played=0,
        got_audio=False,
        cancelled=cancelled,
        ttfa_ms=None,
        llm_ttfs_ms=None,
    )


@dataclass(kw_only=True, repr=False, eq=False)
class IsolatedFishTts:
    """Fish `stream_websocket` on a fresh loop. Do not merge the LLM socket onto this WS."""

    api_key: str
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
        result: IsolatedResult | None = None
        # thread.join does not re-raise. A sink that fails to open would
        # otherwise look like a silent turn.
        error: Exception | None = None

        def worker() -> None:
            nonlocal result, error
            try:
                result = run_isolated(self.speak(text, sink, cancel))
            except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError, GeneratorExit) as e:
                if not is_cancel_noise(e):
                    warn(f"[tts] {e}")
                result = _quiet_result(cancel.is_set())
            except Exception as exc:
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

        Returns
        -------
        IsolatedResult
            Played audio and the spoken prefix.

        Notes
        -----
        Prefer ``speak_isolated`` when the caller is already inside an event
        loop that must stay free for the LLM. Duplex does that.
        """
        prepared = _spoken(text)
        return await run_turn(
            self._spec(),
            text_events(prepared, cancel, self.partial_chars),
            sink,
            cancel,
            sent_text=prepared,
        )

    async def speak_deltas(
        self,
        deltas: Iterable[str] | AsyncIterator[str],
        sink: PlaybackSink,
        cancel: threading.Event,
    ) -> IsolatedResult:
        """Stream model deltas, cutting them into Fish text events.

        Parameters
        ----------
        deltas : Iterable or AsyncIterator of str
            Token stream. Empty pieces are skipped.
        sink : PlaybackSink
            Playback target.
        cancel : threading.Event
            Stops the turn.

        Returns
        -------
        IsolatedResult
            Played audio. ``sent_text`` stays empty because the text arrived
            as deltas, so a retry cannot replay the turn.

        Notes
        -----
        The duplex loop does not call this. It waits for the full reply and
        uses ``speak_isolated``, which can replay on 429 or 5xx. A thought,
        parenthesis, bracket, or URL stays buffered until it closes, so a
        cut cannot speak the inside of a span the closer would remove.
        """
        return await run_turn(
            self._spec(),
            self._delta_events(deltas, cancel),
            sink,
            cancel,
            sent_text="",
        )

    async def _delta_events(
        self,
        deltas: Iterable[str] | AsyncIterator[str],
        cancel: threading.Event,
    ) -> AsyncIterator[Any]:
        raw = ""
        ready = ""
        # The last accepted character. ready is empty once that text is sent,
        # and the next span still needs the neighbor so its gap survives.
        last = ""
        sent = 0
        # Text already sent on this line. An empty ready buffer is not a new
        # line, so a star after "Hello." must not be stripped as a bullet.
        sent_line = ""

        def counted(piece: str) -> TextEvent | None:
            nonlocal sent
            event = self._text_event(piece)
            if event is None:
                return None
            sent += 1
            return event

        def take_ready() -> str | None:
            nonlocal ready
            split = split_tts_piece(ready, self.partial_chars, flush_rest=False)
            if split is None:
                return None
            piece, ready = split
            return piece

        async for tok in as_async(deltas):
            if cancel.is_set():
                break
            # A space-only token is not a TextEvent, but it is the boundary
            # between words. Dropping it here joins those words.
            raw = _fold_stream_breaks(raw + tok)
            sentence_start = not _continues_sentence(ready, raw)
            stable, raw = _stable_prefix(
                raw,
                line_start=_at_line_start(sent_line + ready),
                sentence_start=sentence_start,
                before=ready,
            )
            if stable:
                stable = _drop_orphan_closer(stable, ready)
                ready = _glue_sentence_stop(
                    ready,
                    _scrub_chunk(
                        stable,
                        lead=sentence_start,
                        line_start=_at_line_start(sent_line + ready),
                        before=ready[-1:] or last,
                        after=raw[:1],
                        continued=bool((sent_line + ready).strip()),
                    ),
                )
                if ready:
                    last = ready[-1]
            while True:
                piece = take_ready()
                if piece is None:
                    break
                sent_line = (sent_line + piece).rsplit("\n", 1)[-1]
                event = counted(piece)
                if event is not None:
                    yield event
        if not cancel.is_set() and raw:
            lead = not _continues_sentence(ready, raw)
            raw = _drop_orphan_closer(raw, ready)
            ready = _glue_sentence_stop(
                ready,
                _scrub_chunk(
                    raw,
                    lead=lead,
                    line_start=_at_line_start(sent_line + ready),
                    before=ready[-1:] or last,
                    continued=bool((sent_line + ready).strip()),
                ),
            )
        # A span held until the end can be longer than the send window.
        # One cut would speak the first words and drop the rest.
        while ready and not cancel.is_set():
            split = split_tts_piece(ready, self.partial_chars, flush_rest=True)
            if split is None:
                break
            piece, ready = split
            event = counted(piece)
            if event is not None:
                yield event
        flush = flush_if_sent(sent, cancel)
        if flush is not None:
            yield flush

    def _text_event(self, piece: str) -> TextEvent | None:
        # The piece was already scrubbed, including one edge space. Scrubbing
        # again strips that space and the next cut is spoken as one word.
        if skip_empty_delta(piece):
            return None
        return TextEvent(text=piece)

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
