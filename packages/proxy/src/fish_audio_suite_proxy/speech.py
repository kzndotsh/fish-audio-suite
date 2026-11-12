"""One speech request end to end: the OpenAI body, the spoken text, and the Fish TTS call."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypedDict

import ormsgpack

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    MIN_CHUNK_LENGTH_HI,
    MIN_CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_INTERVAL_HI,
    UNIT_INTERVAL_LO,
    AudioFormat,
    CueStyle,
    FishLatency,
    SuiteDefaults,
    chunk_length_hi,
    clamp_number,
    is_tts_junk,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    parse_number,
    trace_id_of,
    utf8_text,
)
from fish_audio_suite_proxy.audio import (
    SUPPORTED_FORMATS,
    ClientFormat,
    client_format,
    fish_audio_format,
    media_type,
    pcm_sample_rate,
    silent_speech,
)
from fish_audio_suite_proxy.errors import ProxyError, unsupported_format
from fish_audio_suite_proxy.models import resolve_tts_model
from fish_audio_suite_proxy.request_body import decode_audio_b64, read_choice, read_flag
from fish_audio_suite_proxy.settings import ProxySettings
from fish_audio_suite_proxy.upstream import traced_model_headers

__all__ = ["read_speech"]

log: Final[logging.Logger] = logging.getLogger("fish-audio-suite-proxy")
_PREVIEW_CHARS = 160
_PHONEME_MARK_RE = re.compile(r"<\|phoneme_(?:start|end)\|>")
_MAX_PRONUNCIATION_ENTRIES = 512
# ormsgpack rejects integers outside this range. A bigger seed would 500
# the speech route when a reference clip forces MessagePack. The same range
# bounds a sample rate or bitrate, which the encoder would also refuse.
_MSGPACK_INT_LO = -(2**63)
_MSGPACK_INT_HI = 2**64 - 1


def _packable(value: int) -> bool:
    return _MSGPACK_INT_LO <= value <= _MSGPACK_INT_HI


def _clamped_int(body: dict[str, Any], key: str, *, lo: int, hi: int, default: int) -> int:
    # A null chunk_length is an omitted field. Zero is a real value.
    raw = body.get(key)
    return clamp_number(default if raw is None else raw, lo, hi, default, int)


def _scrub_pronunciation_item(item: Any) -> Any:
    if isinstance(item, dict) and isinstance(item.get("value"), str):
        return {**item, "value": _PHONEME_MARK_RE.sub("", item["value"])}
    return item


def _scrub_pronunciation_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    items = entry.get("items")
    if not isinstance(items, list):
        return entry
    return {**entry, "items": [_scrub_pronunciation_item(item) for item in items]}


def _scrub_pronunciation_dictionary(pd: Any) -> Any:
    if not isinstance(pd, list):
        return pd
    return [_scrub_pronunciation_entry(entry) for entry in pd]


_REQUEST_SPEED = 1.0
_CACHE_MODES = frozenset({"on", "off"})


def _read_latency(body: dict[str, Any], default: FishLatency) -> FishLatency:
    raw = read_choice(body, "latency", default=default)
    return known_latency(raw, default)


def _want_quality_guard(body: dict[str, Any], default: bool) -> bool:
    if body.get("quality_guard") is not None:
        return read_flag(body, "quality_guard", default=default)
    features = body.get("features")
    if isinstance(features, list) and "quality-guard" in features:
        return True
    return default


def _decode_clip(value: Any) -> bytes:
    return decode_audio_b64(value, field="reference audio")


def _clip_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        # A surrogate here makes MessagePack raise, and the clip is dropped.
        return utf8_text(text)
    raise ProxyError(400, "reference text must be a string")


def _clip(audio: Any, text: object) -> dict[str, Any]:
    return {"audio": _decode_clip(audio), "text": _clip_text(text)}


def _normalize_references(raw: Any) -> Any:
    if isinstance(raw, list):
        # A null slot is an empty speaker, not a broken clip. Failing the
        # whole list would drop a reference that already decoded.
        clips = [_normalize_references(item) for item in raw if item is not None]
        if not clips:
            raise ProxyError(400, "references must be clips")
        return clips
    if isinstance(raw, Mapping):
        return _clip(raw.get("audio"), raw.get("text"))
    raise ProxyError(400, "references must be clips")


def _clips_from_input_references(items: list[Any]) -> list[dict[str, Any]]:
    clips: list[dict[str, str | bytes]] = []
    pending = ""
    pending_set = False
    got_text = False
    bad_text = False
    for item in items:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "")
        if kind == "input_audio":
            inner = item.get("input_audio")
            if not isinstance(inner, Mapping) or inner.get("data") is None:
                continue
            try:
                decoded = _decode_clip(inner.get("data"))
            except ProxyError:
                # A later empty or invalid part must not wipe a clip that
                # already decoded, and must not fail that request.
                if not clips:
                    raise
                continue
            # Text before this audio belongs to it. A second good part is
            # another clip, so every voice is kept, not only the last.
            clip_text = pending if pending_set else ""
            pending = ""
            pending_set = False
            clips.append({"audio": decoded, "text": clip_text})
        elif kind == "text":
            raw_text = item.get("text")
            # A missing text field is not an explicit clear.
            if raw_text is None:
                continue
            if isinstance(raw_text, str):
                got_text = True
                if clips:
                    clips[-1]["text"] = raw_text
                else:
                    pending = raw_text
                    pending_set = True
                continue
            bad_text = True
    if not clips:
        raise ProxyError(400, "input_references needs one input_audio part")
    if bad_text and not got_text:
        raise ProxyError(400, "reference text must be a string")
    return [_clip(clip["audio"], clip["text"]) for clip in clips]


def _clip_list(body: Mapping[str, Any], key: str) -> list[Any] | None:
    raw = body.get(key)
    if raw is None:
        return None
    # One clip object is the same payload as a one-item list. Dropping it
    # would speak with the library voice and no error.
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, list):
        return raw or None
    raise ProxyError(400, f"{key} must be clips")


def _fish_reference_clips(body: Mapping[str, Any]) -> Any | None:
    raw_in = _clip_list(body, "input_references")
    if raw_in is not None:
        return _clips_from_input_references(raw_in)
    raw = _clip_list(body, "references")
    if raw is not None:
        return _normalize_references(raw)
    return None


def _body_num[T: int | float](
    body: dict[str, Any],
    key: str,
    default: T,
    parse: Callable[[Any], T],
) -> T:
    return parse_number(body.get(key, default), default, parse)


def _body_float(body: dict[str, Any], key: str, default: float) -> float:
    return _body_num(body, key, default, float)


def _body_int(body: dict[str, Any], key: str, default: int) -> int:
    value = _body_num(body, key, default, int)
    return value if _packable(value) else default


def _codec_fields(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    fmt: ClientFormat,
    native_fmt: AudioFormat,
) -> dict[str, Any]:
    if native_fmt == "mp3":
        return {
            "mp3_bitrate": known_mp3_bitrate(_body_int(body, "mp3_bitrate", defaults.mp3_bitrate)),
            "sample_rate": pcm_sample_rate(fmt, body, defaults.sample_rate, hi=_MSGPACK_INT_HI),
        }
    if native_fmt == "opus":
        return {
            "opus_bitrate": known_opus_bitrate(
                _body_int(body, "opus_bitrate", defaults.opus_bitrate)
            ),
            "sample_rate": pcm_sample_rate(
                fmt, body, defaults.opus_sample_rate, hi=_MSGPACK_INT_HI
            ),
        }
    return {"sample_rate": pcm_sample_rate(fmt, body, defaults.sample_rate, hi=_MSGPACK_INT_HI)}


@dataclass(frozen=True, slots=True)
class SpeechControls:
    """The speech fields after clamping, ready to build a Fish request.

    Attributes
    ----------
    model : str
        Fish TTS model id.
    speed : float
        Speech speed multiplier.
    fmt : ClientFormat
        Audio format the client asked for.
    latency : FishLatency
        Fish latency mode.
    chunk_length : int
        Fish chunk length.
    min_chunk_length : int
        Fish minimum chunk length.
    """

    model: str
    speed: float
    fmt: ClientFormat
    latency: FishLatency
    chunk_length: int
    min_chunk_length: int


def _fish_tts_payload(
    body: dict[str, Any],
    settings: ProxySettings,
    controls: SpeechControls,
    spoken: str,
) -> dict[str, Any]:
    defaults = settings.defaults
    fmt = controls.fmt
    native_fmt = fish_audio_format(fmt)
    payload: dict[str, Any] = {
        "text": spoken,
        "format": native_fmt,
        "latency": controls.latency,
        "temperature": clamp_number(
            _body_float(body, "temperature", defaults.temperature),
            UNIT_INTERVAL_LO,
            UNIT_INTERVAL_HI,
            defaults.temperature,
            float,
        ),
        "top_p": clamp_number(
            _body_float(body, "top_p", defaults.top_p),
            UNIT_INTERVAL_LO,
            UNIT_INTERVAL_HI,
            defaults.top_p,
            float,
        ),
        "chunk_length": controls.chunk_length,
        "min_chunk_length": controls.min_chunk_length,
        "normalize": read_flag(body, "normalize", default=defaults.normalize),
        "prosody": {
            "speed": controls.speed,
            "volume": _body_float(body, "volume", defaults.volume),
            "normalize_loudness": read_flag(
                body, "normalize_loudness", default=defaults.normalize_loudness
            ),
        },
        "repetition_penalty": _body_float(body, "repetition_penalty", defaults.repetition_penalty),
        "max_new_tokens": _body_int(body, "max_new_tokens", defaults.max_new_tokens),
        "condition_on_previous_chunks": read_flag(
            body,
            "condition_on_previous_chunks",
            default=defaults.condition_on_previous_chunks,
        ),
        "early_stop_threshold": clamp_number(
            _body_float(body, "early_stop_threshold", defaults.early_stop_threshold),
            UNIT_INTERVAL_LO,
            UNIT_INTERVAL_HI,
            defaults.early_stop_threshold,
            float,
        ),
        **_codec_fields(body, defaults, fmt, native_fmt),
    }
    _optional_tts(payload, body, settings)
    return payload


def _seed(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    # int("42.0") raises. parse_number accepts a whole decimal, so the seed
    # the client asked for reaches Fish instead of being dropped.
    parsed = parse_number(value, _MSGPACK_INT_LO - 1, int)
    return parsed if _packable(parsed) else None


# Fish's own limit for the text's language tag, which it uses to read numbers and dates.
_LANGUAGE_RE: Final = re.compile(r"[A-Za-z0-9_-]{1,35}")


def _optional_tts(payload: dict[str, Any], body: dict[str, Any], settings: ProxySettings) -> None:
    voice = _read_reference_id(body)
    if voice:
        payload["reference_id"] = voice
    seed = _seed(body.get("seed"))
    if seed is not None:
        payload["seed"] = seed
    cache = body.get("use_memory_cache")
    if isinstance(cache, str):
        cache = cache.strip().lower()
    if cache in _CACHE_MODES:
        payload["use_memory_cache"] = cache
    if _want_quality_guard(body, settings.quality_guard):
        payload["features"] = ["quality-guard"]
    language = body.get("language")
    if isinstance(language, str) and (tag := language.strip()):
        if not _LANGUAGE_RE.fullmatch(tag):
            raise ProxyError(400, "language must be a BCP-47 tag such as en or zh-CN")
        payload["language"] = tag
    pd = body.get("pronunciation_dictionary")
    if pd is not None:
        if not isinstance(pd, list) or len(pd) > _MAX_PRONUNCIATION_ENTRIES:
            raise ProxyError(
                400,
                f"pronunciation_dictionary must be a list of at most {_MAX_PRONUNCIATION_ENTRIES} entries",
            )
        if pd:
            payload["pronunciation_dictionary"] = _scrub_pronunciation_dictionary(pd)


def _read_format(body: dict[str, Any], default: ClientFormat) -> ClientFormat:
    # ``format``, then the OpenAI ``response_format``. A name the proxy cannot
    # produce, such as aac or flac, is refused rather than answered in mp3.
    raw = read_choice(body, "format", "response_format", default="")
    if not raw:
        return default
    known = client_format(raw)
    if known is None:
        raise unsupported_format(raw, SUPPORTED_FORMATS)
    return known


def _speech_controls(body: dict[str, Any], settings: ProxySettings) -> SpeechControls:
    defaults = settings.defaults
    # The request's speed multiplies FISH_SPEED.
    raw_speed = _body_float(body, "speed", _REQUEST_SPEED) * defaults.speed
    return SpeechControls(
        model=resolve_tts_model(body.get("model"), defaults.tts_model, settings.model_aliases),
        speed=clamp_number(raw_speed, TTS_SPEED_LO, TTS_SPEED_HI, defaults.speed, float),
        fmt=_read_format(body, settings.tts_format),
        latency=_read_latency(body, defaults.latency),
        chunk_length=_clamped_int(
            body,
            "chunk_length",
            lo=CHUNK_LENGTH_LO,
            hi=chunk_length_hi(defaults.fish_base),
            default=defaults.chunk_length,
        ),
        min_chunk_length=_clamped_int(
            body,
            "min_chunk_length",
            lo=MIN_CHUNK_LENGTH_LO,
            hi=MIN_CHUNK_LENGTH_HI,
            default=defaults.min_chunk_length,
        ),
    )


def _read_reference_id(body: dict[str, Any]) -> str | list[str] | None:
    # ``reference_id`` wins over OpenAI ``voice``. A list is S2 multi-speaker.
    for key in ("reference_id", "voice"):
        value = body.get(key)
        if isinstance(value, list):
            ids = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            # An empty list is blank, same as "". Keep looking at voice.
            if ids:
                return ids
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class TtsBody(TypedDict, total=False):
    """The body part of a Fish TTS request: JSON, or MessagePack bytes with clips."""

    json: dict[str, Any]
    content: bytes


@dataclass(frozen=True, slots=True)
class PackedTts:
    """A Fish TTS request ready for ``fish_send``.

    Attributes
    ----------
    headers : dict
        Request headers, including the trace and the model.
    request_kw : TtsBody
        Either ``json`` or ``content``, to pass on to ``fish_send``.
    media_type : str
        Media type of the audio the client gets back.
    """

    headers: dict[str, str]
    request_kw: TtsBody
    media_type: str


def _json_ready(value: Any) -> Any:
    # Any string on the payload can carry a surrogate. httpx and MessagePack
    # both refuse to encode one, so the rest of the request never leaves.
    if isinstance(value, str):
        return utf8_text(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {
            utf8_text(key) if isinstance(key, str) else key: _json_ready(item)
            for key, item in value.items()
        }
    return value


def _pack(
    payload: dict[str, Any],
    clips: Any | None,
    controls: SpeechControls,
    incoming: Mapping[str, str],
) -> PackedTts:
    if clips is not None:
        payload["references"] = clips
    payload = _json_ready(payload)
    if clips is None:
        content_type = "application/json"
        request_kw: TtsBody = {"json": payload}
    else:
        content_type = "application/msgpack"
        try:
            encoded = ormsgpack.packb(payload)
        # Fields the proxy does not bound, such as a pronunciation_dictionary
        # entry, can still hold a number MessagePack cannot encode.
        except (TypeError, OverflowError, ValueError) as exc:
            raise ProxyError(400, "speech fields contain a number that cannot be encoded") from exc
        request_kw = {"content": encoded}
    return PackedTts(
        headers={"Content-Type": content_type, **traced_model_headers(controls.model, incoming)},
        request_kw=request_kw,
        media_type=media_type(controls.fmt),
    )


@dataclass(frozen=True, slots=True)
class SpeechRequest:
    """One speech request after parsing and text shaping.

    Attributes
    ----------
    body : dict
        The OpenAI speech JSON, read again only for the Fish payload fields.
    controls : SpeechControls
        The clamped model, format, speed, latency and chunk lengths.
    spoken : str
        The shaped text Fish speaks, not the raw ``input``.
    silence : tuple of bytes and str, or None
        Silent audio and its media type when ``spoken`` is junk. The route sends it
        and never calls Fish.
    settings : ProxySettings
        The defaults for every field the request leaves out.
    """

    body: dict[str, Any]
    controls: SpeechControls
    spoken: str
    silence: tuple[bytes, str] | None
    settings: ProxySettings

    def fish_tts(self, incoming: Mapping[str, str]) -> PackedTts:
        """Build the Fish TTS request: JSON, or MessagePack when clips are attached.

        Parameters
        ----------
        incoming : Mapping
            Client headers. A valid trace is forwarded; otherwise one is minted.

        Returns
        -------
        PackedTts
            Headers, body, and the media type of the audio the client gets back.

        Raises
        ------
        ProxyError
            400 when a reference clip cannot be read, ``language`` is not a tag,
            ``pronunciation_dictionary`` is not a list of at most 512 entries, or
            a number cannot be MessagePacked with a clip.

        Notes
        -----
        The route calls this only after it has a Fish key, so a request that would
        fail here still gets the 503 for the missing key first. Fish's live
        websocket has no ``features`` field; quality-guard is HTTP only.
        """
        payload = _fish_tts_payload(self.body, self.settings, self.controls, self.spoken)
        packed = _pack(payload, _fish_reference_clips(self.body), self.controls, incoming)
        log.info(
            "tts model=%s fmt=%s trace=%s",
            self.controls.model,
            self.controls.fmt,
            trace_id_of(packed.headers["traceparent"]),
        )
        return packed


def _shape(body: dict[str, Any], raw_input: str, model: str, settings: ProxySettings) -> str:
    style = CueStyle.for_model(model, official=False, lead=settings.tts_mood_lead)
    dialogue_only = read_flag(body, "dialogue_only", default=settings.tts_dialogue_only)
    spoken = style.shape(raw_input, dialogue_only=dialogue_only)
    log.info(
        "tts scrub model=%s raw_len=%d spoken_len=%d",
        body.get("model") or settings.defaults.tts_model,
        len(raw_input),
        len(spoken),
    )
    # The text people speak is private. It is logged only on request.
    if settings.log_text:
        log.info("tts text preview=%r", _preview(spoken))
    return spoken


def _preview(spoken: str) -> str:
    return spoken.replace("\n", " ")[:_PREVIEW_CHARS]


def read_speech(body: dict[str, Any], settings: ProxySettings) -> SpeechRequest:
    """Parse an OpenAI speech body and shape the text Fish will speak.

    Parameters
    ----------
    body : dict
        OpenAI speech JSON. ``input`` must be a string within
        ``FISH_PROXY_MAX_INPUT_CHARS``.
    settings : ProxySettings
        The defaults, the alias table, and the text settings.

    Returns
    -------
    SpeechRequest
        The resolved request. Its ``silence`` is set when the shaped text is
        junk, such as only a cue or punctuation.

    Raises
    ------
    ProxyError
        400 when ``input`` is not a string or is too long, or when the request
        names a format the proxy cannot produce.

    Notes
    -----
    The model is resolved once, and its cue style shapes the text: scrubbed,
    reduced to quoted speech when ``dialogue_only`` is on, then with its cues
    shaped. Request ``speed`` is multiplied by ``FISH_SPEED``.
    """
    raw_input = body.get("input", "")
    if not isinstance(raw_input, str):
        raise ProxyError(400, "input must be a string")
    if settings.max_input_chars and len(raw_input) > settings.max_input_chars:
        raise ProxyError(400, f"input is longer than {settings.max_input_chars} characters")
    controls = _speech_controls(body, settings)
    spoken = _shape(body, raw_input, controls.model, settings)
    silence = None
    if is_tts_junk(spoken, drop_narration=settings.tts_drop_narration):
        log.info("tts skip junk chars=%d", len(spoken))
        if settings.log_text:
            log.info("tts junk preview=%r", _preview(spoken))
        rate = pcm_sample_rate(controls.fmt, body, settings.defaults.sample_rate)
        silence = silent_speech(controls.fmt, rate)
    return SpeechRequest(body, controls, spoken, silence, settings)
