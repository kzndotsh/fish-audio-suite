"""Fish TTS body built from an OpenAI speech request."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import ormsgpack
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    MIN_CHUNK_LENGTH_HI,
    MIN_CHUNK_LENGTH_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_INTERVAL_HI,
    UNIT_INTERVAL_LO,
    AudioFormat,
    FishLatency,
    SuiteDefaults,
    chunk_length_hi,
    clamp_number,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    parse_number,
    utf8_text,
)
from fish_audio_suite_proxy.audio import (
    AudioDecodeError,
    ClientFormat,
    decode_audio_b64,
    fish_audio_format,
    media_type,
    pcm_sample_rate,
)
from fish_audio_suite_proxy.errors import ProxyError, json_error
from fish_audio_suite_proxy.models import resolve_tts_model
from fish_audio_suite_proxy.request_fields import (
    read_choice,
    read_flag,
    read_format,
    read_reference_id,
    traced_model_headers,
)

__all__ = [
    "ClipError",
    "PackedTts",
    "SpeechControls",
    "TtsBody",
    "pack_tts",
    "speech_controls",
]

_PHONEME_MARK_RE = re.compile(r"<\|phoneme_(?:start|end)\|>")
_MAX_PRONUNCIATION_ENTRIES = 512
# ormsgpack rejects integers outside this range. A bigger seed would 500
# the speech route when a reference clip forces MessagePack. The same range
# bounds a sample rate or bitrate, which the encoder would also refuse.
_MSGPACK_INT_LO = -(2**63)
_MSGPACK_INT_HI = 2**64 - 1


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


class ClipError(AudioDecodeError):
    """A reference clip could not be read. The route returns 400.

    Attributes
    ----------
    status : int
        Always 400.
    message : str
        Shown in the OpenAI error envelope. It names the reference, such as
        ``"reference audio is empty"``.
    """


def _decode_clip(value: Any) -> bytes:
    try:
        return decode_audio_b64(value, field="reference audio")
    except AudioDecodeError as exc:
        raise ClipError(exc.message) from exc


def _clip_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        # A surrogate here makes MessagePack raise, and the clip is dropped.
        return utf8_text(text)
    raise ClipError("reference text must be a string")


def _clip(audio: Any, text: object) -> dict[str, Any]:
    return {"audio": _decode_clip(audio), "text": _clip_text(text)}


def _normalize_references(raw: Any) -> Any:
    if isinstance(raw, list):
        # A null slot is an empty speaker, not a broken clip. Failing the
        # whole list would drop a reference that already decoded.
        clips = [_normalize_references(item) for item in raw if item is not None]
        if not clips:
            raise ClipError("references must be clips")
        return clips
    if isinstance(raw, Mapping):
        return _clip(raw.get("audio"), raw.get("text"))
    raise ClipError("references must be clips")


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
            except ClipError:
                # A later empty or invalid part must not wipe a clip that
                # already decoded, and must not fail that request.
                if not clips:
                    raise
                continue
            # Text before this audio belongs to it. A second good part is
            # another clip; replacing the first spoke with only the last voice.
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
        raise ClipError("input_references needs one input_audio part")
    if bad_text and not got_text:
        raise ClipError("reference text must be a string")
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
    raise ClipError(f"{key} must be clips")


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


def _fit_int(value: int, default: int) -> int:
    if value < _MSGPACK_INT_LO or value > _MSGPACK_INT_HI:
        return default
    return value


def _body_int(body: dict[str, Any], key: str, default: int) -> int:
    return _fit_int(_body_num(body, key, default, int), default)


def _bounded_rate(fmt: ClientFormat, body: dict[str, Any], default: int) -> int:
    rate = pcm_sample_rate(fmt, body, default)
    if _MSGPACK_INT_LO <= rate <= _MSGPACK_INT_HI:
        return rate
    # Ask again with no client rate so pcm16 still falls back to 24 kHz.
    return pcm_sample_rate(fmt, {}, default)


def _codec_fields(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    fmt: ClientFormat,
    native_fmt: AudioFormat,
) -> dict[str, Any]:
    if native_fmt == "mp3":
        return {
            "mp3_bitrate": known_mp3_bitrate(_body_int(body, "mp3_bitrate", defaults.mp3_bitrate)),
            "sample_rate": _bounded_rate(fmt, body, defaults.sample_rate),
        }
    if native_fmt == "opus":
        return {
            "opus_bitrate": known_opus_bitrate(
                _body_int(body, "opus_bitrate", defaults.opus_bitrate)
            ),
            "sample_rate": _bounded_rate(fmt, body, defaults.opus_sample_rate),
        }
    return {"sample_rate": _bounded_rate(fmt, body, defaults.sample_rate)}


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
    defaults: SuiteDefaults,
    controls: SpeechControls,
    spoken: str,
    *,
    quality_guard: bool = False,
) -> dict[str, Any]:
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
    _optional_tts(payload, body, quality_guard=quality_guard)
    return payload


def _seed(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    # int("42.0") raises, so a whole-number decimal was omitted and Fish
    # picked a different seed than the one the client asked for.
    parsed = parse_number(value, _MSGPACK_INT_LO - 1, int)
    if parsed < _MSGPACK_INT_LO or parsed > _MSGPACK_INT_HI:
        return None
    return parsed


def _optional_tts(payload: dict[str, Any], body: dict[str, Any], *, quality_guard: bool) -> None:
    voice = read_reference_id(body)
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
    if _want_quality_guard(body, quality_guard):
        payload["features"] = ["quality-guard"]
    pd = body.get("pronunciation_dictionary")
    if pd:
        if not isinstance(pd, list) or len(pd) > _MAX_PRONUNCIATION_ENTRIES:
            raise ProxyError(
                400,
                f"pronunciation_dictionary must be a list of at most {_MAX_PRONUNCIATION_ENTRIES} entries",
            )
        payload["pronunciation_dictionary"] = _scrub_pronunciation_dictionary(pd)


def speech_controls(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    aliases: Mapping[str, str] | None = None,
    *,
    default_format: ClientFormat | None = None,
) -> SpeechControls:
    """Clamp speed, format, latency, and chunk lengths for one speech call.

    Parameters
    ----------
    body : dict
        OpenAI speech JSON.
    defaults : SuiteDefaults
        Env-backed knobs. Request ``speed`` is multiplied by ``defaults.speed``.
    aliases : Mapping or None, optional
        TTS model alias table. ``None`` maps the OpenAI names to the default.
    default_format : ClientFormat or None, optional
        Format used when the request names none. ``None`` uses
        ``defaults.audio_format``. The proxy passes ``FISH_TTS_FORMAT`` here, which
        can be ``pcm16``, a name Fish itself does not know.

    Returns
    -------
    SpeechControls
        Values safe to put on the Fish payload. Cloud chunk length stays
        within 100-300.

    Raises
    ------
    ProxyError
        With status 400 when the request names an unsupported audio format.
    """
    raw_speed = _body_float(body, "speed", _REQUEST_SPEED) * defaults.speed
    return SpeechControls(
        model=resolve_tts_model(body.get("model"), defaults.tts_model, aliases),
        speed=clamp_number(raw_speed, TTS_SPEED_LO, TTS_SPEED_HI, defaults.speed, float),
        fmt=read_format(body, default_format or defaults.audio_format),
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


def pack_tts(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    incoming: Mapping[str, str],
    controls: SpeechControls,
    spoken: str,
    *,
    quality_guard: bool = False,
) -> PackedTts | JSONResponse:
    """Build the Fish TTS request, JSON or MessagePack when clips are attached.

    Parameters
    ----------
    body : dict
        Original speech JSON, including ``references`` or ``input_references``.
    defaults : SuiteDefaults
        Runtime knobs.
    incoming : Mapping
        Client headers. A valid trace is forwarded; otherwise one is minted.
    controls : SpeechControls
        Already clamped speech fields.
    spoken : str
        Scrubbed text. This is what Fish speaks, not the raw ``input``.
    quality_guard : bool, optional
        Default for the Fish quality-guard feature when the request is silent.

    Returns
    -------
    PackedTts or JSONResponse
        Headers, body, and response media type, or a 400 when a clip fails.

    Notes
    -----
    Fish's live websocket has no ``features`` field. Quality-guard is HTTP only.
    A number outside the 64-bit range cannot be MessagePacked with a clip.
    That is a 400, not an unhandled 500.
    """
    payload = _fish_tts_payload(body, defaults, controls, spoken, quality_guard=quality_guard)
    try:
        clips = _fish_reference_clips(body)
    except ClipError as exc:
        return json_error(400, exc.message)
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
        except (TypeError, OverflowError, ValueError):
            return json_error(400, "speech fields contain a number that cannot be encoded")
        request_kw = {"content": encoded}
    return PackedTts(
        headers={
            "Content-Type": content_type,
            **traced_model_headers(controls.model, incoming),
        },
        request_kw=request_kw,
        media_type=media_type(controls.fmt),
    )
