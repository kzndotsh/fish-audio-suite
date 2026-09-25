"""Fish TTS body built from an OpenAI speech request."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import ormsgpack
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import (
    CHUNK_LENGTH_LO,
    MIN_CHUNK_HI,
    MIN_CHUNK_LO,
    TTS_SPEED_HI,
    TTS_SPEED_LO,
    UNIT_HI,
    UNIT_LO,
    SuiteDefaults,
    clamp_num,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    number_or,
    utf8_text,
)
from fish_audio_suite_proxy.errors import json_error
from fish_audio_suite_proxy.fields import (
    chunk_length_hi,
    explicit_bool,
    first_choice,
    media_type,
    pcm_sample_rate,
    pick_format,
    pick_reference_id,
    quality_guard_env,
    resolve_tts_model,
    traced_model_headers,
)

_PHONEME_MARK_RE = re.compile(r"<\|phoneme_(?:start|end)\|>")
_DATA_URI = "data:"


def _clamped_int(
    body: dict[str, Any],
    *keys: str,
    lo: int,
    hi: int,
    default: int,
) -> int:
    # A null chunk_length is an omitted field. Stopping there used the
    # default and ignored fish_chunk_length. Zero is a real value.
    raw: Any = None
    for key in keys:
        if key not in body or body[key] is None:
            continue
        raw = body[key]
        break
    if raw is None:
        raw = default
    return clamp_num(raw, lo, hi, default, int)


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


def _pick_latency(body: dict[str, Any], default: str) -> str:
    raw = first_choice(body, "latency", "fish_latency", default=default)
    return known_latency(raw, default)


def _want_quality_guard(body: dict[str, Any]) -> bool:
    for key in ("quality_guard", "fish_quality_guard"):
        if key in body and body[key] is not None:
            # explicit_bool stops at the first present value. A null
            # quality_guard is present, so passing both keys ignored
            # fish_quality_guard and used the env default instead.
            return explicit_bool(body, key, default=quality_guard_env())
    features = body.get("features")
    if isinstance(features, list) and "quality-guard" in features:
        return True
    return quality_guard_env()


class ClipError(Exception):
    """A reference clip could not be decoded. The route returns 400.

    Attributes
    ----------
    message : str
        Shown in the OpenAI error envelope.
    """

    def __init__(self, message: str) -> None:
        """Store ``message`` as both the attribute and the exception text.

        Parameters
        ----------
        message : str
            Client-facing reason. No status code; the route maps this to 400.
        """
        self.message = message
        super().__init__(message)


def _b64_audio(value: str) -> bytes:
    raw = "".join(value.strip().split())
    # The data-URI scheme is case-insensitive. "DATA:" failed the base64
    # check, so the speech request was rejected before Fish heard the clip.
    head, sep, tail = raw.partition(",")
    if sep and head.lower().startswith(_DATA_URI):
        raw = tail
    # URL-safe alphabets use - and _. The strict decoder rejects those, so a
    # real clip would 400 and the request would speak with no reference.
    raw = raw.replace("-", "+").replace("_", "/")
    padded = raw + ("=" * ((-len(raw)) % 4))
    try:
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ClipError("reference audio is not valid base64") from exc


def decode_audio_b64(value: Any) -> bytes:
    """Decode reference audio from raw bytes or base64, including a data URI.

    Parameters
    ----------
    value : Any
        ``bytes``, or a base64 string. Whitespace inside the string is removed.

    Returns
    -------
    bytes
        Decoded audio. Never empty.

    Raises
    ------
    ClipError
        When the value is empty or not valid base64.
    """
    if isinstance(value, (bytes, bytearray)):
        audio = bytes(value)
    elif isinstance(value, str) and value.strip():
        audio = _b64_audio(value)
    else:
        audio = b""
    if not audio:
        raise ClipError("reference audio is empty")
    return audio


def _clip_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        # A surrogate here makes MessagePack raise, and the clip is dropped.
        return utf8_text(text)
    raise ClipError("reference text must be a string")


def _clip(audio: Any, text: object) -> dict[str, Any]:
    return {"audio": decode_audio_b64(audio), "text": _clip_text(text)}


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
                decoded = decode_audio_b64(inner.get("data"))
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
    return number_or(body.get(key, default), default, parse)


def _body_float(body: dict[str, Any], key: str, default: float) -> float:
    return _body_num(body, key, default, float)


def _fit_int(value: int, default: int) -> int:
    if value < _SEED_LO or value > _SEED_HI:
        return default
    return value


def _body_int(body: dict[str, Any], key: str, default: int) -> int:
    return _fit_int(_body_num(body, key, default, int), default)


def _bounded_rate(fmt: str, body: dict[str, Any], default: int) -> int:
    rate = pcm_sample_rate(fmt, body, default)
    if _SEED_LO <= rate <= _SEED_HI:
        return rate
    # Ask again with no client rate so pcm16 still falls back to 24 kHz.
    return pcm_sample_rate(fmt, {}, default)


def _codec_fields(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    fmt: str,
    native_fmt: str,
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


@dataclass(frozen=True)
class _SpeechControls:
    model: str
    speed: float
    fmt: str
    latency: str
    chunk_length: int
    min_chunk_length: int


def _fish_tts_payload(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    controls: _SpeechControls,
    spoken: str,
) -> dict[str, Any]:
    fmt = controls.fmt
    native_fmt = "pcm" if fmt == "pcm16" else fmt
    payload: dict[str, Any] = {
        "text": spoken,
        "format": native_fmt,
        "latency": controls.latency,
        "temperature": clamp_num(
            _body_float(body, "temperature", defaults.temperature),
            UNIT_LO,
            UNIT_HI,
            defaults.temperature,
            float,
        ),
        "top_p": clamp_num(
            _body_float(body, "top_p", defaults.top_p), UNIT_LO, UNIT_HI, defaults.top_p, float
        ),
        "chunk_length": controls.chunk_length,
        "min_chunk_length": controls.min_chunk_length,
        "normalize": explicit_bool(body, "normalize", default=defaults.normalize),
        "prosody": {
            "speed": controls.speed,
            "volume": _body_float(body, "volume", defaults.volume),
            "normalize_loudness": explicit_bool(
                body, "normalize_loudness", default=defaults.normalize_loudness
            ),
        },
        "repetition_penalty": _body_float(body, "repetition_penalty", defaults.repetition_penalty),
        "max_new_tokens": _body_int(body, "max_new_tokens", defaults.max_new_tokens),
        "condition_on_previous_chunks": explicit_bool(
            body,
            "condition_on_previous_chunks",
            default=defaults.condition_on_previous_chunks,
        ),
        "early_stop_threshold": clamp_num(
            _body_float(body, "early_stop_threshold", defaults.early_stop_threshold),
            UNIT_LO,
            UNIT_HI,
            defaults.early_stop_threshold,
            float,
        ),
        **_codec_fields(body, defaults, fmt, native_fmt),
    }
    _optional_tts(payload, body)
    return payload


# ormsgpack rejects integers outside this range. A bigger seed would 500
# the speech route when a reference clip forces MessagePack.
_SEED_LO = -(2**63)
_SEED_HI = 2**64 - 1


def _seed(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    # int("42.0") raises, so a whole-number decimal was omitted and Fish
    # picked a different seed than the one the client asked for.
    parsed = number_or(value, _SEED_LO - 1, int)
    if parsed < _SEED_LO or parsed > _SEED_HI:
        return None
    return parsed


def _optional_tts(payload: dict[str, Any], body: dict[str, Any]) -> None:
    voice = pick_reference_id(body)
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
    if _want_quality_guard(body):
        payload["features"] = ["quality-guard"]
    pd = body.get("pronunciation_dictionary")
    if pd:
        payload["pronunciation_dictionary"] = _scrub_pronunciation_dictionary(pd)


def speech_controls(body: dict[str, Any], defaults: SuiteDefaults) -> _SpeechControls:
    """Clamp speed, format, latency, and chunk lengths for one speech call.

    Parameters
    ----------
    body : dict
        OpenAI speech JSON.
    defaults : SuiteDefaults
        Env-backed knobs. Request ``speed`` is multiplied by ``defaults.speed``.

    Returns
    -------
    _SpeechControls
        Values safe to put on the Fish payload. Cloud chunk length stays
        within 100-300.
    """
    raw_speed = _body_float(body, "speed", _REQUEST_SPEED) * defaults.speed
    return _SpeechControls(
        model=resolve_tts_model(body.get("model"), defaults.tts_model),
        speed=clamp_num(raw_speed, TTS_SPEED_LO, TTS_SPEED_HI, defaults.speed, float),
        fmt=pick_format(body, defaults.audio_format),
        latency=_pick_latency(body, defaults.latency),
        chunk_length=_clamped_int(
            body,
            "chunk_length",
            "fish_chunk_length",
            lo=CHUNK_LENGTH_LO,
            hi=chunk_length_hi(defaults.fish_base),
            default=defaults.chunk_length,
        ),
        min_chunk_length=_clamped_int(
            body,
            "min_chunk_length",
            "fish_min_chunk_length",
            lo=MIN_CHUNK_LO,
            hi=MIN_CHUNK_HI,
            default=defaults.min_chunk_length,
        ),
    )


@dataclass(frozen=True)
class _PackedTts:
    headers: dict[str, str]
    request_kw: dict[str, Any]
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
    controls: _SpeechControls,
    spoken: str,
) -> _PackedTts | JSONResponse:
    """Build the Fish TTS request, JSON or MessagePack when clips are attached.

    Parameters
    ----------
    body : dict
        Original speech JSON, including ``references`` or ``input_references``.
    defaults : SuiteDefaults
        Runtime knobs.
    incoming : Mapping
        Client headers. A valid trace is forwarded; otherwise one is minted.
    controls : _SpeechControls
        Already clamped speech fields.
    spoken : str
        Scrubbed text. This is what Fish speaks, not the raw ``input``.

    Returns
    -------
    _PackedTts or JSONResponse
        Headers, body, and response media type, or a 400 when a clip fails.

    Notes
    -----
    Fish's live websocket has no ``features`` field. Quality-guard is HTTP only.
    A number outside the 64-bit range cannot be MessagePacked with a clip.
    That is a 400, not an unhandled 500.
    """
    payload = _fish_tts_payload(body, defaults, controls, spoken)
    try:
        clips = _fish_reference_clips(body)
    except ClipError as exc:
        return json_error(400, exc.message)
    if clips is not None:
        payload["references"] = clips
    payload = _json_ready(payload)
    if clips is None:
        content_type = "application/json"
        request_kw: dict[str, Any] = {"json": payload}
    else:
        content_type = "application/msgpack"
        try:
            encoded = ormsgpack.packb(payload)
        except (TypeError, OverflowError, ValueError):
            return json_error(400, "speech fields contain a number that cannot be encoded")
        request_kw = {"content": encoded}
    return _PackedTts(
        headers={
            "Content-Type": content_type,
            **traced_model_headers(controls.model, incoming),
        },
        request_kw=request_kw,
        media_type=media_type(controls.fmt),
    )
