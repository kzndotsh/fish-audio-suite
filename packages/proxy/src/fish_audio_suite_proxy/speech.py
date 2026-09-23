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
    SuiteDefaults,
    clamp_num,
    known_latency,
    known_mp3_bitrate,
    known_opus_bitrate,
    number_or,
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
    present_value,
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
    raw = present_value(body, *keys)
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
    flag = present_value(body, "quality_guard", "fish_quality_guard")
    if flag is not None:
        return bool(flag)
    features = body.get("features")
    if isinstance(features, list) and "quality-guard" in features:
        return True
    return quality_guard_env()


class ClipError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _b64_audio(value: str) -> bytes:
    raw = "".join(value.strip().split())
    if raw.startswith(_DATA_URI) and "," in raw:
        raw = raw.split(",", 1)[1]
    padded = raw + ("=" * ((-len(raw)) % 4))
    try:
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ClipError("reference audio is not valid base64") from exc


def decode_audio_b64(value: Any) -> bytes:
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
        return text
    raise ClipError("reference text must be a string")


def _clip(audio: Any, text: object) -> dict[str, Any]:
    return {"audio": decode_audio_b64(audio), "text": _clip_text(text)}


def _normalize_references(raw: Any) -> Any:
    if isinstance(raw, list):
        return [_normalize_references(item) for item in raw]
    if isinstance(raw, Mapping):
        return _clip(raw.get("audio"), raw.get("text"))
    raise ClipError("references must be clips")


def _clips_from_input_references(items: list[Any]) -> list[dict[str, Any]]:
    audio: Any = None
    text = ""
    for item in items:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "")
        if kind == "input_audio":
            inner = item.get("input_audio")
            data = inner.get("data") if isinstance(inner, Mapping) else None
            audio = data
        elif kind == "text":
            text = _clip_text(item.get("text"))
    if audio is None:
        raise ClipError("input_references needs one input_audio part")
    return [_clip(audio, text)]


def _clip_list(body: Mapping[str, Any], key: str) -> list[Any] | None:
    raw = body.get(key)
    if isinstance(raw, list) and raw:
        return raw
    return None


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


def _body_int(body: dict[str, Any], key: str, default: int) -> int:
    return _body_num(body, key, default, int)


def _codec_fields(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    fmt: str,
    native_fmt: str,
) -> dict[str, Any]:
    if native_fmt == "mp3":
        return {
            "mp3_bitrate": known_mp3_bitrate(_body_int(body, "mp3_bitrate", defaults.mp3_bitrate)),
            "sample_rate": pcm_sample_rate(fmt, body, defaults.sample_rate),
        }
    if native_fmt == "opus":
        return {
            "opus_bitrate": known_opus_bitrate(
                _body_int(body, "opus_bitrate", defaults.opus_bitrate)
            ),
            "sample_rate": pcm_sample_rate(fmt, body, defaults.opus_sample_rate),
        }
    return {"sample_rate": pcm_sample_rate(fmt, body, defaults.sample_rate)}


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
            0.0,
            1.0,
            defaults.temperature,
            float,
        ),
        "top_p": clamp_num(
            _body_float(body, "top_p", defaults.top_p), 0.0, 1.0, defaults.top_p, float
        ),
        "chunk_length": controls.chunk_length,
        "min_chunk_length": controls.min_chunk_length,
        "normalize": explicit_bool(body, "normalize", default=defaults.normalize),
        "prosody": {
            "speed": controls.speed,
            "volume": _body_float(body, "volume", defaults.volume),
            "normalize_loudness": defaults.normalize_loudness,
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
            0.0,
            1.0,
            defaults.early_stop_threshold,
            float,
        ),
        **_codec_fields(body, defaults, fmt, native_fmt),
    }
    _optional_tts(payload, body)
    return payload


def _seed(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


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


def pack_tts(
    body: dict[str, Any],
    defaults: SuiteDefaults,
    incoming: Mapping[str, str],
    controls: _SpeechControls,
    spoken: str,
) -> _PackedTts | JSONResponse:
    payload = _fish_tts_payload(body, defaults, controls, spoken)
    try:
        clips = _fish_reference_clips(body)
    except ClipError as exc:
        return json_error(400, exc.message)
    if clips is None:
        content_type = "application/json"
        request_kw: dict[str, Any] = {"json": payload}
    else:
        payload["references"] = clips
        content_type = "application/msgpack"
        request_kw = {"content": ormsgpack.packb(payload)}
    return _PackedTts(
        headers={
            "Content-Type": content_type,
            **traced_model_headers(controls.model, incoming),
        },
        request_kw=request_kw,
        media_type=media_type(controls.fmt),
    )
