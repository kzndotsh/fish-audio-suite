"""Fish Audio → OpenAI Audio proxy (TTS + STT).

OpenAI-compatible:
  POST /v1/audio/speech          → Fish POST /v1/tts
  POST /v1/audio/transcriptions  → Fish POST /v1/asr

Unofficial. Not affiliated with Fish Audio.
Docs: https://docs.fish.audio/llms.txt
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import httpx
import ormsgpack
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.datastructures import FormData, UploadFile

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    CaptionCue,
    SuiteDefaults,
    extract_quoted_speech,
    fish_backoff_seconds,
    format_as_srt,
    format_as_vtt,
    is_asr_hallucination,
    is_tts_junk,
    make_traceparent,
    normalize_cues,
    parse_fish_error,
    scrub_asr,
    scrub_tts,
    should_retry_fish_status,
    trace_id_of,
    w3c_trace_headers,
)

_MEDIA = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "wav": "audio/wav",
}

_TTS_MODEL_ALIASES = {
    "tts-1": "s2.1-pro",
    "tts-1-hd": "s2.1-pro",
    "gpt-4o-mini-tts": "s2.1-pro",
    "playai-tts": "s2.1-pro",
}
_ASR_MODEL_ALIASES = {
    "whisper-1",
    "whisper",
    "openai-whisper",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "distil-whisper-large-v3-en",
}

_TTS_MODELS = [
    "s2.1-pro",
    "s2.1-pro-free",
    "s2-pro",
    "s1",
    "drama-3-preview",
]
_ASR_MODELS = [
    "transcribe-1",
    "transcribe-1-pro",
    "whisper-1",
]
_MODELS = _TTS_MODELS + _ASR_MODELS

_SILENT_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 64

_PHONEME_MARK_RE = re.compile(r"<\|phoneme_(?:start|end)\|>")

log = logging.getLogger("fish-audio-suite-proxy")
logging.basicConfig(level=logging.INFO)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _runtime_defaults() -> SuiteDefaults:
    stock = SuiteDefaults()
    return SuiteDefaults(
        tts_model=os.environ.get("FISH_MODEL", stock.tts_model),
        asr_model=os.environ.get("FISH_ASR_MODEL", stock.asr_model),
        asr_language=os.environ.get("FISH_ASR_LANGUAGE", stock.asr_language),
        latency=os.environ.get("FISH_LATENCY", stock.latency),
        chunk_length=int(os.environ.get("FISH_CHUNK_LENGTH", str(stock.chunk_length))),
        min_chunk_length=int(os.environ.get("FISH_MIN_CHUNK_LENGTH", str(stock.min_chunk_length))),
        audio_format=os.environ.get("FISH_FORMAT", stock.audio_format),
        mp3_bitrate=int(os.environ.get("FISH_MP3_BITRATE", str(stock.mp3_bitrate))),
        speed=float(os.environ.get("FISH_SPEED_SCALE", str(stock.speed))),
        fish_base=os.environ.get("FISH_BASE", stock.fish_base).rstrip("/"),
    )


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _chunk_length_hi(fish_base: str) -> int:
    """Cloud OpenAPI max is 300; self-hosted fish-speech allows 1000."""
    return 300 if "api.fish.audio" in fish_base.lower() else 1000


def _pick_reference_id(body: dict[str, Any]) -> str | list[str] | None:
    rid = body.get("reference_id")
    if isinstance(rid, list):
        ids = [str(x) for x in rid if str(x).strip()]
        return ids or None
    if isinstance(rid, str) and rid.strip():
        return rid
    voice = body.get("voice")
    if isinstance(voice, list):
        ids = [str(x) for x in voice if str(x).strip()]
        return ids or None
    if isinstance(voice, str) and voice.strip():
        return voice
    return None


def _scrub_pronunciation_dictionary(pd: Any) -> Any:
    if not isinstance(pd, list):
        return pd
    cleaned: list[Any] = []
    for entry in pd:
        if not isinstance(entry, dict):
            cleaned.append(entry)
            continue
        items = entry.get("items")
        if not isinstance(items, list):
            cleaned.append(entry)
            continue
        new_items: list[Any] = []
        for item in items:
            if isinstance(item, dict) and "value" in item:
                value = _PHONEME_MARK_RE.sub("", str(item["value"]))
                new_items.append({**item, "value": value})
            else:
                new_items.append(item)
        cleaned.append({**entry, "items": new_items})
    return cleaned


def _pick_latency(body: dict[str, Any], default: str) -> str:
    raw = body.get("latency") or body.get("fish_latency") or default
    raw = str(raw).lower().strip()
    return raw if raw in {"low", "balanced", "normal"} else default


def _pick_format(body: dict[str, Any], default: str) -> str:
    raw = body.get("format") or body.get("response_format") or body.get("fish_format") or default
    raw = str(raw).lower().strip()
    if raw in {"mp3", "opus", "pcm", "pcm16", "wav"}:
        return raw
    if raw in {"aac", "flac"}:
        return "mp3"
    return default if default in _MEDIA else "mp3"


def _pcm_sample_rate(fmt: str, body: dict[str, Any], default: int) -> int:
    raw = body.get("sample_rate")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 24000 if fmt == "pcm16" else default
    return 24000 if fmt == "pcm16" else default


def _want_quality_guard(body: dict[str, Any]) -> bool:
    if "quality_guard" in body:
        return bool(body["quality_guard"])
    if "fish_quality_guard" in body:
        return bool(body["fish_quality_guard"])
    features = body.get("features")
    if isinstance(features, list) and "quality-guard" in features:
        return True
    return _env_bool("FISH_QUALITY_GUARD")


def _native_model_id(raw: str) -> str:
    name = raw.strip()
    if name.lower().startswith("fish-audio/"):
        return name.split("/", 1)[1]
    return name


def _resolve_tts_model(model: str | None, default: str) -> str:
    raw = _native_model_id(model or default)
    return _TTS_MODEL_ALIASES.get(raw.lower(), raw)


def _resolve_asr_model(model: str | None, default: str) -> str:
    raw = _native_model_id(model or default)
    key = raw.lower()
    if key in _ASR_MODEL_ALIASES:
        return default
    if key in {"transcribe-1", "transcribe-1-pro"}:
        return key
    return default


def _catalog_ids() -> list[str]:
    ids = list(_MODELS)
    for name in [*_TTS_MODELS, "transcribe-1", "transcribe-1-pro"]:
        prefixed = f"fish-audio/{name}"
        if prefixed not in ids:
            ids.append(prefixed)
    return ids


def prepare_tts_text(raw_input: str, *, dialogue_only: bool) -> str:
    cleaned = scrub_tts(raw_input)
    if dialogue_only:
        cleaned = extract_quoted_speech(cleaned)
    return normalize_cues(cleaned)


def _upstream_trace_headers(headers: Mapping[str, str]) -> dict[str, str]:
    found = w3c_trace_headers(headers)
    if found:
        return found
    return {"traceparent": make_traceparent()}


def _openai_error_type(status: int) -> str:
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 429:
        return "rate_limit_error"
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


def openai_error_body(status: int, message: str, *, provider: bool = False) -> dict[str, Any]:
    err: dict[str, Any] = {
        "code": int(status),
        "message": str(message),
        "type": "provider_error" if provider else _openai_error_type(status),
    }
    if provider:
        err["metadata"] = {"provider_name": "fish-audio"}
    return {"error": err}


def _json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(openai_error_body(status, message), status_code=status)


def _json_from_upstream(status: int, raw: Any) -> JSONResponse:
    detail = parse_fish_error(status, raw)
    code = int(detail["status"])
    return JSONResponse(
        openai_error_body(code, str(detail["message"]), provider=True),
        status_code=code,
    )


class _ReferenceError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _decode_audio_b64(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        audio = bytes(value)
        if not audio:
            raise _ReferenceError("reference audio is empty")
        return audio
    if not isinstance(value, str) or not value.strip():
        raise _ReferenceError("reference audio is empty")
    raw = "".join(value.strip().split())
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    padded = raw + ("=" * ((-len(raw)) % 4))
    try:
        audio = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _ReferenceError("reference audio is not valid base64") from exc
    if not audio:
        raise _ReferenceError("reference audio is empty")
    return audio


def _clip_from_parts(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audio": _decode_audio_b64(item.get("audio")),
        "text": str(item.get("text") or ""),
    }


def _normalize_references(raw: Any) -> Any:
    if isinstance(raw, list):
        return [_normalize_references(item) for item in raw]
    if isinstance(raw, Mapping):
        return _clip_from_parts(raw)
    raise _ReferenceError("references must be clips")


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
            text = str(item.get("text") or "")
    if audio is None:
        raise _ReferenceError("input_references needs one input_audio part")
    return [{"audio": _decode_audio_b64(audio), "text": text}]


def _fish_reference_clips(body: Mapping[str, Any]) -> Any | None:
    raw_in = body.get("input_references")
    if isinstance(raw_in, list) and raw_in:
        return _clips_from_input_references(raw_in)
    raw = body.get("references")
    if isinstance(raw, list) and raw:
        return _normalize_references(raw)
    return None


@dataclass(frozen=True)
class _InboundAsr:
    audio: bytes
    filename: str
    content_type: str
    model: str | None
    language: str
    response_format: str
    granularities: list[str]


def _granularity_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


async def _read_asr(request: Request) -> _InboundAsr | JSONResponse:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            parsed = await request.json()
        except json.JSONDecodeError:
            return _json_error(400, "invalid JSON body")
        if not isinstance(parsed, dict):
            return _json_error(400, "JSON body must be an object")
        inner = parsed.get("input_audio")
        data = inner.get("data") if isinstance(inner, dict) else None
        fmt = str(inner.get("format") or "wav") if isinstance(inner, dict) else "wav"
        try:
            audio = _decode_audio_b64(data)
        except _ReferenceError as exc:
            return _json_error(400, exc.message)
        model = parsed.get("model")
        language = str(parsed.get("language") or "")
        response_format = str(parsed.get("response_format") or "json")
        granularities = _granularity_list(parsed.get("timestamp_granularities"))
        return _InboundAsr(
            audio,
            f"utterance.{fmt}",
            f"audio/{fmt}",
            str(model) if model else None,
            language,
            response_format,
            granularities,
        )

    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        return _json_error(400, "empty audio upload")
    audio = await upload.read()
    if not audio:
        return _json_error(400, "empty audio upload")
    model_field = form.get("model")
    language_field = form.get("language")
    format_field = form.get("response_format")
    granularities = _form_strings(form, "timestamp_granularities", "timestamp_granularities[]")
    return _InboundAsr(
        audio,
        upload.filename or "audio.webm",
        upload.content_type or "application/octet-stream",
        str(model_field) if isinstance(model_field, str) else None,
        str(language_field) if isinstance(language_field, str) else "",
        str(format_field) if isinstance(format_field, str) else "json",
        granularities,
    )


def _form_strings(form: FormData, *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(item for item in form.getlist(name) if isinstance(item, str) and item)
    return out


def _caption_cues(data: dict[str, Any], text: str, *, strip_speakers: bool) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    raw_segments = data.get("segments") or []
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            body = scrub_asr(str(seg.get("text", "")), strip_speakers=strip_speakers)
            if not body:
                continue
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", 0) or 0)
            cues.append(CaptionCue(start, end, body))
    if cues:
        return cues
    if not text:
        return []
    duration = data.get("duration")
    end = float(duration) if isinstance(duration, (int, float)) else 0.0
    return [CaptionCue(0.0, end, text)]


async def _fish_send(
    client: httpx.AsyncClient,
    *,
    stream: bool,
    **request_kwargs: Any,
) -> httpx.Response | JSONResponse:
    last_error: JSONResponse | None = None
    for attempt in range(FISH_RETRY_ATTEMPTS):
        try:
            req = client.build_request(**request_kwargs)
            upstream = await client.send(req, stream=stream)
        except httpx.TimeoutException:
            last_error = _json_error(504, "Fish request timed out")
            if attempt + 1 >= FISH_RETRY_ATTEMPTS:
                return last_error
            await asyncio.sleep(fish_backoff_seconds(attempt))
            continue
        except httpx.RequestError as exc:
            last_error = _json_error(502, str(exc) or "Fish upstream unreachable")
            if attempt + 1 >= FISH_RETRY_ATTEMPTS:
                return last_error
            await asyncio.sleep(fish_backoff_seconds(attempt))
            continue
        if should_retry_fish_status(upstream.status_code):
            err = await upstream.aread()
            await upstream.aclose()
            last_error = _json_from_upstream(upstream.status_code, err)
            log.warning(
                "fish retry status=%s attempt=%s/%s",
                upstream.status_code,
                attempt + 1,
                FISH_RETRY_ATTEMPTS,
            )
            if attempt + 1 >= FISH_RETRY_ATTEMPTS:
                return last_error
            await asyncio.sleep(fish_backoff_seconds(attempt))
            continue
        if upstream.status_code >= 400:
            err = await upstream.aread()
            await upstream.aclose()
            return _json_from_upstream(upstream.status_code, err)
        return upstream
    return last_error or _json_error(502, "Fish upstream unreachable")


@asynccontextmanager
async def lifespan(app: FastAPI):
    defaults = _runtime_defaults()
    key = os.environ.get("FISH_API_KEY", "").strip()
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    else:
        log.warning("FISH_API_KEY unset; speech/transcription routes will fail until set")
    app.state.defaults = defaults
    app.state.fish_api_key = key
    app.state.http = httpx.AsyncClient(
        base_url=defaults.fish_base,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers=headers,
        http2=False,
    )
    yield
    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan, title="fish-audio-suite-proxy")


@app.post("/v1/audio/speech")
async def speech(request: Request):
    defaults: SuiteDefaults = request.app.state.defaults
    body = await request.json()
    model = _resolve_tts_model(body.get("model"), defaults.tts_model)
    try:
        raw_speed = float(body.get("speed", 1)) * defaults.speed
    except (TypeError, ValueError):
        raw_speed = defaults.speed
    speed = _clamp_float(raw_speed, 0.5, 2.0, defaults.speed)
    fmt = _pick_format(body, defaults.audio_format)
    latency = _pick_latency(body, defaults.latency)
    chunk_length = _clamp_int(
        body.get("chunk_length", body.get("fish_chunk_length", defaults.chunk_length)),
        100,
        _chunk_length_hi(defaults.fish_base),
        defaults.chunk_length,
    )
    min_chunk_length = _clamp_int(
        body.get(
            "min_chunk_length",
            body.get("fish_min_chunk_length", defaults.min_chunk_length),
        ),
        0,
        100,
        defaults.min_chunk_length,
    )

    raw_input = body.get("input", "") or ""
    dialogue_only = _env_bool("FISH_TTS_DIALOGUE_ONLY", "0")
    if "dialogue_only" in body:
        dialogue_only = bool(body["dialogue_only"])
    spoken = prepare_tts_text(raw_input, dialogue_only=dialogue_only)
    preview = spoken.replace("\n", " ")[:160]
    log.info(
        "tts scrub model=%s raw_len=%d spoken_len=%d preview=%r",
        body.get("model") or defaults.tts_model,
        len(raw_input),
        len(spoken),
        preview,
    )
    if is_tts_junk(spoken):
        log.info("tts skip junk preview=%r", preview)
        return Response(content=_SILENT_MP3, media_type="audio/mpeg")

    if not request.app.state.fish_api_key:
        return _json_error(401, "Invalid Token")

    native_fmt = "pcm" if fmt == "pcm16" else fmt
    payload: dict[str, Any] = {
        "text": spoken,
        "format": native_fmt,
        "latency": latency,
        "temperature": float(body.get("temperature", defaults.temperature)),
        "top_p": float(body.get("top_p", defaults.top_p)),
        "chunk_length": chunk_length,
        "min_chunk_length": min_chunk_length,
        "normalize": bool(body.get("normalize", True)),
        "prosody": {
            "speed": speed,
            "volume": float(body.get("volume", 0)),
            "normalize_loudness": True,
        },
        "repetition_penalty": float(body.get("repetition_penalty", 1.2)),
        "max_new_tokens": int(body.get("max_new_tokens", 1024)),
        "condition_on_previous_chunks": bool(body.get("condition_on_previous_chunks", True)),
        "early_stop_threshold": float(body.get("early_stop_threshold", 1)),
    }

    if native_fmt == "mp3":
        payload["mp3_bitrate"] = int(body.get("mp3_bitrate", defaults.mp3_bitrate))
        payload["sample_rate"] = int(body.get("sample_rate", defaults.sample_rate))
    elif native_fmt == "opus":
        payload["opus_bitrate"] = int(body.get("opus_bitrate", -1000))
        payload["sample_rate"] = int(body.get("sample_rate", 48000))
    else:
        payload["sample_rate"] = _pcm_sample_rate(fmt, body, defaults.sample_rate)

    voice = _pick_reference_id(body)
    if voice:
        payload["reference_id"] = voice

    seed = body.get("seed")
    if seed is not None:
        with suppress(TypeError, ValueError):
            payload["seed"] = int(seed)

    try:
        clips = _fish_reference_clips(body)
    except _ReferenceError as exc:
        return _json_error(400, exc.message)
    if clips is not None:
        payload["references"] = clips

    cache = body.get("use_memory_cache")
    if cache in {"on", "off"}:
        payload["use_memory_cache"] = cache

    if _want_quality_guard(body):
        payload["features"] = ["quality-guard"]

    pd = body.get("pronunciation_dictionary")
    if pd:
        payload["pronunciation_dictionary"] = _scrub_pronunciation_dictionary(pd)

    headers = {
        "Content-Type": "application/msgpack" if clips is not None else "application/json",
        "model": str(model),
        **_upstream_trace_headers(request.headers),
    }
    log.info(
        "tts model=%s fmt=%s trace=%s",
        model,
        fmt,
        trace_id_of(headers["traceparent"]),
    )

    client: httpx.AsyncClient = request.app.state.http
    upstream = await _fish_send(
        client,
        stream=True,
        method="POST",
        url="/v1/tts",
        headers=headers,
        **({"content": ormsgpack.packb(payload)} if clips is not None else {"json": payload}),
    )
    if isinstance(upstream, JSONResponse):
        return upstream

    async def stream_audio():
        try:
            async for chunk in upstream.aiter_bytes(4096):
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_audio(),
        media_type=_MEDIA.get(fmt, "audio/mpeg"),
    )


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    defaults: SuiteDefaults = request.app.state.defaults
    inbound = await _read_asr(request)
    if isinstance(inbound, JSONResponse):
        return inbound
    asr_model = _resolve_asr_model(inbound.model, defaults.asr_model)
    if not request.app.state.fish_api_key:
        return _json_error(401, "Invalid Token")

    granularities = inbound.granularities
    fmt = (inbound.response_format or "json").lower().strip()
    want_ts = fmt in {"verbose_json", "vtt", "srt"} or bool(granularities)
    ignore_timestamps = "false" if want_ts else "true"

    files = {"audio": (inbound.filename, inbound.audio, inbound.content_type)}
    form: dict[str, str] = {"ignore_timestamps": ignore_timestamps}
    lang = (inbound.language or defaults.asr_language or "").strip()
    if lang:
        form["language"] = lang

    asr_headers = {"model": asr_model, **_upstream_trace_headers(request.headers)}
    client: httpx.AsyncClient = request.app.state.http
    r = await _fish_send(
        client,
        stream=False,
        method="POST",
        url="/v1/asr",
        headers=asr_headers,
        files=files,
        data=form,
    )
    if isinstance(r, JSONResponse):
        return r

    raw = r.json()
    data: dict[str, Any] = raw if isinstance(raw, dict) else {}
    strip_speakers = _env_bool("FISH_ASR_STRIP_SPEAKERS", "0")
    text = scrub_asr(data.get("text") or "", strip_speakers=strip_speakers)
    log.info(
        "asr model=%s lang=%s chars=%d trace=%s",
        asr_model,
        data.get("language") or lang,
        len(text),
        trace_id_of(asr_headers["traceparent"]),
    )
    detected_lang = data.get("language") or data.get("language_code") or lang
    if is_asr_hallucination(text):
        log.info("asr drop hallucination lang=%r chars=%d", detected_lang, len(text))
        text = ""
        cues: list[CaptionCue] = []
    else:
        cues = _caption_cues(data, text, strip_speakers=strip_speakers)
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(format_as_srt(cues), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(format_as_vtt(cues), media_type="text/vtt")
    if fmt == "verbose_json":
        body: dict[str, Any] = {
            "task": "transcribe",
            "language": data.get("language_code") or data.get("language") or inbound.language,
            "duration": data.get("duration"),
            "text": text,
            "segments": [{"text": c.text, "start": c.start, "end": c.end} for c in cues],
        }
        if any(g.lower() == "word" for g in granularities):
            body["words"] = [{"word": c.text, "start": c.start, "end": c.end} for c in cues]
        return body
    return {"text": text}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in _catalog_ids()],
    }


@app.get("/health")
async def health(request: Request):
    defaults: SuiteDefaults = request.app.state.defaults
    return {
        "status": "ok",
        "defaults": {
            "model": defaults.tts_model,
            "asr_model": defaults.asr_model,
            "asr_language": defaults.asr_language,
            "latency": defaults.latency,
            "chunk_length": defaults.chunk_length,
            "format": defaults.audio_format,
            "speed_scale": defaults.speed,
            "quality_guard": _env_bool("FISH_QUALITY_GUARD"),
            "asr_strip_speakers": _env_bool("FISH_ASR_STRIP_SPEAKERS", "0"),
            "tts_dialogue_only": _env_bool("FISH_TTS_DIALOGUE_ONLY", "0"),
        },
    }


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _uvicorn_run_kwargs() -> dict[str, Any]:
    workers = max(_int_env("FISH_PROXY_WORKERS", _int_env("WEB_CONCURRENCY", 1)), 1)
    kwargs: dict[str, Any] = {
        "host": os.environ.get("FISH_PROXY_HOST", "0.0.0.0"),
        "port": _int_env("FISH_PROXY_PORT", 8849),
        "workers": workers,
        "loop": "auto",
        "http": "auto",
        "ws": "none",
        "timeout_keep_alive": _int_env("FISH_PROXY_KEEP_ALIVE", 5),
        "timeout_graceful_shutdown": _int_env("FISH_PROXY_GRACEFUL_SHUTDOWN", 120),
        "proxy_headers": True,
    }
    limit_raw = os.environ.get("FISH_PROXY_LIMIT_CONCURRENCY", "").strip()
    if limit_raw:
        with suppress(ValueError):
            limit = int(limit_raw)
            if limit > 0:
                kwargs["limit_concurrency"] = limit
    return kwargs


def main() -> None:
    # Import string so --workers / FISH_PROXY_WORKERS can spawn processes.
    uvicorn.run("fish_audio_suite_proxy.server:app", **_uvicorn_run_kwargs())


if __name__ == "__main__":
    main()
