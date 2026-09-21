"""Fish Audio → OpenAI Audio proxy (TTS + STT).

OpenAI-compatible:
  POST /v1/audio/speech          → Fish POST /v1/tts
  POST /v1/audio/transcriptions  → Fish POST /v1/asr

Unofficial. Not affiliated with Fish Audio.
Docs: https://docs.fish.audio/llms.txt
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from fish_audio_suite_kit import (
    SuiteDefaults,
    extract_quoted_speech,
    is_asr_hallucination,
    is_tts_junk,
    normalize_cues,
    scrub_asr,
    scrub_tts,
)

_MEDIA = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "wav": "audio/wav",
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
    if raw in {"mp3", "opus", "pcm", "wav"}:
        return raw
    if raw in {"aac", "flac"}:
        return "mp3"
    return default if default in _MEDIA else "mp3"


def _want_quality_guard(body: dict[str, Any]) -> bool:
    if "quality_guard" in body:
        return bool(body["quality_guard"])
    if "fish_quality_guard" in body:
        return bool(body["fish_quality_guard"])
    features = body.get("features")
    if isinstance(features, list) and "quality-guard" in features:
        return True
    return _env_bool("FISH_QUALITY_GUARD")


def _resolve_asr_model(model: str | None, default: str) -> str:
    raw = (model or default).strip()
    if raw in {"whisper-1", "whisper", "openai-whisper"}:
        return default
    if raw in {"transcribe-1", "transcribe-1-pro"}:
        return raw
    return default


def prepare_tts_text(raw_input: str, *, dialogue_only: bool) -> str:
    cleaned = scrub_tts(raw_input)
    if dialogue_only:
        cleaned = extract_quoted_speech(cleaned)
    return normalize_cues(cleaned)


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
    model = body.get("model") or defaults.tts_model
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

    payload: dict[str, Any] = {
        "text": spoken,
        "format": fmt,
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

    if fmt == "mp3":
        payload["mp3_bitrate"] = int(body.get("mp3_bitrate", defaults.mp3_bitrate))
        payload["sample_rate"] = int(body.get("sample_rate", defaults.sample_rate))
    elif fmt == "opus":
        payload["opus_bitrate"] = int(body.get("opus_bitrate", -1000))
        payload["sample_rate"] = int(body.get("sample_rate", 48000))
    else:
        payload["sample_rate"] = int(body.get("sample_rate", defaults.sample_rate))

    voice = _pick_reference_id(body)
    if voice:
        payload["reference_id"] = voice

    seed = body.get("seed")
    if seed is not None:
        with suppress(TypeError, ValueError):
            payload["seed"] = int(seed)

    references = body.get("references")
    if isinstance(references, list) and references:
        payload["references"] = references

    cache = body.get("use_memory_cache")
    if cache in {"on", "off"}:
        payload["use_memory_cache"] = cache

    if _want_quality_guard(body):
        payload["features"] = ["quality-guard"]

    pd = body.get("pronunciation_dictionary")
    if pd:
        payload["pronunciation_dictionary"] = _scrub_pronunciation_dictionary(pd)

    headers = {
        "Content-Type": "application/json",
        "model": str(model),
    }

    client: httpx.AsyncClient = request.app.state.http
    req = client.build_request("POST", "/v1/tts", json=payload, headers=headers)
    upstream = await client.send(req, stream=True)
    if upstream.status_code >= 400:
        err = await upstream.aread()
        await upstream.aclose()
        try:
            detail: Any = json.loads(err)
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = {
                "message": err.decode("utf-8", errors="replace"),
                "status": upstream.status_code,
            }
        if not isinstance(detail, dict):
            detail = {"message": str(detail), "status": upstream.status_code}
        return JSONResponse(detail, status_code=upstream.status_code)

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
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: list[str] | None = Form(None),
):
    defaults: SuiteDefaults = request.app.state.defaults
    asr_model = _resolve_asr_model(model, defaults.asr_model)
    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse({"error": "empty audio upload"}, status_code=400)

    filename = file.filename or "audio.webm"
    content_type = file.content_type or "application/octet-stream"

    want_ts = response_format in {"verbose_json", "vtt", "srt"} or bool(timestamp_granularities)
    ignore_timestamps = "false" if want_ts else "true"

    files = {"audio": (filename, audio_bytes, content_type)}
    form: dict[str, str] = {"ignore_timestamps": ignore_timestamps}
    lang = (language or defaults.asr_language or "").strip()
    if lang:
        form["language"] = lang

    client: httpx.AsyncClient = request.app.state.http
    r = await client.post(
        "/v1/asr",
        headers={"model": asr_model},
        files=files,
        data=form,
    )
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = {"message": r.text, "status": r.status_code}
        return JSONResponse(detail, status_code=r.status_code)

    raw = r.json()
    data: dict[str, Any] = raw if isinstance(raw, dict) else {}
    strip_speakers = _env_bool("FISH_ASR_STRIP_SPEAKERS", "0")
    text = scrub_asr(data.get("text") or "", strip_speakers=strip_speakers)
    log.info(
        "asr model=%s lang=%s chars=%d",
        asr_model,
        data.get("language") or lang,
        len(text),
    )
    detected_lang = data.get("language") or data.get("language_code") or lang
    if is_asr_hallucination(text):
        log.info("asr drop hallucination lang=%r chars=%d", detected_lang, len(text))
        text = ""

    fmt = (response_format or "json").lower().strip()
    if fmt == "text":
        return PlainTextResponse(text)

    if fmt == "verbose_json":
        segments: list[dict[str, float | str]] = []
        raw_segments = data.get("segments") or []
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                segments.append(
                    {
                        "text": str(seg.get("text", "")),
                        "start": float(seg.get("start", 0) or 0),
                        "end": float(seg.get("end", 0) or 0),
                    }
                )
        return {
            "task": "transcribe",
            "language": data.get("language_code") or data.get("language") or language,
            "duration": data.get("duration"),
            "text": text,
            "segments": segments,
        }

    return {"text": text}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in _MODELS],
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
