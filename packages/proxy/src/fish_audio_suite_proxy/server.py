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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from fish_audio_suite_kit import (
    FISH_ASR_PATH,
    FISH_RETRY_ATTEMPTS,
    FISH_TTS_PATH,
    CaptionCue,
    FishHttpError,
    SuiteDefaults,
    bearer,
    env_int,
    env_text,
    env_token,
    fish_non_json,
    fish_request_error,
    fish_retry_pause,
    fish_unreachable,
    is_asr_hallucination,
    is_tts_junk,
    parse_asr_body,
    scrub_asr,
    should_retry_fish_status,
    trace_id_of,
)
from fish_audio_suite_proxy.errors import json_error, json_from_upstream, read_json_object
from fish_audio_suite_proxy.fields import (
    SILENT_MP3,
    catalog_ids,
    dialogue_only_env,
    explicit_bool,
    media_type,
    prepare_tts_text,
    quality_guard_env,
    resolve_asr_model,
    runtime_defaults,
    strip_speakers_env,
    traced_model_headers,
)
from fish_audio_suite_proxy.speech import pack_tts, speech_controls
from fish_audio_suite_proxy.transcribe import (
    asr_upload,
    caption_cues,
    read_asr,
    transcription_body,
)

log = logging.getLogger("fish-audio-suite-proxy")
_TTS_CHUNK = 4096
_FISH_TIMEOUT_S = 120.0
_FISH_CONNECT_S = 10.0
_PREVIEW_CHARS = 160
_DEFAULT_PORT = 8849
_PORT_MAX = 65535
_DEFAULT_KEEP_ALIVE_S = 5
_DEFAULT_GRACEFUL_S = 120

logging.basicConfig(level=logging.INFO)


async def _closed_error(upstream: httpx.Response) -> JSONResponse:
    body = await upstream.aread()
    await upstream.aclose()
    return json_from_upstream(upstream.status_code, body)


class _FishHttp(Protocol):
    def build_request(
        self,
        method: str,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        headers: Any = None,
    ) -> Any: ...

    async def send(self, request: Any, *, stream: bool = False) -> Any: ...


async def _fish_send(
    client: _FishHttp,
    *,
    stream: bool,
    **request_kwargs: Any,
) -> httpx.Response | JSONResponse:
    last_error: JSONResponse | None = None
    for attempt in range(FISH_RETRY_ATTEMPTS):
        try:
            req = client.build_request(**request_kwargs)
            upstream = await client.send(req, stream=stream)
        except httpx.RequestError as exc:
            status, message = fish_request_error(exc, httpx.TimeoutException)
            last_error = json_error(status, message)
        else:
            if upstream.status_code < 400:
                return upstream
            last_error = await _closed_error(upstream)
            if not should_retry_fish_status(upstream.status_code):
                return last_error
            log.warning(
                "fish retry status=%s attempt=%s/%s",
                upstream.status_code,
                attempt + 1,
                FISH_RETRY_ATTEMPTS,
            )
        if await fish_retry_pause(attempt):
            return last_error
    status, message = fish_unreachable()
    return last_error or json_error(status, message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    defaults = runtime_defaults()
    key = env_text("FISH_API_KEY")
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = bearer(key)
    else:
        log.warning("FISH_API_KEY unset; speech/transcription routes will fail until set")
    app.state.defaults = defaults
    app.state.fish_api_key = key
    app.state.http = httpx.AsyncClient(
        base_url=defaults.fish_base,
        timeout=httpx.Timeout(_FISH_TIMEOUT_S, connect=_FISH_CONNECT_S),
        headers=headers,
        http2=False,
    )
    yield
    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan, title="fish-audio-suite-proxy")


def _spoken_line(body: dict[str, Any], defaults: SuiteDefaults) -> tuple[str, str]:
    raw_input = body.get("input", "") or ""
    dialogue_only = explicit_bool(body, "dialogue_only", default=dialogue_only_env())
    spoken = prepare_tts_text(raw_input, dialogue_only=dialogue_only)
    preview = spoken.replace("\n", " ")[:_PREVIEW_CHARS]
    log.info(
        "tts scrub model=%s raw_len=%d spoken_len=%d preview=%r",
        body.get("model") or defaults.tts_model,
        len(raw_input),
        len(spoken),
        preview,
    )
    return spoken, preview


def _asr_text(
    data: dict[str, Any],
    transcript: str,
    lang: str,
    asr_model: str,
    traceparent: str,
) -> tuple[str, list[CaptionCue]]:
    strip_speakers = strip_speakers_env()
    text = scrub_asr(transcript, strip_speakers=strip_speakers)
    log.info(
        "asr model=%s lang=%s chars=%d trace=%s",
        asr_model,
        data.get("language") or lang,
        len(text),
        trace_id_of(traceparent),
    )
    detected_lang = data.get("language") or data.get("language_code") or lang
    if is_asr_hallucination(text):
        log.info("asr drop hallucination lang=%r chars=%d", detected_lang, len(text))
        return "", []
    return text, caption_cues(data, text, strip_speakers=strip_speakers)


async def _iter_upstream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_bytes(_TTS_CHUNK):
            yield chunk
    finally:
        await upstream.aclose()


def _defaults(request: Request) -> SuiteDefaults:
    return request.app.state.defaults


def _fish_client(request: Request) -> httpx.AsyncClient | JSONResponse:
    if not request.app.state.fish_api_key:
        return json_error(401, "Invalid Token")
    return request.app.state.http


@app.post("/v1/audio/speech")
async def speech(request: Request):
    defaults = _defaults(request)
    body = await read_json_object(request)
    if isinstance(body, JSONResponse):
        return body
    raw_input = body.get("input", "")
    if not isinstance(raw_input, str):
        return json_error(400, "input must be a string")
    controls = speech_controls(body, defaults)
    spoken, preview = _spoken_line(body, defaults)
    if is_tts_junk(spoken):
        log.info("tts skip junk preview=%r", preview)
        return Response(content=SILENT_MP3, media_type=media_type("mp3"))

    client = _fish_client(request)
    if isinstance(client, JSONResponse):
        return client

    packed = pack_tts(body, defaults, request.headers, controls, spoken)
    if isinstance(packed, JSONResponse):
        return packed
    log.info(
        "tts model=%s fmt=%s trace=%s",
        controls.model,
        controls.fmt,
        trace_id_of(packed.headers["traceparent"]),
    )

    upstream = await _fish_send(
        client,
        stream=True,
        method="POST",
        url=FISH_TTS_PATH,
        headers=packed.headers,
        **packed.request_kw,
    )
    if isinstance(upstream, JSONResponse):
        return upstream
    return StreamingResponse(_iter_upstream(upstream), media_type=packed.media_type)


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    defaults = _defaults(request)
    inbound = await read_asr(request)
    if isinstance(inbound, JSONResponse):
        return inbound
    asr_model = resolve_asr_model(inbound.model, defaults.asr_model)
    client = _fish_client(request)
    if isinstance(client, JSONResponse):
        return client

    granularities = inbound.granularities
    fmt = (inbound.response_format or "json").lower().strip()
    files, form, lang = asr_upload(inbound, defaults, fmt, granularities)

    asr_headers = traced_model_headers(asr_model, request.headers)
    r = await _fish_send(
        client,
        stream=False,
        method="POST",
        url=FISH_ASR_PATH,
        headers=asr_headers,
        files=files,
        data=form,
    )
    if isinstance(r, JSONResponse):
        return r

    try:
        raw = r.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        status, message = fish_non_json()
        return json_error(status, message)
    try:
        data, transcript = parse_asr_body(raw)
    except FishHttpError as exc:
        return json_error(exc.status, exc.message)
    text, cues = _asr_text(data, transcript, lang, asr_model, asr_headers["traceparent"])
    return transcription_body(
        fmt,
        text,
        cues,
        data,
        language=inbound.language,
        granularities=granularities,
    )


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model"} for m in catalog_ids()],
    }


@app.get("/health")
async def health(request: Request):
    defaults = _defaults(request)
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
            "quality_guard": quality_guard_env(),
            "asr_strip_speakers": strip_speakers_env(),
            "tts_dialogue_only": dialogue_only_env(),
        },
    }


def _listen_port() -> int:
    port = env_int("FISH_PROXY_PORT", _DEFAULT_PORT)
    if 1 <= port <= _PORT_MAX:
        return port
    return _DEFAULT_PORT


def _non_negative_s(name: str, default: int) -> int:
    value = env_int(name, default)
    return value if value >= 0 else default


def _uvicorn_run_kwargs() -> dict[str, Any]:
    workers = max(env_int("FISH_PROXY_WORKERS", env_int("WEB_CONCURRENCY", 1)), 1)
    kwargs: dict[str, Any] = {
        "host": env_token("FISH_PROXY_HOST", "0.0.0.0"),
        "port": _listen_port(),
        "workers": workers,
        "loop": "auto",
        "http": "auto",
        "ws": "none",
        "timeout_keep_alive": _non_negative_s("FISH_PROXY_KEEP_ALIVE", _DEFAULT_KEEP_ALIVE_S),
        "timeout_graceful_shutdown": _non_negative_s(
            "FISH_PROXY_GRACEFUL_SHUTDOWN", _DEFAULT_GRACEFUL_S
        ),
        "proxy_headers": True,
    }
    limit = env_int("FISH_PROXY_LIMIT_CONCURRENCY", 0)
    if limit > 0:
        kwargs["limit_concurrency"] = limit
    return kwargs


def main() -> None:
    # Import string so --workers / FISH_PROXY_WORKERS can spawn processes.
    uvicorn.run("fish_audio_suite_proxy.server:app", **_uvicorn_run_kwargs())


if __name__ == "__main__":
    main()
