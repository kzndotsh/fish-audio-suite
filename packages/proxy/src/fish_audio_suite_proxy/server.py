"""Fish Audio → OpenAI Audio proxy (TTS + STT).

OpenAI-compatible:
  POST /v1/audio/speech          → Fish POST /v1/tts
  POST /v1/audio/transcriptions  → Fish POST /v1/asr

Unofficial. Not affiliated with Fish Audio.
Docs: https://docs.fish.audio/llms.txt
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from fish_audio_suite_kit import (
    FISH_ASR_PATH,
    FISH_TTS_PATH,
    AsrBody,
    CaptionCue,
    FishHttpError,
    bearer,
    env_text,
    is_asr_hallucination,
    is_caption_watermark,
    is_insecure_fish_base,
    is_tts_junk,
    parse_asr_body,
    scrub_asr,
    trace_id_of,
    without_watermark_segments,
)
from fish_audio_suite_proxy.audio import pcm_sample_rate, silent_speech
from fish_audio_suite_proxy.body_limit import BodyLimitMiddleware
from fish_audio_suite_proxy.errors import (
    ProxyError,
    json_error,
    json_from_call_failure,
    proxy_error_response,
    read_json_object,
)
from fish_audio_suite_proxy.models import catalog_ids, resolve_asr_model
from fish_audio_suite_proxy.phrases import caption_cues
from fish_audio_suite_proxy.request_fields import prepare_tts_text, read_flag, traced_model_headers
from fish_audio_suite_proxy.settings import ProxySettings, load_settings
from fish_audio_suite_proxy.speech import pack_tts, speech_controls
from fish_audio_suite_proxy.transcribe import (
    PRO_ASR_MODEL,
    asr_upload,
    read_asr_format,
    read_asr_request,
    transcription_body,
)
from fish_audio_suite_proxy.upstream import RetryPolicy, fish_send

__all__ = ["app", "lifespan", "main"]

log: Final[logging.Logger] = logging.getLogger("fish-audio-suite-proxy")
_PREVIEW_CHARS = 160
_MAX_CONNECTIONS = 100
# Keep every pooled connection warm, so a burst over the old limit of 20 did
# not pay new TLS handshakes to Fish on the next burst.
_MAX_KEEPALIVE = _MAX_CONNECTIONS
_KEEPALIVE_EXPIRY_S = 30.0
_STARTED_AT = int(time.time())
_OWNER = "fish-audio"


def _user_agent() -> str:
    try:
        pkg = version("fish-audio-suite-proxy")
    except PackageNotFoundError:
        pkg = "unknown"
    return f"fish-audio-suite-proxy/{pkg}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Open one Fish httpx client for the process and close it on shutdown.

    Parameters
    ----------
    app : FastAPI
        Receives ``settings``, ``defaults``, ``fish_api_key``, and ``http``
        on ``app.state``.

    Notes
    -----
    ``FISH_API_KEY`` is read here, not at import, so ``GET /health`` works
    with the key unset. Speech and transcription then return 503, because
    the proxy is misconfigured, not the caller. The client keeps HTTP/2 off,
    sends a User-Agent, and caps the connection pool.
    """
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    key = env_text("FISH_API_KEY")
    headers: dict[str, str] = {"User-Agent": _user_agent()}
    if key:
        headers["Authorization"] = bearer(key)
    else:
        log.warning("FISH_API_KEY unset; speech/transcription routes will fail until set")
    if is_insecure_fish_base(settings.defaults.fish_base):
        log.warning(
            "FISH_BASE host %s is plain http and not loopback; the Fish API key "
            "travels in cleartext. Use https unless this network is trusted",
            urlsplit(settings.defaults.fish_base).hostname,
        )
    if settings.exposed and not settings.auth_required:
        log.warning(
            "listening on %s with no FISH_PROXY_API_KEYS; anyone who can reach this port "
            "spends your Fish credits",
            settings.host,
        )
    app.state.settings = settings
    app.state.defaults = settings.defaults
    app.state.fish_api_key = key
    app.state.http = httpx.AsyncClient(
        base_url=settings.defaults.fish_base,
        timeout=httpx.Timeout(
            connect=settings.connect_timeout_s,
            read=settings.read_timeout_s,
            write=settings.read_timeout_s,
            pool=settings.pool_timeout_s,
        ),
        limits=httpx.Limits(
            max_connections=_MAX_CONNECTIONS,
            max_keepalive_connections=_MAX_KEEPALIVE,
            keepalive_expiry=_KEEPALIVE_EXPIRY_S,
        ),
        headers=headers,
        http2=False,
    )
    yield
    await app.state.http.aclose()


app: Final[FastAPI] = FastAPI(lifespan=lifespan, title="fish-audio-suite-proxy")
app.add_middleware(BodyLimitMiddleware)
app.add_exception_handler(ProxyError, proxy_error_response)


def _settings(request: Request) -> ProxySettings:
    return request.app.state.settings


def _client_allowed(request: Request, keys: tuple[str, ...]) -> bool:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    supplied = token.strip().encode()
    # Compare against every key, so the time does not say which one matched.
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(supplied, key.encode())
    return matched


def _unauthorized(request: Request) -> JSONResponse | None:
    keys = _settings(request).api_keys
    if not keys or _client_allowed(request, keys):
        return None
    return json_error(401, "Invalid API key")


def _spoken_line(body: dict[str, Any], settings: ProxySettings) -> tuple[str, str]:
    raw_input = body.get("input", "") or ""
    dialogue_only = read_flag(body, "dialogue_only", default=settings.tts_dialogue_only)
    spoken = prepare_tts_text(
        raw_input,
        dialogue_only=dialogue_only,
        mood_lead=settings.tts_mood_lead,
    )
    preview = spoken.replace("\n", " ")[:_PREVIEW_CHARS]
    log.info(
        "tts scrub model=%s raw_len=%d spoken_len=%d",
        body.get("model") or settings.defaults.tts_model,
        len(raw_input),
        len(spoken),
    )
    # The text people speak is private. It is logged only on request.
    if settings.log_text:
        log.info("tts text preview=%r", preview)
    return spoken, preview


def _asr_text(
    data: AsrBody,
    transcript: str,
    *,
    lang: str,
    asr_model: str,
    traceparent: str,
    request_id: str,
    settings: ProxySettings,
) -> tuple[str, list[CaptionCue]]:
    strip_speakers = settings.asr_strip_speakers
    strip_cues = settings.asr_strip_cues
    text = scrub_asr(transcript, strip_speakers=strip_speakers, strip_cues=strip_cues)
    log.info(
        "asr model=%s lang=%s chars=%d trace=%s request_id=%s",
        asr_model,
        data.get("language") or lang,
        len(text),
        trace_id_of(traceparent),
        request_id or "-",
    )
    detected_lang = data.get("language") or data.get("language_code") or lang
    if is_asr_hallucination(text):
        # A watermark or blank text still drops words that Fish put only in
        # segments. A watermark segment next to real speech is left out, or
        # the caption line is spoken with the sentence.
        cues = [
            cue
            for cue in caption_cues(data, "", strip_speakers=strip_speakers, strip_cues=strip_cues)
            if not is_caption_watermark(cue.text)
        ]
        joined = " ".join(cue.text for cue in cues).strip()
        if joined and not is_asr_hallucination(joined):
            return joined, cues
        log.info("asr drop hallucination lang=%r chars=%d", detected_lang, len(text))
        return "", []
    cues = caption_cues(data, text, strip_speakers=strip_speakers, strip_cues=strip_cues)
    # A watermark segment is omitted from the captions. The top-level text
    # still contains that phrase, so JSON says "thanks for watching" and
    # the SRT file does not.
    trimmed = without_watermark_segments(
        text, data.get("segments"), strip_speakers=strip_speakers, strip_cues=strip_cues
    )
    if trimmed != text and trimmed and not is_asr_hallucination(trimmed):
        # Segment cues already omit the watermark. When every segment was
        # one, the fallback cue was the untrimmed text, so the SRT file
        # still said "thanks for watching".
        return trimmed, caption_cues(
            data, trimmed, strip_speakers=strip_speakers, strip_cues=strip_cues
        )
    return text, cues


_REQUEST_ID_MAX = 128


def _request_id(*candidates: object) -> str:
    # transcribe-1-pro sends request_id in the body and the x-request-id header.
    # Only a short single-line value reaches the log.
    for value in candidates:
        if isinstance(value, str) and (line := value.strip()):
            return line.splitlines()[0][:_REQUEST_ID_MAX]
    return ""


async def _iter_upstream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    try:
        # No chunk size: httpx would hold bytes back until a full chunk had
        # arrived, which delays the first audio when Fish sends small pieces.
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()


def _fish_client(request: Request) -> httpx.AsyncClient | JSONResponse:
    if not request.app.state.fish_api_key:
        # The caller did nothing wrong. A 401 would tell them to fix a key
        # that is fine, while the proxy has none to send.
        return json_error(503, "proxy has no FISH_API_KEY configured")
    return request.app.state.http


def _policy(settings: ProxySettings, *, deadline_s: float | None = None) -> RetryPolicy:
    deadline = settings.retry_deadline_s if deadline_s is None else deadline_s
    return RetryPolicy(attempts=settings.retry_attempts, deadline_s=deadline)


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    """OpenAI ``/v1/audio/speech`` forwarded to Fish ``POST /v1/tts``.

    Parameters
    ----------
    request : Request
        JSON body. ``input`` must be a string within ``FISH_PROXY_MAX_INPUT_CHARS``.

    Returns
    -------
    Response
        Audio bytes, silence in the requested format when the scrubbed text
        is junk, or an OpenAI error JSON. Fish 429 and 5xx are retried.
        Other 4xx are not. An unsupported ``response_format`` is a 400.

    Notes
    -----
    Reference clips are MessagePack. A missing server key is a 503 from this
    route, not an import error.
    """
    if (refused := _unauthorized(request)) is not None:
        return refused
    settings = _settings(request)
    defaults = settings.defaults
    body = await read_json_object(request)
    if isinstance(body, JSONResponse):
        return body
    raw_input = body.get("input", "")
    if not isinstance(raw_input, str):
        return json_error(400, "input must be a string")
    if settings.max_input_chars and len(raw_input) > settings.max_input_chars:
        return json_error(400, f"input is longer than {settings.max_input_chars} characters")
    controls = speech_controls(
        body, defaults, settings.tts_aliases, default_format=settings.tts_format
    )
    spoken, preview = _spoken_line(body, settings)
    if is_tts_junk(spoken, drop_narration=settings.tts_drop_narration):
        log.info("tts skip junk chars=%d", len(spoken))
        if settings.log_text:
            log.info("tts junk preview=%r", preview)
        rate = pcm_sample_rate(controls.fmt, body, defaults.sample_rate)
        audio, mime = silent_speech(controls.fmt, rate)
        return Response(content=audio, media_type=mime)

    client = _fish_client(request)
    if isinstance(client, JSONResponse):
        return client

    packed = pack_tts(
        body,
        defaults,
        request.headers,
        controls,
        spoken,
        quality_guard=settings.quality_guard,
    )
    if isinstance(packed, JSONResponse):
        return packed
    log.info(
        "tts model=%s fmt=%s trace=%s",
        controls.model,
        controls.fmt,
        trace_id_of(packed.headers["traceparent"]),
    )

    upstream = await fish_send(
        client,
        stream=True,
        policy=_policy(settings),
        is_disconnected=request.is_disconnected,
        method="POST",
        url=FISH_TTS_PATH,
        headers=packed.headers,
        **packed.request_kw,
    )
    if isinstance(upstream, JSONResponse):
        return upstream
    return StreamingResponse(_iter_upstream(upstream), media_type=packed.media_type)


@app.post("/v1/audio/transcriptions", response_model=None)
async def transcriptions(request: Request) -> Response | dict[str, Any]:
    """OpenAI ``/v1/audio/transcriptions`` forwarded to Fish ``POST /v1/asr``.

    Parameters
    ----------
    request : Request
        Multipart ``file`` or JSON ``input_audio``. ``srt`` and ``vtt``
        response formats return caption files, not JSON.

    Returns
    -------
    Response
        Transcript JSON, caption text, or an OpenAI error. Language is omitted
        upstream unless the client or ``FISH_ASR_LANGUAGE`` sets it.
    """
    if (refused := _unauthorized(request)) is not None:
        return refused
    settings = _settings(request)
    defaults = settings.defaults
    inbound = await read_asr_request(request)
    if isinstance(inbound, JSONResponse):
        return inbound
    asr_model = resolve_asr_model(inbound.model, defaults.asr_model)
    client = _fish_client(request)
    if isinstance(client, JSONResponse):
        return client

    granularities = inbound.granularities
    fmt = read_asr_format(inbound.response_format or "json")
    files, form, lang = asr_upload(inbound, defaults, fmt, granularities, model=asr_model)
    if asr_model != PRO_ASR_MODEL and inbound.pro.form_fields():
        log.info("asr pro-only fields not sent to model=%s", asr_model)

    asr_headers = traced_model_headers(asr_model, request.headers)
    r = await fish_send(
        client,
        stream=False,
        # A long recording can take Fish minutes, past the speech budgets.
        policy=_policy(settings, deadline_s=settings.asr_timeout_s),
        read_timeout_s=settings.asr_timeout_s,
        is_disconnected=request.is_disconnected,
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
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return json_from_call_failure(FishHttpError.for_non_json())
    try:
        data, transcript = parse_asr_body(raw)
    except FishHttpError as exc:
        return json_from_call_failure(exc)
    text, cues = _asr_text(
        data,
        transcript,
        lang=lang,
        asr_model=asr_model,
        traceparent=asr_headers["traceparent"],
        request_id=_request_id(dict(data).get("request_id"), r.headers.get("x-request-id")),
        settings=settings,
    )
    return transcription_body(
        fmt,
        text,
        cues,
        data,
        language=lang or None,
        granularities=granularities,
        strip_speakers=settings.asr_strip_speakers,
        strip_cues=settings.asr_strip_cues,
    )


@app.get("/v1/models", response_model=None)
async def models(request: Request) -> Response | dict[str, Any]:
    """List the model ids the routes accept.

    Parameters
    ----------
    request : Request
        Used for the key check and the alias table.

    Returns
    -------
    dict
        OpenAI ``{object: list, data: [...]}``. Native Fish ids, the TTS
        aliases (``tts-1`` and any from ``FISH_PROXY_TTS_ALIASES``), ``whisper-1``,
        and the ``fish-audio/`` prefixed native ids.
    """
    if (refused := _unauthorized(request)) is not None:
        return refused
    aliases = _settings(request).tts_aliases
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": _STARTED_AT, "owned_by": _OWNER}
            for m in catalog_ids(aliases)
        ],
    }


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness plus the clamped runtime defaults. Does not call Fish.

    Parameters
    ----------
    request : Request
        Used only to read the settings stored at startup.

    Returns
    -------
    dict
        ``status`` is ``ok`` even when ``FISH_API_KEY`` is unset. No key
        value is included, only whether client auth is required.
    """
    return {"status": "ok", "defaults": _settings(request).health()}


def _uvicorn_run_kwargs() -> dict[str, Any]:
    return load_settings().uvicorn_kwargs()


def main() -> None:
    """Run uvicorn on ``fish_audio_suite_proxy.server:app``.

    Notes
    -----
    The import string is required so ``FISH_PROXY_WORKERS`` can spawn
    processes. Websockets are disabled. ``forwarded-allow-ips`` is left
    at uvicorn's default, not ``*``. The default host is ``127.0.0.1``.
    """
    # Import string so --workers / FISH_PROXY_WORKERS can spawn processes.
    uvicorn.run("fish_audio_suite_proxy.server:app", **_uvicorn_run_kwargs())


if __name__ == "__main__":
    main()
