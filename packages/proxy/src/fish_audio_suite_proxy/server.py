"""OpenAI-compatible audio API in front of Fish Audio.

Routes:

- ``POST /v1/audio/speech`` calls Fish ``POST /v1/tts``.
- ``POST /v1/audio/transcriptions`` calls Fish ``POST /v1/asr``.
- ``GET /v1/models`` lists the model ids the routes accept.
- ``GET /health`` reports liveness and the runtime defaults.

Unofficial. Not affiliated with Fish Audio. Fish API docs:
https://docs.fish.audio/llms.txt
"""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from fish_audio_suite_kit import (
    FISH_ASR_PATH,
    FISH_TTS_PATH,
    bearer,
    env_text,
    is_insecure_fish_base,
)
from fish_audio_suite_proxy.body_limit import BodyLimitMiddleware
from fish_audio_suite_proxy.errors import ProxyError, proxy_error_response
from fish_audio_suite_proxy.models import catalog_ids
from fish_audio_suite_proxy.request_body import read_json_object
from fish_audio_suite_proxy.settings import ProxySettings, load_settings
from fish_audio_suite_proxy.speech import read_speech
from fish_audio_suite_proxy.transcribe import (
    fish_asr_request,
    read_asr_request,
    transcription_reply,
)
from fish_audio_suite_proxy.upstream import RetryPolicy, fish_send

__all__ = ["app", "lifespan", "main"]

log: Final[logging.Logger] = logging.getLogger("fish-audio-suite-proxy")
_MAX_CONNECTIONS = 100
# Keep every pooled connection alive, so the next burst reuses them instead of
# paying new TLS handshakes to Fish.
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
        Receives ``settings``, ``fish_api_key``, and ``http`` on ``app.state``.

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
    try:
        yield
    finally:
        await app.state.http.aclose()


app: Final[FastAPI] = FastAPI(lifespan=lifespan, title="fish-audio-suite-proxy")


def _settings(request: Request) -> ProxySettings:
    return request.app.state.settings


def _max_body_bytes() -> int:
    settings: ProxySettings = app.state.settings
    return settings.max_body_bytes


app.add_middleware(BodyLimitMiddleware, limit=_max_body_bytes)
app.add_exception_handler(ProxyError, proxy_error_response)


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


async def _client_key(request: Request) -> None:
    # Runs before the route reads the body, so a refused client sends nothing to Fish.
    keys = _settings(request).api_keys
    if keys and not _client_allowed(request, keys):
        raise ProxyError(401, "Invalid API key")


_CLIENT_KEY: Final = [Depends(_client_key)]


async def _iter_upstream(upstream: httpx.Response) -> AsyncIterator[bytes]:
    try:
        # No chunk size: httpx would hold bytes back until a full chunk had
        # arrived, which delays the first audio when Fish sends small pieces.
        async for chunk in upstream.aiter_bytes():
            yield chunk
    finally:
        await upstream.aclose()


def _fish_client(request: Request) -> httpx.AsyncClient:
    # Not a route dependency: speech answers junk text with silence before it
    # needs Fish, and a bad upload is a 400 before the missing key is a 503.
    if not request.app.state.fish_api_key:
        # The caller did nothing wrong. A 401 would tell them to fix a key
        # that is fine, while the proxy has none to send.
        raise ProxyError(503, "proxy has no FISH_API_KEY configured")
    return request.app.state.http


def _policy(settings: ProxySettings, *, deadline_s: float | None = None) -> RetryPolicy:
    deadline = settings.retry_deadline_s if deadline_s is None else deadline_s
    return RetryPolicy(attempts=settings.retry_attempts, deadline_s=deadline)


@app.post("/v1/audio/speech", dependencies=_CLIENT_KEY)
async def speech(request: Request) -> Response:
    """OpenAI ``/v1/audio/speech`` forwarded to Fish ``POST /v1/tts``.

    Parameters
    ----------
    request : Request
        JSON body. ``input`` must be a string within ``FISH_PROXY_MAX_INPUT_CHARS``.

    Returns
    -------
    Response
        Audio bytes streamed from Fish, or silence in the requested format when
        the scrubbed text is junk. Fish 429 and 5xx are retried. Other 4xx are not.

    Raises
    ------
    ProxyError
        An OpenAI error: 400 for a bad body or an unsupported ``response_format``,
        503 when the proxy has no Fish key, and Fish's own error otherwise.

    Notes
    -----
    Reference clips are MessagePack. A missing server key is a 503 from this
    route, not an import error.
    """
    settings = _settings(request)
    spoken = read_speech(await read_json_object(request), settings)
    if spoken.silence is not None:
        audio, mime = spoken.silence
        return Response(content=audio, media_type=mime)
    client = _fish_client(request)
    fish = spoken.fish_tts(request.headers)
    upstream = await fish_send(
        client,
        stream=True,
        policy=_policy(settings),
        is_disconnected=request.is_disconnected,
        method="POST",
        url=FISH_TTS_PATH,
        headers=fish.headers,
        **fish.request_kw,
    )
    return StreamingResponse(_iter_upstream(upstream), media_type=fish.media_type)


@app.post("/v1/audio/transcriptions", response_model=None, dependencies=_CLIENT_KEY)
async def transcriptions(request: Request) -> Response | dict[str, Any]:
    """OpenAI ``/v1/audio/transcriptions`` forwarded to Fish ``POST /v1/asr``.

    Parameters
    ----------
    request : Request
        Multipart ``file`` or JSON ``input_audio``. ``srt`` and ``vtt``
        response formats return caption files, not JSON.

    Returns
    -------
    Response or dict
        Transcript JSON or caption text. Language is omitted upstream unless the
        client or ``FISH_ASR_LANGUAGE`` sets it.

    Raises
    ------
    ProxyError
        An OpenAI error: 400 for a bad upload or option, 503 when the proxy has no
        Fish key, 502 for a Fish reply that is not a transcript, and Fish's own
        error otherwise.
    """
    settings = _settings(request)
    inbound = await read_asr_request(request)
    client = _fish_client(request)
    fish = fish_asr_request(inbound, settings, request.headers)
    upstream = await fish_send(
        client,
        stream=False,
        # A long recording can take Fish minutes, past the speech budgets.
        policy=_policy(settings, deadline_s=settings.asr_timeout_s),
        read_timeout_s=settings.asr_timeout_s,
        is_disconnected=request.is_disconnected,
        method="POST",
        url=FISH_ASR_PATH,
        headers=fish.headers,
        files=fish.files,
        data=fish.form,
    )
    return transcription_reply(inbound, fish, upstream, settings)


@app.get("/v1/models", response_model=None, dependencies=_CLIENT_KEY)
async def models(request: Request) -> dict[str, Any]:
    """List the model ids the routes accept.

    Parameters
    ----------
    request : Request
        Used for the alias table.

    Returns
    -------
    dict
        OpenAI ``{object: list, data: [...]}``. Native Fish ids, the TTS
        aliases (``tts-1`` and any from ``FISH_PROXY_TTS_ALIASES``), ``whisper-1``,
        and the ``fish-audio/`` prefixed native ids.
    """
    aliases = _settings(request).model_aliases
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


def main() -> None:
    """Run uvicorn on ``fish_audio_suite_proxy.server:app``.

    Notes
    -----
    The import string is required so ``FISH_PROXY_WORKERS`` can spawn
    processes. Websockets are disabled. ``forwarded-allow-ips`` is left
    at uvicorn's default, not ``*``. The default host is ``127.0.0.1``.
    """
    uvicorn.run("fish_audio_suite_proxy.server:app", **load_settings().uvicorn_kwargs())


if __name__ == "__main__":
    main()
