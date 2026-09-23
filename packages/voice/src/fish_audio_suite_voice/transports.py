"""Duplex chat transports: OpenRouter SDK, or httpx SSE for any other base."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from fish_audio_suite_kit import MS_PER_S, bearer, strip_base
from fish_audio_suite_voice.debug import debug, env_debug, header_meta, warn

LLM_REFERER = "https://github.com/kzndotsh/fish-audio-suite"
LLM_TITLE = "fish-audio-suite-voice"
LLM_TEMPERATURE = 0.8
LLM_TIMEOUT_S = 120.0
_ABORT_CHARS = 800
_SSE_DATA = "data:"


class _AbortStats(Protocol):
    aborted: bool


@dataclass
class ChatCall:
    """One duplex chat request, shared by the OpenRouter SDK and httpx SSE paths.

    Notes
    -----
    ``route_model`` may already include ``:nitro``. ``use_openrouter`` is true
    only when ``base`` contains ``openrouter.ai``. ``stats`` is filled while
    events are consumed and is how a HTTP error aborts the stream.
    """

    messages: list[dict[str, str]]
    base: str
    key: str
    route_model: str
    max_tokens: int
    nitro: bool
    client: Any | None
    session_id: str | None
    trace_id: str | None
    stats: _AbortStats
    use_openrouter: bool


@asynccontextmanager
async def openrouter_client(key: str, base: str) -> AsyncGenerator[Any, None]:
    """Open an OpenRouter client for the duplex loop.

    Parameters
    ----------
    key : str
        API key.
    base : str
        Server URL passed through to the SDK.

    Yields
    ------
    Any
        The SDK client. Closed when the context exits.

    Notes
    -----
    ``openrouter`` is imported inside this function so ``import fish_audio_suite_voice``
    works without the ``cli`` extra.
    """
    from openrouter import OpenRouter

    async with OpenRouter(
        api_key=key,
        http_referer=LLM_REFERER,
        x_open_router_title=LLM_TITLE,
        x_open_router_categories="cli-agent",
        server_url=strip_base(base),
    ) as owned:
        yield owned


@asynccontextmanager
async def _or_client_ctx(key: str, base: str, client: Any | None) -> AsyncGenerator[Any, None]:
    if client is not None:
        yield client
        return
    async with openrouter_client(key, base) as owned:
        yield owned


def _chat_body(
    messages: list[dict[str, str]],
    route_model: str,
    max_tokens: int,
    token_field: str,
) -> dict[str, Any]:
    return {
        "messages": messages,
        "model": route_model,
        "stream": True,
        "temperature": LLM_TEMPERATURE,
        token_field: max_tokens,
    }


def _send_kwargs(call: ChatCall) -> dict[str, Any]:
    send_kw = _chat_body(call.messages, call.route_model, call.max_tokens, "max_completion_tokens")
    send_kw["timeout_ms"] = int(LLM_TIMEOUT_S * MS_PER_S)
    if call.nitro:
        send_kw["provider"] = {"sort": "throughput"}
    if call.session_id:
        send_kw["session_id"] = call.session_id
    if call.trace_id:
        send_kw["trace"] = {"trace_id": call.trace_id, "trace_name": LLM_TITLE}
    if env_debug():
        _note_usage(send_kw)
        send_kw["x_open_router_metadata"] = "enabled"
    return send_kw


async def _iter_openrouter_events(call: ChatCall) -> AsyncIterator[object]:
    from openrouter.errors import OpenRouterError

    try:
        async with _or_client_ctx(call.key, call.base, call.client) as or_client:
            res = await or_client.chat.send_async(**_send_kwargs(call))
            async with res as event_stream:
                async for event in event_stream:
                    yield event
    except OpenRouterError as e:
        _abort_http(call.stats, e.status_code, call.route_model, _abort_preview(e.body))


def _note_usage(payload: dict[str, Any]) -> None:
    payload["stream_options"] = {"include_usage": True}


def _abort_preview(body: object) -> str:
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)[:_ABORT_CHARS].decode("utf-8", "replace")
    return str(body)[:_ABORT_CHARS]


def _abort_http(stats: _AbortStats, status: object, model: str, body: str) -> None:
    warn(f"[llm] HTTP {status} model={model}: {body}")
    stats.aborted = True


def _sse_object(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _data_line(line: str) -> str | None:
    if not line or line.startswith(":") or not line.startswith(_SSE_DATA):
        return None
    return line.removeprefix(_SSE_DATA).strip()


def chat_completions_url(base: str) -> str:
    """Return the OpenAI-compatible chat completions URL for ``base``."""
    return f"{strip_base(base)}/chat/completions"


async def _iter_httpx_sse_events(call: ChatCall) -> AsyncIterator[object]:
    url = chat_completions_url(call.base)
    headers = {
        "Authorization": bearer(call.key),
        "Content-Type": "application/json",
        "HTTP-Referer": LLM_REFERER,
        "X-Title": LLM_TITLE,
    }
    payload = _chat_body(call.messages, call.route_model, call.max_tokens, "max_tokens")
    if env_debug():
        _note_usage(payload)
    async with (
        httpx.AsyncClient(timeout=LLM_TIMEOUT_S) as http,
        http.stream("POST", url, headers=headers, json=payload) as resp,
    ):
        if resp.status_code >= 400:
            body = _abort_preview(await resp.aread())
            _abort_http(call.stats, resp.status_code, call.route_model, body)
            return
        debug(
            "llm.response status={} headers={}",
            resp.status_code,
            header_meta(resp.headers),
        )
        async for line in resp.aiter_lines():
            data = _data_line(line)
            if data is None:
                continue
            if data == "[DONE]":
                return
            parsed = _sse_object(data)
            if parsed is not None:
                yield parsed


def chat_events(call: ChatCall) -> AsyncIterator[object]:
    """Pick the OpenRouter or httpx event stream for one chat call.

    Parameters
    ----------
    call : ChatCall
        Request. ``use_openrouter`` selects the SDK.

    Returns
    -------
    AsyncIterator
        Raw chat events. ``llm_token_stream`` is the only consumer and applies
        the same field reads to both transports.
    """
    if call.use_openrouter:
        return _iter_openrouter_events(call)
    return _iter_httpx_sse_events(call)
