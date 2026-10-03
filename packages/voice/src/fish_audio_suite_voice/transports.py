"""Duplex chat transports: OpenRouter SDK, or httpx SSE for any OpenAI-compatible server."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final, Protocol

import httpx

from fish_audio_suite_kit import (
    MS_PER_S,
    ChatMessage,
    bearer,
    retry_after_s,
    strip_base,
    utf8_text,
)
from fish_audio_suite_voice.debug import debug, debug_enabled, warn
from fish_audio_suite_voice.tune import LlmSettings

__all__ = [
    "ChatCall",
    "chat_completions_url",
    "chat_events",
    "http_client",
    "openrouter_client",
]

_ABORT_CHARS: Final = 800
_SSE_DATA: Final = "data:"


class _AbortStats(Protocol):
    aborted: bool
    http_status: int | None
    retry_after_s: float | None
    quota_exhausted: bool


@dataclass(slots=True)
class ChatCall:
    """One duplex chat request, shared by the OpenRouter SDK and httpx SSE paths.

    Notes
    -----
    ``route_model`` may already include ``:nitro``. ``tune.uses_openrouter_sdk`` picks the
    SDK path. ``stats`` is filled while events are consumed and is how a HTTP
    error aborts the stream. ``http`` is the session's pooled client for the
    OpenAI-compatible path.
    """

    messages: list[ChatMessage]
    tune: LlmSettings
    route_model: str
    client: Any | None
    http: httpx.AsyncClient | None
    session_id: str | None
    trace_id: str | None
    stats: _AbortStats


def http_client(tune: LlmSettings) -> httpx.AsyncClient:
    """Build the pooled client for the OpenAI-compatible backend.

    Parameters
    ----------
    tune : LlmSettings
        Supplies the request timeout.

    Returns
    -------
    httpx.AsyncClient
        Reused for every reply in a session. The caller closes it.
    """
    return httpx.AsyncClient(timeout=tune.timeout_s)


@asynccontextmanager
async def openrouter_client(tune: LlmSettings) -> AsyncGenerator[Any, None]:
    """Open an OpenRouter client for the duplex loop.

    Parameters
    ----------
    tune : LlmSettings
        Supplies the key, server URL, and attribution fields. An empty
        referer, title, or category is not sent.

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

    # The SDK writes the key into Authorization and adds Bearer itself.
    # A newline raises before the request is sent, so the reply is empty.
    async with OpenRouter(
        api_key=bearer(tune.api_key).removeprefix("Bearer "),
        http_referer=tune.referer or None,
        x_open_router_title=tune.title or None,
        x_open_router_categories=tune.categories or None,
        server_url=strip_base(tune.base),
    ) as owned:
        yield owned


@asynccontextmanager
async def _or_client_ctx(tune: LlmSettings, client: Any | None) -> AsyncGenerator[Any, None]:
    if client is not None:
        yield client
        return
    async with openrouter_client(tune) as owned:
        yield owned


def _json_ready(value: Any) -> Any:
    # A lone surrogate cannot be encoded as UTF-8. httpx then raises and the
    # duplex turn speaks nothing, including when the character is only in the
    # system prompt.
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


def _chat_body(call: ChatCall, token_field: str) -> dict[str, Any]:
    tune = call.tune
    return _json_ready(
        {
            "messages": call.messages,
            "model": call.route_model,
            "stream": True,
            "temperature": tune.temperature,
            token_field: tune.max_tokens,
        }
    )


def _send_kwargs(call: ChatCall) -> dict[str, Any]:
    tune = call.tune
    send_kw = _chat_body(call, "max_completion_tokens")
    send_kw["timeout_ms"] = int(tune.timeout_s * MS_PER_S)
    if tune.nitro and tune.provider_sort and call.route_model.endswith(":nitro"):
        send_kw["provider"] = {"sort": tune.provider_sort}
    if call.session_id:
        send_kw["session_id"] = call.session_id
    if call.trace_id:
        send_kw["trace"] = {"trace_id": call.trace_id, "trace_name": tune.title or "fish-voice"}
    if debug_enabled():
        _note_usage(send_kw)
        send_kw["x_open_router_metadata"] = "enabled"
    return _json_ready(send_kw)


async def _iter_openrouter_events(call: ChatCall) -> AsyncIterator[object]:
    from openrouter.errors import OpenRouterError

    try:
        async with _or_client_ctx(call.tune, call.client) as or_client:
            res = await or_client.chat.send_async(**_send_kwargs(call))
            async with res as event_stream:
                async for event in event_stream:
                    yield event
    except OpenRouterError as e:
        _abort_http(
            call.stats,
            e.status_code,
            call.route_model,
            _abort_text(e.body),
            e.headers,
        )


def _note_usage(payload: dict[str, Any]) -> None:
    payload["stream_options"] = {"include_usage": True}


def _abort_text(body: object) -> str:
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    return str(body)


def _abort_http(
    stats: _AbortStats,
    status: object,
    model: str,
    body: str,
    headers: httpx.Headers | None = None,
) -> None:
    warn(f"[llm] HTTP {status} model={model}: {body[:_ABORT_CHARS]}")
    stats.aborted = True
    stats.http_status = status if isinstance(status, int) else None
    if stats.http_status == 429:
        stats.retry_after_s = _retry_after_seconds(headers, body)
        # A spent quota (Experiential's insufficient_quota, a daily free limit) is
        # also a 429, but waiting a second does not refill it.
        stats.quota_exhausted = "quota" in body.lower()


def _retry_after_seconds(headers: httpx.Headers | None, body: str) -> float | None:
    found = retry_after_s(headers)
    if found is not None:
        return found
    return _seconds_in_json(body)


def _seconds_in_json(body: str) -> float | None:
    try:
        parsed: object = json.loads(body)
    except json.JSONDecodeError:
        return None
    return _find_retry_seconds(parsed)


def _json_seconds(raw: object) -> float | None:
    # The kit owns what counts as a usable wait. A JSON number is read the same
    # way a Retry-After header is, so a negative, nan or huge value is dropped.
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    return retry_after_s({"Retry-After": str(raw)})


def _find_retry_seconds(value: object) -> float | None:
    if isinstance(value, dict):
        for key in ("retry_after_seconds", "Retry-After", "retry-after"):
            if key in value:
                found = _json_seconds(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _find_retry_seconds(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_retry_seconds(item)
            if found is not None:
                return found
    return None


def _sse_object(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _data_line(line: str) -> str | None:
    # A leading BOM is not part of the field name. The first token was
    # ignored, so a one-event reply never started.
    if line.startswith("\ufeff"):
        line = line.removeprefix("\ufeff")
    if not line or line.startswith(":") or not line.startswith(_SSE_DATA):
        return None
    return line.removeprefix(_SSE_DATA).strip()


def _feed_sse(buf: str, line: str) -> tuple[str, dict[str, Any] | None, bool]:
    # One JSON object may be split across data lines. Parsing each line alone
    # drops the token. A line that is already JSON is returned at once.
    if line == "":
        parsed = _sse_object(buf) if buf else None
        return "", parsed, False
    data = _data_line(line)
    if data is None:
        return buf, None, False
    if data == "[DONE]":
        return "", None, True
    alone = _sse_object(data)
    if alone is not None:
        return "", alone, False
    piece = f"{buf}\n{data}" if buf else data
    parsed = _sse_object(piece)
    if parsed is not None:
        return "", parsed, False
    return piece, None, False


def _gateway_note(headers: httpx.Headers) -> str:
    """Describe the request id and route a gateway reports, for the debug log."""
    fields = (
        ("request", "x-request-id"),
        ("via", "x-gateway-provider"),
        ("zdr", "x-gateway-zdr"),
    )
    found = [f"{label}={headers[name]}" for label, name in fields if name in headers]
    return f" {' '.join(found)}" if found else ""


def chat_completions_url(base: str) -> str:
    """Return the OpenAI-compatible chat completions URL for ``base``."""
    return f"{strip_base(base)}/chat/completions"


@asynccontextmanager
async def _pooled(call: ChatCall) -> AsyncGenerator[httpx.AsyncClient, None]:
    if call.http is not None:
        yield call.http
        return
    async with http_client(call.tune) as owned:
        yield owned


async def _iter_httpx_sse_events(call: ChatCall) -> AsyncIterator[object]:
    url = chat_completions_url(call.tune.base)
    # No Referer or X-Title: those attribute an app to OpenRouter and should
    # not leak to an arbitrary chat-completions server.
    headers = {
        "Authorization": bearer(call.tune.api_key),
        "Content-Type": "application/json",
    }
    payload = _chat_body(call, "max_tokens")
    if call.tune.reasoning_effort:
        # Only the OpenAI-compatible path. The OpenRouter SDK has its own
        # reasoning object, and a provider rejects fields it does not know.
        payload["reasoning_effort"] = call.tune.reasoning_effort
    if debug_enabled():
        _note_usage(payload)
    async with (
        _pooled(call) as http,
        http.stream("POST", url, headers=headers, json=payload) as resp,
    ):
        if resp.status_code >= 400:
            body = _abort_text(await resp.aread())
            _abort_http(call.stats, resp.status_code, call.route_model, body, resp.headers)
            return
        debug("llm.response status={}{}", resp.status_code, _gateway_note(resp.headers))
        buf = ""
        async for line in resp.aiter_lines():
            buf, parsed, stop = _feed_sse(buf, line)
            if parsed is not None:
                yield parsed
            if stop:
                return


def chat_events(call: ChatCall) -> AsyncIterator[object]:
    """Pick the OpenRouter or httpx event stream for one chat call.

    Parameters
    ----------
    call : ChatCall
        Request. ``tune.uses_openrouter_sdk`` selects the SDK.

    Returns
    -------
    AsyncIterator
        Raw chat events. ``llm_token_stream`` is the only consumer and applies
        the same field reads to both transports.
    """
    if call.tune.uses_openrouter_sdk:
        return _iter_openrouter_events(call)
    return _iter_httpx_sse_events(call)
