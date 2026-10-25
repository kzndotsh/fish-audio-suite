"""Duplex chat transports: OpenRouter SDK, or httpx SSE for any OpenAI-compatible server."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final, cast

import httpx

from fish_audio_suite_kit import (
    MS_PER_S,
    ChatMessage,
    bearer,
    retry_after_s,
    strip_base,
    utf8_text,
)
from fish_audio_suite_voice.debug import debug, debug_enabled, trace, warn
from fish_audio_suite_voice.llm_tune import LlmTune
from fish_audio_suite_voice.tune import HTTP_KEEPALIVE_S

__all__ = [
    "ChatCall",
    "ChatChunk",
    "ChatHttpError",
    "chat_completions_url",
    "chat_events",
    "describe_http_error",
    "http_client",
    "openrouter_client",
]

_ABORT_CHARS: Final = 800
_EVENT_TRACE_CHARS: Final = 600
# One warning line stays short enough to read. The whole body is in the debug log.
_ERROR_LINE_CHARS: Final = 220
_SSE_DATA: Final = "data:"


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One streamed chat event, read the same way from the SDK and from SSE.

    Notes
    -----
    Only the first choice is read. ``text`` is None when that choice's delta has
    no content, and empty when the content holds nothing speakable, such as
    only reasoning parts. ``error`` is None unless the event reports a stream
    error. A field the event does not carry is empty.
    """

    text: str | None = None
    refusal: str = ""
    message: str = ""
    message_refusal: str = ""
    finish: str | None = None
    native_finish: str = ""
    usage: dict[str, Any] | None = None
    id: str = ""
    model: str = ""
    provider: str = ""
    provider_summary: str = ""
    service_tier: str = ""
    error: str | None = None


class ChatHttpError(Exception):
    """The chat server refused the request. ``warn`` has already reported it.

    Parameters
    ----------
    status : int or None
        HTTP status, or None when the SDK gave something else.
    retry_after_s : float or None
        The wait a 429 asked for, from its header or its body.
    quota_exhausted : bool
        True for a 429 that says the quota is spent, where waiting does not help.
    """

    def __init__(
        self, status: int | None, retry_after_s: float | None, quota_exhausted: bool
    ) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after_s = retry_after_s
        self.quota_exhausted = quota_exhausted


@dataclass(slots=True)
class ChatCall:
    """One duplex chat request, shared by the OpenRouter SDK and httpx SSE paths.

    Notes
    -----
    ``route_model`` may already include ``:nitro``. ``tune.uses_openrouter_sdk`` picks the
    SDK path. ``http`` is the session's pooled client for the OpenAI-compatible
    path.
    """

    messages: list[ChatMessage]
    tune: LlmTune
    route_model: str
    client: Any | None
    http: httpx.AsyncClient | None
    session_id: str | None
    trace_id: str | None


def http_client(tune: LlmTune) -> httpx.AsyncClient:
    """Build the pooled client for the OpenAI-compatible backend.

    Parameters
    ----------
    tune : LlmTune
        Supplies the request timeout.

    Returns
    -------
    httpx.AsyncClient
        Reused for every reply in a session. The caller closes it.
    """
    return httpx.AsyncClient(
        timeout=tune.timeout_s, limits=httpx.Limits(keepalive_expiry=HTTP_KEEPALIVE_S)
    )


@asynccontextmanager
async def openrouter_client(tune: LlmTune) -> AsyncGenerator[Any, None]:
    """Open an OpenRouter client for the duplex loop.

    Parameters
    ----------
    tune : LlmTune
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

    # The SDK's own client keeps idle connections for 5 s, so pass one that
    # lives across turns. The SDK does not close a client it was given. It
    # writes the key into Authorization and adds Bearer itself, and a newline
    # raises before the request is sent, so the reply is empty.
    async with (
        httpx.AsyncClient(
            follow_redirects=True, limits=httpx.Limits(keepalive_expiry=HTTP_KEEPALIVE_S)
        ) as http,
        OpenRouter(
            api_key=bearer(tune.api_key).removeprefix("Bearer "),
            http_referer=tune.referer or None,
            x_open_router_title=tune.title or None,
            x_open_router_categories=tune.categories or None,
            server_url=strip_base(tune.base),
            async_client=http,
        ) as owned,
    ):
        yield owned


@asynccontextmanager
async def _or_client_ctx(tune: LlmTune, client: Any | None) -> AsyncGenerator[Any, None]:
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
    if tune.provider_sort:
        send_kw["provider"] = {"sort": tune.provider_sort}
    if tune.service_tier:
        send_kw["service_tier"] = tune.service_tier
    if tune.reasoning_effort:
        # OpenRouter's unified reasoning control; the openai path sends the
        # flat reasoning_effort field instead.
        send_kw["reasoning"] = {"effort": tune.reasoning_effort}
    if call.session_id:
        send_kw["session_id"] = call.session_id
    if call.trace_id:
        send_kw["trace"] = {"trace_id": call.trace_id, "trace_name": tune.title or "fish-voice"}
    if debug_enabled():
        _note_usage(send_kw)
        send_kw["x_open_router_metadata"] = "enabled"
    return _json_ready(send_kw)


async def _iter_openrouter_events(call: ChatCall) -> AsyncIterator[ChatChunk]:
    from openrouter.errors import OpenRouterError

    try:
        async with _or_client_ctx(call.tune, call.client) as or_client:
            res = await or_client.chat.send_async(**_send_kwargs(call))
            async with res as event_stream:
                async for event in event_stream:
                    yield _chunk_of(event)
    except OpenRouterError as e:
        raise _abort_http(e.status_code, call.route_model, _abort_text(e.body), e.headers) from e


def _note_usage(payload: dict[str, Any]) -> None:
    payload["stream_options"] = {"include_usage": True}


def _abort_text(body: object) -> str:
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    return str(body)


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed: object = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


def _error_detail(body: str) -> tuple[str, str]:
    """Pull the provider name and the most specific message out of an error body."""
    outer = _json_object(body)
    error = outer.get("error") if outer else None
    if isinstance(error, str):
        return "", error
    if not isinstance(error, dict):
        return "", body
    info = cast(dict[str, Any], error)
    meta = info.get("metadata")
    meta_dict = cast(dict[str, Any], meta) if isinstance(meta, dict) else {}
    provider = meta_dict.get("provider_name")
    message = info.get("message")
    detail = message if isinstance(message, str) else ""
    # OpenRouter wraps the real answer from the provider as a JSON string in
    # metadata.raw, and says only "Provider returned error" itself.
    raw = meta_dict.get("raw")
    if isinstance(raw, str):
        inner = _json_object(raw)
        specific = None
        if inner:
            specific = inner.get("error") or inner.get("message")
            if isinstance(specific, dict):
                specific = cast(dict[str, Any], specific).get("message")
        detail = specific if isinstance(specific, str) and specific else (raw or detail)
    return (provider if isinstance(provider, str) else ""), detail


def describe_http_error(status: object, model: str, body: str) -> str:
    """Say what a failed chat request returned in one readable line.

    Parameters
    ----------
    status : object
        HTTP status of the reply.
    model : str
        The model that was asked.
    body : str
        Raw response body. OpenRouter puts the provider's own message in a JSON
        string inside it, which is unwrapped.

    Returns
    -------
    str
        ``HTTP 429 from Parasail, model=org/m: temporarily rate-limited upstream``.
        Falls back to the start of the body when it is not a known error shape.
        The full body goes to the debug log, not here.
    """
    provider, detail = _error_detail(body)
    detail = " ".join(detail.split())[:_ERROR_LINE_CHARS] or "no detail in the reply"
    who = f" from {provider}" if provider else ""
    return f"HTTP {status}{who}, model={model}: {detail}"


def _abort_http(
    status: object,
    model: str,
    body: str,
    headers: httpx.Headers | None = None,
) -> ChatHttpError:
    warn(f"[llm] {describe_http_error(status, model, body)}")
    debug("llm.error body={}", body[:_ABORT_CHARS])
    code = status if isinstance(status, int) else None
    if code != 429:
        return ChatHttpError(code, None, quota_exhausted=False)
    # A spent quota (Experiential's insufficient_quota, a daily free limit) is
    # also a 429, but waiting a second does not refill it.
    return ChatHttpError(
        code, _retry_after_seconds(headers, body), quota_exhausted="quota" in body.lower()
    )


def _retry_after_seconds(headers: httpx.Headers | None, body: str) -> float | None:
    found = retry_after_s(headers)
    if found is not None:
        return found
    return _seconds_in_json(body)


def _seconds_in_json(body: str) -> float | None:
    try:
        parsed: object = json.loads(body)
    except (json.JSONDecodeError, RecursionError):
        return None
    return _find_retry_seconds(parsed)


def _json_seconds(raw: object) -> float | None:
    # The kit owns what counts as a usable wait. A JSON number is read the same
    # way a Retry-After header is, so a negative, nan or huge value is dropped.
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    return retry_after_s({"Retry-After": str(raw)})


# Retry hints sit near the top of an error body. A cap keeps a deeply nested
# body from recursing to the interpreter limit.
_RETRY_HINT_DEPTH: Final = 8


def _find_retry_seconds(value: object, depth: int = 0) -> float | None:
    if depth > _RETRY_HINT_DEPTH:
        return None
    if isinstance(value, dict):
        for key in ("retry_after_seconds", "Retry-After", "retry-after"):
            if key in value:
                found = _json_seconds(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _find_retry_seconds(item, depth + 1)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_retry_seconds(item, depth + 1)
            if found is not None:
                return found
    return None


def _sse_object(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, RecursionError):
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


async def _iter_httpx_sse_events(call: ChatCall) -> AsyncIterator[ChatChunk]:
    url = chat_completions_url(call.tune.base)
    # No Referer or X-Title: those attribute an app to OpenRouter and should
    # not leak to an arbitrary chat-completions server.
    headers = {
        "Authorization": bearer(call.tune.api_key),
        "Content-Type": "application/json",
    }
    payload = _chat_body(call, "max_tokens")
    if call.tune.reasoning_effort:
        # The flat OpenAI field. The OpenRouter SDK path sends its reasoning
        # object instead, and a provider rejects fields it does not know.
        payload["reasoning_effort"] = call.tune.reasoning_effort
    if debug_enabled():
        _note_usage(payload)
    async with (
        _pooled(call) as http,
        http.stream("POST", url, headers=headers, json=payload) as resp,
    ):
        if resp.status_code >= 400:
            body = _abort_text(await resp.aread())
            raise _abort_http(resp.status_code, call.route_model, body, resp.headers)
        debug("llm.response status={}{}", resp.status_code, _gateway_note(resp.headers))
        buf = ""
        async for line in resp.aiter_lines():
            buf, parsed, stop = _feed_sse(buf, line)
            if parsed is not None:
                trace("llm.event {}", json.dumps(parsed, ensure_ascii=False)[:_EVENT_TRACE_CHARS])
                yield _chunk_of(parsed)
            if stop:
                trace("llm.event [DONE]")
                return
        trace("llm.event stream closed without [DONE]")


def _field(event: object, name: str) -> Any:
    # The SDK gives objects and SSE gives dicts, with the same field names.
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        return event.get(name)
    return value


def _text_field(event: object, name: str) -> str:
    value = _field(event, name)
    return value if isinstance(value, str) else ""


_TEXT_PARTS: Final = frozenset({"text", "output_text"})


def _part_text(part: object) -> str:
    if isinstance(part, str):
        return part
    kind = _field(part, "type")
    # Reasoning and image parts are not speech.
    if isinstance(kind, str) and kind not in _TEXT_PARTS:
        return ""
    return _text_field(part, "text")


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_part_text(part) for part in content)
    # One part object is the same payload as a one-item list. Dropping it
    # speaks nothing for that token.
    if content is None:
        return ""
    return _part_text(content)


def _finish_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if value is not None and not isinstance(value, str):
        return str(value)
    return None


def _usage_dict(usage: object) -> dict[str, Any] | None:
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, dict):
            return cast(dict[str, Any], dumped)
    if isinstance(usage, dict):
        return cast(dict[str, Any], usage)
    return None


def _chunk_of(event: object) -> ChatChunk:
    error = _field(event, "error")
    choices = _field(event, "choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    delta = _field(choice, "delta")
    content = _field(delta, "content")
    message = _field(choice, "message")
    return ChatChunk(
        text=None if content is None else _content_text(content),
        # A refusal is the reply. Content is null, so reading only content
        # speaks nothing and the next turn has no record of what was said.
        refusal=_text_field(delta, "refusal"),
        message=_content_text(_field(message, "content")),
        message_refusal=_text_field(message, "refusal"),
        finish=_finish_text(_field(choice, "finish_reason")),
        native_finish=_text_field(choice, "native_finish_reason"),
        usage=_usage_dict(_field(event, "usage")),
        id=_text_field(event, "id"),
        model=_text_field(event, "model"),
        provider=_text_field(event, "provider"),
        provider_summary=_text_field(_field(event, "openrouter_metadata"), "summary"),
        service_tier=_text_field(event, "service_tier"),
        error=str(error) if error else None,
    )


def chat_events(call: ChatCall) -> AsyncIterator[ChatChunk]:
    """Pick the OpenRouter or httpx event stream for one chat call.

    Parameters
    ----------
    call : ChatCall
        Request. ``tune.uses_openrouter_sdk`` selects the SDK.

    Returns
    -------
    AsyncIterator
        One ``ChatChunk`` per event. A refused request raises ``ChatHttpError``.
    """
    if call.tune.uses_openrouter_sdk:
        return _iter_openrouter_events(call)
    return _iter_httpx_sse_events(call)
