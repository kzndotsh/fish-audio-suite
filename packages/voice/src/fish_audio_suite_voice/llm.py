"""Duplex LLM token stream. OpenRouter SDK, or httpx SSE for any other base."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from fish_audio_suite_kit import MS_PER_S, env_int, env_off, utf8_text
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.live import is_cancel_noise
from fish_audio_suite_voice.transports import ChatCall, chat_completions_url, chat_events

_DEFAULT_MAX_TOKENS = 600
_MODEL_LOOKUP_S = 15
_MODEL_LOOKUP_MS = _MODEL_LOOKUP_S * MS_PER_S
_LOOKUP_BODY_CHARS = 200


def openrouter_base(base: str) -> bool:
    return "openrouter.ai" in base


def _route_suffix(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1]


def _want_nitro(model: str, base: str) -> bool:
    if not openrouter_base(base) or env_off("FISH_LLM_NITRO"):
        return False
    return ":" not in _route_suffix(model)


def _nitro_route(model: str, *, nitro: bool) -> str:
    if nitro:
        return f"{model}:nitro"
    return model


def _model_author_slug(model: str) -> tuple[str, str] | None:
    if "/" not in model:
        return None
    author, slug = model.split("/", maxsplit=1)
    if not author or not slug:
        return None
    return author, slug


async def check_openrouter_model(client: Any, model: str, base: str) -> None:
    """Resolve FISH_LLM_MODEL via models.get. Warn on 404; never abort duplex."""
    route = _nitro_route(model, nitro=_want_nitro(model, base))
    parts = _model_author_slug(route)
    if parts is None:
        return
    from openrouter.errors import OpenRouterError

    author, slug = parts
    try:
        res = await client.models.get_async(author=author, slug=slug, timeout_ms=_MODEL_LOOKUP_MS)
    except OpenRouterError as e:
        if e.status_code == 404:
            print(f"[llm] unknown model {route}", file=sys.stderr)
        else:
            print(
                f"[llm] models.get HTTP {e.status_code}: {e.body[:_LOOKUP_BODY_CHARS]}",
                file=sys.stderr,
            )
        return
    except Exception as e:
        print(f"[llm] models.get {e}", file=sys.stderr)
        return
    data = _event_field(res, "data")
    if data is None:
        data = res
    mid = _event_field(data, "id")
    name = _event_field(data, "name")
    ctx = _event_field(data, "context_length")
    debug(
        "llm.model id={mid} display={display} context={ctx} requested={requested}",
        mid=mid,
        display=name,
        ctx=ctx,
        requested=route,
    )
    if isinstance(mid, str) and mid and mid != route:
        print(f"  [llm model {route} → {mid}]", flush=True)


def _event_field(event: object, name: str) -> Any:
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        return event.get(name)
    return value


def _message_text(choice: object) -> str:
    msg = _event_field(choice, "message")
    if msg is None:
        return ""
    piece = _event_field(msg, "content")
    return piece if isinstance(piece, str) else ""


def _first_choice(event: object) -> object | None:
    choices = _event_field(event, "choices")
    if isinstance(choices, list) and choices:
        return choices[0]
    return None


def _delta_content(chunk: object) -> str:
    ch0 = _first_choice(chunk)
    if ch0 is None:
        return ""
    delta = _event_field(ch0, "delta")
    content = _event_field(delta, "content") if delta is not None else None
    if isinstance(content, str) and content:
        return content
    return _message_text(ch0)


@dataclass
class _ChatStats:
    yielded: int = 0
    last_finish: str | None = None
    last_usage: dict[str, Any] | None = None
    last_id: str = ""
    last_model: str = ""
    last_provider: str = ""
    last_native: str = ""
    aborted: bool = False


def _usage_dict(usage: object) -> dict[str, Any] | None:
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        dumped = dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(usage, dict):
        return usage
    return None


def _remember_text(current: str, value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return current


def _finish_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if value is not None and not isinstance(value, str):
        return str(value)
    return None


def _note_identity(event: object, stats: _ChatStats) -> None:
    stats.last_id = _remember_text(stats.last_id, _event_field(event, "id"))
    stats.last_model = _remember_text(stats.last_model, _event_field(event, "model"))
    stats.last_provider = _remember_text(stats.last_provider, _event_field(event, "provider"))
    meta = _event_field(event, "openrouter_metadata")
    if stats.last_provider == "" and meta is not None:
        stats.last_provider = _remember_text(stats.last_provider, _event_field(meta, "summary"))


def _note_choice(choice: object, stats: _ChatStats) -> None:
    finish = _finish_text(_event_field(choice, "finish_reason"))
    if finish is not None:
        stats.last_finish = finish
    stats.last_native = _remember_text(
        stats.last_native, _event_field(choice, "native_finish_reason")
    )


def _note_chat_event(event: object, stats: _ChatStats) -> str:
    err = _event_field(event, "error")
    if err:
        print(f"[llm] stream error: {err}", file=sys.stderr)
        stats.aborted = True
        return ""
    usage = _event_field(event, "usage")
    if usage is not None:
        dumped = _usage_dict(usage)
        if dumped is not None:
            stats.last_usage = dumped
    _note_identity(event, stats)
    choice = _first_choice(event)
    if choice is not None:
        _note_choice(choice, stats)
    piece = _delta_content(event)
    if piece:
        stats.yielded += 1
    return utf8_text(piece)


async def _consume_chat_events(
    events: AsyncIterator[object],
    cancel: asyncio.Event | None,
    stats: _ChatStats,
) -> AsyncIterator[str]:
    async for event in events:
        if cancel is not None and cancel.is_set():
            return
        piece = _note_chat_event(event, stats)
        if stats.aborted:
            return
        if piece:
            yield piece


def _finish_llm(stats: _ChatStats, route_model: str) -> None:
    if stats.aborted:
        return
    debug(
        "llm.done id={} model={} provider={} finish={} native_finish={} usage={} deltas={}",
        stats.last_id,
        stats.last_model or route_model,
        stats.last_provider,
        stats.last_finish,
        stats.last_native,
        stats.last_usage,
        stats.yielded,
    )
    if stats.yielded == 0:
        print(
            f"[llm] empty reply (model={route_model} finish={stats.last_finish!r})",
            file=sys.stderr,
        )


async def llm_token_stream(
    messages: list[dict[str, str]],
    *,
    base: str,
    key: str,
    model: str,
    cancel: asyncio.Event | None = None,
    client: Any | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> AsyncIterator[str]:
    nitro = _want_nitro(model, base)
    route_model = _nitro_route(model, nitro=nitro)
    max_tokens = env_int("FISH_LLM_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
    if max_tokens <= 0:
        max_tokens = _DEFAULT_MAX_TOKENS
    debug(
        "llm.request url={} model={} nitro={} msgs={}",
        chat_completions_url(base),
        route_model,
        nitro,
        len(messages),
    )
    stats = _ChatStats()
    try:
        events = chat_events(
            ChatCall(
                messages=messages,
                base=base,
                key=key,
                route_model=route_model,
                max_tokens=max_tokens,
                nitro=nitro,
                client=client,
                session_id=session_id,
                trace_id=trace_id,
                stats=stats,
                use_openrouter=openrouter_base(base),
            )
        )
        async for piece in _consume_chat_events(events, cancel, stats):
            yield piece
    except (asyncio.CancelledError, GeneratorExit, BaseExceptionGroup):
        return
    except Exception as e:
        if is_cancel_noise(e):
            return
        print(f"[llm] {e}", file=sys.stderr)
    _finish_llm(stats, route_model)
