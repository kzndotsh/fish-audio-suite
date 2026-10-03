"""Duplex LLM token stream behind a swappable ``ChatBackend``.

Two backends ship: an OpenAI-compatible httpx client for any chat-completions
server, and the OpenRouter SDK as an opt-in adapter. ``LlmTune.backend`` picks
one. Both share the event consumer, the 429 retry, and the cut-off continuation.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx

from fish_audio_suite_kit import MS_PER_S, ends_sentence, utf8_text
from fish_audio_suite_voice.debug import console_print, debug, short_model, trace, warn
from fish_audio_suite_voice.live import is_cancel_noise
from fish_audio_suite_voice.pause import sleep_unless
from fish_audio_suite_voice.transports import (
    ChatCall,
    chat_events,
    http_client,
    openrouter_client,
)
from fish_audio_suite_voice.tune import LlmTune, openrouter_host
from fish_audio_suite_voice.wire import own_cancel, reap

# A longer Retry-After than this ends the reply instead of stalling the turn.
_LLM_429_CAP_S = 15.0
# A cue after a finished sentence is not a cut-off. "Hello. [break]" is done.
_TRAIL_CUE_RE = re.compile(r"(?:\s*\[[^\[\]]{0,80}\])+\s*$")
# A follow-up with no sentence end past this is a runaway, not the missing words.
_CONT_SCAN = 240
_CONT_BARE = 80
_MODEL_LOOKUP_S = 15
_MODEL_LOOKUP_MS = _MODEL_LOOKUP_S * MS_PER_S
_LOOKUP_BODY_CHARS = 200


def _route_suffix(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1]


def _want_nitro(model: str, tune: LlmTune) -> bool:
    if not tune.openrouter or not tune.nitro:
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


async def check_openrouter_model(client: Any, tune: LlmTune) -> None:
    """Resolve the configured model via ``models.get``. Warn on 404; never abort duplex.

    Parameters
    ----------
    client : Any
        An open OpenRouter SDK client.
    tune : LlmTune
        Supplies the model id and whether ``:nitro`` is added.
    """
    route = _nitro_route(tune.model, nitro=_want_nitro(tune.model, tune))
    parts = _model_author_slug(route)
    if parts is None:
        return
    from openrouter.errors import OpenRouterError

    author, slug = parts
    try:
        res = await client.models.get_async(author=author, slug=slug, timeout_ms=_MODEL_LOOKUP_MS)
    except OpenRouterError as e:
        if e.status_code == 404:
            warn(f"[llm] unknown model {route}")
        else:
            warn(f"[llm] models.get HTTP {e.status_code}: {e.body[:_LOOKUP_BODY_CHARS]}")
        return
    except Exception as e:  # noqa: BLE001 - best-effort lookup must never stop the session
        warn(f"[llm] models.get {e}")
        return
    data = _event_field(res, "data")
    if data is None:
        data = res
    mid = _event_field(data, "id")
    name = _event_field(data, "name")
    ctx = _event_field(data, "context_length")
    routed = f", routed from {route}" if isinstance(mid, str) and mid and mid != route else ""
    # The ready line already names the model. A remap is announced below.
    trace("llm.model {} ({} token context){}", name or short_model(mid or route), ctx, routed)
    if isinstance(mid, str) and mid and mid != route:
        console_print(f"  [llm model {route} → {mid}]", flush=True)


def _event_field(event: object, name: str) -> Any:
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        return event.get(name)
    return value


_TEXT_PARTS = frozenset({"text", "output_text"})


def _part_text(part: object) -> str:
    if isinstance(part, str):
        return part
    kind = _event_field(part, "type")
    # Reasoning and image parts are not speech.
    if isinstance(kind, str) and kind not in _TEXT_PARTS:
        return ""
    text = _event_field(part, "text")
    return text if isinstance(text, str) else ""


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


def _message_text(choice: object) -> str:
    msg = _event_field(choice, "message")
    if msg is None:
        return ""
    return _content_text(_event_field(msg, "content"))


def _refusal_text(obj: object | None) -> str:
    # A refusal is the reply. Content is null, so reading only content
    # speaks nothing and the next turn has no record of what was said.
    if obj is None:
        return ""
    refusal = _event_field(obj, "refusal")
    return refusal if isinstance(refusal, str) else ""


def _first_choice(event: object) -> object | None:
    choices = _event_field(event, "choices")
    if isinstance(choices, list) and choices:
        return choices[0]
    return None


def _delta_content(chunk: object, *, already: bool = False) -> str:
    ch0 = _first_choice(chunk)
    if ch0 is None:
        return ""
    delta = _event_field(ch0, "delta")
    content = _event_field(delta, "content") if delta is not None else None
    refused = _refusal_text(delta)
    # An empty delta is the end of a streamed reply. Falling through would
    # speak message.content again when the provider also sends the full text.
    if content is not None:
        text = _content_text(content)
        # A first chunk often sets content to "" with the role. That is not
        # the end of the reply. The words can still be on message or refusal.
        if text:
            return text
        if already:
            return refused
    elif already:
        return refused
    if refused:
        return refused
    # A null content on the first event is the whole reply. After tokens
    # have already been spoken, the same null is a second copy.
    text = _message_text(ch0)
    if text:
        return text
    return _refusal_text(_event_field(ch0, "message"))


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
    http_status: int | None = None
    retry_after_s: float | None = None


class _HeldStats:
    def __init__(self) -> None:
        self.stats = _ChatStats()
        self.stopped = ""


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
        warn(f"[llm] stream error: {err}")
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
    piece = _delta_content(event, already=stats.yielded > 0)
    if piece:
        stats.yielded += 1
    return utf8_text(piece)


async def _next_chat_event(events: AsyncIterator[object]) -> object:
    return await anext(events)


async def _consume_chat_events(
    events: AsyncIterator[object],
    cancel: asyncio.Event | None,
    stats: _ChatStats,
) -> AsyncIterator[str]:
    # async for only notices cancel after the next token. Ctrl+C during a
    # stalled reply would wait until the provider sends something.
    incoming = aiter(events)
    pending: asyncio.Task[object] = asyncio.create_task(_next_chat_event(incoming))
    try:
        while True:
            if not await _event_or_cancel(pending, cancel):
                return
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            pending = asyncio.create_task(_next_chat_event(incoming))
            piece = _note_chat_event(event, stats)
            if stats.aborted:
                return
            if piece:
                yield piece
    finally:
        if not pending.done():
            pending.cancel()
            await reap(pending)


async def _event_or_cancel(pending: asyncio.Task[object], cancel: asyncio.Event | None) -> bool:
    """Return whether ``pending`` finished. False means cancel won."""
    if cancel is None:
        await asyncio.wait({pending})
        return True
    if cancel.is_set():
        return False
    stopped = asyncio.create_task(cancel.wait())
    try:
        done, _pending = await asyncio.wait(
            {pending, stopped},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        if not stopped.done():
            stopped.cancel()
            await reap(stopped)
    return pending in done


_SELECTED_PROVIDER = re.compile(r"selected=([^,]+)")


def _provider_brief(provider: object) -> str:
    """Return the provider that served the reply from OpenRouter's routing text."""
    text = str(provider or "")
    found = _SELECTED_PROVIDER.search(text)
    return found.group(1).strip() if found else text


def _usage_brief(usage: object) -> str:
    """Render token counts and cost from an OpenRouter usage dict, or ``usage=?``."""
    if not isinstance(usage, dict):
        return "usage=?"
    fields = cast(dict[str, Any], usage)
    parts = [
        f"in={fields.get('prompt_tokens')}",
        f"out={fields.get('completion_tokens')}",
    ]
    cost = fields.get("cost")
    if isinstance(cost, (int, float)):
        parts.append(f"cost=${cost:.5f}")
    return " ".join(parts)


def _finish_llm(stats: _ChatStats, route_model: str) -> None:
    if stats.aborted:
        return
    debug(
        "llm.stream_end finish={} {} chunks={} provider={} model={}",
        stats.last_finish,
        _usage_brief(stats.last_usage),
        stats.yielded,
        _provider_brief(stats.last_provider),
        short_model(stats.last_model or route_model),
    )
    if stats.yielded == 0:
        warn(f"[llm] empty reply (model={route_model} finish={stats.last_finish!r})")


def _cancel_group(exc: BaseException, *, cancelled: bool = False) -> bool:
    # A task group on barge-in is only cancel noise. A group that holds a
    # provider error was swallowed, so the reply stopped and nothing was logged.
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            _cancel_group(item, cancelled=cancelled) for item in exc.exceptions
        )
    return is_cancel_noise(exc, cancelled=cancelled)


def _group_text(exc: BaseException) -> str:
    # str(ExceptionGroup) is "llm (1 sub-exception)". The provider message
    # is on the nested error, so the log never said why the reply stopped.
    if isinstance(exc, BaseExceptionGroup):
        inner = "; ".join(_group_text(item) for item in exc.exceptions)
        return inner or str(exc)
    return str(exc)


class ChatBackend(Protocol):
    """A chat-completions provider the duplex loop can stream from.

    Notes
    -----
    ``stream`` yields text deltas for one reply and ends on cancel, an error
    that was already logged, or the end of the reply. ``aclose`` releases the
    connection pool or SDK client.
    """

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        cancel: asyncio.Event | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant text for ``messages``.

        Parameters
        ----------
        messages : list of dict
            OpenAI-style chat messages, including the system prompt.
        cancel : asyncio.Event or None, optional
            Stops the stream. Cancel is not a fatal duplex error.
        trace_id : str or None, optional
            Logged with the request.

        Returns
        -------
        AsyncIterator
            Text deltas. Empty deltas are omitted.
        """
        ...

    async def aclose(self) -> None:
        """Release the client held for this session."""
        ...


class _Backend:
    """Shared plumbing: one tune, one session id, one pooled client."""

    def __init__(
        self,
        tune: LlmTune,
        *,
        session_id: str | None = None,
        client: Any | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.tune = tune
        self.session_id = session_id
        self._client = client
        self._http = http

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        cancel: asyncio.Event | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant text. See ``ChatBackend.stream``."""
        return llm_token_stream(
            messages,
            tune=self.tune,
            cancel=cancel,
            client=self._client,
            http=self._http,
            session_id=self.session_id,
            trace_id=trace_id,
        )

    async def aclose(self) -> None:
        """Close the pooled httpx client if this backend opened one."""
        http, self._http = self._http, None
        if http is not None:
            with contextlib.suppress(Exception):
                await http.aclose()


@asynccontextmanager
async def open_chat_backend(
    tune: LlmTune,
    *,
    session_id: str | None = None,
) -> AsyncGenerator[ChatBackend, None]:
    """Open the backend that ``tune.backend`` names for one duplex session.

    Parameters
    ----------
    tune : LlmTune
        Backend, base URL, key, model, and request settings.
    session_id : str or None, optional
        Sent with every OpenRouter request so its logs group by session.

    Yields
    ------
    ChatBackend
        Ready to stream. Closed when the context exits. The OpenRouter backend
        also resolves the model once and warns when it is unknown.
    """
    if tune.openrouter:
        async with openrouter_client(tune) as sdk:
            await check_openrouter_model(sdk, tune)
            yield _Backend(tune, session_id=session_id, client=sdk)
        return
    backend = _Backend(tune, session_id=session_id, http=http_client(tune))
    try:
        yield backend
    finally:
        await backend.aclose()


async def llm_token_stream(
    messages: list[dict[str, str]],
    *,
    tune: LlmTune | None = None,
    base: str = "",
    key: str = "",
    model: str = "",
    cancel: asyncio.Event | None = None,
    client: Any | None = None,
    http: httpx.AsyncClient | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> AsyncIterator[str]:
    """Stream assistant text from OpenRouter or a generic chat-completions URL.

    Parameters
    ----------
    messages : list of dict
        OpenAI-style chat messages, including the system prompt.
    tune : LlmTune or None, optional
        Backend and request settings. When omitted one is built from ``base``,
        ``key`` and ``model``, choosing OpenRouter only for an ``openrouter.ai``
        host.
    base, key, model : str, optional
        Used only to build ``tune`` when it is omitted.
    cancel : asyncio.Event or None, optional
        Stops the stream. Cancel is not a fatal duplex error.
    client : Any or None, optional
        An already-open OpenRouter client. Created when omitted.
    http : httpx.AsyncClient or None, optional
        A pooled client for the OpenAI-compatible backend. A one-shot client
        is used when omitted.
    session_id : str or None, optional
        One duplex session id sent on every OpenRouter call.
    trace_id : str or None, optional
        Logged with the request. Not an OpenTelemetry span.

    Yields
    ------
    str
        Text deltas. Empty deltas are omitted.

    Notes
    -----
    A 429 waits for ``Retry-After`` and tries once more when that wait is at
    most 15 seconds. A longer wait, a missing wait, or cancel ends the
    iterator. Any other stream error is logged and ends the iterator. With
    ``tune.continuation`` on, a ``stop`` before a sentence end sends one more
    request with that text as the assistant line, and only a suffix that
    continues the sentence is yielded. A ``reasoning`` field on a stream event
    is not spoken.
    """
    tune = tune or LlmTune(
        backend="openrouter" if openrouter_host(base) else "openai",
        base=base,
        key=key,
        model=model,
    )
    route_model = _nitro_route(tune.model, nitro=_want_nitro(tune.model, tune))
    batch = messages
    spoken: list[str] = []
    generations = (1, 2) if tune.continuation else (1,)
    for generation in generations:
        if generation == 2:
            console_print("  [llm cut off, continuing]", flush=True)
        held = _HeldStats()
        extra: list[str] = []
        async for piece in _stream_generation(
            batch,
            tune=tune,
            route_model=route_model,
            cancel=cancel,
            client=client,
            http=http,
            session_id=session_id,
            trace_id=trace_id,
            held=held,
        ):
            if generation == 1:
                spoken.append(piece)
                yield piece
            else:
                extra.append(piece)
        if held.stopped != "done" or generation == 2:
            suffix = _continuation_suffix("".join(spoken), "".join(extra))
            if suffix:
                yield suffix
            return
        text = "".join(spoken)
        if not _cut_off(text, held.stats, cancel):
            return
        batch = [*messages, {"role": "assistant", "content": text}]


def _cut_off(text: str, stats: _ChatStats, cancel: asyncio.Event | None) -> bool:
    # finish=length used the cap. Another call would spend the cap again.
    if stats.aborted or stats.last_finish != "stop":
        return False
    if cancel is not None and cancel.is_set():
        return False
    body = _TRAIL_CUE_RE.sub("", text.strip()).strip()
    if not body:
        return False
    return not ends_sentence(body)


def _continuation_suffix(partial: str, more: str) -> str:
    # A prefill reply sometimes repeats the line it was given. A capital after
    # "You can" is a new document, not the missing words.
    more = more.removeprefix(partial)
    lead = more.lstrip()
    if not lead or lead[0].isupper():
        return ""
    if partial and not partial[-1].isspace() and not more[0].isspace() and more[0].isalnum():
        more = " " + more
    return _first_sentence(more)


def _first_sentence(text: str) -> str:
    window = text[:_CONT_SCAN]
    for index in range(1, len(window) + 1):
        if ends_sentence(window[:index]):
            return window[:index]
    if len(text) <= _CONT_BARE:
        return text
    return ""


async def _stream_generation(
    messages: list[dict[str, str]],
    *,
    tune: LlmTune,
    route_model: str,
    cancel: asyncio.Event | None,
    client: Any | None,
    http: httpx.AsyncClient | None,
    session_id: str | None,
    trace_id: str | None,
    held: _HeldStats,
) -> AsyncIterator[str]:
    debug(
        "llm.request model={} msgs={} nitro={}",
        short_model(route_model),
        len(messages),
        tune.nitro,
    )
    for attempt in (1, 2):
        stats = _ChatStats()
        held.stats = stats
        try:
            events = chat_events(
                ChatCall(
                    messages=messages,
                    tune=tune,
                    route_model=route_model,
                    client=client,
                    http=http,
                    session_id=session_id,
                    trace_id=trace_id,
                    stats=stats,
                )
            )
            async for piece in _consume_chat_events(events, cancel, stats):
                yield piece
        except asyncio.CancelledError:
            held.stopped = "cancel"
            # Only our own cancel flag may end the stream quietly. An outer
            # cancellation (asyncio.timeout, a task group) has to propagate.
            if not own_cancel(cancel):
                raise
            return
        except GeneratorExit:
            held.stopped = "cancel"
            return
        except BaseExceptionGroup as exc:
            if _cancel_group(exc, cancelled=cancel is not None and cancel.is_set()):
                held.stopped = "cancel"
                return
            warn(f"[llm] {_group_text(exc)}")
        except Exception as e:  # noqa: BLE001 - stream boundary: reported by warn, then retry or finish
            if is_cancel_noise(e, cancelled=cancel is not None and cancel.is_set()):
                held.stopped = "cancel"
                return
            warn(f"[llm] {e}")
        if attempt == 1 and await _maybe_retry_429(stats, cancel):
            continue
        _finish_llm(stats, route_model)
        held.stopped = "done"
        return


def _retry_wait(stats: _ChatStats, cancel: asyncio.Event | None) -> float | None:
    wait = stats.retry_after_s
    if stats.http_status != 429 or stats.yielded or wait is None:
        return None
    if not math.isfinite(wait) or wait < 0 or wait > _LLM_429_CAP_S:
        return None
    if cancel is not None and cancel.is_set():
        return None
    return wait


def _wait_label(seconds: float) -> str:
    if seconds == int(seconds):
        return str(int(seconds))
    return f"{seconds:.1f}"


async def _pause_for_retry(cancel: asyncio.Event | None, seconds: float) -> bool:
    """Wait out a 429. True means cancel landed first."""
    stopped = cancel.is_set if cancel is not None else (lambda: False)
    return await sleep_unless(seconds, stopped)


async def _maybe_retry_429(stats: _ChatStats, cancel: asyncio.Event | None) -> bool:
    wait = _retry_wait(stats, cancel)
    if wait is None:
        return False
    console_print(f"  [llm 429, retrying in {_wait_label(wait)}s]", flush=True)
    return not await _pause_for_retry(cancel, wait)
