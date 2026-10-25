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
from typing import Any, Final, Protocol, cast

import httpx

from fish_audio_suite_kit import MS_PER_S, ChatMessage, ends_sentence, utf8_text
from fish_audio_suite_voice.cancel import is_cancel_noise, is_own_cancel, reap
from fish_audio_suite_voice.debug import debug, short_model, trace, warn
from fish_audio_suite_voice.events import notice
from fish_audio_suite_voice.llm_tune import LlmTune, is_openrouter_host
from fish_audio_suite_voice.pause import sleep_unless
from fish_audio_suite_voice.transports import (
    ChatCall,
    ChatChunk,
    ChatHttpError,
    chat_events,
    http_client,
    openrouter_client,
)

__all__ = [
    "ChatBackend",
    "check_openrouter_model",
    "llm_token_stream",
    "open_chat_backend",
]

# A longer Retry-After than this ends the reply instead of stalling the turn.
_LLM_429_CAP_S: Final = 15.0
# A rate-limit 429 with no Retry-After is retried once after this pause. A shared
# provider pool (one host behind an OpenRouter model) often clears in a second.
_LLM_429_DEFAULT_S: Final = 1.0
# A cue after a finished sentence is not a cut-off. "Hello. [break]" is done.
_TRAIL_CUE_RE: Final = re.compile(r"(?:\s*\[[^\[\]]{0,80}\])+\s*$")
# A follow-up with no sentence end past this is a runaway, not the missing words.
_CONT_SCAN: Final = 240
_CONT_BARE: Final = 80
_MODEL_LOOKUP_S: Final = 15
_MODEL_LOOKUP_MS: Final = _MODEL_LOOKUP_S * MS_PER_S
_LOOKUP_BODY_CHARS: Final = 200


def _route_suffix(model: str) -> str:
    return model.rsplit("/", maxsplit=1)[-1]


def _want_nitro(model: str, tune: LlmTune) -> bool:
    if not tune.uses_openrouter_sdk or not tune.nitro:
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


def _lookup_field(found: object, name: str) -> Any:
    value = getattr(found, name, None)
    if value is None and isinstance(found, dict):
        return cast(dict[str, Any], found).get(name)
    return value


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
    data = _lookup_field(res, "data")
    if data is None:
        data = res
    mid = _lookup_field(data, "id")
    name = _lookup_field(data, "name")
    ctx = _lookup_field(data, "context_length")
    routed = f", routed from {route}" if isinstance(mid, str) and mid and mid != route else ""
    # The ready line already names the model. A remap is announced below.
    trace("llm.model {} ({} token context){}", name or short_model(mid or route), ctx, routed)
    if isinstance(mid, str) and mid and mid != route:
        notice(f"  [llm model {route} → {mid}]")


def _delta_content(chunk: ChatChunk, *, already: bool) -> str:
    # An empty delta is the end of a streamed reply. After tokens have been
    # spoken, the full message on it is a second copy, so only a refusal counts.
    # Before any token, a null or blank delta leaves the reply on the message.
    if chunk.text:
        return chunk.text
    if already:
        return chunk.refusal
    return chunk.refusal or chunk.message or chunk.message_refusal


@dataclass(slots=True)
class _ChatStats:
    yielded: int = 0
    last_finish: str | None = None
    last_usage: dict[str, Any] | None = None
    last_id: str = ""
    last_model: str = ""
    last_provider: str = ""
    last_native: str = ""
    aborted: bool = False
    refused: ChatHttpError | None = None


class _HeldStats:
    def __init__(self) -> None:
        self.stats = _ChatStats()
        self.stopped = ""


def _note_chat_event(chunk: ChatChunk, stats: _ChatStats) -> str:
    if chunk.error is not None:
        warn(f"[llm] stream error: {chunk.error}")
        stats.aborted = True
        return ""
    if chunk.usage is not None:
        stats.last_usage = chunk.usage
    stats.last_id = chunk.id or stats.last_id
    stats.last_model = chunk.model or stats.last_model
    stats.last_provider = chunk.provider or stats.last_provider or chunk.provider_summary
    if chunk.finish is not None:
        stats.last_finish = chunk.finish
    stats.last_native = chunk.native_finish or stats.last_native
    piece = _delta_content(chunk, already=stats.yielded > 0)
    if piece:
        stats.yielded += 1
    return utf8_text(piece)


async def _next_chat_event(events: AsyncIterator[ChatChunk]) -> ChatChunk:
    return await anext(events)


async def _consume_chat_events(
    events: AsyncIterator[ChatChunk],
    cancel: asyncio.Event | None,
    stats: _ChatStats,
) -> AsyncGenerator[str, None]:
    # async for only notices cancel after the next token. Ctrl+C during a
    # stalled reply would wait until the provider sends something.
    incoming = aiter(events)
    pending: asyncio.Task[ChatChunk] = asyncio.create_task(_next_chat_event(incoming))
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
        elif not pending.cancelled():
            # A read that failed after the consumer stopped is not news. Taking
            # its exception keeps asyncio from logging it as never retrieved.
            pending.exception()


async def _event_or_cancel(pending: asyncio.Task[ChatChunk], cancel: asyncio.Event | None) -> bool:
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


_SELECTED_PROVIDER: Final = re.compile(r"selected=([^,]+)")


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
        messages: list[ChatMessage],
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
        messages: list[ChatMessage],
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
    if tune.uses_openrouter_sdk:
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
    messages: list[ChatMessage],
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
    A 429 that arrives before the first text delta waits for ``Retry-After`` and
    tries once more when that wait is at most 15 seconds. With no
    ``Retry-After`` it waits about one second and tries once more, unless the
    quota is exhausted. Once any text has been yielded there is no retry. A longer wait, a spent
    quota, or cancel ends the iterator. Any other stream error is logged and ends the iterator. With
    ``tune.continuation`` on, a ``stop`` before a sentence end sends one more
    request with that text as the assistant line, and only a suffix that
    continues the sentence is yielded. A ``reasoning`` field on a stream event
    is not spoken.
    """
    tune = tune or LlmTune(
        backend="openrouter" if is_openrouter_host(base) else "openai",
        base=base,
        api_key=key,
        model=model,
    )
    route_model = _nitro_route(tune.model, nitro=_want_nitro(tune.model, tune))
    batch: list[ChatMessage] = messages
    spoken: list[str] = []
    generations = (1, 2) if tune.continuation else (1,)
    for generation in generations:
        if generation == 2:
            notice("  [llm cut off, continuing]")
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
    messages: list[ChatMessage],
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
                )
            )
            async for piece in _consume_chat_events(events, cancel, stats):
                yield piece
        except ChatHttpError as refused:
            stats.aborted = True
            stats.refused = refused
        except asyncio.CancelledError:
            held.stopped = "cancel"
            # Only our own cancel flag may end the stream quietly. An outer
            # cancellation (asyncio.timeout, a task group) has to propagate.
            if not is_own_cancel(cancel):
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
    refused = stats.refused
    if refused is None or refused.status != 429 or stats.yielded or refused.quota_exhausted:
        return None
    wait = refused.retry_after_s
    if wait is None:
        wait = _LLM_429_DEFAULT_S
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
    notice(f"  [llm 429, retrying in {_wait_label(wait)}s]")
    return not await _pause_for_retry(cancel, wait)
