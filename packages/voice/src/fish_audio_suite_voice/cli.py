"""Duplex CLI recipe: mic → Fish ASR → LLM → IsolatedFishTts → sink.

Not the only way to use the voice library. Apps should import IsolatedFishTts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from fish_audio_suite_kit import (
    FISH_RETRY_ATTEMPTS,
    FishHttpError,
    LatencySnapshot,
    SuiteDefaults,
    ensure_lead_cue,
    fish_backoff_seconds,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_tts_junk,
    make_traceparent,
    normalize_cues,
    parse_fish_error,
    scrub_asr,
    scrub_tts,
    should_retry_fish_status,
    trace_id_of,
)
from fish_audio_suite_voice.barge import (
    BargeGate,
    post_speak_cooldown_s,
    record_utterance,
)
from fish_audio_suite_voice.debug import (
    configure_voice_logging,
    env_debug,
    header_meta,
    logger,
    public_meta,
)
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.playback import FileSink, PortAudioMissingError, make_sink

HISTORY_TURNS = 20
STOP_RECORD = threading.Event()
DEFAULT_ENV_FILE = Path(".env")


class _TurnSignals:
    cancel: threading.Event | None = None
    llm_cancel: asyncio.Event | None = None


TURN = _TurnSignals()


def request_quit() -> None:
    """SIGINT: stop mic listen and cancel in-flight LLM/TTS. Sticky until process exit."""
    STOP_RECORD.set()
    if TURN.cancel is not None:
        TURN.cancel.set()
    if TURN.llm_cancel is not None:
        TURN.llm_cancel.set()


def _load_dotenv(path: Path) -> bool:
    """Fill os.environ from KEY=VAL lines. Existing keys win. Returns whether the file existed."""
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val
    return True


def apply_cli_env_files(paths: list[Path], *, required: bool) -> list[Path]:
    """Load dotenv files in order. First file wins per key. Process env already wins."""
    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            resolved = expanded.resolve()
        except OSError:
            resolved = expanded
        if resolved in seen:
            continue
        if _load_dotenv(expanded):
            seen.add(resolved)
            loaded.append(path)
        elif required:
            print(f"fish-voice: --env-file not found: {path}", file=sys.stderr)
    return loaded


def _parse_device(raw: str | None) -> str | int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def cfg() -> dict[str, Any]:
    d = SuiteDefaults()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    return {
        "fish_api_key": os.environ.get("FISH_API_KEY", "").strip(),
        "fish_base": os.environ.get("FISH_BASE", d.fish_base).rstrip("/"),
        "fish_voice_id": voice_id,
        "fish_asr_language": os.environ.get("FISH_ASR_LANGUAGE", d.asr_language).strip(),
        "fish_tts_model": os.environ.get("FISH_TTS_MODEL", d.tts_model),
        "fish_latency": os.environ.get("FISH_LATENCY", d.latency),
        "fish_speed": float(os.environ.get("FISH_SPEED", str(d.speed))),
        "fish_temperature": float(os.environ.get("FISH_TEMPERATURE", str(d.temperature))),
        "fish_top_p": float(os.environ.get("FISH_TOP_P", str(d.top_p))),
        "fish_rep_penalty": float(
            os.environ.get("FISH_REPETITION_PENALTY", str(d.repetition_penalty))
        ),
        "fish_chunk": int(os.environ.get("FISH_CHUNK_LENGTH", str(d.chunk_length))),
        "fish_min_chunk": int(os.environ.get("FISH_MIN_CHUNK_LENGTH", str(d.min_chunk_length))),
        "fish_volume": float(os.environ.get("FISH_VOLUME", "0")),
        "fish_sample_rate": int(os.environ.get("FISH_SAMPLE_RATE", str(d.sample_rate))),
        "playback": os.environ.get("FISH_PLAYBACK", "sounddevice"),
        "system_prompt": os.environ.get("FISH_SYSTEM_PROMPT", d.system_prompt),
        "device": os.environ.get("FISH_VOICE_DEVICE"),
        "llm_backend": os.environ.get("FISH_LLM_BACKEND", "openrouter"),
        "llm_base": os.environ.get(
            "FISH_LLM_BASE",
            os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
        "llm_key": os.environ.get(
            "FISH_LLM_KEY",
            os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        ),
        "llm_model": (
            os.environ.get("FISH_LLM_MODEL") or os.environ.get("OPENROUTER_MODEL") or ""
        ).strip(),
    }


LLM_REFERER = "https://github.com/kzndotsh/fish-audio-suite"
LLM_TITLE = "fish-audio-suite-voice"


def _openrouter_base(base: str) -> bool:
    return "openrouter.ai" in base


def _want_nitro(model: str, base: str) -> bool:
    if not _openrouter_base(base):
        return False
    raw = os.environ.get("FISH_LLM_NITRO", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return ":" not in model.rsplit("/", maxsplit=1)[-1]


def _model_author_slug(model: str) -> tuple[str, str] | None:
    if "/" not in model:
        return None
    author, slug = model.split("/", maxsplit=1)
    if not author or not slug:
        return None
    return author, slug


async def check_openrouter_model(client: Any, model: str, base: str) -> None:
    """Resolve FISH_LLM_MODEL via models.get. Warn on 404; never abort duplex."""
    route = f"{model}:nitro" if _want_nitro(model, base) else model
    parts = _model_author_slug(route)
    if parts is None:
        return
    from openrouter.errors import OpenRouterError

    author, slug = parts
    try:
        res = await client.models.get_async(author=author, slug=slug, timeout_ms=15_000)
    except OpenRouterError as e:
        if e.status_code == 404:
            print(f"[llm] unknown model {route}", file=sys.stderr)
        else:
            print(f"[llm] models.get HTTP {e.status_code}: {e.body[:200]}", file=sys.stderr)
        return
    except Exception as e:
        print(f"[llm] models.get {e}", file=sys.stderr)
        return
    data = getattr(res, "data", res)
    mid = getattr(data, "id", None)
    if mid is None and isinstance(data, dict):
        mid = data.get("id")
    name = getattr(data, "name", None)
    if name is None and isinstance(data, dict):
        name = data.get("name")
    ctx = getattr(data, "context_length", None)
    if ctx is None and isinstance(data, dict):
        ctx = data.get("context_length")
    if env_debug():
        logger.debug(
            "llm.model id={mid} display={display} context={ctx} requested={requested}",
            mid=mid,
            display=name,
            ctx=ctx,
            requested=route,
        )
    if isinstance(mid, str) and mid and mid != route:
        print(f"  [llm model {route} → {mid}]", flush=True)


def _delta_content(chunk: object) -> str:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    ch0 = choices[0]
    delta = getattr(ch0, "delta", None)
    if delta is None and isinstance(ch0, dict):
        delta = ch0.get("delta")
    content = getattr(delta, "content", None) if delta is not None else None
    if content is None and isinstance(delta, dict):
        content = delta.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(ch0, dict):
        msg = ch0.get("message") or {}
        if isinstance(msg, dict):
            piece = msg.get("content")
            if isinstance(piece, str):
                return piece
    return ""


@asynccontextmanager
async def _or_client_ctx(key: str, base: str, client: Any | None) -> AsyncGenerator[Any, None]:
    if client is not None:
        yield client
        return
    from openrouter import OpenRouter

    async with OpenRouter(
        api_key=key,
        http_referer=LLM_REFERER,
        x_open_router_title=LLM_TITLE,
        x_open_router_categories="cli-agent",
        server_url=base.rstrip("/"),
    ) as owned:
        yield owned


@asynccontextmanager
async def _chat_event_stream(res: Any) -> AsyncGenerator[Any, None]:
    enter = getattr(res, "__aenter__", None)
    if callable(enter):
        async with res as event_stream:
            yield event_stream
        return
    yield res


async def fish_asr(
    audio_wav: bytes,
    api_key: str,
    *,
    base: str,
    language: str = "",
    extra_headers: dict[str, str] | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "model": "transcribe-1",
        **(extra_headers or {}),
    }
    files = {"audio": ("utterance.wav", audio_wav, "audio/wav")}
    data: dict[str, str] = {}
    if language:
        data["language"] = language
    last_error: FishHttpError | None = None
    body: Any = None
    asr_status = 0
    asr_headers: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(FISH_RETRY_ATTEMPTS):
            try:
                r = await client.post(
                    f"{base}/v1/asr",
                    headers=headers,
                    files=files,
                    data=data or None,
                )
            except httpx.TimeoutException as exc:
                last_error = FishHttpError(504, "Fish request timed out")
                if attempt + 1 >= FISH_RETRY_ATTEMPTS:
                    raise last_error from exc
                await asyncio.sleep(fish_backoff_seconds(attempt))
                continue
            except httpx.RequestError as exc:
                last_error = FishHttpError(502, str(exc) or "Fish upstream unreachable")
                if attempt + 1 >= FISH_RETRY_ATTEMPTS:
                    raise last_error from exc
                await asyncio.sleep(fish_backoff_seconds(attempt))
                continue
            if r.status_code >= 400:
                detail = parse_fish_error(r.status_code, r.text)
                last_error = FishHttpError(int(detail["status"]), str(detail["message"]))
                if (
                    should_retry_fish_status(last_error.status)
                    and attempt + 1 < FISH_RETRY_ATTEMPTS
                ):
                    await asyncio.sleep(fish_backoff_seconds(attempt))
                    continue
                raise last_error
            body = r.json()
            asr_status = r.status_code
            raw_headers = getattr(r, "headers", None)
            asr_headers = dict(raw_headers) if raw_headers is not None else {}
            break
        else:
            raise last_error or FishHttpError(502, "Fish upstream unreachable")
    if not isinstance(body, dict):
        return ""
    if env_debug():
        logger.debug(
            "fish.asr status={} language_sent={!r} headers={} meta={}",
            asr_status,
            language or "auto",
            header_meta(asr_headers),
            public_meta(body),
        )
    return scrub_asr((body.get("text") or "").strip())


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
    route_model = f"{model}:nitro" if _want_nitro(model, base) else model
    max_tokens = int(os.environ.get("FISH_LLM_MAX_TOKENS", "600"))
    nitro = _want_nitro(model, base)
    if env_debug():
        logger.debug(
            "llm.request url={} model={} nitro={} msgs={}",
            f"{base.rstrip('/')}/chat/completions",
            route_model,
            nitro,
            len(messages),
        )
    yielded = 0
    last_finish: str | None = None
    last_usage: dict[str, Any] | None = None
    last_id = ""
    last_model = ""
    last_provider = ""
    last_native = ""
    try:
        if _openrouter_base(base):
            from openrouter.errors import OpenRouterError

            send_kw: dict[str, Any] = {
                "messages": messages,
                "model": route_model,
                "stream": True,
                "temperature": 0.8,
                "max_completion_tokens": max_tokens,
                "timeout_ms": 120_000,
            }
            if nitro:
                send_kw["provider"] = {"sort": "throughput"}
            if session_id:
                send_kw["session_id"] = session_id
            if trace_id:
                send_kw["trace"] = {
                    "trace_id": trace_id,
                    "trace_name": LLM_TITLE,
                }
            if env_debug():
                send_kw["stream_options"] = {"include_usage": True}
                send_kw["x_open_router_metadata"] = "enabled"
            try:
                async with _or_client_ctx(key, base, client) as or_client:
                    res = await or_client.chat.send_async(**send_kw)
                    async with _chat_event_stream(res) as event_stream:
                        async for event in event_stream:
                            if cancel is not None and cancel.is_set():
                                break
                            err = getattr(event, "error", None)
                            if err is None and isinstance(event, dict):
                                err = event.get("error")
                            if err:
                                print(f"[llm] stream error: {err}", file=sys.stderr)
                                break
                            usage = getattr(event, "usage", None)
                            if usage is None and isinstance(event, dict):
                                usage = event.get("usage")
                            if usage is not None:
                                dump = getattr(usage, "model_dump", None)
                                if callable(dump):
                                    dumped = dump(mode="json")
                                    if isinstance(dumped, dict):
                                        last_usage = dumped
                                elif isinstance(usage, dict):
                                    last_usage = usage
                            oid = getattr(event, "id", None)
                            if isinstance(oid, str) and oid:
                                last_id = oid
                            omodel = getattr(event, "model", None)
                            if isinstance(omodel, str) and omodel:
                                last_model = omodel
                            oprov = getattr(event, "provider", None)
                            if isinstance(oprov, str) and oprov:
                                last_provider = oprov
                            meta = getattr(event, "openrouter_metadata", None)
                            if meta is None and isinstance(event, dict):
                                meta = event.get("openrouter_metadata")
                            if last_provider == "" and meta is not None:
                                summary = getattr(meta, "summary", None)
                                if summary is None and isinstance(meta, dict):
                                    summary = meta.get("summary")
                                if isinstance(summary, str) and summary:
                                    last_provider = summary
                            choices = getattr(event, "choices", None)
                            if isinstance(choices, list) and choices:
                                ch0 = choices[0]
                                fr = getattr(ch0, "finish_reason", None)
                                if fr is None and isinstance(ch0, dict):
                                    fr = ch0.get("finish_reason")
                                if isinstance(fr, str) and fr:
                                    last_finish = fr
                                elif fr is not None and not isinstance(fr, str):
                                    last_finish = str(fr)
                            piece = _delta_content(event)
                            if piece:
                                yielded += 1
                                yield piece
            except OpenRouterError as e:
                print(
                    f"[llm] HTTP {e.status_code} model={route_model}: {e.body[:800]}",
                    file=sys.stderr,
                )
                return
        else:
            url = f"{base.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": LLM_REFERER,
                "X-Title": LLM_TITLE,
            }
            payload: dict[str, Any] = {
                "model": route_model,
                "messages": messages,
                "stream": True,
                "temperature": 0.8,
                "max_tokens": max_tokens,
            }
            if env_debug():
                payload["stream_options"] = {"include_usage": True}
            async with (
                httpx.AsyncClient(timeout=120.0) as http,
                http.stream("POST", url, headers=headers, json=payload) as resp,
            ):
                if resp.status_code >= 400:
                    body = (await resp.aread())[:800].decode("utf-8", "replace")
                    print(
                        f"[llm] HTTP {resp.status_code} model={route_model}: {body}",
                        file=sys.stderr,
                    )
                    return
                if env_debug():
                    logger.debug(
                        "llm.response status={} headers={}",
                        resp.status_code,
                        header_meta(resp.headers),
                    )
                async for line in resp.aiter_lines():
                    if cancel is not None and cancel.is_set():
                        break
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    obj: dict[str, Any] = parsed
                    err = obj.get("error")
                    if err:
                        print(f"[llm] stream error: {err}", file=sys.stderr)
                        break
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        last_usage = usage
                    oid = obj.get("id")
                    if isinstance(oid, str) and oid:
                        last_id = oid
                    omodel = obj.get("model")
                    if isinstance(omodel, str) and omodel:
                        last_model = omodel
                    oprov = obj.get("provider")
                    if isinstance(oprov, str) and oprov:
                        last_provider = oprov
                    choices = obj.get("choices") or []
                    if not isinstance(choices, list) or not choices:
                        continue
                    ch0 = choices[0]
                    if not isinstance(ch0, dict):
                        continue
                    fr = ch0.get("finish_reason")
                    if isinstance(fr, str):
                        last_finish = fr
                    nfr = ch0.get("native_finish_reason")
                    if isinstance(nfr, str) and nfr:
                        last_native = nfr
                    piece = _delta_content(obj)
                    if piece:
                        yielded += 1
                        yield piece
    except (asyncio.CancelledError, GeneratorExit):
        return
    except BaseExceptionGroup:
        return
    except Exception as e:
        msg = str(e).lower()
        if "athrow" in msg or "generator didn't stop" in msg or "cancel scope" in msg:
            return
        print(f"[llm] {e}", file=sys.stderr)
    if env_debug():
        logger.debug(
            "llm.done id={} model={} provider={} finish={} native_finish={} usage={} deltas={}",
            last_id,
            last_model or route_model,
            last_provider,
            last_finish,
            last_native,
            last_usage,
            yielded,
        )
    if yielded == 0:
        print(
            f"[llm] empty reply (model={route_model} finish={last_finish!r})",
            file=sys.stderr,
        )


def _require_fish(c: dict[str, Any]) -> int | None:
    if not c["fish_api_key"]:
        print("BLOCKER: FISH_API_KEY missing", file=sys.stderr)
        return 2
    if not c["fish_voice_id"]:
        print("BLOCKER: FISH_VOICE_ID missing", file=sys.stderr)
        return 2
    return None


async def smoke_test(c: dict[str, Any]) -> int:
    missing = _require_fish(c)
    if missing is not None:
        return missing
    out = Path(tempfile.gettempdir()) / "fish-audio-suite-smoke.wav"
    tts = IsolatedFishTts(
        api_key=c["fish_api_key"],
        voice_id=c["fish_voice_id"],
        model=c["fish_tts_model"],
        latency=c["fish_latency"],
        speed=c["fish_speed"],
        audio_format="pcm",
        sample_rate=c["fish_sample_rate"],
        temperature=c["fish_temperature"],
        top_p=c["fish_top_p"],
        repetition_penalty=c["fish_rep_penalty"],
        chunk_length=c["fish_chunk"],
        min_chunk_length=c["fish_min_chunk"],
        volume=c["fish_volume"],
        base_url=c["fish_base"],
    )
    sink = FileSink(out, sample_rate=c["fish_sample_rate"], wav=True)
    result = tts.speak_isolated("[clear] Hello there.", sink)
    if result.error_status is not None:
        print(
            f"smoke: FAIL {result.error_status} {result.error_message}",
            file=sys.stderr,
        )
        return 1
    ok = result.bytes_played > 1000
    print(f"smoke: wrote {result.bytes_played} bytes → {out} ({'OK' if ok else 'FAIL <1k'})")
    print("sdk: fishaudio; playback=file format=pcm")
    return 0 if ok else 1


async def run_loop(c: dict[str, Any]) -> int:
    missing = _require_fish(c)
    if missing is not None:
        return missing
    if not c["llm_key"]:
        print("BLOCKER: FISH_LLM_KEY / OPENROUTER_API_KEY missing", file=sys.stderr)
        return 2
    if not c["llm_model"]:
        print("BLOCKER: FISH_LLM_MODEL / OPENROUTER_MODEL missing", file=sys.stderr)
        return 2

    device = _parse_device(c["device"])
    playback = c["playback"]
    print(
        f"fish-voice ready | tts={c['fish_tts_model']} voice={c['fish_voice_id']} "
        f"asr_lang={c['fish_asr_language'] or 'auto'} latency={c['fish_latency']} "
        f"playback={playback} | "
        f"llm={c['llm_backend']}:{c['llm_model']} | Ctrl+C quit",
        flush=True,
    )

    tts = IsolatedFishTts(
        api_key=c["fish_api_key"],
        voice_id=c["fish_voice_id"],
        model=c["fish_tts_model"],
        latency=c["fish_latency"],
        speed=c["fish_speed"],
        audio_format="mp3" if playback == "mpv" else "pcm",
        sample_rate=c["fish_sample_rate"],
        temperature=c["fish_temperature"],
        top_p=c["fish_top_p"],
        repetition_penalty=c["fish_rep_penalty"],
        chunk_length=c["fish_chunk"],
        min_chunk_length=c["fish_min_chunk"],
        volume=c["fish_volume"],
        base_url=c["fish_base"],
    )

    async with AsyncExitStack() as stack:
        or_client: Any | None = None
        if _openrouter_base(c["llm_base"]):
            from openrouter import OpenRouter

            or_client = await stack.enter_async_context(
                OpenRouter(
                    api_key=c["llm_key"],
                    http_referer=LLM_REFERER,
                    x_open_router_title=LLM_TITLE,
                    x_open_router_categories="cli-agent",
                    server_url=c["llm_base"].rstrip("/"),
                )
            )
        if or_client is not None:
            await check_openrouter_model(or_client, c["llm_model"], c["llm_base"])
        return await _duplex_turns(c, tts, device, or_client)


async def _duplex_turns(
    c: dict[str, Any],
    tts: IsolatedFishTts,
    device: str | int | None,
    or_client: Any | None,
) -> int:
    history: list[dict[str, str]] = [
        {"role": "system", "content": c["system_prompt"]},
    ]
    last_user = ""
    pending: list[str] = []
    tts_playing = False
    llm_session = uuid.uuid4().hex
    while True:
        print("listening…")
        if env_debug():
            logger.debug("listen.waiting device={}", device)
        try:
            wav = await asyncio.to_thread(record_utterance, device, STOP_RECORD)
        except PortAudioMissingError as e:
            print(e, file=sys.stderr)
            return 2
        if STOP_RECORD.is_set():
            print("\nbye")
            return 0
        if not wav:
            if env_debug():
                logger.debug("listen.dropped (too short or none)")
            continue
        if env_debug():
            logger.debug("listen.wav bytes={}", len(wav))
        t0 = time.perf_counter()
        asr_parent = make_traceparent()
        turn_trace = trace_id_of(asr_parent)
        try:
            text = await fish_asr(
                wav,
                c["fish_api_key"],
                base=c["fish_base"],
                language=c["fish_asr_language"],
                extra_headers={"traceparent": asr_parent},
            )
        except FishHttpError as e:
            print(f"[asr] {e.status} {e.message}", file=sys.stderr)
            if e.status in {401, 402, 403}:
                return 2
            continue
        except Exception as e:
            print(f"[asr] {e}", file=sys.stderr)
            continue
        asr_ms = (time.perf_counter() - t0) * 1000
        if is_backchannel(text):
            if env_debug():
                logger.debug("asr skip backchannel chars={}", len(text.strip()))
            continue
        if is_asr_hallucination(text):
            print("[asr skip hallucination]", file=sys.stderr)
            if env_debug():
                logger.debug("asr skip hallucination chars={}", len(text.strip()))
            continue
        if is_quit_utterance(text):
            print("\nbye")
            return 0
        if tts_playing:
            continue
        if text.strip() == last_user.strip():
            continue
        pending.append(text)
        text = " ".join(pending).strip()
        pending.clear()
        last_user = text
        print(f"you: {text}  [asr {asr_ms:.0f}ms]")
        history.append({"role": "user", "content": text})
        while len(history) > 1 + HISTORY_TURNS * 2:
            history.pop(1)

        cancel = threading.Event()
        llm_cancel = asyncio.Event()
        TURN.cancel = cancel
        TURN.llm_cancel = llm_cancel
        reply_parts: list[str] = []
        t_llm = time.perf_counter()
        first_tok_ms: list[float] = []
        try:
            async for tok in llm_token_stream(
                history,
                base=c["llm_base"],
                key=c["llm_key"],
                model=c["llm_model"],
                cancel=llm_cancel,
                client=or_client,
                session_id=llm_session,
                trace_id=turn_trace,
            ):
                if not first_tok_ms:
                    first_tok_ms.append((time.perf_counter() - t_llm) * 1000)
                    print(f"  [llm ttft {first_tok_ms[0]:.0f}ms]", flush=True)
                reply_parts.append(tok)
                print(tok, end="", flush=True)
            print()
        except (asyncio.CancelledError, BaseExceptionGroup, RuntimeError) as e:
            msg = str(e).lower()
            if not any(x in msg for x in ("cancel scope", "athrow", "generator didn't stop")):
                print(f"[llm] {e}", file=sys.stderr)

        reply = "".join(reply_parts).strip()
        print(f"  [got {len(reply)} chars]", flush=True)
        snapshot = LatencySnapshot(
            srt=asr_ms,
            llm_ttft=first_tok_ms[0] if first_tok_ms else None,
            trace_id=turn_trace,
        )
        if STOP_RECORD.is_set():
            print("\nbye")
            return 0
        if reply:
            scrubbed = ensure_lead_cue(normalize_cues(scrub_tts(reply)))
            if is_tts_junk(scrubbed):
                print("  (skip junk TTS)", flush=True)
            else:
                barge = BargeGate(device=device)
                barge.start_after_bleed(cancel)
                tts_playing = True
                sink = make_sink(
                    c["playback"],
                    path=None,
                    sample_rate=c["fish_sample_rate"],
                    device=device,
                    cancel=cancel,
                )
                tts.trace_headers = {
                    "traceparent": make_traceparent(trace_id=turn_trace),
                }
                result = await asyncio.to_thread(tts.speak_isolated, scrubbed, sink, cancel)
                tts_playing = False
                if env_debug():
                    logger.debug(
                        "tts.done cancelled={} audio={} bytes={} spoken_chars={} err={} {}",
                        result.cancelled,
                        result.got_audio,
                        result.bytes_played,
                        len(result.spoken_so_far),
                        result.error_status,
                        result.error_message or "",
                    )
                if result.error_status in {401, 402, 403}:
                    return 2
                if not result.got_audio:
                    if result.cancelled:
                        print("  [tts cancelled before audio]", flush=True)
                    elif result.error_status is None:
                        print(
                            f"  [tts silent] voice={c['fish_voice_id']} model={c['fish_tts_model']}",
                            flush=True,
                        )
                snapshot = LatencySnapshot(
                    srt=asr_ms,
                    llm_ttft=first_tok_ms[0] if first_tok_ms else None,
                    llm_ttfs=result.llm_ttfs_ms,
                    ttfa=result.ttfa_ms,
                    voice_to_voice=(time.perf_counter() - t0) * 1000,
                    trace_id=turn_trace,
                )
                if result.cancelled and not result.got_audio:
                    pass
                elif result.spoken_so_far:
                    history.append({"role": "assistant", "content": result.spoken_so_far})
        print(f"  {snapshot.log_line()}", flush=True)
        cancel.set()
        llm_cancel.set()
        TURN.cancel = None
        TURN.llm_cancel = None
        if STOP_RECORD.is_set() or await asyncio.to_thread(
            STOP_RECORD.wait, post_speak_cooldown_s()
        ):
            print("\nbye")
            return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fish Audio duplex CLI recipe (kit + live WS + sinks)")
    p.add_argument("--smoke", action="store_true", help="TTS smoke test to a wav file")
    p.add_argument(
        "--playback",
        default=None,
        help="sounddevice | file | stdout | mpv (overrides FISH_PLAYBACK)",
    )
    p.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="dotenv to load (repeatable). Default: ./.env if it exists. Process env wins",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="verbose stderr logs: VAD, barge, Fish WS events, ASR/LLM meta (or FISH_VOICE_DEBUG=1)",
    )
    args = p.parse_args(argv)
    if args.env_file:
        loaded = apply_cli_env_files(args.env_file, required=True)
    else:
        loaded = apply_cli_env_files([DEFAULT_ENV_FILE], required=False)
    if loaded:
        print("env: " + " ".join(str(p) for p in loaded), flush=True)
    debug = bool(args.debug or env_debug())
    configure_voice_logging(debug=debug)
    c = cfg()
    if args.playback:
        c["playback"] = args.playback

    def _sigint(*_a: Any) -> None:
        request_quit()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _sigint)

    if args.smoke:
        try:
            return asyncio.run(smoke_test(c))
        except KeyboardInterrupt:
            print("\nbye")
            return 0
    while True:
        try:
            return asyncio.run(run_loop(c))
        except asyncio.CancelledError:
            if STOP_RECORD.is_set():
                print("\nbye")
                return 0
            print("  (loop cancelled — restarting)", flush=True)
            continue
        except KeyboardInterrupt:
            print("\nbye")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
