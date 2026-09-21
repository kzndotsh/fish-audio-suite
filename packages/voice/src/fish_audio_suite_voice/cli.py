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
from collections.abc import AsyncIterator
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
    spread_cues,
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


def _want_nitro(model: str, base: str) -> bool:
    if "openrouter.ai" not in base:
        return False
    raw = os.environ.get("FISH_LLM_NITRO", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return ":" not in model.rsplit("/", maxsplit=1)[-1]


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
) -> AsyncIterator[str]:
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/kzndotsh/fish-audio-suite",
        "X-Title": "fish-audio-suite-voice",
    }
    route_model = f"{model}:nitro" if _want_nitro(model, base) else model
    payload: dict[str, Any] = {
        "model": route_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.8,
        "max_tokens": int(os.environ.get("FISH_LLM_MAX_TOKENS", "600")),
    }
    if "openrouter.ai" in base and _want_nitro(model, base):
        payload["provider"] = {"sort": "throughput"}
    if env_debug():
        payload["stream_options"] = {"include_usage": True}
        logger.debug(
            "llm.request url={} model={} nitro={} msgs={}",
            url,
            route_model,
            _want_nitro(model, base),
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
        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream("POST", url, headers=headers, json=payload) as resp,
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
                delta = ch0.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}
                piece = delta.get("content") or ""
                if not isinstance(piece, str):
                    piece = ""
                if not piece:
                    msg = ch0.get("message") or {}
                    if not isinstance(msg, dict):
                        msg = {}
                    piece = msg.get("content") or ""
                    if not isinstance(piece, str):
                        piece = ""
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

    history: list[dict[str, str]] = [
        {"role": "system", "content": c["system_prompt"]},
    ]
    device = _parse_device(c["device"])
    last_user = ""
    pending: list[str] = []
    tts_playing = False
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
            scrubbed = spread_cues(ensure_lead_cue(normalize_cues(scrub_tts(reply))))
            if is_tts_junk(scrubbed):
                print("  (skip junk TTS)", flush=True)
            else:
                barge = BargeGate(device=device)
                barge.start_after_bleed(cancel)
                tts_playing = True
                sink = make_sink(
                    playback,
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
