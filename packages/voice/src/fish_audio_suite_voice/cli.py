"""Duplex CLI recipe: mic → Fish ASR → LLM → IsolatedFishTts → sink.

Not the only way to use the voice library. Apps should import IsolatedFishTts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
    LatencySnapshot,
    SuiteDefaults,
    is_asr_hallucination,
    is_backchannel,
    is_quit_utterance,
    is_tts_junk,
    normalize_cues,
    scrub_asr,
    scrub_tts,
)
from fish_audio_suite_voice.barge import (
    POST_SPEAK_COOLDOWN_S,
    BargeGate,
    record_utterance,
)
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.playback import FileSink, make_sink

FISH_BASE = "https://api.fish.audio"
HISTORY_TURNS = 20
STOP_RECORD = threading.Event()
SECRETS_PATH = Path.home() / ".secrets" / "ai.env"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _parse_device(raw: str | None) -> str | int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def cfg() -> dict[str, Any]:
    _load_dotenv(SECRETS_PATH)
    d = SuiteDefaults()
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    return {
        "fish_api_key": os.environ.get("FISH_API_KEY", "").strip(),
        "fish_voice_id": voice_id,
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


async def fish_asr(audio_wav: bytes, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "model": "transcribe-1",
    }
    files = {"audio": ("utterance.wav", audio_wav, "audio/wav")}
    data = {"language": "en"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{FISH_BASE}/v1/asr",
            headers=headers,
            files=files,
            data=data,
        )
        r.raise_for_status()
        body = r.json()
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
    yielded = 0
    last_finish: str | None = None
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
                choices = obj.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue
                ch0 = choices[0]
                if not isinstance(ch0, dict):
                    continue
                fr = ch0.get("finish_reason")
                if isinstance(fr, str):
                    last_finish = fr
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
    )
    sink = FileSink(out, sample_rate=c["fish_sample_rate"], wav=True)
    result = tts.speak_isolated("[clear] Hello there.", sink)
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
        f"fish-voice ready | tts={c['fish_tts_model']} latency={c['fish_latency']} "
        f"playback={playback} | llm={c['llm_backend']}:{c['llm_model']} | Ctrl+C quit",
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
    )

    while True:
        print("listening…")
        STOP_RECORD.clear()
        wav = await asyncio.to_thread(record_utterance, device, STOP_RECORD)
        if STOP_RECORD.is_set():
            print("\nbye")
            return 0
        if not wav:
            continue
        t0 = time.perf_counter()
        try:
            text = await fish_asr(wav, c["fish_api_key"])
        except Exception as e:
            print(f"[asr] {e}", file=sys.stderr)
            continue
        asr_ms = (time.perf_counter() - t0) * 1000
        if is_asr_hallucination(text):
            continue
        if is_quit_utterance(text):
            print("\nbye")
            return 0
        if tts_playing:
            continue
        if is_backchannel(text):
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
        )
        if reply:
            scrubbed = normalize_cues(scrub_tts(reply))
            if is_tts_junk(scrubbed):
                print("  (skip junk TTS)", flush=True)
            else:
                cancel.clear()
                barge = BargeGate(device=device)
                barge.start_after_bleed(cancel)
                tts_playing = True
                sink = make_sink(
                    playback,
                    path=None,
                    sample_rate=c["fish_sample_rate"],
                    device=device,
                )
                result = await asyncio.to_thread(tts.speak_isolated, scrubbed, sink, cancel)
                tts_playing = False
                snapshot = LatencySnapshot(
                    srt=asr_ms,
                    llm_ttft=first_tok_ms[0] if first_tok_ms else None,
                    llm_ttfs=result.llm_ttfs_ms,
                    ttfa=result.ttfa_ms,
                    voice_to_voice=(time.perf_counter() - t0) * 1000,
                )
                if result.cancelled and not result.got_audio:
                    pass
                elif result.spoken_so_far:
                    history.append({"role": "assistant", "content": result.spoken_so_far})
        print(f"  {snapshot.log_line()}", flush=True)
        cancel.set()
        llm_cancel.set()
        await asyncio.sleep(0)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(POST_SPEAK_COOLDOWN_S)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fish Audio duplex CLI recipe (kit + live WS + sinks)")
    p.add_argument("--smoke", action="store_true", help="TTS smoke test to a wav file")
    p.add_argument(
        "--playback",
        default=None,
        help="sounddevice | file | stdout | mpv (overrides FISH_PLAYBACK)",
    )
    args = p.parse_args(argv)
    c = cfg()
    if args.playback:
        c["playback"] = args.playback

    def _sigint(*_a: Any) -> None:
        STOP_RECORD.set()

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
