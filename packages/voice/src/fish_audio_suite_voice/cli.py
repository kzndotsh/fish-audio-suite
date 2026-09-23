"""Duplex CLI recipe: mic → Fish ASR → LLM → IsolatedFishTts → sink.

Not the only way to use the voice library. Apps should import IsolatedFishTts.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import tempfile
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from fish_audio_suite_voice.config import VoiceCliConfig, cfg
from fish_audio_suite_voice.debug import (
    configure_voice_logging,
    end_reply_line,
    env_debug,
)
from fish_audio_suite_voice.duplex import EXIT_FATAL, EXIT_OK, bye, duplex_turns
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.llm import check_openrouter_model, openrouter_base
from fish_audio_suite_voice.playback import (
    FileSink,
    audio_format_for,
    duplex_playback_problem,
    playback_key,
)
from fish_audio_suite_voice.signals import STOP_RECORD, request_quit
from fish_audio_suite_voice.transports import openrouter_client

DEFAULT_ENV_FILE = Path(".env")
_EXPORT_PREFIX = "export "
_SMOKE_MIN_BYTES = 1000
_SMOKE_FAIL = 1


def _load_dotenv(path: Path) -> bool:
    """Fill os.environ from KEY=VAL lines. Existing keys win. Returns whether the file was read."""
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"fish-voice: env file is not utf-8: {path}", file=sys.stderr)
        return False
    except OSError as exc:
        print(f"fish-voice: could not read env file {path}: {exc.strerror}", file=sys.stderr)
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith(_EXPORT_PREFIX):
            line = line.removeprefix(_EXPORT_PREFIX).strip()
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
        if not expanded.is_file():
            if required:
                print(f"fish-voice: --env-file not found: {path}", file=sys.stderr)
            continue
        if _load_dotenv(expanded):
            seen.add(resolved)
            loaded.append(path)
    return loaded


def _parse_device(raw: str | None) -> str | int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _blocker(label: str) -> int:
    print(f"BLOCKER: {label} missing", file=sys.stderr)
    return EXIT_FATAL


def _require_fish(c: VoiceCliConfig) -> int | None:
    if not c.fish_api_key:
        return _blocker("FISH_API_KEY")
    if not c.fish_voice_id:
        return _blocker("FISH_VOICE_ID")
    return None


def _fish_tts(c: VoiceCliConfig, audio_format: str) -> IsolatedFishTts:
    return IsolatedFishTts(
        api_key=c.fish_api_key,
        voice_id=c.fish_voice_id,
        model=c.tts_model,
        latency=c.latency,
        speed=c.speed,
        audio_format=audio_format,
        sample_rate=c.sample_rate,
        temperature=c.temperature,
        top_p=c.top_p,
        repetition_penalty=c.repetition_penalty,
        chunk_length=c.chunk_length,
        min_chunk_length=c.min_chunk_length,
        volume=c.volume,
        base_url=c.fish_base,
    )


async def smoke_test(c: VoiceCliConfig) -> int:
    missing = _require_fish(c)
    if missing is not None:
        return missing
    out = Path(tempfile.gettempdir()) / "fish-audio-suite-smoke.wav"
    tts = _fish_tts(c, "pcm")
    sink = FileSink(out, sample_rate=c.sample_rate, wav=True)
    result = tts.speak_isolated("[clear] Hello there.", sink)
    if result.error_status is not None:
        print(
            f"smoke: FAIL {result.error_status} {result.error_message}",
            file=sys.stderr,
        )
        return _SMOKE_FAIL
    ok = result.bytes_played > _SMOKE_MIN_BYTES
    print(f"smoke: wrote {result.bytes_played} bytes → {out} ({'OK' if ok else 'FAIL <1k'})")
    print("sdk: fishaudio; playback=file format=pcm")
    return EXIT_OK if ok else _SMOKE_FAIL


async def run_loop(c: VoiceCliConfig) -> int:
    missing = _require_fish(c)
    if missing is not None:
        return missing
    if not c.llm_key:
        return _blocker("FISH_LLM_KEY / OPENROUTER_API_KEY")
    if not c.llm_model:
        return _blocker("FISH_LLM_MODEL / OPENROUTER_MODEL")
    playback_problem = duplex_playback_problem(c.playback)
    if playback_problem is not None:
        print(f"BLOCKER: {playback_problem}", file=sys.stderr)
        return EXIT_FATAL

    device = _parse_device(c.device)
    playback = c.playback
    print(
        f"fish-voice ready | tts={c.tts_model} voice={c.fish_voice_id} "
        f"asr_lang={c.fish_asr_language or 'auto'} latency={c.latency} "
        f"playback={playback} | "
        f"llm={c.llm_backend}:{c.llm_model} | Ctrl+C quit",
        flush=True,
    )

    tts = _fish_tts(c, audio_format_for(playback))

    async with AsyncExitStack() as stack:
        or_client: Any | None = None
        if openrouter_base(c.llm_base):
            or_client = await stack.enter_async_context(openrouter_client(c.llm_key, c.llm_base))
        if or_client is not None:
            await check_openrouter_model(or_client, c.llm_model, c.llm_base)
        return await duplex_turns(c, tts, device, or_client)


def _parser() -> argparse.ArgumentParser:
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
        c = replace(c, playback=playback_key(args.playback))

    def _sigint(*_a: Any) -> None:
        end_reply_line()
        sys.stderr.write("\n")
        sys.stderr.flush()
        request_quit()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _sigint)

    if args.smoke:
        try:
            return asyncio.run(smoke_test(c))
        except KeyboardInterrupt:
            return bye()
    while True:
        try:
            return asyncio.run(run_loop(c))
        except asyncio.CancelledError:
            if STOP_RECORD.is_set():
                return bye()
            print("  (loop cancelled — restarting)", flush=True)
            continue
        except KeyboardInterrupt:
            return bye()


if __name__ == "__main__":
    raise SystemExit(main())
