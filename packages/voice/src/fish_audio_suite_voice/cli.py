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
    warn,
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
_EXPORT_WORD = "export"
_SMOKE_MIN_BYTES = 1000
_SMOKE_FAIL = 1


def _quoted_span(val: str) -> tuple[str, str] | None:
    # The closer is the next unescaped quote, not the last character.
    # "sk" # "old" used to keep the comment because the line also ended
    # on a quote. \" inside the value is not that closer.
    if not val or val[0] not in {'"', "'"}:
        return None
    quote = val[0]
    index = 1
    while index < len(val):
        # A single-quoted value is literal. "C:\temp\" used to skip the
        # closer, so the next key was swallowed and the voice id was empty.
        if quote == '"' and val[index] == "\\" and index + 1 < len(val):
            index += 2
            continue
        if val[index] == quote:
            return val[1:index], val[index + 1 :]
        index += 1
    return None


def _unescape_double(inner: str) -> str:
    # "Say \"hi\"\nthere" is a prompt, not the letters backslash and n.
    out: list[str] = []
    index = 0
    while index < len(inner):
        if inner[index] == "\\" and index + 1 < len(inner):
            nxt = inner[index + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            else:
                out.append(inner[index : index + 2])
            index += 2
            continue
        out.append(inner[index])
        index += 1
    return "".join(out)


def _env_value(raw: str) -> str:
    """Unquote a dotenv value. An unquoted ` #` starts a comment."""
    val = raw.strip()
    span = _quoted_span(val)
    if span is not None:
        inner, rest = span
        rest = rest.lstrip()
        if not rest or rest.startswith("#"):
            if val[0] == '"':
                return _unescape_double(inner)
            return inner
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
        return val[1:-1]
    hashed = val.find(" #")
    if hashed >= 0:
        val = val[:hashed].rstrip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
        return val[1:-1]
    return val


def _without_export(line: str) -> str:
    # `export KEY=value` is a shell prefix. A tab after export is legal and
    # must not become part of the key name.
    if (
        line.startswith(_EXPORT_WORD)
        and len(line) > len(_EXPORT_WORD)
        and line[len(_EXPORT_WORD)].isspace()
    ):
        return line[len(_EXPORT_WORD) :].strip()
    return line


def _unclosed_quote(stripped: str) -> str | None:
    # "You are helpful. keeps going on the next line. A one-line "key" is
    # already balanced, including "key # not a comment".
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    _, _, val = _without_export(stripped).partition("=")
    val = val.lstrip()
    if val[:1] not in {'"', "'"}:
        return None
    if _quoted_span(val) is None:
        return val[0]
    return None


def _logical_lines(text: str) -> list[str]:
    raw_lines = text.splitlines()
    folded: list[str] = []
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        quote = _unclosed_quote(line.strip())
        if quote is None:
            folded.append(line)
            index += 1
            continue
        parts = [line]
        index += 1
        while index < len(raw_lines):
            parts.append(raw_lines[index])
            index += 1
            # \" counts as a quote character, so a raw count closes too early
            # and the next line of the prompt is dropped.
            if _unclosed_quote("\n".join(parts).strip()) is None:
                break
        folded.append("\n".join(parts))
    return folded


def _load_dotenv(path: Path) -> bool:
    """Fill os.environ from KEY=VAL lines. Existing keys win. Returns whether the file was read."""
    if not path.is_file():
        return False
    try:
        # utf-8-sig drops a leading BOM. Left in place it sticks to the first key.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        warn(f"fish-voice: env file is not utf-8: {path}")
        return False
    except OSError as exc:
        warn(f"fish-voice: could not read env file {path}: {exc.strerror}")
        return False
    for raw in _logical_lines(text):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = _without_export(line)
        key, _, val = line.partition("=")
        key = key.strip()
        val = _env_value(val)
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
                warn(f"fish-voice: --env-file not found: {path}")
            continue
        if _load_dotenv(expanded):
            seen.add(resolved)
            loaded.append(path)
    return loaded


def _parse_device(raw: str | None) -> str | int | None:
    if raw is None:
        return None
    text = raw.strip()
    # A newline is not part of a PortAudio name. The mic open then fails
    # and the duplex loop stops.
    cut = next((index for index, ch in enumerate(text) if ord(ch) < 32), None)
    if cut is not None:
        text = text[:cut].strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        # "1.0" is device 1. Leaving it as a name made the mic open fail
        # and the duplex loop stop.
        try:
            value = float(text)
        except ValueError:
            return text
        if value.is_integer():
            return int(value)
        return text


def _blocker(label: str) -> int:
    warn(f"BLOCKER: {label} missing")
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
    """Speak one line to a WAV file. Does not open the microphone.

    Parameters
    ----------
    c : VoiceCliConfig
        Needs ``FISH_API_KEY`` and ``FISH_VOICE_ID``.

    Returns
    -------
    int
        0 when the file is at least 1000 bytes. 1 when Fish returns audio that
        is shorter than that. 2 when a required setting is missing.
    """
    missing = _require_fish(c)
    if missing is not None:
        return missing
    out = Path(tempfile.gettempdir()) / "fish-audio-suite-smoke.wav"
    tts = _fish_tts(c, "pcm")
    sink = FileSink(out, sample_rate=c.sample_rate, wav=True)
    result = tts.speak_isolated("[clear] Hello there.", sink)
    if result.error_status is not None:
        warn(f"smoke: FAIL {result.error_status} {result.error_message}")
        return _SMOKE_FAIL
    ok = result.bytes_played > _SMOKE_MIN_BYTES
    print(f"smoke: wrote {result.bytes_played} bytes → {out} ({'OK' if ok else 'FAIL <1k'})")
    print("sdk: fishaudio; playback=file format=pcm")
    return EXIT_OK if ok else _SMOKE_FAIL


async def run_loop(c: VoiceCliConfig) -> int:
    """Run duplex until quit, after checking keys, model, and the playback sink.

    Parameters
    ----------
    c : VoiceCliConfig
        Environment snapshot.

    Returns
    -------
    int
        2 when a key, the LLM model, or the playback sink is unusable.
        Otherwise the code from ``duplex_turns``.
    """
    missing = _require_fish(c)
    if missing is not None:
        return missing
    if not c.llm_key:
        return _blocker("FISH_LLM_KEY / OPENROUTER_API_KEY")
    if not c.llm_model:
        return _blocker("FISH_LLM_MODEL / OPENROUTER_MODEL")
    playback_problem = duplex_playback_problem(c.playback)
    if playback_problem is not None:
        warn(f"BLOCKER: {playback_problem}")
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
    """``fish-voice`` entry. Load dotenv, then smoke or duplex.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments without the program name. None reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code. A second SIGINT uses the default terminate handler.

    Notes
    -----
    Process environment wins, then ``--env-file``, otherwise ``./.env`` when
    it exists. ``--debug`` or ``FISH_VOICE_DEBUG=1`` turns on the stderr sink.
    Logging is configured here, not at import.
    """
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
