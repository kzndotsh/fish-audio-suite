"""Duplex CLI recipe: mic → Fish ASR → LLM → IsolatedFishTts → sink.

Not the only way to use the voice library. Apps should import IsolatedFishTts.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Final

from fish_audio_suite_kit import SuiteDefaults
from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.config import (
    VoiceCliConfig,
    load_config,
    system_prompt_from_file,
    warn_if_insecure_base,
)
from fish_audio_suite_voice.debug import (
    DebugLevel,
    configure_voice_logging,
    console_print,
    debug_level,
    end_reply_line,
    short_model,
    warn,
)
from fish_audio_suite_voice.duplex import EXIT_FATAL, EXIT_OK, bye, duplex_turns
from fish_audio_suite_voice.envfile import apply_cli_env_files
from fish_audio_suite_voice.live import IsolatedFishTts
from fish_audio_suite_voice.llm import open_chat_backend
from fish_audio_suite_voice.playback import (
    FileSink,
    audio_format_for,
    duplex_playback_problem,
    playback_key,
)
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.tune import LlmSettings, provider_for_base

__all__ = [
    "apply_cli_env_files",
    "load_config",
    "main",
    "run_loop",
    "smoke_test",
]

DEFAULT_ENV_FILE: Final = Path(".env")
_SMOKE_MIN_BYTES: Final = 1000
_SMOKE_FAIL: Final = 1


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


def llm_setting_names(llm: LlmSettings) -> tuple[str, str]:
    """Name the key and model variables the selected provider reads."""
    provider = provider_for_base(llm.base)
    if provider is None:
        return (
            "OPENROUTER_API_KEY" if llm.uses_openrouter_sdk else "OPENAI_API_KEY"
        ), "OPENROUTER_MODEL"
    models = [provider.model_env]
    if provider.name == "openrouter":
        models.append("OPENROUTER_MODEL")
    return provider.key_env, " / ".join(models)


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
        mood_lead=c.mood_lead,
        base_url=c.fish_base,
    )


async def smoke_test(c: VoiceCliConfig, out: Path | None = None) -> int:
    """Speak one line to a WAV file. Does not open the microphone.

    Parameters
    ----------
    c : VoiceCliConfig
        Needs ``FISH_API_KEY`` and ``FISH_VOICE_ID``.
    out : Path or None, optional
        Where to write the WAV. None uses a new file in the temp directory,
        so two users on one host never share a path.

    Returns
    -------
    int
        0 when the file is at least 1000 bytes. 1 when Fish returns audio that
        is shorter than that. 2 when a required setting is missing.
    """
    missing = _require_fish(c)
    if missing is not None:
        return missing
    if out is None:
        with tempfile.NamedTemporaryFile(
            prefix="fish-audio-suite-smoke-", suffix=".wav", delete=False
        ) as handle:
            out = Path(handle.name)
    tts = _fish_tts(c, "pcm")
    sink = FileSink(out, sample_rate=c.sample_rate, wav=True)
    result = tts.speak_isolated("Hello there.", sink)
    if result.error_status is not None:
        warn(f"smoke: FAIL {result.error_status} {result.error_message}")
        return _SMOKE_FAIL
    ok = result.bytes_played > _SMOKE_MIN_BYTES
    console_print(
        f"smoke: wrote {result.bytes_played} bytes → {out} ({'OK' if ok else 'FAIL <1k'})"
    )
    console_print("sdk: fishaudio; playback=file format=pcm")
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

    Notes
    -----
    Ctrl+C is routed to ``DuplexSession.request_quit`` on the running loop. A
    second Ctrl+C uses the default handler.
    """
    missing = _require_fish(c)
    if missing is not None:
        return missing
    key_name, model_names = llm_setting_names(c.llm)
    if not c.llm.api_key:
        return _blocker(f"FISH_LLM_API_KEY / {key_name}")
    if not c.llm.model:
        return _blocker(f"FISH_LLM_MODEL / {model_names}")
    playback_problem = duplex_playback_problem(c.playback)
    if playback_problem is not None:
        warn(f"BLOCKER: {playback_problem}")
        return EXIT_FATAL

    device = _parse_device(c.device)
    playback = c.playback
    console_print(
        f"fish-voice ready | tts={c.tts_model} voice={c.fish_voice_id} "
        f"asr_lang={c.fish_asr_language or 'auto'} latency={c.latency} "
        f"playback={playback} | "
        f"llm={c.llm.provider}:{short_model(c.llm.model)} | Ctrl+C quit",
        flush=True,
    )

    tts = _fish_tts(c, audio_format_for(playback))
    session = DuplexSession(aec=EchoCanceller(c.aec))

    def _before_quit() -> None:
        end_reply_line()
        sys.stderr.write("\n")
        sys.stderr.flush()

    session.install_sigint(asyncio.get_running_loop(), before=_before_quit)
    async with open_chat_backend(c.llm, session_id=uuid.uuid4().hex) as backend:
        return await duplex_turns(c, tts, device, backend, session)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fish Audio duplex CLI recipe (kit + live WS + sinks)")
    p.add_argument("--smoke", action="store_true", help="TTS smoke test to a wav file")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="where --smoke writes its wav (default: a new temp file)",
    )
    p.add_argument(
        "--playback",
        default=None,
        help="sounddevice | file | stdout | mpv (overrides FISH_VOICE_PLAYBACK)",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="character or scene text for the system prompt. The voice rules follow it "
        "(overrides FISH_VOICE_SYSTEM_PROMPT_FILE)",
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
        help="stderr event log: listen, ASR, LLM, TTS and barge-in steps (or FISH_VOICE_DEBUG=1)",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="--debug plus mic heartbeats, raw audio events and HTTP lines (or FISH_VOICE_DEBUG=2)",
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
        An unexpected ``CancelledError`` exits 2.

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
    level = DebugLevel.TRACE if args.trace else DebugLevel(int(bool(args.debug)))
    level = max(level, debug_level())
    debug = level >= DebugLevel.EVENTS
    configure_voice_logging(debug=level)
    c = load_config()
    warn_if_insecure_base(c)
    if args.playback:
        c = replace(c, playback=playback_key(args.playback))
    if args.prompt_file:
        composed = system_prompt_from_file(str(args.prompt_file), SuiteDefaults().system_prompt)
        if composed is None:
            return EXIT_FATAL
        c = replace(c, system_prompt=composed, pin_seed=True)

    if args.smoke:
        try:
            return asyncio.run(smoke_test(c, args.out))
        except KeyboardInterrupt:
            return bye()
    try:
        return asyncio.run(run_loop(c))
    except KeyboardInterrupt:
        return bye()
    except asyncio.CancelledError:
        # run_loop returns on quit. A cancel that gets here is a bug, so say
        # so instead of restarting with the same state.
        warn("fish-voice: loop cancelled unexpectedly")
        if debug:
            traceback.print_exc()
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
