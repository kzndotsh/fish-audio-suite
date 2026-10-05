"""Duplex CLI recipe: mic → Fish ASR → LLM → FishSpeaker → sink.

Not the only way to use the voice library. Apps should import FishSpeaker.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
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
from fish_audio_suite_voice.console import console_print, end_reply_line
from fish_audio_suite_voice.debug import (
    DebugLevel,
    configure_voice_logging,
    debug_level,
    short_model,
    warn,
)
from fish_audio_suite_voice.duplex import EXIT_FATAL, EXIT_OK, bye, duplex_turns
from fish_audio_suite_voice.envfile import apply_cli_env_files
from fish_audio_suite_voice.events import forward_logs
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.llm import open_chat_backend
from fish_audio_suite_voice.llm_tune import LlmTune, provider_for_base
from fish_audio_suite_voice.playback import (
    FileSink,
    audio_format_for,
    duplex_playback_problem,
    playback_key,
)
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.ws_tap import install_fish_ws_tap

__all__ = [
    "apply_cli_env_files",
    "load_config",
    "main",
    "run_loop",
    "smoke_test",
]

DEFAULT_ENV_FILE: Final = Path(".env")
_SMOKE_MIN_BYTES: Final = 1000
# Seconds a Python thread may hold the GIL before another is let in. The default of 5 ms lets the
# screen's drawing keep the playback thread waiting longer than the sound card's buffer lasts,
# which is heard as clicks and drop-outs. Shorter hands over sooner, at a small cost in speed.
_TUI_SWITCH_INTERVAL_S: Final = 0.0005
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


def llm_setting_names(llm: LlmTune) -> tuple[str, str]:
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


def _fish_tts(c: VoiceCliConfig, audio_format: str) -> FishSpeaker:
    return FishSpeaker(
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
        fade_ms=c.fade_ms,
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
    result = tts.speak("Hello there.", sink)
    if result.error_status is not None:
        warn(f"smoke: FAIL {result.error_status} {result.error_message}")
        return _SMOKE_FAIL
    ok = result.bytes_played > _SMOKE_MIN_BYTES
    console_print(
        f"smoke: wrote {result.bytes_played} bytes → {out} ({'OK' if ok else 'FAIL <1k'})"
    )
    console_print("sdk: fishaudio; playback=file format=pcm")
    return EXIT_OK if ok else _SMOKE_FAIL


def _preflight(c: VoiceCliConfig) -> int | None:
    """Return an exit code when a key, the model or the playback sink is unusable."""
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
    return _require_deepgram(c)


def _require_deepgram(c: VoiceCliConfig) -> int | None:
    """Return an exit code when Deepgram is chosen but cannot be used."""
    if c.stt.provider != "deepgram":
        return None
    if not c.stt.deepgram_key:
        return _blocker("DEEPGRAM_API_KEY (FISH_VOICE_STT=deepgram)")
    if importlib.util.find_spec("websockets") is None:
        warn(
            "BLOCKER: FISH_VOICE_STT=deepgram needs the deepgram extra: "
            "pip install 'fish-audio-suite-voice[deepgram]' (or uv sync --extra deepgram)"
        )
        return EXIT_FATAL
    return None


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
    blocked = _preflight(c)
    if blocked is not None:
        return blocked

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


async def run_tui(c: VoiceCliConfig, *, debug: DebugLevel) -> int:
    """Run the same session inside the full-screen app, until it ends.

    Parameters
    ----------
    c : VoiceCliConfig
        Environment snapshot.
    debug : DebugLevel
        The log level chosen on the command line, kept when logging moves into the app.

    Returns
    -------
    int
        2 when a key, the LLM model, or the playback sink is unusable. Otherwise the
        exit code of the session, which the app reports.

    Notes
    -----
    Needs the ``tui`` extra. Log lines go to the app's log pane, not to stderr, which
    the screen owns. Quit is Ctrl+Q, since Ctrl+C is copy inside the app.
    """
    blocked = _preflight(c)
    if blocked is not None:
        return blocked
    # Imported here because Textual is an optional extra, checked for in ``main``.
    from fish_audio_suite_voice.tui import VoiceApp  # noqa: PLC0415 - optional extra

    live = LiveInput()
    session = DuplexSession(aec=EchoCanceller(c.aec))
    tts = _fish_tts(c, audio_format_for(c.playback))
    device = _parse_device(c.device)
    # From here the screen owns the terminal, so logging moves into the log pane.
    configure_voice_logging(debug=debug, to_stderr=False)
    stop_forwarding = forward_logs()
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(_TUI_SWITCH_INTERVAL_S)
    try:
        async with open_chat_backend(c.llm, session_id=uuid.uuid4().hex) as backend:

            async def runner() -> int:
                return await duplex_turns(
                    c, tts, device, backend, session, source=live, console=False
                )

            info = (
                f"{c.llm.provider}:{short_model(c.llm.model)} | tts {c.tts_model} | "
                f"voice {c.fish_voice_id[:8]}"
            )
            app = VoiceApp(runner, live=live, session=session, info=info)
            code = await app.run_async()
    finally:
        sys.setswitchinterval(switch_interval)
        stop_forwarding()
    return code or EXIT_OK


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
        "--tui",
        action="store_true",
        help="full-screen app: conversation, mic meter, timings and log, with typed lines "
        "(needs the tui extra)",
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


def _quit_line() -> int:
    # A second Ctrl+C lands here, after the session and its display are gone.
    console_print("\nbye")
    return bye()


def _can_draw_a_screen() -> bool:
    """Say whether the full-screen app has a terminal to draw on."""
    return sys.stdin.isatty() and sys.stdout.isatty()


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
    if args.tui and importlib.util.find_spec("textual") is None:
        # Checked first, while stderr still reaches the user.
        sys.stderr.write(
            "fish-voice --tui needs the tui extra: "
            "pip install 'fish-audio-suite-voice[tui]' (or uv sync --extra tui)\n"
        )
        return EXIT_FATAL
    if args.tui and not _can_draw_a_screen():
        sys.stderr.write(
            "fish-voice --tui needs a terminal (stdin and stdout on a TTY): "
            "running the plain loop instead\n"
        )
        args.tui = False
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
    if debug:
        install_fish_ws_tap()
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
            return _quit_line()
    try:
        if args.tui:
            return asyncio.run(run_tui(c, debug=level))
        return asyncio.run(run_loop(c))
    except KeyboardInterrupt:
        return _quit_line()
    except asyncio.CancelledError:
        # run_loop returns on quit. A cancel that gets here is a bug, so say
        # so instead of restarting with the same state.
        warn("fish-voice: loop cancelled unexpectedly")
        if debug:
            traceback.print_exc()
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
