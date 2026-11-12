"""Duplex CLI recipe: mic → speech to text → LLM → FishSpeaker → sink.

Not the only way to use the voice library. Apps should import FishSpeaker.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.util
import sys
import tempfile
import traceback
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from fish_audio_suite_voice.config import (
    VoiceCliConfig,
    fish_problem,
    load_config,
    log_dir_from_env,
    prompt_from_file,
    warn_if_insecure_base,
)
from fish_audio_suite_voice.console_sink import ConsoleSink, console_print
from fish_audio_suite_voice.debug import DebugLevel, debug_level, warn
from fish_audio_suite_voice.duplex import duplex_turns
from fish_audio_suite_voice.envfile import apply_cli_env_files
from fish_audio_suite_voice.events import EVENTS, ExitCode
from fish_audio_suite_voice.inputs import LiveInput
from fish_audio_suite_voice.llm import ChatBackend, open_chat_backend
from fish_audio_suite_voice.llm_tune import llm_problem
from fish_audio_suite_voice.log_setup import (
    configure_voice_logging,
    forward_logs,
    mirror_conversation,
)
from fish_audio_suite_voice.playback import FileSink, duplex_playback_problem
from fish_audio_suite_voice.playback_kind import playback_key
from fish_audio_suite_voice.portaudio import parse_device
from fish_audio_suite_voice.recognisers import recogniser_problem
from fish_audio_suite_voice.signals import DuplexSession
from fish_audio_suite_voice.speaker import FishSpeaker
from fish_audio_suite_voice.transports import ChatUnusableError
from fish_audio_suite_voice.turn_stream import RecogniserSettings
from fish_audio_suite_voice.vad import vad_problem
from fish_audio_suite_voice.wording import short_model
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
# The garbage collector stops every thread while it runs, and a big full-screen app has plenty to
# scan. Collecting less often, and not scanning what was built at startup, keeps those stops rare.
_TUI_GC_THRESHOLD: Final = (20_000, 50, 100)
_SMOKE_FAIL: Final = 1


# What a session cannot start without, in the order they are checked. Each check belongs to
# the module that knows the setting, and says why in words the docs quote.
_CHECKS: Final[tuple[Callable[[VoiceCliConfig], str | None], ...]] = (
    fish_problem,
    lambda c: llm_problem(c.llm),
    lambda c: duplex_playback_problem(c.playback),
    lambda c: vad_problem(c.listen.vad_engine, c.listen.vad_aggressiveness),
    lambda c: recogniser_problem(
        RecogniserSettings(c.stt, fish_key=c.fish_api_key, fish_base=c.fish_base)
    ),
)


def _preflight(
    c: VoiceCliConfig, checks: tuple[Callable[[VoiceCliConfig], str | None], ...] = _CHECKS
) -> ExitCode | None:
    """Say the first problem as a BLOCKER line and return ``FATAL``, or None when there is none."""
    for check in checks:
        problem = check(c)
        if problem is not None:
            warn(f"BLOCKER: {problem}")
            return ExitCode.FATAL
    return None


def _fish_tts(c: VoiceCliConfig) -> FishSpeaker:
    return FishSpeaker(
        api_key=c.fish_api_key,
        voice_id=c.fish_voice_id,
        model=c.tts_model,
        tune=c.tts,
        cue_style=c.cue_style,
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
        0 when Fish returned more than 1000 bytes of audio. 1 when Fish returned an
        error or less audio than that. 2 when a required setting is missing.
    """
    blocked = _preflight(c, (fish_problem,))
    if blocked is not None:
        return blocked
    if out is None:
        with tempfile.NamedTemporaryFile(
            prefix="fish-audio-suite-smoke-", suffix=".wav", delete=False
        ) as handle:
            out = Path(handle.name)
    tts = _fish_tts(c)
    sink = FileSink(out, sample_rate=c.tts.sample_rate, wav=True)
    result = tts.speak("Hello there.", sink)
    if result.error_status is not None:
        warn(f"smoke: FAIL {result.error_status} {result.error_message}")
        return _SMOKE_FAIL
    ok = result.bytes_played > _SMOKE_MIN_BYTES
    console_print(
        f"smoke: wrote {result.bytes_played} bytes → {out} ({'OK' if ok else 'FAIL <1k'})"
    )
    console_print("sdk: fishaudio; playback=file format=pcm")
    return ExitCode.OK if ok else _SMOKE_FAIL


@dataclass(frozen=True, slots=True)
class _Runtime:
    """What both run modes build from the settings before a session starts."""

    config: VoiceCliConfig
    tts: FishSpeaker
    device: str | int | None
    session: DuplexSession

    def chat(self) -> AbstractAsyncContextManager[ChatBackend]:
        """Open the chat backend for this session. It checks the model first."""
        return open_chat_backend(self.config.llm, session_id=uuid.uuid4().hex)


def _start(c: VoiceCliConfig, *, full_screen: bool) -> _Runtime | None:
    """Check the settings and build the session's runtime, or say why not and return None."""
    if _preflight(c) is not None:
        return None
    c = replace(c, full_screen=full_screen)
    return _Runtime(
        config=c, tts=_fish_tts(c), device=parse_device(c.device), session=DuplexSession()
    )


async def run_loop(c: VoiceCliConfig) -> int:
    """Run duplex until quit, after checking the keys, model, sink and voice detector.

    Parameters
    ----------
    c : VoiceCliConfig
        Environment snapshot.

    Returns
    -------
    int
        2 when a key or the LLM model is missing, the playback sink, the voice
        detector or Deepgram cannot be used, or the chat model is unusable.
        Otherwise the code from ``duplex_turns``.

    Notes
    -----
    The plain terminal display (``ConsoleSink``) prints the session. Ctrl+C is routed
    to ``DuplexSession.request_quit`` on the running loop. A second Ctrl+C uses the
    default handler.
    """
    run = _start(c, full_screen=False)
    if run is None:
        return ExitCode.FATAL
    c = run.config
    console_print(
        f"fish-voice ready | tts={c.tts_model} voice={c.fish_voice_id} "
        f"asr_lang={c.fish_asr_language or 'auto'} latency={c.tts.latency} "
        f"playback={c.playback} | "
        f"llm={c.llm.provider}:{short_model(c.llm.model)} | Ctrl+C quit",
        flush=True,
    )
    with ConsoleSink(EVENTS) as display:

        def before_quit() -> None:
            display.end_reply_line()
            sys.stderr.write("\n")
            sys.stderr.flush()

        run.session.install_sigint(asyncio.get_running_loop(), before=before_quit)
        try:
            async with run.chat() as backend:
                return await duplex_turns(c, run.tts, run.device, backend, run.session)
        except ChatUnusableError:
            return ExitCode.FATAL


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
        2 when a key or the LLM model is missing, the playback sink, the voice detector
        or Deepgram cannot be used, or the chat model is unusable. Otherwise the exit
        code of the session, which the app reports.

    Notes
    -----
    Needs the ``tui`` extra. The chat backend is opened, and its model checked, while
    the terminal is still plain, so a model that cannot answer is reported there. Then
    log lines go to the app's log pane, not to stderr, which the screen owns. Quit is
    Ctrl+Q, since Ctrl+C is copy inside the app.
    """
    run = _start(c, full_screen=True)
    if run is None:
        return ExitCode.FATAL
    c = run.config
    # Imported here because Textual is an optional extra, checked for in ``main``.
    from fish_audio_suite_voice.tui import VoiceApp  # noqa: PLC0415 - optional extra

    live = LiveInput()
    try:
        async with run.chat() as backend:
            # From here the screen owns the terminal, so logging moves into the log pane.
            configure_voice_logging(debug=debug, to_stderr=False)
            stop_forwarding = forward_logs()
            switch_interval = sys.getswitchinterval()
            sys.setswitchinterval(_TUI_SWITCH_INTERVAL_S)
            gc_threshold = gc.get_threshold()
            gc.collect()
            gc.freeze()
            gc.set_threshold(*_TUI_GC_THRESHOLD)
            try:

                async def runner() -> int:
                    return await duplex_turns(
                        c, run.tts, run.device, backend, run.session, source=live
                    )

                info = (
                    f"{c.llm.provider}:{short_model(c.llm.model)} | tts {c.tts_model} | "
                    f"voice {c.fish_voice_id[:8]}"
                )
                app = VoiceApp(runner, live=live, session=run.session, info=info)
                code = await app.run_async()
            finally:
                sys.setswitchinterval(switch_interval)
                gc.set_threshold(*gc_threshold)
                gc.unfreeze()
                stop_forwarding()
    except ChatUnusableError:
        return ExitCode.FATAL
    return code or ExitCode.OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fish Audio duplex CLI recipe (kit + live WS + sinks)")
    p.add_argument("--smoke", action="store_true", help="TTS smoke test to a wav file")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="write a complete log of this run to a new file in DIR "
        "(overrides FISH_VOICE_LOG_DIR). It holds what was said",
    )
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
    return ExitCode.OK


def _can_draw_a_screen() -> bool:
    """Say whether the full-screen app has a terminal to draw on."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _start_logging(args: argparse.Namespace) -> DebugLevel:
    """Configure logging from the flags and the environment, and return the level shown."""
    level = DebugLevel.TRACE if args.trace else DebugLevel(int(bool(args.debug)))
    level = max(level, debug_level())
    log_file = configure_voice_logging(
        debug=level, log_dir=str(args.log_dir) if args.log_dir else log_dir_from_env()
    )
    if level >= DebugLevel.EVENTS or log_file:
        install_fish_ws_tap()
    if log_file:
        mirror_conversation()
        print(f"log: {log_file}", flush=True)
    return level


def main(argv: list[str] | None = None) -> int:
    """``fish-voice`` entry. Load dotenv, then the smoke test, the full-screen app or the loop.

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
        return ExitCode.FATAL
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
    level = _start_logging(args)
    debug = level >= DebugLevel.EVENTS
    c = load_config()
    warn_if_insecure_base(c)
    if args.playback:
        c = replace(c, playback=playback_key(args.playback))
    if args.prompt_file:
        prompt = prompt_from_file(str(args.prompt_file), required=True)
        if prompt is None:
            return ExitCode.FATAL
        c = replace(c, system_prompt=prompt.text, pin_seed=prompt.pin_seed)

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
        return ExitCode.FATAL


if __name__ == "__main__":
    raise SystemExit(main())
