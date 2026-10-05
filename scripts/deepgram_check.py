r"""Try Deepgram Flux with your mic: see the words arrive, and how long after you stop the turn ends.

It streams your speech to Deepgram the way `FISH_VOICE_STT=deepgram` does, with the same mic
gate, and prints each turn: the words as they come in, the final text, and the time from your
last bit of speech to the turn being final. That last number is what streaming saves you, so
compare it with the "asr" figure and the silence wait of the batch path.

It calls the live Deepgram API and spends a little of your credit (a few tenths of a cent a
minute), but nothing else: no Fish, no language model. It needs `DEEPGRAM_API_KEY` (in `.env`
or the environment) and the `speakers`, `vad` and `deepgram` extras.

    uv run python scripts/deepgram_check.py             # three turns
    uv run python scripts/deepgram_check.py --turns 5
    uv run python scripts/deepgram_check.py --device 3  # a PortAudio device number

Say a sentence, then stop. Ctrl+C ends it. On NixOS, set `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from fish_audio_suite_voice.envfile import apply_cli_env_files
from fish_audio_suite_voice.events import EVENTS, Event, Interim
from fish_audio_suite_voice.streaming import StreamedTurn, StreamFallback, stream_turn
from fish_audio_suite_voice.tune import ListenTune, SttTune


async def run(turns: int, device: str | int | None) -> int:
    """Stream a few turns from the mic and print what Deepgram makes of them.

    Parameters
    ----------
    turns : int
        How many turns to try.
    device : str or int or None
        PortAudio input.

    Returns
    -------
    int
        0 when it ran, 2 when Deepgram could not be used.
    """
    stt = SttTune.from_env()
    if not stt.deepgram_key:
        print("DEEPGRAM_API_KEY is not set (put it in .env or the environment)", file=sys.stderr)
        return 2
    quit_requested = threading.Event()

    def show(event: Event) -> None:
        if isinstance(event, Interim):
            print(f"  ... {event.text}")

    stop_showing = EVENTS.subscribe(show)
    try:
        for number in range(1, turns + 1):
            print(f"\nturn {number}: say a sentence, then stop")
            outcome = await stream_turn(
                stt=stt,
                listen=ListenTune.from_env(),
                device=device,
                aec=None,
                quit_requested=quit_requested,
                stop=quit_requested,
            )
            if isinstance(outcome, StreamedTurn):
                print(f"  final: {outcome.text}")
                print(f"  {outcome.asr_ms:.0f} ms from your last speech to the final text")
            else:
                print(
                    f"  no turn: {'could not reach Deepgram' if isinstance(outcome, StreamFallback) else outcome}"
                )
                if isinstance(outcome, StreamFallback) or outcome == "fatal":
                    return 2
    finally:
        stop_showing()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments. ``sys.argv`` by default.

    Returns
    -------
    int
        0 when it ran, 2 when Deepgram could not be used.
    """
    parser = argparse.ArgumentParser(description="Try Deepgram Flux with your mic.")
    parser.add_argument("--turns", type=int, default=3, help="how many turns to try")
    parser.add_argument("--device", default=None, help="a PortAudio device name or number")
    args = parser.parse_args(argv)
    device: str | int | None = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    apply_cli_env_files([Path(".env")], required=False)
    try:
        return asyncio.run(run(max(1, args.turns), device))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
