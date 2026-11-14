r"""Record your mic level for a few seconds and print it, to tune the waveform in `fish-voice --tui`.

It opens the default mic (or `--device`), reads it in the 30 ms frames `fish-voice` uses, and
prints one line about every 90 ms, the rate of the app's `MicLevel` events: the time, the RMS on the
int16 scale, the level in dB below full scale and a small bar. It ends with a summary of
what the room and your voice look like in numbers: the noise floor and speech level the app's
waveform (`session_view.SpeechScale`) would take from them, then the average level while you
speak, the loudest moment and how far apart they are. The levels are the raw mic.
The app's automatic gain and noise suppression are not applied, so with `FISH_VOICE_AGC` on
the app hears a quiet voice louder than this shows.

It calls no API and sends nothing anywhere, and it does not save any audio. It needs the
`speakers` extra (PortAudio).

    uv run python scripts/mic_levels.py                 # 12 seconds
    uv run python scripts/mic_levels.py --seconds 20
    uv run python scripts/mic_levels.py --device 3      # a PortAudio device name or number

Stay quiet for the first two seconds, then talk as you normally do to the assistant: a few
sentences with some pauses, as if answering "Hey, can you hear me?". On NixOS, set
`LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH` first (see docs/INSTALL.md).
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections.abc import Sequence

from fish_audio_suite_voice.capture.mic import SessionMic
from fish_audio_suite_voice.capture.pcm import FULL_SCALE, MIC_LEVEL_EVERY_FRAMES, pcm_rms
from fish_audio_suite_voice.capture.portaudio import parse_device
from fish_audio_suite_voice.cli.session_view import noise_floor_db, speech_level_db
from fish_audio_suite_voice.session.latency import percentile

_SILENCE_DB = -90.0
_BAR_FROM_DB = -70.0  # a bar is empty at this level and full at 0 dB


def to_db(rms: float) -> float:
    """Return an RMS on the int16 scale as dB below full scale.

    Parameters
    ----------
    rms : float
        The RMS of a block of audio.

    Returns
    -------
    float
        0 at full scale, negative below it, ``-90`` for silence.

    Examples
    --------
    >>> round(to_db(32768.0), 1), round(to_db(3276.8), 1), to_db(0.0)
    (0.0, -20.0, -90.0)
    """
    ratio = rms / FULL_SCALE
    return 20 * math.log10(ratio) if ratio > 0 else _SILENCE_DB


def scale_levels(levels_db: Sequence[float]) -> tuple[float, float | None]:
    """Return the noise floor and speech level the app's waveform reads from a recording.

    Parameters
    ----------
    levels_db : Sequence[float]
        The level of each block, in dB. Must not be empty.

    Returns
    -------
    tuple of float and float or None
        ``SpeechScale``'s noise floor and speech level, from its own functions, over the
        whole recording instead of its last six seconds. The speech level is None when
        too few blocks were speech.
    """
    noise = noise_floor_db(levels_db)
    return noise, speech_level_db(levels_db, noise)


def summarize(levels_db: Sequence[float]) -> dict[str, float]:
    """Boil a recording down to the numbers that decide how a waveform should be scaled.

    Parameters
    ----------
    levels_db : Sequence[float]
        The level of each block, in dB. Must not be empty.

    Returns
    -------
    dict[str, float]
        ``noise`` (the quiet end: the 10th percentile), ``speech`` (the mean of the blocks
        more than 10 dB above the noise, which is when you were talking), ``p95`` and
        ``peak`` (the loudest block), ``spread`` (``peak - speech``, how far a plosive or
        bump sits above your speech) and ``active`` (the share of blocks counted as speech).
    """
    noise = percentile(levels_db, 0.10)
    active = [level for level in levels_db if level > noise + 10.0]
    speech = sum(active) / len(active) if active else noise
    peak = max(levels_db)
    return {
        "noise": noise,
        "speech": speech,
        "p95": percentile(levels_db, 0.95),
        "peak": peak,
        "spread": peak - speech,
        "active": len(active) / len(levels_db),
    }


def _bar(level_db: float, width: int = 40) -> str:
    fill = round(min(1.0, max(0.0, (level_db - _BAR_FROM_DB) / -_BAR_FROM_DB)) * width)
    return "#" * fill


def main(argv: Sequence[str] | None = None) -> int:
    """Record, print and summarise.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments. ``sys.argv`` by default.

    Returns
    -------
    int
        0 when it recorded something, 1 when it heard no audio at all.
    """
    parser = argparse.ArgumentParser(description="Print your mic level for a few seconds.")
    parser.add_argument("--seconds", type=float, default=12.0, help="how long to record")
    parser.add_argument("--device", default=None, help="a PortAudio device name or number")
    args = parser.parse_args(argv)
    device = parse_device(args.device)

    stop = threading.Event()
    levels: list[float] = []
    started = time.monotonic()
    print("t(s)    rms     dB  level", file=sys.stderr)
    with SessionMic(device) as mic:
        for index, frame in enumerate(mic.frames(stop)):
            if index % MIC_LEVEL_EVERY_FRAMES == 0:
                rms = pcm_rms(frame)
                level = to_db(rms)
                levels.append(level)
                elapsed = time.monotonic() - started
                print(f"{elapsed:5.1f} {rms:7.0f} {level:6.1f}  {_bar(level)}")
            if time.monotonic() - started >= args.seconds:
                stop.set()
    if not levels:
        print("no audio heard: check the device and the PortAudio setup", file=sys.stderr)
        return 1
    stats = summarize(levels)
    scale_noise, scale_speech = scale_levels(levels)
    scale_speech_text = f"{scale_speech:6.1f}" if scale_speech is not None else "   n/a"
    print(
        "\nsummary (dB below full scale, 0 is the loudest the mic can record)\n"
        "  the app's waveform (session_view.SpeechScale) would take:\n"
        f"  waveform floor   {scale_noise:6.1f}   the noise floor it measures\n"
        f"  waveform speech  {scale_speech_text}   its speech level (90th percentile of speech)\n"
        "  more about the recording:\n"
        f"  noise floor      {stats['noise']:6.1f}   the quiet end (10th percentile)\n"
        f"  your speech      {stats['speech']:6.1f}   the average while you were talking\n"
        f"  95th percentile  {stats['p95']:6.1f}\n"
        f"  loudest block    {stats['peak']:6.1f}\n"
        f"  peak above speech{stats['spread']:6.1f}   how far a plosive or bump sits above it\n"
        f"  talking          {stats['active'] * 100:5.0f}%   of the blocks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
