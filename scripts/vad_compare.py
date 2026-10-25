r"""Play one recording through each voice detector and show where each would interrupt the bot.

Each detector hears the same audio through the real barge-in gate, with the same loudness
floor and hit count `fish-voice` uses. For each one it prints a timeline of the frames it
called speech and the moments the gate would have stopped a reply. Sounds that are not you
talking (typing, a cough, a chair) should leave the timeline empty and trip nothing.

It calls no API and sends nothing anywhere. `--record` needs the `speakers` extra (PortAudio)
and saves the WAV it records. Silero needs the `silero` extra.

    uv run python scripts/vad_compare.py --record 30 take.wav    # record, then compare
    uv run python scripts/vad_compare.py take.wav                # compare a 16 kHz mono WAV
    uv run python scripts/vad_compare.py take.wav --engines webrtc,silero --strictness 1,2
    uv run python scripts/vad_compare.py take.wav --process      # as the app hears it

A useful take: ten seconds of typing, ten of talking, then typing while you talk. The WAV
is the raw mic. `--process` first runs it through the app's mic cleanup (automatic gain and
noise suppression, with FISH_VOICE_AGC and FISH_VOICE_NOISE_SUPPRESSION from the environment),
so the detectors hear what `fish-voice` hears. The gate runs as if the bot were silent. On NixOS, set `LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH`
first (see docs/INSTALL.md).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import wave
from collections.abc import Iterator, Sequence
from pathlib import Path

from fish_audio_suite_voice.aec import EchoCanceller
from fish_audio_suite_voice.barge import BargeGate
from fish_audio_suite_voice.mic import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, SessionMic, StopFlag
from fish_audio_suite_voice.tune import AecTune, ListenTune
from fish_audio_suite_voice.vad import frame_is_speech, open_vad, vad_problem

_COLUMN_FRAMES = 5  # one timeline column is 150 ms
_RULER_EVERY_S = 5


class _FileMic:
    """A ``Mic`` that plays recorded frames from a start position."""

    def __init__(self, frames: Sequence[bytes]) -> None:
        self._frames = frames
        self.aec = EchoCanceller(AecTune.off())
        self.start = 0

    @property
    def position(self) -> int:
        return self.start

    def frames(self, stop: StopFlag | None, since: int | None = None) -> Iterator[bytes]:
        for frame in self._frames[self.start if since is None else since :]:
            if stop is not None and stop.is_set():
                return
            yield frame

    def release(self) -> None:
        """Nothing to close."""

    def recent(self, since: int) -> list[bytes]:
        return list(self._frames[since:])


def barge_trips(frames: Sequence[bytes], listen: ListenTune) -> list[int]:
    """Return the frame index of every barge-in, restarting the gate after each one."""
    mic = _FileMic(frames)
    trips: list[int] = []
    while mic.start < len(frames):
        gate = BargeGate(mic=mic, listen=listen, bleed_delay_s=0)
        gate.watch(threading.Event())
        if gate.resume_at is None:
            break
        trips.append(gate.resume_at)
        mic.start = gate.resume_at
    return trips


def timeline(voiced: Sequence[bool], trips: Sequence[int]) -> tuple[str, str]:
    """Draw voiced frames as a bar row and the barge-ins as carets under it."""
    bars: list[str] = []
    marks: list[str] = []
    tripped = {t // _COLUMN_FRAMES for t in trips}
    for col in range(0, len(voiced), _COLUMN_FRAMES):
        share = sum(voiced[col : col + _COLUMN_FRAMES]) / _COLUMN_FRAMES
        bars.append("█" if share > 0.6 else "▄" if share > 0.2 else "·" if share else " ")
        marks.append("^" if col // _COLUMN_FRAMES in tripped else " ")
    return "".join(bars), "".join(marks)


def ruler(n_frames: int) -> str:
    """Seconds under the timeline, one label every few seconds."""
    columns = -(-n_frames // _COLUMN_FRAMES)
    column_ms = FRAME_MS * _COLUMN_FRAMES
    out = [" "] * columns
    for second in range(0, columns * column_ms // 1000 + 1, _RULER_EVERY_S):
        col = round(second * 1000 / column_ms)
        label = f"{second}s"
        out[col : col + len(label)] = label
    return "".join(out)[:columns]


def read_wav(path: Path) -> list[bytes]:
    """Return a 16 kHz mono 16-bit WAV as 30 ms frames."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            msg = (
                f"{path} must be 16 kHz mono 16-bit, not {wav.getframerate()} Hz, "
                f"{wav.getnchannels()} channels, {wav.getsampwidth() * 8}-bit"
            )
            raise SystemExit(msg)
        pcm = wav.readframes(wav.getnframes())
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]


def record(path: Path, seconds: float, device: int | None) -> None:
    """Record the mic the way ``fish-voice`` reads it and save it as a WAV."""
    stop = threading.Event()
    timer = threading.Timer(seconds, stop.set)
    timer.daemon = True  # a mic that fails to open must not leave the script waiting
    frames: list[bytes] = []
    print(f"recording {seconds:g} s ...", file=sys.stderr)
    try:
        with SessionMic(device) as mic:
            timer.start()
            started = time.monotonic()
            frames.extend(mic.frames(stop))
    finally:
        timer.cancel()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(frames))
    print(f"saved {time.monotonic() - started:.1f} s to {path}", file=sys.stderr)


def processed(frames: Sequence[bytes]) -> list[bytes]:
    """Run frames through the app's mic cleanup, with no speaker playing."""
    cleanup = EchoCanceller(AecTune.from_env())
    if cleanup.load() is None:
        msg = "--process needs the aec extra (pywebrtc-audio) and at least one cleanup stage on"
        raise SystemExit(msg)
    return [cleanup.clean(frame) for frame in frames]


def compare(frames: Sequence[bytes], engines: Sequence[str], strictness: Sequence[int]) -> None:
    """Print a timeline and the barge-ins for every engine and strictness."""
    width = max(len(f"{e} {s}") for e in engines for s in strictness)
    print(f"{'':{width}}  {ruler(len(frames))}")
    for engine in engines:
        for level in strictness:
            name = f"{engine} {level}"
            problem = vad_problem(engine, level)
            if problem is not None:
                print(f"{name:{width}}  skipped: {problem}")
                continue
            vad = open_vad(engine, level)
            voiced = [frame_is_speech(vad, frame) for frame in frames]
            listen = ListenTune(vad_aggressiveness=level, vad_engine=engine)
            trips = barge_trips(frames, listen)
            bars, marks = timeline(voiced, trips)
            at = ", ".join(f"{t * FRAME_MS / 1000:.1f}s" for t in trips) or "none"
            print(f"{name:{width}}  {bars}")
            print(f"{'':{width}}  {marks}")
            share = 100 * sum(voiced) / max(len(voiced), 1)
            print(f"{'':{width}}  speech {share:.0f}% of frames, barge-ins at: {at}")


def strictness_levels(text: str) -> list[int]:
    """Parse ``--strictness``: comma-separated levels from 0 to 3."""
    try:
        levels = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        levels = []
    if not levels or any(not 0 <= level <= 3 for level in levels):
        msg = f"expected levels from 0 to 3 separated by commas, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return levels


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, record if asked, then compare.

    Parameters
    ----------
    argv : Sequence of str or None, optional
        Arguments without the program name. None reads ``sys.argv``.

    Returns
    -------
    int
        0 on success.
    """
    parser = argparse.ArgumentParser(description="Compare voice detectors on one recording.")
    parser.add_argument("wav", type=Path, help="16 kHz mono 16-bit WAV (written by --record)")
    parser.add_argument("--record", type=float, metavar="SECONDS", help="record this long first")
    parser.add_argument("--device", type=int, help="PortAudio input device number")
    parser.add_argument(
        "--process", action="store_true", help="hear it through the app's mic cleanup first"
    )
    parser.add_argument("--engines", default="webrtc,silero", help="comma-separated engines")
    parser.add_argument(
        "--strictness",
        type=strictness_levels,
        default=[1],
        help="comma-separated levels, 0 to 3",
    )
    args = parser.parse_args(argv)
    if args.record:
        record(args.wav, args.record, args.device)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    frames = read_wav(args.wav)
    compare(processed(frames) if args.process else frames, engines, args.strictness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
