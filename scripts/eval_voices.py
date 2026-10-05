r"""Speak the same text with several Fish voices and models, and save each to listen to.

For every voice and TTS model it makes one Fish turn through `FishSpeaker`, the same code
`fish-voice` speaks with, and writes a WAV file to `tmp/`. The file name holds the voice
id, the model and a timestamp, so you can play them one after another and compare:

    tmp/20261004-123456_<voice id>_s2.1-pro.wav

It also prints the time to the first audio, the total time and the length of each clip.

It calls the live Fish API and spends credits, a few hundredths of a cent for a short
line.

    uv run python scripts/eval_voices.py            # the VOICES and TTS_MODELS at the top
    uv run python scripts/eval_voices.py -t "[happy] Hello there, how are you?"
    uv run python scripts/eval_voices.py --find "narrator" --language en

`--find` lists public Fish voices (id, title, popularity) so you can copy ids into
`VOICES`. Your key comes from `.env` (`FISH_API_KEY`), and `FISH_VOICE_ID` is used when
`VOICES` is empty.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fishaudio import FishAudio
from fishaudio.core import OMIT

from fish_audio_suite_kit import SuiteDefaults, known_latency, normalize_cues, scrub_tts
from fish_audio_suite_voice import FileSink, FishSpeaker
from fish_audio_suite_voice.envfile import load_dotenv

# ---------------------------------------------------------------------------
# EDIT THESE. The voices and models the script runs when you pass none.
# ---------------------------------------------------------------------------
# Fish voice ids (the long hex id on a voice's page). A label after the id is only a
# note for you. Leave it empty to use FISH_VOICE_ID from .env. `--find` lists voices.
VOICES: tuple[str, ...] = ()

# Fish TTS models to try with each voice.
TTS_MODELS: tuple[str, ...] = ("s2.1-pro",)

# What to say. Fish [cue] tags work in it. A long line shows more of a voice.
TEXT = (
    "[happy] Hey, it's so good to hear from you! [curious] What have you been up to? "
    "[calm] I've got a few minutes, so tell me everything."
)

# Where the files go, relative to the repository root. The folder is ignored by git.
OUT_DIR = "tmp"

__all__ = [
    "OUT_DIR",
    "TEXT",
    "TTS_MODELS",
    "VOICES",
    "Clip",
    "clip_path",
    "main",
]

ROOT = Path(__file__).resolve().parent.parent
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class Clip:
    """One synthesized clip, or the reason it failed.

    Attributes
    ----------
    voice : str
        The Fish voice id.
    model : str
        The Fish TTS model.
    path : Path
        Where the WAV was written.
    error : str
        Empty when the clip was made.
    first_audio_ms : float
        Milliseconds from the start of the turn to the first audio, or 0.
    total_ms : float
        Milliseconds until the turn ended.
    seconds : float
        Length of the audio.
    """

    voice: str
    model: str
    path: Path
    error: str = ""
    first_audio_ms: float = 0.0
    total_ms: float = 0.0
    seconds: float = 0.0


def clip_path(out_dir: Path, voice: str, model: str, stamp: str) -> Path:
    """Build the file name for a clip: timestamp, voice id and model.

    Parameters
    ----------
    out_dir : Path
        The output folder.
    voice : str
        The Fish voice id. Anything that is not safe in a file name is replaced.
    model : str
        The Fish TTS model.
    stamp : str
        A timestamp shared by every clip in one run, so they sort together.

    Returns
    -------
    Path
        ``out_dir/<stamp>_<voice>_<model>.wav``.
    """
    safe_voice = _UNSAFE.sub("-", voice).strip("-") or "voice"
    safe_model = _UNSAFE.sub("-", model).strip("-") or "model"
    return out_dir / f"{stamp}_{safe_voice}_{safe_model}.wav"


def voice_ids(voices: Sequence[str]) -> list[str]:
    """Clean a list of voice ids: first word of each entry, no blanks or repeats.

    Parameters
    ----------
    voices : sequence of str
        Entries such as ``"0123456789ab... # my voice"``.

    Returns
    -------
    list of str
        The ids in order.
    """
    found = [
        entry.split("#", 1)[0].split()[0] for entry in voices if entry.split("#", 1)[0].strip()
    ]
    return list(dict.fromkeys(found))


def speak_clip(
    api_key: str, voice: str, model: str, text: str, path: Path, *, latency: str, sample_rate: int
) -> Clip:
    """Make one clip with ``FishSpeaker`` and write it to ``path``.

    Parameters
    ----------
    api_key : str
        The Fish API key.
    voice : str
        The Fish voice id.
    model : str
        The Fish TTS model.
    text : str
        What to say.
    path : Path
        Where to write the WAV.
    latency : str
        Fish latency mode.
    sample_rate : int
        Sample rate of the audio.

    Returns
    -------
    Clip
        The timings and length, or the error.
    """
    defaults = SuiteDefaults()
    speaker = FishSpeaker(
        api_key=api_key,
        voice_id=voice,
        model=model,
        latency=known_latency(latency, defaults.latency),
        sample_rate=sample_rate,
    )
    sink = FileSink(path, sample_rate=sample_rate, wav=True)
    first: list[float] = []
    started = time.perf_counter()
    result = speaker.speak(
        text,
        sink,
        cancel=threading.Event(),
        on_first_audio=lambda: first.append(time.perf_counter()),
    )
    total = (time.perf_counter() - started) * 1000
    if result.error_message or result.error_status is not None or not result.got_audio:
        status = f"HTTP {result.error_status}" if result.error_status is not None else ""
        return Clip(
            voice, model, path, error=result.error_message or status or "no audio came back"
        )
    seconds = result.bytes_played / (2 * sample_rate)
    return Clip(
        voice,
        model,
        path,
        first_audio_ms=(first[0] - started) * 1000 if first else 0.0,
        total_ms=total,
        seconds=seconds,
    )


def _table(clips: Sequence[Clip]) -> list[str]:
    header = ("voice", "model", "first audio ms", "total ms", "seconds", "file")
    rows: list[tuple[str, ...]] = [header]
    for c in clips:
        if c.error:
            rows.append((c.voice, c.model, "-", "-", "-", f"FAILED: {c.error}"))
        else:
            rows.append(
                (
                    c.voice,
                    c.model,
                    f"{c.first_audio_ms:.0f}",
                    f"{c.total_ms:.0f}",
                    f"{c.seconds:.1f}",
                    str(c.path),
                )
            )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]


def find_voices(api_key: str, query: str, language: str, count: int) -> list[str]:
    """List public Fish voices that match a search, most used first.

    Parameters
    ----------
    api_key : str
        The Fish API key.
    query : str
        Words from the voice's title. Empty lists the most used.
    language : str
        A language code such as ``en``. Empty for any.
    count : int
        How many to list.

    Returns
    -------
    list of str
        One line per voice: id, title, languages and how many times it was used.
    """
    client = FishAudio(api_key=api_key)
    try:
        page = client.voices.list(
            page_size=count,
            title=query or OMIT,
            language=language or OMIT,
            sort_by="task_count",
        )
        return [
            f"{v.id}  {v.title}  [{', '.join(v.languages or [])}]  used {v.task_count} times"
            for v in page.items
        ]
    finally:
        client.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speak the same text with several Fish voices and models, and save the clips.",
    )
    parser.add_argument("-t", "--text", help="What to say. Replaces TEXT.")
    parser.add_argument("--text-file", type=Path, help="Read the text from a file. Replaces TEXT.")
    parser.add_argument("--latency", default=SuiteDefaults().latency, help="Fish latency mode.")
    parser.add_argument("--sample-rate", type=int, default=SuiteDefaults().sample_rate)
    parser.add_argument("--out-dir", help="Where to write the clips (default: OUT_DIR at the top).")
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="Settings file (default .env)."
    )
    parser.add_argument(
        "--find", metavar="WORDS", help="List public voices whose title matches, then stop."
    )
    parser.add_argument(
        "--language", default="", help="With --find: only this language, such as en."
    )
    parser.add_argument(
        "--count", type=int, default=15, help="With --find: how many to list (default 15)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run and make no requests."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, without the program name. Defaults to ``sys.argv``.

    Returns
    -------
    int
        0 when at least one clip was made, 2 when something is missing or every clip failed.
    """
    args = _parser().parse_args(argv)
    load_dotenv(args.env_file)
    api_key = os.environ.get("FISH_API_KEY", "").strip()
    if args.find is not None:
        if not api_key:
            print("No FISH_API_KEY found. Set it in .env or pass --env-file.", file=sys.stderr)
            return 2
        print("\n".join(find_voices(api_key, args.find, args.language, args.count)))
        return 0
    voices = voice_ids(VOICES) or voice_ids([os.environ.get("FISH_VOICE_ID", "")])
    models = list(dict.fromkeys(m.strip() for m in TTS_MODELS if m.strip()))
    if not voices or not models:
        print(
            "No voices or models to run. Fill in VOICES and TTS_MODELS at the top of the script, "
            "or set FISH_VOICE_ID in .env.",
            file=sys.stderr,
        )
        return 2
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        text = (args.text or TEXT).strip()
    # The same cleanup the voice app applies before it sends text to Fish.
    spoken = scrub_tts(normalize_cues(text))
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / OUT_DIR
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    print(
        f"{len(voices)} voices x {len(models)} models = {len(voices) * len(models)} clips -> {out_dir}",
        file=sys.stderr,
    )
    if args.dry_run:
        for voice in voices:
            for model in models:
                print(clip_path(out_dir, voice, model, stamp))
        return 0
    if not api_key:
        print("No FISH_API_KEY found. Set it in .env or pass --env-file.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    for voice in voices:
        for model in models:
            path = clip_path(out_dir, voice, model, stamp)
            print(f"  speaking {voice} with {model} ...", file=sys.stderr, flush=True)
            try:
                clip = speak_clip(
                    api_key,
                    voice,
                    model,
                    spoken,
                    path,
                    latency=args.latency,
                    sample_rate=args.sample_rate,
                )
            except Exception as exc:  # noqa: BLE001 - one bad voice must not stop the sweep
                clip = Clip(voice, model, path, error=f"{type(exc).__name__}: {exc}")
            clips.append(clip)
    print("\n".join(_table(clips)))
    return 0 if any(not c.error for c in clips) else 2


if __name__ == "__main__":
    raise SystemExit(main())
