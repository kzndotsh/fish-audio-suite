r"""Measure how varied the cue tags are that a chat model writes under each system prompt.

Give it one or more system prompts as files. It sends the same user lines to the model
under each prompt, several times, and counts the `[cue]` tags in the replies: how many
different ones, how evenly they are used, how often one cue dominates, and how many are
outside Fish's own list (which `FISH_TTS_CUES=official` would map or drop). The built-in
prompt `fish-voice` uses is always included as `default`, so a prompt is measured
against it.

It calls a live chat API and spends credits. Nothing is sent to Fish, and nothing is
sent at all until you run it.

    uv run python scripts/eval_cue_tags.py                      # the default prompt alone

    uv run python scripts/eval_cue_tags.py wide.txt strict.txt -n 5

    uv run python scripts/eval_cue_tags.py wide.txt --model sao10k/l3.1-euryale-70b --out cues.md

A prompt file is used as written, as `FISH_VOICE_SYSTEM_PROMPT` is, so it gets no opening
example unless you pass `--seed`. The user lines are the LINES below, picked to call for different feelings;
`-l` and `--lines-file` replace them. The LLM settings come from the environment and
`--env-file` (default `.env`), the same variables `fish-voice` reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import eval_models

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    CueStyle,
    last_emotion,
    official_cue,
    sentence_cuts,
    split_cues,
)
from fish_audio_suite_voice.config.dotenv import load_dotenv
from fish_audio_suite_voice.config.settings import SystemPrompt
from fish_audio_suite_voice.llm.tune import LlmTune

# ---------------------------------------------------------------------------
# EDIT THESE. The user lines sent under every prompt when you pass none. Each one
# calls for a different feeling, so a model that answers them all in the same two
# moods is not reading the moment.
# ---------------------------------------------------------------------------
LINES: tuple[str, ...] = (
    "Hey, can you hear me? Tell me something fun.",
    "I got the job! I start on Monday.",
    "My dog died this morning. I don't really know what to do with myself.",
    "Someone scratched my car in the parking lot and just drove off.",
    "I have to give a speech tomorrow and I can't stop shaking.",
    "I found my old school photos last night. I haven't thought about those years in ages.",
    "I called my boss by my ex's name in front of everyone today.",
    "Ugh, there's nothing to do. This whole weekend is dragging.",
    "Wait, so is the meeting on Tuesday or Thursday? You said both.",
    "My sister got the promotion I was up for. Good for her, I guess.",
    "I forgot my mom's birthday. She hasn't said anything, which is worse.",
    "Tell me a really bad joke.",
)

DEFAULT_NAME = "default"
_TOP_CUES = 12


@dataclass(frozen=True, slots=True)
class CueStats:
    """How one system prompt's replies used cues.

    Attributes
    ----------
    name : str
        The prompt's name: ``default`` or the file's name without its extension.
    replies : int
        Replies that came back with text.
    failed : int
        Requests that failed or came back empty.
    sentences : int
        Sentences across the replies.
    cues : int
        Cue tags across the replies, as the model wrote them.
    distinct : int
        Different cues written.
    distinct_sent : int
        Different cues left after the ``official`` mapping, which is what Fish hears by
        default.
    evenness : float
        From 0 to 1: 1 when every cue written is used equally often, 0 when one cue is
        the only one. The entropy of the cue counts over its largest possible value.
    top_cue : str
        The most used cue.
    top_share : float
        The share of all cues that are ``top_cue``.
    top3_share : float
        The share of all cues that are among the three most used.
    off_list_share : float
        The share of cues that are not Fish's own, so ``official`` maps or drops them.
    dropped_share : float
        The share of cues ``official`` removes, having found no official cue near them.
    sound_share : float
        The share of cues that are a sound or a pause, not an emotion.
    bare_sentence_share : float
        The share of sentences the model wrote without any cue.
    counts : dict
        Each cue written and how often, most used first.
    cost : float
        Dollars the requests reported, when the provider reports any.
    """

    name: str
    replies: int = 0
    failed: int = 0
    sentences: int = 0
    cues: int = 0
    distinct: int = 0
    distinct_sent: int = 0
    evenness: float = 0.0
    top_cue: str = ""
    top_share: float = 0.0
    top3_share: float = 0.0
    off_list_share: float = 0.0
    dropped_share: float = 0.0
    sound_share: float = 0.0
    bare_sentence_share: float = 0.0
    counts: dict[str, int] = field(default_factory=dict[str, int])
    cost: float = 0.0


def cues_in(text: str, *, official: bool = False) -> list[str]:
    """List the cue tags in a reply, without their brackets.

    Parameters
    ----------
    text : str
        A reply as the model wrote it.
    official : bool, optional
        Apply the ``official`` mapping first, so the result is what Fish is sent with
        ``FISH_TTS_CUES=official``.

    Returns
    -------
    list of str
        The cues in order, lowercased and with spelling aliases applied
        (``laugh`` is ``laughing``), as the app shapes them.

    Examples
    --------
    >>> cues_in("[Happy] Hi! [soft chuckle] Ha. [in a storytelling voice] So.")
    ['happy', 'soft chuckle', 'in a storytelling voice']
    >>> cues_in("[Happy] Hi! [soft chuckle] Ha. [in a storytelling voice] So.", official=True)
    ['happy', 'chuckling']
    """
    shaped = CueStyle("official" if official else "free").shape(text)
    return [piece.strip()[1:-1].strip() for piece, is_cue in split_cues(shaped) if is_cue]


def _evenness(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if len(counts) < 2 or not total:
        return 0.0
    entropy = -sum(n / total * math.log(n / total) for n in counts.values())
    return entropy / math.log(len(counts))


def _bare_sentences(text: str) -> tuple[int, int]:
    """Count a reply's sentences, and how many of them carry no cue."""
    shaped = CueStyle().shape(text)
    starts = [0, *sentence_cuts(shaped)]
    ends = [*starts[1:], len(shaped)]
    pieces = [shaped[a:b] for a, b in zip(starts, ends, strict=True) if shaped[a:b].strip()]
    bare = sum(1 for piece in pieces if not any(is_cue for _, is_cue in split_cues(piece)))
    return len(pieces), bare


def cue_stats(name: str, replies: Sequence[str], *, failed: int = 0, cost: float = 0.0) -> CueStats:
    """Count how a set of replies used cues.

    Parameters
    ----------
    name : str
        The prompt's name.
    replies : sequence of str
        The replies, as the model wrote them.
    failed : int, optional
        Requests that returned no reply.
    cost : float, optional
        Dollars the requests reported.

    Returns
    -------
    CueStats
        The counts and shares. Every share is 0 when no cue was written.
    """
    written: Counter[str] = Counter()
    sent: set[str] = set()
    sentences = bare = 0
    for reply in replies:
        written.update(cues_in(reply))
        sent.update(cues_in(reply, official=True))
        total, without = _bare_sentences(reply)
        sentences += total
        bare += without
    cues = sum(written.values())
    ranked = written.most_common()

    def share(count: int) -> float:
        return count / cues if cues else 0.0

    off_list = sum(n for cue, n in ranked if official_cue(cue) != cue)
    dropped = sum(n for cue, n in ranked if official_cue(cue) is None)
    sounds = sum(n for cue, n in ranked if last_emotion(f"[{cue}]") is None)
    return CueStats(
        name=name,
        replies=len(replies),
        failed=failed,
        sentences=sentences,
        cues=cues,
        distinct=len(written),
        distinct_sent=len(sent),
        evenness=_evenness(written),
        top_cue=ranked[0][0] if ranked else "",
        top_share=share(ranked[0][1]) if ranked else 0.0,
        top3_share=share(sum(n for _, n in ranked[:3])),
        off_list_share=share(off_list),
        dropped_share=share(dropped),
        sound_share=share(sounds),
        bare_sentence_share=bare / sentences if sentences else 0.0,
        counts=dict(ranked),
        cost=cost,
    )


def read_system_prompts(paths: Sequence[Path], *, default: bool = True) -> dict[str, str]:
    """Read the system prompts to compare.

    Parameters
    ----------
    paths : sequence of Path
        One file per prompt. The whole file is the prompt.
    default : bool, optional
        Put the built-in prompt first, named ``default``.

    Returns
    -------
    dict
        Name to prompt text, in the order given. A file is named by its stem, with a
        number added when two share one.

    Raises
    ------
    ValueError
        When a file is empty.
    """
    prompts: dict[str, str] = {DEFAULT_NAME: DEFAULT_SYSTEM_PROMPT} if default else {}
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{path} is empty")
        name, count = path.stem, 2
        while name in prompts:
            name, count = f"{path.stem}-{count}", count + 1
        prompts[name] = text
    return prompts


async def evaluate(
    tune: LlmTune,
    prompts: dict[str, str],
    lines: Sequence[str],
    *,
    runs: int,
    seed: bool = False,
    show: bool = False,
) -> tuple[list[CueStats], dict[str, list[eval_models.Run]]]:
    """Send every user line under every system prompt and count the cues.

    Parameters
    ----------
    tune : LlmTune
        The LLM settings, with the model to ask.
    prompts : dict
        Name to system prompt.
    lines : sequence of str
        The user lines.
    runs : int
        Repeats of each line under each prompt.
    seed : bool, optional
        True pins the app's opening example under every prompt. False pins it as the app
        would for a prompt written out in full: under the built-in prompt only.
    show : bool, optional
        Print each reply as it arrives.

    Returns
    -------
    tuple
        The stats, one per prompt in order, and every request by prompt name.
    """
    stats: list[CueStats] = []
    raw: dict[str, list[eval_models.Run]] = {}

    def printer(name: str) -> Callable[[eval_models.Run], None]:
        def on_run(run: eval_models.Run) -> None:
            said = run.error or run.text
            print(f"[{name}] line {run.prompt + 1}/{len(lines)}, run {run.run + 1}/{runs}: {said}")

        return on_run

    for name, system_prompt in prompts.items():
        results = await eval_models.evaluate(
            tune,
            [tune.model],
            lines,
            system_prompt,
            runs=runs,
            seed=seed or SystemPrompt.written(system_prompt).pin_seed,
            warmup=False,
            on_run=printer(name) if show else None,
        )
        raw[name] = results
        ok = [run for run in results if not run.error]
        stats.append(
            cue_stats(
                name,
                [run.text for run in ok],
                failed=len(results) - len(ok),
                cost=sum(run.cost or 0.0 for run in ok),
            )
        )
    return stats, raw


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def table(stats: Sequence[CueStats]) -> list[str]:
    """Format the comparison as a Markdown table, one row per prompt.

    Parameters
    ----------
    stats : sequence of CueStats
        One entry per prompt.

    Returns
    -------
    list of str
        The table's lines.
    """
    rows = [
        "| prompt | replies | cues per reply | distinct | distinct sent | evenness | top cue | "
        "top 3 | off list | dropped | sounds | bare sentences |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in stats:
        per_reply = s.cues / s.replies if s.replies else 0.0
        top = f"{s.top_cue} {_pct(s.top_share)}" if s.top_cue else "none"
        rows.append(
            f"| `{s.name}` | {s.replies}/{s.replies + s.failed} | {per_reply:.1f} | {s.distinct} | "
            f"{s.distinct_sent} | {s.evenness:.2f} | {top} | {_pct(s.top3_share)} | "
            f"{_pct(s.off_list_share)} | {_pct(s.dropped_share)} | {_pct(s.sound_share)} | "
            f"{_pct(s.bare_sentence_share)} |"
        )
    return rows


def cue_lists(stats: Sequence[CueStats]) -> list[str]:
    """List each prompt's most used cues with their counts.

    Parameters
    ----------
    stats : sequence of CueStats
        One entry per prompt.

    Returns
    -------
    list of str
        One line per prompt.
    """
    lines: list[str] = []
    for s in stats:
        ranked = list(s.counts.items())
        shown = ", ".join(f"{cue} {n}" for cue, n in ranked[:_TOP_CUES])
        more = f", and {len(ranked) - _TOP_CUES} more" if len(ranked) > _TOP_CUES else ""
        lines.append(f"- `{s.name}`: {shown or 'no cues'}{more}")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare how varied a chat model's cue tags are under different system prompts.",
    )
    parser.add_argument(
        "prompts", nargs="*", type=Path, help="System prompt files. Each whole file is one prompt."
    )
    parser.add_argument(
        "--no-default", action="store_true", help="Leave the built-in prompt out of the comparison."
    )
    parser.add_argument(
        "-l", "--line", action="append", default=[], metavar="TEXT", help="A user line. Repeatable."
    )
    parser.add_argument(
        "--lines-file", type=Path, help="User lines separated by a line of dashes (---)."
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=3,
        help="Repeats of each line under each prompt (default 3).",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Pin the app's opening example under every prompt, not only the built-in one. "
        "Use it to compare a variant of the built-in prompt like for like.",
    )
    parser.add_argument("--model", help="The chat model. Default: the one fish-voice would use.")
    parser.add_argument(
        "--temperature", type=float, help="Sampling temperature. Default FISH_LLM_TEMPERATURE."
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="Settings file (default .env)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write a report: .md for Markdown, .json for the stats and replies.",
    )
    parser.add_argument("--show", action="store_true", help="Print each reply as it arrives.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run and make no requests."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison from the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, without the program name. Defaults to ``sys.argv``.

    Returns
    -------
    int
        0 on success, 2 when something is missing or every request failed.
    """
    args = _parser().parse_args(argv)
    load_dotenv(args.env_file)
    if args.runs < 1:
        print("--runs must be at least 1.", file=sys.stderr)
        return 2
    try:
        prompts = read_system_prompts(args.prompts, default=not args.no_default)
    except (OSError, ValueError) as exc:
        print(f"Could not read a prompt file: {exc}", file=sys.stderr)
        return 2
    if not prompts:
        print(
            "No prompts to compare: --no-default needs at least one prompt file.", file=sys.stderr
        )
        return 2
    lines = (
        eval_models.read_prompts(args.line, args.lines_file)
        if args.line or args.lines_file
        else list(LINES)
    )
    tune = LlmTune.from_env()
    if args.model:
        tune = replace(tune, model=args.model)
    if args.temperature is not None:
        tune = replace(tune, temperature=args.temperature)
    requests = len(prompts) * len(lines) * args.runs
    print(
        f"model {tune.model}: {len(prompts)} prompts x {len(lines)} lines x {args.runs} runs "
        f"= {requests} requests",
        file=sys.stderr,
    )
    if args.dry_run:
        for name in prompts:
            print(name)
        return 0
    if not tune.model:
        print("No model. Pass --model or set FISH_LLM_MODEL.", file=sys.stderr)
        return 2
    if not tune.api_key:
        print(
            "No API key found. Set OPENROUTER_API_KEY or FISH_LLM_API_KEY, or pass --env-file.",
            file=sys.stderr,
        )
        return 2
    eval_models.listen_for_usage()
    stats, raw = asyncio.run(
        evaluate(tune, prompts, lines, runs=args.runs, seed=args.seed, show=args.show)
    )
    report = [*table(stats), "", *cue_lists(stats)]
    if args.show:
        print()
    print("\n".join(report))
    spent = sum(s.cost for s in stats)
    if spent:
        print(f"\nSpent ${spent:.5f} on the requests that reported a cost.")
    if args.out is not None:
        if args.out.suffix == ".json":
            payload = {
                "model": tune.model,
                "lines": lines,
                "stats": [asdict(s) for s in stats],
                "replies": {name: [asdict(run) for run in runs] for name, runs in raw.items()},
            }
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            head = [
                f"# Cue variety: `{tune.model}`",
                "",
                f"{len(lines)} lines, {args.runs} runs each.",
                "",
            ]
            args.out.write_text("\n".join([*head, *report]) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0 if any(s.replies for s in stats) else 2


if __name__ == "__main__":
    raise SystemExit(main())
