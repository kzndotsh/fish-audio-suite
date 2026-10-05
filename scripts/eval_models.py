r"""Run the same prompts through several chat models and compare how they answer.

It sends each prompt to each model through the same backend code `fish-voice` uses
(the OpenRouter SDK for OpenRouter, the OpenAI-compatible client for everything
else), with the same system prompt and the same opening example. For every reply
it records the time to the first token and the total time, and checks how well the
reply suits a voice: does it start with a cue, end without one, stay short, and
avoid markdown and emoji.

It calls live APIs and spends credits, a few hundredths of a cent per request on a
small model. Nothing is sent until you run it.

    uv run python scripts/eval_models.py          # the MODELS and PROMPTS at the top

    uv run python scripts/eval_models.py -p "Hey, can you hear me? Tell me something fun."

    uv run python scripts/eval_models.py --prompts-file prompts.txt --runs 5 --out report.md

The LLM settings come from the environment and `--env-file` (default `.env`), the
same variables `fish-voice` reads: `OPENROUTER_API_KEY` or `FISH_LLM_API_KEY`,
`FISH_LLM_BASE`, `FISH_LLM_REASONING_EFFORT`, `FISH_LLM_PROVIDER_SORT` and the rest.
Flags override them.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from loguru import logger

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    ChatMessage,
    is_tts_junk,
    normalize_cues,
    scrub_tts,
)
from fish_audio_suite_voice.config import system_prompt_from_file
from fish_audio_suite_voice.debug import configure_voice_logging
from fish_audio_suite_voice.envfile import load_dotenv
from fish_audio_suite_voice.history import opening_history
from fish_audio_suite_voice.llm import open_chat_backend
from fish_audio_suite_voice.llm_tune import LlmTune

# ---------------------------------------------------------------------------
# EDIT THESE. The models and prompts the script runs when you pass none.
# `-p` and `--prompts-file` replace the prompts for one run. The models always come
# from this list.
# ---------------------------------------------------------------------------
MODELS: tuple[str, ...] = (
    # The number is how many providers serve it on OpenRouter.
    "sao10k/l3.1-euryale-70b",  # 2
    "sao10k/l3.3-euryale-70b",  # 1
    "sao10k/l3-lunaris-8b",  # 3
    "thedrummer/cydonia-24b-v4.1",  # 1
    "thedrummer/skyfall-36b-v2",  # 1
    "thedrummer/unslopnemo-12b",  # 1
    "anthracite-org/magnum-v4-72b",  # 1
    "gryphe/mythomax-l2-13b",  # 3
    "undi95/remm-slerp-l2-13b",  # 2
    "cognitivecomputations/dolphin-mistral-24b-venice-edition",  # 1
    "aion-labs/aion-rp-llama-3.1-8b",  # 1
    "nousresearch/hermes-3-llama-3.1-70b",  # 1
    "mistralai/mistral-nemo",  # 6
)

PROMPTS: tuple[str, ...] = ("Hey, can you hear me? Tell me something fun.",)

# Or read the prompts from a file instead: a path from the repository root, used in
# place of the list above. Prompts are separated by a line of dashes (---).
PROMPTS_FILE: str = ""

__all__ = [
    "MODELS",
    "PROMPTS",
    "PROMPTS_FILE",
    "Checks",
    "Run",
    "Summary",
    "main",
    "read_prompts",
    "summarize",
    "voice_checks",
]

_WARNING = 30  # loguru's number for WARNING

ROOT = Path(__file__).resolve().parent.parent

_CUE = re.compile(r"\[[^\[\]\n]{1,40}\]")
_LEADING_CUE = re.compile(r"^\s*\[[^\[\]\n]{1,40}\]")
_TRAILING_CUE = re.compile(r"\[[^\[\]\n]{1,40}\]\s*$")
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*]\s|```|\[[^\]]*\]\(https?://)", re.MULTILINE)
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿]")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
_PROMPT_SEPARATOR = re.compile(r"^-{3,}\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Checks:
    """How well one reply suits being spoken aloud.

    Attributes
    ----------
    words : int
        Words in the reply with the cues removed.
    sentences : int
        Sentence ends in the spoken text.
    cues : int
        How many ``[cue]`` tags the reply holds.
    starts_with_cue : bool
        The reply opens with a cue, as the system prompt asks.
    ends_with_cue : bool
        The reply ends on a cue, which Fish plays as a stray sound.
    markdown : bool
        Markdown a voice cannot say: bold, headings, bullets, code or links.
    emoji : bool
        An emoji, which the voice would skip or read out.
    junk : bool
        The text is too thin to speak once the markup is scrubbed away.
    """

    words: int
    sentences: int
    cues: int
    starts_with_cue: bool
    ends_with_cue: bool
    markdown: bool
    emoji: bool
    junk: bool


@dataclass(frozen=True, slots=True)
class Run:
    """One request to one model.

    Attributes
    ----------
    model : str
        The model id.
    prompt : int
        Index of the prompt, from 0.
    run : int
        Index of the repeat, from 0.
    error : str
        Empty when the request returned text.
    first_token_ms : float
        Milliseconds to the first token, or 0 on an error.
    total_ms : float
        Milliseconds until the reply ended.
    text : str
        The reply as the model wrote it.
    checks : Checks or None
        Voice-suitability checks, or None on an error.
    cost : float or None
        Dollars the provider charged for this request, or None when the provider did
        not report one (OpenRouter does, most other servers do not).
    provider : str
        Which provider served the reply, as OpenRouter reports it. Empty when unknown.
    """

    model: str
    prompt: int
    run: int
    error: str = ""
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    text: str = ""
    checks: Checks | None = None
    cost: float | None = None
    provider: str = ""


@dataclass(frozen=True, slots=True)
class Summary:
    """All the runs of one model, reduced to the numbers worth comparing.

    Attributes
    ----------
    model : str
        The model id.
    ok : int
        Requests that returned text.
    failed : int
        Requests that returned nothing.
    first_token_ms : tuple of float
        Median, minimum and maximum time to the first token.
    total_ms : float
        Median time until the reply ended.
    words : float
        Mean words per reply.
    starts_with_cue : float
        Fraction of replies that open with a cue.
    ends_with_cue : float
        Fraction of replies that end on a cue.
    markdown : float
        Fraction of replies with markdown.
    emoji : float
        Fraction of replies with an emoji.
    cost : float or None
        Mean dollars per reply, or None when no reply reported a cost.
    total_cost : float
        Dollars spent across all of this model's requests that reported a cost.
    provider : str
        The provider that served most replies. Empty when unknown.
    errors : list of str
        The distinct error messages.
    """

    model: str
    ok: int
    failed: int
    first_token_ms: tuple[float, float, float]
    total_ms: float
    words: float
    starts_with_cue: float
    ends_with_cue: float
    markdown: float
    emoji: float
    cost: float | None = None
    total_cost: float = 0.0
    provider: str = ""
    errors: list[str] = field(default_factory=list[str])


def voice_checks(text: str) -> Checks:
    """Check whether a reply suits being spoken aloud.

    Parameters
    ----------
    text : str
        The reply exactly as the model wrote it.

    Returns
    -------
    Checks
        The counts and flags described on ``Checks``.
    """
    spoken = scrub_tts(normalize_cues(text))
    plain = _CUE.sub(" ", spoken)
    return Checks(
        words=len(plain.split()),
        sentences=len(_SENTENCE_END.findall(plain)),
        cues=len(_CUE.findall(text)),
        starts_with_cue=bool(_LEADING_CUE.match(text)),
        ends_with_cue=bool(_TRAILING_CUE.search(text)),
        markdown=bool(_MARKDOWN.search(text)),
        emoji=bool(_EMOJI.search(text)),
        junk=is_tts_junk(spoken),
    )


def _listed(path: str) -> Path | None:
    """Resolve the ``PROMPTS_FILE`` setting against the repository root."""
    return (ROOT / path) if path else None


def read_prompts(prompts: Sequence[str], prompts_file: Path | None) -> list[str]:
    """Collect prompts from ``--prompt`` values and a file.

    Parameters
    ----------
    prompts : sequence of str
        Values of ``--prompt``.
    prompts_file : Path or None
        A file of prompts separated by a line of three or more dashes.

    Returns
    -------
    list of str
        The prompts in the order given. When neither flag was used: the file named by
        ``PROMPTS_FILE`` at the top of this script if set, else the ``PROMPTS`` list.
    """
    found = [p.strip() for p in prompts if p.strip()]
    if not prompts and prompts_file is None:
        prompts_file = _listed(PROMPTS_FILE)
        if prompts_file is None:
            return list(PROMPTS)
    if prompts_file is not None:
        text = prompts_file.read_text(encoding="utf-8")
        found.extend(part.strip() for part in _PROMPT_SEPARATOR.split(text) if part.strip())
    return found


def summarize(model: str, runs: Sequence[Run]) -> Summary:
    """Reduce one model's runs to a summary.

    Parameters
    ----------
    model : str
        The model id, used when there are no runs.
    runs : sequence of Run
        Every request made to that model.

    Returns
    -------
    Summary
        Timing is over the successful runs only. Cost counts every request that
        reported one, failed or not.
    """
    good = [r for r in runs if not r.error and r.checks is not None]
    errors = sorted({r.error for r in runs if r.error})
    costs = [r.cost for r in runs if r.cost is not None]
    spent = sum(costs)
    if not good:
        return Summary(
            model,
            0,
            len(runs),
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            spent,
            "",
            errors,
        )
    firsts = [r.first_token_ms for r in good]
    checks = [r.checks for r in good if r.checks is not None]
    providers = [r.provider for r in good if r.provider]

    def share(flag: str) -> float:
        return sum(1 for c in checks if getattr(c, flag)) / len(checks)

    return Summary(
        model=model,
        ok=len(good),
        failed=len(runs) - len(good),
        first_token_ms=(statistics.median(firsts), min(firsts), max(firsts)),
        total_ms=statistics.median(r.total_ms for r in good),
        words=statistics.fmean(c.words for c in checks),
        starts_with_cue=share("starts_with_cue"),
        ends_with_cue=share("ends_with_cue"),
        markdown=share("markdown"),
        emoji=share("emoji"),
        cost=statistics.fmean(costs) if costs else None,
        total_cost=spent,
        provider=max(set(providers), key=providers.count) if providers else "",
        errors=errors,
    )


# The backend logs one line per finished reply at debug level, with the token counts,
# the cost OpenRouter reports, and which provider served it. Reading that line gets the
# cost without changing the package.
_STREAM_END = re.compile(
    r"^llm\.stream_end .*?\bin=(?P<tin>\d+) out=(?P<tout>\d+)(?: cost=\$(?P<cost>[0-9.]+))?"
    r".*?\bprovider=(?P<provider>.*?) model="
)


@dataclass(slots=True)
class _Usage:
    """What the backend reported for the reply that just finished."""

    cost: float | None = None
    provider: str = ""


def parse_stream_end(message: str) -> _Usage | None:
    """Read the cost and provider from the backend's ``llm.stream_end`` log message.

    Parameters
    ----------
    message : str
        A log message.

    Returns
    -------
    _Usage or None
        The cost (None when the provider reported none) and the provider, or None when
        the message is not an ``llm.stream_end`` line.
    """
    found = _STREAM_END.match(message)
    if found is None:
        return None
    cost = found["cost"]
    return _Usage(float(cost) if cost else None, found["provider"].strip())


_latest = _Usage()


def _capture(message: object) -> None:
    """Loguru sink: remember the usage line, and pass warnings on to stderr."""
    record = getattr(message, "record", None)
    if not isinstance(record, dict):
        return
    fields = cast(dict[str, Any], record)
    text = str(fields.get("message", ""))
    usage = parse_stream_end(text)
    if usage is not None:
        _latest.cost, _latest.provider = usage.cost, usage.provider
    elif getattr(fields.get("level"), "no", 0) >= _WARNING:
        print(text, file=sys.stderr)


async def _one(
    tune: LlmTune, messages: list[ChatMessage]
) -> tuple[str, float, float, float | None, str]:
    """Stream one reply and return its text, timings, cost and provider."""
    _latest.cost, _latest.provider = None, ""
    started = time.perf_counter()
    first = 0.0
    parts: list[str] = []
    async with open_chat_backend(tune) as backend:
        async for token in backend.stream(messages):
            if not parts:
                first = (time.perf_counter() - started) * 1000
            parts.append(token)
    total = (time.perf_counter() - started) * 1000
    return "".join(parts).strip(), first, total, _latest.cost, _latest.provider


async def evaluate(
    tune: LlmTune,
    models: Sequence[str],
    prompts: Sequence[str],
    system_prompt: str,
    *,
    runs: int,
    seed: bool | None,
    warmup: bool,
    on_run: Callable[[Run], None] | None = None,
) -> list[Run]:
    """Send every prompt to every model, one request at a time.

    Parameters
    ----------
    tune : LlmTune
        The LLM settings to start from. ``model`` is replaced for each model.
    models : sequence of str
        Model ids.
    prompts : sequence of str
        The user lines to send.
    system_prompt : str
        The system prompt, sent with the opening example the app pins.
    runs : int
        Repeats of each prompt on each model.
    seed : bool or None
        Whether to pin the opening example. None pins it for the default prompt only.
    warmup : bool
        Send one throwaway request per model first, so connection setup is not timed.
    on_run : Callable or None, optional
        Called with each ``Run`` as soon as it finishes, so a caller can show replies
        while the sweep is still going.

    Returns
    -------
    list of Run
        One entry per request, in the order they ran. Requests run one at a time so
        they do not compete for the same connection or rate limit.
    """
    results: list[Run] = []

    def record(run: Run) -> None:
        results.append(run)
        if on_run is not None:
            on_run(run)

    for model in models:
        model_tune = replace(tune, model=model)
        if warmup:
            opening, _ = opening_history(system_prompt, seed=seed)
            # Best effort: the real runs report the error if the model is broken.
            with contextlib.suppress(Exception):
                await _one(model_tune, [*opening, {"role": "user", "content": "Say hi."}])
        for index, prompt in enumerate(prompts):
            opening, _ = opening_history(system_prompt, seed=seed)
            messages: list[ChatMessage] = [*opening, {"role": "user", "content": prompt}]
            for run in range(runs):
                try:
                    text, first, total, cost, provider = await _one(model_tune, messages)
                except Exception as exc:  # noqa: BLE001 - one bad model must not stop the sweep
                    record(Run(model, index, run, error=f"{type(exc).__name__}: {exc}"))
                    continue
                if not text:
                    record(Run(model, index, run, error="empty reply (see the warning above)"))
                    continue
                record(
                    Run(
                        model,
                        index,
                        run,
                        "",
                        first,
                        total,
                        text,
                        voice_checks(text),
                        cost,
                        provider,
                    )
                )
    return results


def _money(value: float | None) -> str:
    """Format a cost in dollars, with enough digits to see a hundredth of a cent."""
    return "-" if value is None else f"${value:.5f}"


def _pct(value: float) -> str:
    return f"{round(value * 100):d}%"


def _table(summaries: Sequence[Summary]) -> Iterator[str]:
    """Yield the summary as aligned text lines."""
    header = (
        "model",
        "ok",
        "first token ms (med/min/max)",
        "total ms",
        "words",
        "cue first",
        "cue last",
        "md",
        "emoji",
        "cost/reply",
        "provider",
    )
    rows: list[tuple[str, ...]] = [header]
    for s in summaries:
        med, low, high = s.first_token_ms
        rows.append(
            (
                s.model,
                f"{s.ok}/{s.ok + s.failed}",
                f"{med:.0f} / {low:.0f} / {high:.0f}" if s.ok else "-",
                f"{s.total_ms:.0f}" if s.ok else "-",
                f"{s.words:.0f}" if s.ok else "-",
                _pct(s.starts_with_cue) if s.ok else "-",
                _pct(s.ends_with_cue) if s.ok else "-",
                _pct(s.markdown) if s.ok else "-",
                _pct(s.emoji) if s.ok else "-",
                _money(s.cost),
                s.provider or "-",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        yield "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()


def _markdown(
    summaries: Sequence[Summary], runs: Sequence[Run], prompts: Sequence[str], system_prompt: str
) -> str:
    """Render the report as Markdown: the summary table, then every reply."""
    lines = ["# Model evaluation", "", f"System prompt: {len(system_prompt)} characters.", ""]
    lines += [
        "| Model | OK | First token ms (median / min / max) | Total ms | Words | Cue first | Cue last | Markdown | Emoji | Cost per reply | Provider |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        med, low, high = s.first_token_ms
        if s.ok:
            lines.append(
                f"| `{s.model}` | {s.ok}/{s.ok + s.failed} | {med:.0f} / {low:.0f} / {high:.0f} | "
                f"{s.total_ms:.0f} | {s.words:.0f} | {_pct(s.starts_with_cue)} | "
                f"{_pct(s.ends_with_cue)} | {_pct(s.markdown)} | {_pct(s.emoji)} | "
                f"{_money(s.cost)} | {s.provider or '-'} |"
            )
        else:
            lines.append(f"| `{s.model}` | 0/{s.failed} | - | - | - | - | - | - | - | - | - |")
    errors = [(s.model, e) for s in summaries for e in s.errors]
    if errors:
        lines += ["", "## Errors", ""] + [f"- `{model}`: {error}" for model, error in errors]
    for index, prompt in enumerate(prompts):
        lines += ["", f"## Prompt {index + 1}", "", "> " + prompt.replace("\n", "\n> "), ""]
        for run in runs:
            if run.prompt != index or run.error:
                continue
            flags: list[str] = []
            if run.checks is not None:
                if not run.checks.starts_with_cue:
                    flags.append("no opening cue")
                if run.checks.ends_with_cue:
                    flags.append("ends on a cue")
                if run.checks.markdown:
                    flags.append("markdown")
                if run.checks.emoji:
                    flags.append("emoji")
            note = f" ({', '.join(flags)})" if flags else ""
            lines += [
                f"**`{run.model}`**, run {run.run + 1}, {run.first_token_ms:.0f} ms{note}",
                "",
                run.text,
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def _print_run(run: Run, prompts: int, runs: int) -> None:
    """Print one reply, with its timing and anything a voice would trip on."""
    where = f"prompt {run.prompt + 1}/{prompts}, run {run.run + 1}/{runs}"
    if run.error:
        print(f"\n[{run.model}] {where}: FAILED, {run.error}", flush=True)
        return
    notes: list[str] = []
    if run.checks is not None:
        if not run.checks.starts_with_cue:
            notes.append("no opening cue")
        if run.checks.ends_with_cue:
            notes.append("ends on a cue")
        if run.checks.markdown:
            notes.append("markdown")
        if run.checks.emoji:
            notes.append("emoji")
    flagged = f"  [{', '.join(notes)}]" if notes else ""
    paid = f", {_money(run.cost)}" if run.cost is not None else ""
    via = f", via {run.provider}" if run.provider else ""
    print(
        f"\n[{run.model}] {where}: first token {run.first_token_ms:.0f} ms, "
        f"total {run.total_ms:.0f} ms{paid}{via}{flagged}\n{run.text}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the same prompts to several chat models and compare the replies.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        action="append",
        default=[],
        metavar="TEXT",
        help="A user line to send. Repeatable.",
    )
    parser.add_argument(
        "--prompts-file", type=Path, help="Prompts separated by a line of dashes (---)."
    )
    parser.add_argument(
        "--system-file",
        type=Path,
        help="A character file. The voice rules follow it, as with fish-voice --prompt-file.",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=1,
        help="Repeats of each prompt on each model (default 1).",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Send one unmeasured request per model first, so connection setup is not timed.",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Completion cap. Default FISH_LLM_MAX_TOKENS."
    )
    parser.add_argument(
        "--temperature", type=float, help="Sampling temperature. Default FISH_LLM_TEMPERATURE."
    )
    parser.add_argument(
        "--reasoning-effort",
        help="none, minimal, low, medium, high or max. Default FISH_LLM_REASONING_EFFORT.",
    )
    parser.add_argument(
        "--provider-sort",
        help="OpenRouter latency, throughput, price or off. Default FISH_LLM_PROVIDER_SORT.",
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="Settings file (default .env)."
    )
    parser.add_argument(
        "--out", type=Path, help="Write a report: .md for Markdown, .json for the raw runs."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print each reply as it arrives, only the summary table.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run and make no requests."
    )
    return parser


def _tune(args: argparse.Namespace) -> LlmTune:
    """Build the LLM settings from the environment, then apply the flags."""
    tune = LlmTune.from_env()
    changes: dict[str, object] = {}
    if args.max_tokens is not None:
        changes["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        changes["temperature"] = args.temperature
    if args.reasoning_effort is not None:
        changes["reasoning_effort"] = (
            "" if args.reasoning_effort == "off" else args.reasoning_effort
        )
    if args.provider_sort is not None:
        changes["provider_sort"] = "" if args.provider_sort == "off" else args.provider_sort
    return replace(tune, **changes) if changes else tune


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation from the command line.

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
    models = list(dict.fromkeys(m.strip() for m in MODELS if m.strip()))
    if not models:
        print("No models to run. Fill in MODELS at the top of the script.", file=sys.stderr)
        return 2
    if args.runs < 1:
        print("--runs must be at least 1.", file=sys.stderr)
        return 2
    prompts = read_prompts(args.prompt, args.prompts_file)
    system_prompt = DEFAULT_SYSTEM_PROMPT
    seed: bool | None = None
    if args.system_file is not None:
        composed = system_prompt_from_file(str(args.system_file), DEFAULT_SYSTEM_PROMPT)
        if composed is None:
            print(f"Could not use {args.system_file} as a character file.", file=sys.stderr)
            return 2
        system_prompt, seed = composed, True
    tune = _tune(args)
    requests = len(models) * len(prompts) * args.runs
    print(
        f"{len(models)} models x {len(prompts)} prompts x {args.runs} runs = {requests} requests",
        file=sys.stderr,
    )
    if args.dry_run:
        for model in models:
            print(model)
        return 0
    if not tune.api_key:
        print(
            "No API key found. Set OPENROUTER_API_KEY or FISH_LLM_API_KEY, or pass --env-file.",
            file=sys.stderr,
        )
        return 2
    # The backend reports each reply's cost in a debug log line, so turn debug on and
    # read those lines. Everything else the logger says is dropped, except warnings.
    configure_voice_logging(debug=True)
    logger.remove()
    logger.add(_capture, level="DEBUG")
    runs = asyncio.run(
        evaluate(
            tune,
            models,
            prompts,
            system_prompt,
            runs=args.runs,
            seed=seed,
            warmup=args.warmup,
            on_run=None if args.quiet else lambda run: _print_run(run, len(prompts), args.runs),
        )
    )
    summaries = [summarize(m, [r for r in runs if r.model == m]) for m in models]
    if not args.quiet:
        print()
    print("\n".join(_table(summaries)))
    priced = sum(1 for r in runs if r.cost is not None)
    if priced:
        total = sum(s.total_cost for s in summaries)
        print(f"\nSpent ${total:.5f} across {priced} of {len(runs)} requests that reported a cost.")
    else:
        print("\nNo request reported a cost (only OpenRouter does).")
    for summary in summaries:
        for error in summary.errors:
            print(f"  {summary.model}: {error}", file=sys.stderr)
    if args.out is not None:
        if args.out.suffix == ".json":
            payload = {
                "summaries": [asdict(s) for s in summaries],
                "runs": [asdict(r) for r in runs],
            }
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            args.out.write_text(
                _markdown(summaries, runs, prompts, system_prompt), encoding="utf-8"
            )
        print(f"wrote {args.out}", file=sys.stderr)
    return 0 if any(s.ok for s in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
