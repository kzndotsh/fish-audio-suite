"""The settings tables must match the settings the code reads, with their defaults and ranges.

The settings are found by running the code that reads them, not by listing them here: voice
and the proxy load their settings in an empty environment while every reader call is
recorded with what it declares (the variable, its default, and its range), and every lookup
in ``os.environ`` is recorded too. A reader is any function in ``config.env`` or kit's ``env``
module whose first parameter is the variable name. The docs are parsed as text. Each test
compares the two and names the variable, the file and what each side says.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import TypeGuard

import pytest

import fish_audio_suite_kit as kit
from fish_audio_suite_proxy import settings as proxy_settings
from fish_audio_suite_voice.config import env as env_read
from fish_audio_suite_voice.config import settings as voice_config
from fish_audio_suite_voice.llm.tune import LLM_PROVIDERS
from fish_audio_suite_voice.logutil import debug as voice_debug

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("kit", "proxy", "voice")
README = {name: ROOT / "packages" / name / "README.md" for name in PACKAGES}
ROOT_README = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"
DEV_SH = ROOT / "packages" / "voice" / "dev.sh"

ENV_NAME = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
OURS = re.compile(r"^(FISH|OPENROUTER|OPENAI|EXPLABS|DEEPGRAM)_[A-Z0-9_]+$")
SHELL_REF = re.compile(r"\$\{?(FISH_[A-Z0-9_]+)")
TICKED = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")
ENV_LINE = re.compile(r"^(#\s*)?([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)=(.*)$", re.MULTILINE)
NUMBER = r"-?\d+(?:\.\d+)?"
RANGE_TO = re.compile(rf"(?<![\w.])({NUMBER}) to ({NUMBER})(?!\w|\.\d)")
RANGE_DASH = re.compile(r"(?<![\w.-])(\d+)-(\d+)(?![\w.-])")
AT_LEAST = re.compile(rf"\bat least ({NUMBER})", re.IGNORECASE)
ABOVE = re.compile(rf"\babove ({NUMBER})", re.IGNORECASE)
EITHER = re.compile(rf"(?<![\w.])({NUMBER}) or ({NUMBER})(?!\w|\.\d)")

# Settings read only when another one is set. Each environment below is loaded once, so
# every variable that can be read is read. A setting that needs a new one fails
# test_every_documented_variable_is_read_somewhere until its environment is added here.
ENVIRONMENTS: tuple[Mapping[str, str], ...] = (
    {},
    # The loudness target is read only when loudness is normalised.
    {"FISH_VOICE_NORMALIZE_LOUDNESS": "1"},
    # Each provider reads its own key and model variables.
    *({"FISH_LLM_PROVIDER": provider.name} for provider in LLM_PROVIDERS),
    # A server no provider owns reads OPENAI_API_KEY.
    {"FISH_LLM_BASE": "https://llm.example.test/v1"},
)

# Documented on purpose but not read through a reader, with the reason.
DOC_ONLY = {
    "LD_LIBRARY_PATH": "the loader variable the nix wrapper and dev.sh set for PortAudio",
    "NO_COLOR": "the standard switch, honoured by Textual and by the stderr debug sink",
    "TEXTUAL_ANIMATIONS": "Textual's own switch, which the app honours through animation_level",
}
# Variables the code reads that .env.example leaves out on purpose, with the reason.
# They are still documented in the README of the package that reads them.
ENV_EXAMPLE_SKIP = {
    "WEB_CONCURRENCY": "the standard uvicorn fallback for FISH_PROXY_WORKERS",
    "XDG_DATA_HOME": "the desktop's data folder, which voice saves chats under unless told",
}
# Strings that look like a variable name but are not one, with the reason. Empty today.
NOT_ENV: dict[str, str] = {}
# Defaults the docs describe in words instead of showing, with the reason.
DEFAULT_IN_WORDS = {
    "FISH_VOICE_SYSTEM_PROMPT": "the default prompt is pages long; the docs call it the kit default",
    "FISH_LLM_REFERER": "this project's URL; the README row covers three attribution fields",
    "FISH_LLM_TITLE": "this project's name; the README row covers three attribution fields",
    "FISH_LLM_CATEGORIES": "this project's category; the README row covers three fields",
}
# Ranges the docs state in words, because the code's rule is not one interval.
RANGE_IN_WORDS = {
    "FISH_VOICE_EAGER_EOT_THRESHOLD": "0 is off; 0 to 0.3 is raised to 0.3 with a warning",
    "FISH_VOICE_NOISE_SUPPRESSION": "only the steps 0, 6, 12, 18 and 21, listed by name",
}


@dataclass(frozen=True, slots=True)
class Read:
    """One reader call: the variable and what the call declares about it."""

    package: str
    name: str
    default: object = None
    lo: float | None = None
    hi: float | None = None
    positive: bool = False


def _is_reader(fn: object) -> TypeGuard[Callable[..., object]]:
    if not inspect.isfunction(fn):
        return False
    homes = {env_read.__name__, kit.env_int.__module__}
    params = list(inspect.signature(fn).parameters.values())
    return fn.__module__ in homes and bool(params) and params[0].name in {"name", "names"}


def _package(module: ModuleType) -> str:
    return module.__name__.split(".")[0].removeprefix("fish_audio_suite_")


class _Lookups(MutableMapping[str, str]):
    """An environment that records every name looked up in it."""

    def __init__(self, values: Mapping[str, str], seen: set[str]) -> None:
        self._values = dict(values)
        self._seen = seen

    def __getitem__(self, key: str) -> str:
        # get() and ``in`` both come here, through Mapping.
        self._seen.add(key)
        return self._values[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._values[key] = value

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _recording(fn: Callable[..., object], package: str, reads: list[Read]) -> Callable[..., object]:
    signature = inspect.signature(fn)

    def record(*args: object, **kwargs: object) -> object:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        given = bound.arguments
        names = given["names"] if "names" in given else (given["name"],)
        # A first-of-several read is empty when none is set.
        default = given.get("default", "" if "names" in given else None)
        reads.extend(
            Read(
                package,
                str(name),
                default,
                _number(given.get("lo")),
                _number(given.get("hi")),
                given.get("positive") is True,
            )
            for name in names
        )
        return fn(*args, **kwargs)

    return record


type _Setting = env_read.IntSetting | env_read.FloatSetting | env_read.ChoiceSetting[str]


def _recording_setting[S: _Setting](
    read: Callable[[S], object], reads: list[Read]
) -> Callable[[S], object]:
    """Record a declared setting's read: the record holds what a reader call would pass."""

    def record(setting: S) -> object:
        lo = hi = None
        if not isinstance(setting, env_read.ChoiceSetting):
            lo, hi = setting.lo, setting.hi
        positive = isinstance(setting, env_read.FloatSetting) and setting.positive
        reads.append(
            Read("voice", setting.name, setting.default, _number(lo), _number(hi), positive)
        )
        return read(setting)

    return record


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _load_everything() -> None:
    voice_config.load_config()
    voice_config.log_dir_from_env()
    voice_debug.debug_level()
    proxy_settings.load_settings()


@cache
def recorded() -> tuple[tuple[Read, ...], frozenset[str]]:
    """Run every settings entry point in each environment and return what was read."""
    reads: list[Read] = []
    seen: set[str] = set()
    readers: list[tuple[ModuleType, str, Callable[..., object]]] = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith(("fish_audio_suite_voice.", "fish_audio_suite_proxy.")):
            continue
        if name == env_read.__name__:
            continue
        for attr, value in list(vars(module).items()):
            if _is_reader(value):
                readers.append((module, attr, value))
    with pytest.MonkeyPatch.context() as patch:
        for module, attr, value in readers:
            patch.setattr(module, attr, _recording(value, _package(module), reads))
        # A setting a tune also checks in code is declared once and read through its record.
        patch.setattr(
            env_read.IntSetting, "read", _recording_setting(env_read.IntSetting.read, reads)
        )
        patch.setattr(
            env_read.FloatSetting, "read", _recording_setting(env_read.FloatSetting.read, reads)
        )
        choice_read = env_read.ChoiceSetting[str].read
        patch.setattr(env_read.ChoiceSetting, "read", _recording_setting(choice_read, reads))
        # Unfixed, the debug level is read from the environment on each call.
        voice_debug.reset_debug_level()
        for environment in ENVIRONMENTS:
            patch.setattr(os, "environ", _Lookups(environment, seen))
            _load_everything()
    return tuple(reads), frozenset(seen)


def read_by_package() -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for read in recorded()[0]:
        found[read.name].add(read.package)
    return found


def declared() -> dict[tuple[str, str], list[Read]]:
    """Reader calls per (package, variable), the ones that declare a default."""
    found: dict[tuple[str, str], list[Read]] = defaultdict(list)
    for read in recorded()[0]:
        found[read.package, read.name].append(read)
    return found


def _in_dunder_all(tree: ast.Module) -> set[int]:
    """Ids of the string nodes inside an ``__all__`` assignment, which are exports."""
    skip: set[int] = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            skip.update(id(n) for n in ast.walk(node))
    return skip


def scan_mentions() -> dict[str, set[str]]:
    """Variable-like string constants in the source, by name, with the files they are in."""
    mentioned: dict[str, set[str]] = defaultdict(set)
    for package in PACKAGES:
        for path in sorted((ROOT / "packages" / package / "src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            exports = _in_dunder_all(tree)
            for node in ast.walk(tree):
                if (
                    id(node) not in exports
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and OURS.match(node.value)
                ):
                    mentioned[node.value].add(str(path.relative_to(ROOT)))
    return mentioned


def documented() -> dict[str, set[str]]:
    """Variable names per documentation file: backticked in a README, or set in .env.example."""
    found: dict[str, set[str]] = {}
    for label, path in [("README.md", ROOT_README)] + [
        (f"packages/{n}/README.md", README[n]) for n in PACKAGES
    ]:
        found[label] = set(TICKED.findall(path.read_text(encoding="utf-8")))
    found[".env.example"] = {m[1] for m in ENV_LINE.findall(ENV_EXAMPLE.read_text("utf-8"))}
    return found


def shell_names() -> set[str]:
    return set(SHELL_REF.findall(DEV_SH.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class Row:
    """A settings-table row: the default column, and the whole row for ranges."""

    shown: str | bool | None
    text: str


def readme_rows(readme: Path) -> dict[str, Row]:
    """Each variable named in a table row's first cell, with the default that row shows.

    A row naming several variables pairs them with as many backticked defaults, or gives
    them all its one default. ``on`` and ``off`` are booleans; words are no default (None).
    """
    rows: dict[str, Row] = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 2:
            continue
        names = [n for n in TICKED.findall(cells[0]) if ENV_NAME.match(n)]
        if not names:
            continue
        lead = re.match(r"^`([^`]*)`((?:\s*,\s*`[^`]*`)*)", cells[1])
        values: list[str | bool | None]
        if lead:
            values = [lead[1], *re.findall(r"`([^`]*)`", lead[2])]
        elif re.match(r"^(on|off)\b", cells[1]):
            values = [cells[1].startswith("on")]
        else:
            values = [None]
        text = " ".join(cells[1:])
        for i, name in enumerate(names):
            shown = values[i] if len(values) == len(names) else values[0]
            rows.setdefault(name, Row(shown, text))
    return rows


def env_example_lines() -> dict[str, list[tuple[str, str]]]:
    """Each variable in .env.example with every (value, comment) it is shown with."""
    lines: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for match in ENV_LINE.finditer(ENV_EXAMPLE.read_text(encoding="utf-8")):
        value, _, comment = match[3].partition("#")
        lines[match[2]].append((value.strip(), comment.strip()))
    return lines


def _defaults(package: str, name: str) -> list[object]:
    """The defaults the code declares for one variable, without repeats. None is none."""
    found: list[object] = []
    for read in declared()[package, name]:
        if read.default is not None and read.default not in found:
            found.append(read.default)
    return found


def _words_for(default: object) -> set[str]:
    if isinstance(default, bool):
        return {"1", "true", "on", "yes"} if default else {"0", "false", "off", "no"}
    return set()


def _same(shown: str, default: object) -> bool:
    if isinstance(default, bool):
        return shown.lower() in _words_for(default)
    if isinstance(default, (int, float)):
        try:
            return float(shown.replace("_", "")) == float(default)
        except ValueError:
            return False
    return shown == default


READ = read_by_package()
MENTIONED = scan_mentions()
DOCS = documented()
EVERY_DOC_NAME = set[str]().union(*DOCS.values())


def test_the_recording_finds_the_settings() -> None:
    assert len(READ) > 80, f"the recorded reads found only {len(READ)} environment variables"
    for name in ("FISH_API_KEY", "FISH_PROXY_PORT", "FISH_VOICE_DEBUG", "OPENAI_API_KEY"):
        assert name in READ, name


def test_every_lookup_goes_through_a_reader() -> None:
    """A setting read straight from ``os.environ`` would escape every check below."""
    _, seen = recorded()
    loose = sorted(seen - set(READ))
    assert not loose, (
        "read from os.environ without an env_read or kit env reader, so its default and "
        "range are not declared: " + ", ".join(loose)
    )


def test_every_variable_the_code_reads_is_in_the_readme_of_the_package_that_reads_it() -> None:
    missing: list[str] = []
    for name, packages in sorted(READ.items()):
        for package in sorted(packages):
            own = DOCS[f"packages/{package}/README.md"]
            if name not in own and name not in DOCS["README.md"]:
                missing.append(
                    f"{name} is read by {package} but is not in packages/{package}/README.md"
                )
    assert not missing, "undocumented settings:\n  " + "\n  ".join(missing)


def test_every_variable_the_code_reads_is_in_env_example() -> None:
    missing = sorted(n for n in READ if n not in DOCS[".env.example"] and n not in ENV_EXAMPLE_SKIP)
    assert not missing, (
        ".env.example does not list these variables, which the code reads: "
        + ", ".join(missing)
        + ". Add each one, commented out, with its default."
    )


def test_every_documented_variable_is_read_somewhere() -> None:
    known = set(READ) | shell_names() | set(DOC_ONLY) | set(kit.__all__)
    stale: list[str] = []
    for label, names in sorted(DOCS.items()):
        stale.extend(
            f"{name} is documented in {label} but nothing reads it (a setting read only "
            "when another is set needs an entry in ENVIRONMENTS)"
            for name in sorted(names - known - set(MENTIONED))
        )
    assert not stale, "stale documentation:\n  " + "\n  ".join(stale)


def test_no_prefixed_name_is_left_unexplained() -> None:
    """A variable-like string in the source that nothing reads must be documented.

    If this fails, either the code reads a variable in a way the recording does not see
    (add an entry to ENVIRONMENTS), or the string is not a variable and belongs in NOT_ENV
    with a reason.
    """
    loose = sorted(
        f"{name} (in {', '.join(sorted(files))})"
        for name, files in MENTIONED.items()
        if name not in READ and name not in EVERY_DOC_NAME and name not in NOT_ENV
    )
    assert not loose, "unexplained variable-like strings: " + "; ".join(loose)


def _settings() -> Iterator[tuple[str, str]]:
    yield from sorted({(read.package, read.name) for read in recorded()[0]})


SETTINGS = list(_settings())


@pytest.mark.parametrize(("package", "name"), SETTINGS, ids=[f"{p}:{n}" for p, n in SETTINGS])
def test_the_readme_shows_the_default_the_code_declares(package: str, name: str) -> None:
    defaults = _defaults(package, name)
    if not defaults or name in DEFAULT_IN_WORDS:
        return
    assert len(defaults) == 1, f"{package} reads {name} with different defaults: {defaults}"
    default = defaults[0]
    readme = f"packages/{package}/README.md"
    row = readme_rows(README[package]).get(name)
    if row is None and default == "":
        return  # nothing shown to contradict; test_every_variable_... checks it is named
    assert row is not None, f"{name} has no settings-table row in {readme}"
    shown = row.shown
    if default == "":
        assert not isinstance(shown, str) or shown == "", (
            f"{readme} shows {name} defaulting to `{shown}`, but the code reads it as empty "
            "when unset; say so in words (unset, none)"
        )
        return
    if isinstance(shown, bool):
        shown = "on" if shown else "off"
    assert shown is not None, (
        f"{readme} describes the default of {name} in words, but the code's default is "
        f"{default!r}: show it, or add {name} to DEFAULT_IN_WORDS with the reason"
    )
    assert _same(shown, default), (
        f"{readme} says {name} defaults to {shown}, but the code's default is {default!r}"
    )


@pytest.mark.parametrize(("package", "name"), SETTINGS, ids=[f"{p}:{n}" for p, n in SETTINGS])
def test_env_example_shows_the_default_the_code_declares(package: str, name: str) -> None:
    defaults = _defaults(package, name)
    if not defaults or name in DEFAULT_IN_WORDS or name in ENV_EXAMPLE_SKIP:
        return
    lines = env_example_lines().get(name, [])
    default = defaults[0]
    if default == "":
        ok = any(value == "" or "unset" in comment for value, comment in lines)
    else:
        ok = any(_same(value, default) for value, _ in lines)
    shown = ", ".join(repr(value) for value, _ in lines) or "nothing"
    assert ok, (
        f".env.example shows {name} as {shown}, but no line has the code's default "
        f"{default!r} (an empty value, or a comment saying unset, for an empty default)"
    )


def _declared_range(name: str) -> tuple[float | None, float | None, bool] | None:
    reads = [r for r in declared()["voice", name] if r.lo is not None or r.hi is not None]
    reads += [r for r in declared()["voice", name] if r.positive]
    if not reads:
        return None
    first = reads[0]
    return first.lo, first.hi, first.positive


RANGED = [
    name
    for package, name in SETTINGS
    if package == "voice" and _declared_range(name) is not None and name not in RANGE_IN_WORDS
]


def _range_problems(where: str, name: str, text: str, *, complete: bool) -> list[str]:
    declared_range = _declared_range(name)
    assert declared_range is not None
    lo, hi, positive = declared_range
    code = (
        f"{lo:g} to {hi:g}"
        if lo is not None and hi is not None
        else f"at least {lo:g}"
        if lo is not None
        else f"at most {hi:g}"
        if hi is not None
        else "above 0"
    )
    problems: list[str] = []
    pairs = [(float(a), float(b)) for a, b in RANGE_TO.findall(text)]
    if where == ".env.example":
        pairs += [(float(a), float(b)) for a, b in RANGE_DASH.findall(text)]
    if lo is not None and hi is not None and hi - lo == 1:
        pairs += [(float(a), float(b)) for a, b in EITHER.findall(text)]
    problems.extend(
        f"{where} gives {name} the range {a:g} to {b:g}, but the code accepts {code}"
        for a, b in pairs
        if (a, b) != (lo, hi)
    )
    problems.extend(
        f"{where} says {name} is at least {least}, but the code accepts {code}"
        for least in AT_LEAST.findall(text)
        if float(least) != lo
    )
    problems.extend(
        f"{where} says {name} is above {above}, but the code accepts {code}"
        for above in ABOVE.findall(text)
        if not (positive and float(above) == 0)
    )
    if complete and lo is not None and hi is not None and (lo, hi) not in pairs:
        problems.append(f"{where} does not give the range of {name}, which is {code}")
    return problems


@pytest.mark.parametrize("name", RANGED)
def test_the_documented_range_is_the_range_the_code_declares(name: str) -> None:
    """Where the README or .env.example states a range, it is the code's.

    The README row must also state a two-sided range the code declares.
    """
    row = readme_rows(README["voice"]).get(name)
    assert row is not None, f"{name} has no settings-table row in packages/voice/README.md"
    problems = _range_problems("packages/voice/README.md", name, row.text, complete=True)
    comments = " ".join(comment for _, comment in env_example_lines().get(name, []))
    problems += _range_problems(".env.example", name, comments, complete=False)
    assert not problems, "\n".join(problems)
