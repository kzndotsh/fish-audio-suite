"""The settings tables must match the environment variables the code reads.

The code is scanned with the AST, the docs are parsed as text, and the two sets
are compared both ways. A failure names the variable and the file to fix.
"""

from __future__ import annotations

import ast
import importlib
import re
from collections import defaultdict
from pathlib import Path

import pytest

import fish_audio_suite_kit as kit

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("kit", "proxy", "voice")
README = {name: ROOT / "packages" / name / "README.md" for name in PACKAGES}
ROOT_README = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"
DEV_SH = ROOT / "packages" / "voice" / "dev.sh"

ENV_NAME = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
OURS = re.compile(r"^(FISH|OPENROUTER|OPENAI|EXPLABS)_[A-Z0-9_]+$")
# Helpers that read an environment variable. A string argument is its name. If
# someone adds a reader with another name, the guard in
# test_no_prefixed_name_is_left_unexplained fails and says to add it here.
READER_CALL = re.compile(
    r"^(env_\w+|read_\w+|_first_\w+|_existing|_non_negative|_positive_float|_positive_int"
    r"|_raw|LlmProvider)$"
)
SHELL_REF = re.compile(r"\$\{?(FISH_[A-Z0-9_]+)")
TICKED = re.compile(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`")
ENV_LINE = re.compile(r"^#?\s*([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*=", re.MULTILINE)

# Documented on purpose but not read through the helpers above, with the reason.
DOC_ONLY = {
    "LD_LIBRARY_PATH": "the loader variable the nix wrapper and dev.sh set for PortAudio",
}
# Variables read through a helper that .env.example leaves out on purpose, with the reason.
# They are still documented in the README of the package that reads them.
ENV_EXAMPLE_SKIP = {
    "OPENROUTER_BASE_URL": "fallback name for FISH_LLM_BASE",
    "OPENROUTER_MODEL": "fallback name for FISH_LLM_MODEL",
    "WEB_CONCURRENCY": "the standard uvicorn fallback for FISH_PROXY_WORKERS",
}
# Strings that look like a variable name but are not one, with the reason. Empty today.
NOT_ENV: dict[str, str] = {}

# Where each default lives: variable -> (module, constant, README that owns the row).
DEFAULTS = {
    "FISH_PROXY_HOST": ("fish_audio_suite_proxy.settings", "DEFAULT_HOST", "proxy"),
    "FISH_PROXY_PORT": ("fish_audio_suite_proxy.settings", "DEFAULT_PORT", "proxy"),
    "FISH_PROXY_MAX_BODY_BYTES": (
        "fish_audio_suite_proxy.settings",
        "DEFAULT_MAX_BODY_BYTES",
        "proxy",
    ),
    "FISH_PROXY_MAX_INPUT_CHARS": (
        "fish_audio_suite_proxy.settings",
        "DEFAULT_MAX_INPUT_CHARS",
        "proxy",
    ),
    "FISH_PROXY_CONNECT_TIMEOUT": ("fish_audio_suite_proxy.settings", "DEFAULT_CONNECT_S", "proxy"),
    "FISH_PROXY_READ_TIMEOUT": ("fish_audio_suite_proxy.settings", "DEFAULT_READ_S", "proxy"),
    "FISH_PROXY_POOL_TIMEOUT": ("fish_audio_suite_proxy.settings", "DEFAULT_POOL_S", "proxy"),
    "FISH_PROXY_RETRY_DEADLINE": (
        "fish_audio_suite_proxy.settings",
        "DEFAULT_RETRY_DEADLINE_S",
        "proxy",
    ),
    "FISH_PROXY_ASR_TIMEOUT": (
        "fish_audio_suite_proxy.settings",
        "DEFAULT_ASR_TIMEOUT_S",
        "proxy",
    ),
    "FISH_PROXY_KEEP_ALIVE": ("fish_audio_suite_proxy.settings", "DEFAULT_KEEP_ALIVE_S", "proxy"),
    "FISH_PROXY_GRACEFUL_SHUTDOWN": (
        "fish_audio_suite_proxy.settings",
        "DEFAULT_GRACEFUL_S",
        "proxy",
    ),
    "FISH_PROXY_RETRY_ATTEMPTS": ("fish_audio_suite_kit", "FISH_RETRY_ATTEMPTS", "proxy"),
    "FISH_VOICE_HISTORY_TURNS": (
        "fish_audio_suite_voice.llm_tune",
        "DEFAULT_HISTORY_TURNS",
        "voice",
    ),
    "FISH_VOICE_REPEAT_WINDOW": (
        "fish_audio_suite_voice.config",
        "DEFAULT_REPEAT_WINDOW_S",
        "voice",
    ),
    "FISH_LLM_TIMEOUT": ("fish_audio_suite_voice.llm_tune", "DEFAULT_LLM_TIMEOUT_S", "voice"),
    "FISH_LLM_MAX_TOKENS": ("fish_audio_suite_voice.llm_tune", "DEFAULT_LLM_MAX_TOKENS", "voice"),
    "FISH_LLM_TEMPERATURE": ("fish_audio_suite_voice.llm_tune", "DEFAULT_LLM_TEMPERATURE", "voice"),
    "FISH_LLM_PROVIDER_SORT": (
        "fish_audio_suite_voice.llm_tune",
        "DEFAULT_LLM_PROVIDER_SORT",
        "voice",
    ),
    "FISH_VOICE_BARGE_FRAMES": ("fish_audio_suite_voice.tune", "DEFAULT_BARGE_HIT_FRAMES", "voice"),
    "FISH_VOICE_BARGE_RMS": ("fish_audio_suite_voice.tune", "DEFAULT_BARGE_RMS", "voice"),
    "FISH_VOICE_BARGE_PLAYING_GAIN": (
        "fish_audio_suite_voice.tune",
        "DEFAULT_BARGE_PLAYING_GAIN",
        "voice",
    ),
    "FISH_VOICE_COOLDOWN": (
        "fish_audio_suite_voice.tune",
        "DEFAULT_POST_SPEAK_COOLDOWN_S",
        "voice",
    ),
    "FISH_VOICE_BLEED_DELAY": ("fish_audio_suite_voice.tune", "DEFAULT_BLEED_DELAY_S", "voice"),
    "FISH_VOICE_AEC_WET": ("fish_audio_suite_voice.tune", "DEFAULT_AEC_WET", "voice"),
    "FISH_VOICE_AEC_BLEED_DELAY": (
        "fish_audio_suite_voice.tune",
        "DEFAULT_AEC_BLEED_DELAY_S",
        "voice",
    ),
    "FISH_VOICE_SILENCE_FRAMES": (
        "fish_audio_suite_voice.tune",
        "DEFAULT_END_SILENCE_FRAMES",
        "voice",
    ),
    "FISH_VOICE_SPEECH_FRAMES": (
        "fish_audio_suite_voice.tune",
        "DEFAULT_START_SPEECH_FRAMES",
        "voice",
    ),
    "FISH_VOICE_MIN_RMS": ("fish_audio_suite_voice.tune", "DEFAULT_MIN_SPEECH_RMS", "voice"),
    "FISH_VOICE_MIN_VOICED_FRAMES": (
        "fish_audio_suite_voice.tune",
        "DEFAULT_MIN_VOICED_FRAMES",
        "voice",
    ),
    "FISH_VOICE_PRE_PAD_FRAMES": ("fish_audio_suite_voice.tune", "DEFAULT_PRE_PAD_FRAMES", "voice"),
    "FISH_VOICE_VAD": ("fish_audio_suite_voice.tune", "DEFAULT_VAD_AGGRESSIVENESS", "voice"),
}


def _string_constants(node: ast.AST) -> list[str]:
    """String literals directly in ``node`` or inside a tuple or list held by it."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [text for item in node.elts for text in _string_constants(item)]
    return []


def _is_environ_read(call: ast.Call) -> bool:
    """True for ``os.environ.get(...)``, ``environ.get(...)`` and ``os.getenv(...)``."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "getenv":
        return True
    owner = func.value
    named_environ = isinstance(owner, ast.Attribute) and owner.attr == "environ"
    return func.attr == "get" and (
        named_environ or (isinstance(owner, ast.Name) and owner.id == "environ")
    )


def _in_dunder_all(tree: ast.Module) -> set[int]:
    """Ids of the string nodes inside an ``__all__`` assignment, which are exports."""
    skip: set[int] = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            skip.update(id(n) for n in ast.walk(node))
    return skip


def scan_sources() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return (name -> packages that read it, name -> files that mention it)."""
    read: dict[str, set[str]] = defaultdict(set)
    mentioned: dict[str, set[str]] = defaultdict(set)
    for package in PACKAGES:
        for path in sorted((ROOT / "packages" / package / "src").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            exports = _in_dunder_all(tree)
            for node in ast.walk(tree):
                if id(node) in exports:
                    continue
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and OURS.match(node.value)
                ):
                    mentioned[node.value].add(str(path.relative_to(ROOT)))
                if isinstance(node, ast.Call):
                    callee = node.func
                    name = (
                        callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", "")
                    )
                    if READER_CALL.match(name) or _is_environ_read(node):
                        for arg in node.args:
                            for text in _string_constants(arg):
                                if ENV_NAME.match(text):
                                    read[text].add(package)
    return read, mentioned


def documented() -> dict[str, set[str]]:
    """Variable names per documentation file: backticked in a README, or set in .env.example."""
    found: dict[str, set[str]] = {}
    for label, path in [("README.md", ROOT_README)] + [
        (f"packages/{n}/README.md", README[n]) for n in PACKAGES
    ]:
        found[label] = set(TICKED.findall(path.read_text(encoding="utf-8")))
    found[".env.example"] = set(ENV_LINE.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))
    return found


def shell_names() -> set[str]:
    return set(SHELL_REF.findall(DEV_SH.read_text(encoding="utf-8")))


READ, MENTIONED = scan_sources()
DOCS = documented()
EVERY_DOC_NAME = set[str]().union(*DOCS.values())


def test_the_scan_finds_the_settings() -> None:
    assert len(READ) > 40, f"the AST scan found only {len(READ)} environment variables"
    assert "FISH_API_KEY" in READ
    assert "FISH_PROXY_PORT" in READ


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
    known = set(READ) | set(MENTIONED) | shell_names() | set(DOC_ONLY) | set(kit.__all__)
    stale: list[str] = []
    for label, names in sorted(DOCS.items()):
        stale.extend(
            f"{name} is documented in {label} but nothing reads it"
            for name in sorted(names - known)
        )
    assert not stale, "stale documentation:\n  " + "\n  ".join(stale)


def test_no_prefixed_name_is_left_unexplained() -> None:
    """A FISH_, OPENROUTER_ or OPENAI_ string the scan did not see read must be documented.

    If this fails, either the code reads a variable through a helper that
    READER_CALL does not list, or the string is not a variable and belongs in
    NOT_ENV with a reason.
    """
    loose = sorted(
        f"{name} (in {', '.join(sorted(files))})"
        for name, files in MENTIONED.items()
        if name not in READ and name not in EVERY_DOC_NAME and name not in NOT_ENV
    )
    assert not loose, "unexplained variable-like strings: " + "; ".join(loose)


def _readme_default(readme: Path, name: str) -> str | None:
    """The first backticked value in the second column of ``name``'s table row."""
    row = re.search(
        rf"^\|\s*`{re.escape(name)}`\s*\|\s*`([^`]*)`",
        readme.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return row.group(1) if row else None


def _same_value(documented_value: str, actual: object) -> bool:
    if isinstance(actual, bool):
        return documented_value.lower() in {str(actual).lower(), "on" if actual else "off"}
    if isinstance(actual, (int, float)):
        try:
            return float(documented_value.replace("_", "")) == float(actual)
        except ValueError:
            return False
    return documented_value == str(actual)


@pytest.mark.parametrize("name", sorted(DEFAULTS))
def test_the_documented_default_matches_the_constant(name: str) -> None:
    module_name, constant, owner = DEFAULTS[name]
    actual = getattr(importlib.import_module(module_name), constant)
    shown = _readme_default(README[owner], name)
    assert shown is not None, (
        f"{name} has no row with a backticked default in packages/{owner}/README.md, "
        f"but {module_name}.{constant} is its default"
    )
    assert _same_value(shown, actual), (
        f"packages/{owner}/README.md says {name} defaults to {shown}, "
        f"but {module_name}.{constant} is {actual!r}"
    )
