"""Minimal dotenv loader for ``fish-voice --env-file``. No new dependency.

Handles ``KEY=VAL``, ``export KEY=VAL``, single and double quotes, a trailing
`` # comment``, double-quoted escapes, a BOM, and a quoted value that spans
lines. Existing process environment always wins.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from fish_audio_suite_voice.debug import warn

__all__ = [
    "apply_cli_env_files",
    "load_dotenv",
]

_EXPORT_WORD: Final = "export"


def _quoted_span(val: str) -> tuple[str, str] | None:
    # The closer is the next unescaped quote, not the last character.
    # "sk" # "old" used to keep the comment because the line also ended
    # on a quote. \" inside the value is not that closer.
    if not val or val[0] not in {'"', "'"}:
        return None
    quote = val[0]
    index = 1
    while index < len(val):
        # A single-quoted value is literal. "C:\temp\" used to skip the
        # closer, so the next key was swallowed and the voice id was empty.
        if quote == '"' and val[index] == "\\" and index + 1 < len(val):
            index += 2
            continue
        if val[index] == quote:
            return val[1:index], val[index + 1 :]
        index += 1
    return None


def _unescape_double(inner: str) -> str:
    # "Say \"hi\"\nthere" is a prompt, not the letters backslash and n.
    out: list[str] = []
    index = 0
    while index < len(inner):
        if inner[index] == "\\" and index + 1 < len(inner):
            nxt = inner[index + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            else:
                out.append(inner[index : index + 2])
            index += 2
            continue
        out.append(inner[index])
        index += 1
    return "".join(out)


def _env_value(raw: str) -> str:
    """Unquote a dotenv value. An unquoted ` #` starts a comment."""
    val = raw.strip()
    span = _quoted_span(val)
    if span is not None:
        inner, rest = span
        rest = rest.lstrip()
        if not rest or rest.startswith("#"):
            if val[0] == '"':
                return _unescape_double(inner)
            return inner
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
        return val[1:-1]
    hashed = val.find(" #")
    if hashed >= 0:
        val = val[:hashed].rstrip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
        return val[1:-1]
    return val


def _without_export(line: str) -> str:
    # `export KEY=value` is a shell prefix. A tab after export is legal and
    # must not become part of the key name.
    if (
        line.startswith(_EXPORT_WORD)
        and len(line) > len(_EXPORT_WORD)
        and line[len(_EXPORT_WORD)].isspace()
    ):
        return line[len(_EXPORT_WORD) :].strip()
    return line


def _unclosed_quote(stripped: str) -> str | None:
    # "You are helpful. keeps going on the next line. A one-line "key" is
    # already balanced, including "key # not a comment".
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    _, _, val = _without_export(stripped).partition("=")
    val = val.lstrip()
    if val[:1] not in {'"', "'"}:
        return None
    if _quoted_span(val) is None:
        return val[0]
    return None


def _logical_lines(text: str) -> list[str]:
    raw_lines = text.splitlines()
    folded: list[str] = []
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        quote = _unclosed_quote(line.strip())
        if quote is None:
            folded.append(line)
            index += 1
            continue
        parts = [line]
        index += 1
        while index < len(raw_lines):
            parts.append(raw_lines[index])
            index += 1
            # \" counts as a quote character, so a raw count closes too early
            # and the next line of the prompt is dropped.
            if _unclosed_quote("\n".join(parts).strip()) is None:
                break
        folded.append("\n".join(parts))
    return folded


def load_dotenv(path: Path) -> bool:
    """Fill ``os.environ`` from ``KEY=VAL`` lines. Existing keys win.

    Parameters
    ----------
    path : Path
        The dotenv file.

    Returns
    -------
    bool
        Whether the file was read.
    """
    if not path.is_file():
        return False
    try:
        # utf-8-sig drops a leading BOM. Left in place it sticks to the first key.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        warn(f"fish-voice: env file is not utf-8: {path}")
        return False
    except OSError as exc:
        warn(f"fish-voice: could not read env file {path}: {exc.strerror}")
        return False
    for raw in _logical_lines(text):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = _without_export(line)
        key, _, val = line.partition("=")
        key = key.strip()
        val = _env_value(val)
        if key and key not in os.environ:
            try:
                os.environ[key] = val
            except ValueError:
                # os.environ rejects an embedded NUL. Skip that key, keep the rest.
                warn(f"fish-voice: skipped {key!r} in {path}: value or name has a NUL byte")
    return True


def apply_cli_env_files(paths: list[Path], *, required: bool) -> list[Path]:
    """Load dotenv files in order. First file wins per key. Process env already wins.

    Parameters
    ----------
    paths : list of Path
        Files to load, in priority order.
    required : bool
        Warn when a file is missing.

    Returns
    -------
    list of Path
        The files that were read.
    """
    loaded: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            resolved = expanded.resolve()
        except OSError:
            resolved = expanded
        if resolved in seen:
            continue
        if not expanded.is_file():
            if required:
                warn(f"fish-voice: --env-file not found: {path}")
            continue
        if load_dotenv(expanded):
            seen.add(resolved)
            loaded.append(path)
    return loaded
