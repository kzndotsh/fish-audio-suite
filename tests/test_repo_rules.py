"""Rules the contributors' docs used to state in prose, checked mechanically.

Each test names the rule and the file to fix, so a failure explains itself.
"""

from __future__ import annotations

import ast
import re
import sys
from functools import cache
from pathlib import Path

from fish_audio_suite_proxy.settings import DEFAULT_GRACEFUL_S

ROOT = Path(__file__).resolve().parents[1]
KIT_SRC = ROOT / "packages" / "kit" / "src"
KIT_PACKAGE = KIT_SRC / "fish_audio_suite_kit"
VOICE_SRC = ROOT / "packages" / "voice" / "src" / "fish_audio_suite_voice"


def _modules(base: Path) -> list[Path]:
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


@cache
def _kit_root_exports() -> frozenset[str]:
    tree = ast.parse((KIT_PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return frozenset(ast.literal_eval(node.value))
    return frozenset()


def _imports(path: Path) -> list[tuple[str, int]]:
    """Return ``(module, line)`` for every import in a file, absolute imports only.

    ``from pkg import name`` also reports ``pkg.name`` when that is a module on
    disk and the root does not export something of that name, because it then
    loads the submodule just as ``import pkg.name`` does. ``scrub_tts`` is both a
    module and a root function, and the root function wins.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, node.lineno))
            if node.module == "fish_audio_suite_kit":
                found.extend(
                    (f"fish_audio_suite_kit.{alias.name}", node.lineno)
                    for alias in node.names
                    if (KIT_PACKAGE / f"{alias.name}.py").exists()
                    and alias.name not in _kit_root_exports()
                )
    return found


def _reads_environment(path: Path) -> list[int]:
    """Return the lines where a file reads ``os.environ`` or calls ``os.getenv``.

    Aliases are followed (``import os as host``, ``from os import environ as env``),
    and a mention in a docstring or comment is not a read.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    os_names: set[str] = set()
    env_names: set[str] = set()
    getenv_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(a.asname or a.name for a in node.names if a.name == "os")
        elif isinstance(node, ast.ImportFrom) and node.module == "os" and node.level == 0:
            for alias in node.names:
                target = alias.asname or alias.name
                if alias.name in {"environ", "environb"}:
                    env_names.add(target)
                elif alias.name in {"getenv", "getenvb"}:
                    getenv_names.add(target)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in os_names and node.attr in {
                "environ",
                "environb",
                "getenv",
                "getenvb",
            }:
                lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id in env_names | getenv_names:
            lines.append(node.lineno)
    return lines


def test_the_justfile_and_the_ci_action_use_the_same_coverage_floors() -> None:
    """Keep ``justfile`` and ``.github/actions/gates/action.yml`` equal."""
    pattern = re.compile(
        r'(?:--include="(?P<what>[^"]+)" )?--fail-under=(?P<n>\d+)|--cov-fail-under=(?P<total>\d+)'
    )

    def floors(path: Path) -> dict[str, int]:
        out: dict[str, int] = {}
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match["total"]:
                out["total"] = int(match["total"])
            else:
                out[match["what"]] = int(match["n"])
        return out

    just = floors(ROOT / "justfile")
    action = floors(ROOT / ".github" / "actions" / "gates" / "action.yml")
    assert len(just) == 4, (
        f"expected a total and three package floors in the justfile, found {just}"
    )
    assert just == action, f"coverage floors differ: justfile {just}, gates action {action}"


def test_kit_imports_only_the_standard_library() -> None:
    """Kit has no runtime dependencies, so every import resolves to the stdlib or to kit."""
    allowed = set(sys.stdlib_module_names) | {"fish_audio_suite_kit"}
    offenders = [
        f"{path.relative_to(ROOT)}:{line} imports {module}"
        for path in _modules(KIT_SRC)
        for module, line in _imports(path)
        if module.split(".")[0] not in allowed
    ]
    assert not offenders, "kit must stay pure text and dependency-free:\n" + "\n".join(offenders)


def test_proxy_and_voice_import_kit_only_through_its_root() -> None:
    """A submodule or private kit import breaks the curated, golden-checked API."""
    offenders = [
        f"{path.relative_to(ROOT)}:{line} imports {module}"
        for package in ("proxy", "voice")
        for path in _modules(ROOT / "packages" / package)
        if ".venv" not in path.parts
        for module, line in _imports(path)
        if module.startswith("fish_audio_suite_kit.")
    ]
    assert not offenders, "import from fish_audio_suite_kit, never a submodule:\n" + "\n".join(
        offenders
    )


def test_the_nixos_module_default_matches_the_proxy_graceful_shutdown() -> None:
    """The module's drain time must equal ``DEFAULT_GRACEFUL_S``; its timeouts add 10 s to it."""
    nix = (ROOT / "nix" / "module.nix").read_text(encoding="utf-8")
    block = nix[nix.index("gracefulShutdownSeconds = lib.mkOption") :]
    default = re.search(r"default = (\d+);", block)
    assert default, "could not find the gracefulShutdownSeconds default in nix/module.nix"
    assert int(default[1]) == DEFAULT_GRACEFUL_S


def test_voice_reads_the_environment_only_in_its_settings_layer() -> None:
    """Settings come from ``config`` and the ``*.from_env`` tunes; the rest take arguments."""
    allowed = {"config.py", "tune.py", "llm_tune.py", "debug.py", "envfile.py"}
    offenders = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in _modules(VOICE_SRC)
        if path.name not in allowed
        for line in _reads_environment(path)
    ]
    assert not offenders, "read settings in config/tune and pass them down:\n" + "\n".join(
        offenders
    )


def _is_terminal_write(node: ast.Call | ast.Attribute) -> bool:
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Name) and node.func.id == "print"
    return (
        isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr in {"stdout", "stderr"}
    )


def _writes_to_the_terminal(path: Path) -> list[int]:
    """Return the lines that call ``print`` or touch ``sys.stdout`` or ``sys.stderr``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call | ast.Attribute) and _is_terminal_write(node)
    )


def test_the_voice_session_reports_through_events_and_never_writes_to_the_terminal() -> None:
    """A full-screen display owns the terminal, so session code reports through ``events``.

    Only the modules that make up the plain display, the startup in ``cli`` and the raw
    stdout audio sink may write to a stream.
    """
    allowed = {"console.py", "console_sink.py", "debug.py", "cli.py", "playback.py"}
    offenders = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in _modules(VOICE_SRC)
        if path.name not in allowed
        for line in _writes_to_the_terminal(path)
    ]
    assert not offenders, (
        "emit an event (events.notice, or a new event) instead of writing to the terminal; "
        "console_sink prints it:\n" + "\n".join(offenders)
    )
