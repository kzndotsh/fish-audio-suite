"""Rules the contributors' docs used to state in prose, checked mechanically.

Each test names the rule and the file to fix, so a failure explains itself.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from fish_audio_suite_proxy.settings import DEFAULT_GRACEFUL_S

ROOT = Path(__file__).resolve().parents[1]
KIT_SRC = ROOT / "packages" / "kit" / "src"
VOICE_SRC = ROOT / "packages" / "voice" / "src" / "fish_audio_suite_voice"


def _modules(base: Path) -> list[Path]:
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> list[tuple[str, int]]:
    """Return ``(module, line)`` for every import in a file, absolute imports only."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, node.lineno))
    return found


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
    offenders: list[str] = []
    for path in _modules(VOICE_SRC):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bos\.(environ|getenv)\b|\benviron\b", text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "read settings in config/tune and pass them down:\n" + "\n".join(
        offenders
    )
