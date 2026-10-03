"""Contract test for the kit's public API.

Proxy and voice depend on ``fish_audio_suite_kit`` through a pinned range, so a
change to its root exports or to the signatures they call is a breaking change.
The golden file records both. A failure here means the API moved. If the move is
intended, bump the kit version, then regenerate the file with::

    UPDATE_GOLDEN=1 uv run pytest packages/kit/tests/test_api_contract.py

and review the diff of ``golden/kit_api.json`` like any other API change.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import re
from collections.abc import Sized
from pathlib import Path
from typing import Any

import fish_audio_suite_kit as kit

PACKAGES = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden" / "kit_api.json"
CONSUMERS = ("proxy", "voice")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_SHORT = 60


def _kit_imports(source_dir: Path) -> set[str]:
    """Names a package imports from the kit root, found with the AST."""
    names: set[str] = set()
    for path in source_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "fish_audio_suite_kit":
                names.update(alias.name for alias in node.names)
    return names


def _describe(value: object) -> str:
    """A stable one-line description of an exported name."""
    if inspect.isclass(value):
        fields = (
            f" fields={[f.name for f in dataclasses.fields(value)]}"
            if dataclasses.is_dataclass(value)
            else ""
        )
        return f"class{_signature(value)}{fields}"
    if callable(value):
        return f"def{_signature(value)}"
    if isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= _SHORT):
        return f"{type(value).__name__}={value!r}"
    # Prompts and other long text or collections are tuned freely. Their type and
    # size are the contract, not their contents.
    if isinstance(value, str):
        return "str"
    size = len(value) if isinstance(value, Sized) else "?"
    return f"{type(value).__name__}[len={size}]"


_FORWARD_REF = re.compile(r"ForwardRef\('([^']*)'\)")


def _signature(value: Any) -> str:
    """The signature as text that does not depend on the Python version.

    NamedTuple fields render as ``ForwardRef('float')`` on 3.12 and as ``float``
    on 3.14, and string annotations keep their quotes, so both are reduced to the
    bare annotation text. A changed name, order, default or annotation still shows.
    """
    try:
        text = _ADDRESS.sub("", str(inspect.signature(value)))
    except (TypeError, ValueError):
        return "(...)"
    return _FORWARD_REF.sub(r"\1", text).replace("'", "")


def snapshot() -> dict[str, Any]:
    """The kit's root exports and the shape of every name proxy and voice import."""
    used: set[str] = set()
    for consumer in CONSUMERS:
        used |= _kit_imports(PACKAGES / consumer / "src")
    return {
        "exports": sorted(kit.__all__),
        "used_by_proxy_and_voice": {name: _describe(getattr(kit, name)) for name in sorted(used)},
    }


def test_every_name_proxy_and_voice_import_is_a_root_export() -> None:
    used: set[str] = set()
    for consumer in CONSUMERS:
        used |= _kit_imports(PACKAGES / consumer / "src")
    missing = sorted(name for name in used if name not in kit.__all__)
    assert not missing, f"imported from the kit root but not in kit.__all__: {missing}"


def test_the_kit_api_matches_the_golden_file() -> None:
    current = snapshot()
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    recorded = json.loads(GOLDEN.read_text(encoding="utf-8"))
    removed = sorted(set(recorded["exports"]) - set(current["exports"]))
    added = sorted(set(current["exports"]) - set(recorded["exports"]))
    changed = sorted(
        name
        for name, shape in current["used_by_proxy_and_voice"].items()
        if recorded["used_by_proxy_and_voice"].get(name) != shape
    )
    dropped = sorted(
        set(recorded["used_by_proxy_and_voice"]) - set(current["used_by_proxy_and_voice"])
    )
    problems: list[str] = []
    if removed:
        problems.append(f"removed from kit.__all__ (breaking): {removed}")
    if added:
        problems.append(f"added to kit.__all__: {added}")
    if changed:
        problems.append(f"signature or value changed for names proxy/voice use: {changed}")
    if dropped:
        problems.append(f"no longer imported by proxy or voice: {dropped}")
    assert not problems, (
        "The kit's public API changed:\n  "
        + "\n  ".join(problems)
        + "\nIf this is intended, bump the kit version and regenerate with "
        "UPDATE_GOLDEN=1 (see the docstring of this file)."
    )
