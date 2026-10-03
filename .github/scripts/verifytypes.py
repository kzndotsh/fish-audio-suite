# ruff: noqa: INP001
# A standalone script run by path, not an importable package.
"""Fail when a package's public API is less fully typed than its floor allows.

``basedpyright --verifytypes`` scores how much of a package's public interface has
known types. It exits 1 for anything under 100 %, so this script reads the score from
its JSON output and compares it with the floor in ``.github/verifytypes-floors.json``.
CI and the ``justfile`` both run it, so the floors live in that one file.

Run it through uv so the dev tools are on the path::

    uv run python .github/scripts/verifytypes.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

FLOORS_FILE = Path(__file__).resolve().parent.parent / "verifytypes-floors.json"
# A score this far over its floor is a hint to raise the floor, never a failure.
RATCHET_MARGIN = 1.0


def score(package: str, basedpyright: str) -> float:
    """Measure one package's type completeness.

    Parameters
    ----------
    package : str
        Importable name of the package to score.
    basedpyright : str
        Path of the basedpyright executable.

    Returns
    -------
    float
        The completeness score as a percentage.

    Raises
    ------
    SystemExit
        When basedpyright prints no JSON report for the package.
    """
    # The argument list is fixed and no shell is involved.
    result = subprocess.run(  # noqa: S603
        [basedpyright, "--verifytypes", package, "--ignoreexternal", "--outputjson"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        report = json.loads(result.stdout)
        return float(report["typeCompleteness"]["completenessScore"]) * 100
    except (json.JSONDecodeError, KeyError, TypeError):
        detail = (result.stderr or result.stdout).strip()[:500]
        raise SystemExit(
            f"basedpyright gave no type completeness for {package}: {detail}"
        ) from None


def main() -> int:
    """Print the score table and compare each package with its floor.

    Returns
    -------
    int
        0 when every package meets its floor, 1 when any is below it.

    Raises
    ------
    SystemExit
        When basedpyright is not installed or a package cannot be scored.
    """
    basedpyright = shutil.which("basedpyright")
    if basedpyright is None:
        raise SystemExit("basedpyright is not on PATH. Run this through `uv run`.")
    floors: dict[str, float] = json.loads(FLOORS_FILE.read_text(encoding="utf-8"))
    rows = [(package, score(package, basedpyright), floor) for package, floor in floors.items()]

    width = max(len(package) for package, _, _ in rows)
    out = [f"{'package':<{width}}  {'score':>7}  {'floor':>7}  status"]
    failed = False
    for package, value, floor in rows:
        if value < floor:
            status, failed = "BELOW FLOOR", True
        elif value >= floor + RATCHET_MARGIN:
            status = f"ok, raise the floor to {math.floor(value * 10 + 1e-6) / 10:.1f}"
        else:
            status = "ok"
        out.append(f"{package:<{width}}  {value:>6.1f}%  {floor:>6.1f}%  {status}")
    sys.stdout.write("\n".join(out) + "\n")
    if failed:
        sys.stderr.write(
            "A public API lost type information. Annotate it, or lower the floor in "
            f"{FLOORS_FILE.name} only for a reviewed reason.\n"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
