"""The root conftest must not stand in the way of pytest's own ignore options."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def test_ignore_options_still_work_with_doctest_modules(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_kept="def test_kept():\n    pass\n", test_dropped="def test_dropped():\n    pass\n"
    )
    conftest = Path(__file__).resolve().parent.parent / "conftest.py"
    pytester.makeconftest(conftest.read_text(encoding="utf-8"))
    result = pytester.runpytest(
        "--collect-only", "-q", "--doctest-modules", "--ignore=test_dropped.py"
    )
    output = result.stdout.str()
    assert "test_kept.py::test_kept" in output
    assert "test_dropped" not in output
