"""Pytest fixtures shared by every package."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Only the kit documents its API with runnable Examples. With --doctest-modules on,
# pytest would otherwise import every proxy and voice module just to find nothing.
_NO_DOCTESTS = ("proxy", "voice")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Keep the proxy and voice sources out of doctest collection.

    Parameters
    ----------
    collection_path : Path
        The file or directory pytest is about to collect.
    config : pytest.Config
        The run configuration.

    Returns
    -------
    bool or None
        True for a file or directory under ``packages/proxy/src`` or
        ``packages/voice/src`` while ``--doctest-modules`` is on, so nothing there
        is imported. None leaves every other path to pytest.
    """
    if not config.option.doctestmodules:
        return None
    return any(
        collection_path.is_relative_to(config.rootpath / "packages" / name / "src")
        for name in _NO_DOCTESTS
    )


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Give each test the environment it started with.

    ``apply_cli_env_files`` and similar code write to ``os.environ`` directly.
    ``monkeypatch.delenv(..., raising=False)`` records nothing to restore when the
    key was absent, so those values would otherwise reach later tests.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)
