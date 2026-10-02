"""Pytest fixtures shared by every package."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


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
