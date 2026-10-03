from __future__ import annotations

from collections.abc import Iterator

import pytest

from fish_audio_suite_proxy import settings


@pytest.fixture(autouse=True)
def _reset_rename_warnings() -> Iterator[None]:
    """Clear the once-per-process rename warnings so test order cannot matter."""
    settings._warn.cache_clear()
    yield
    settings._warn.cache_clear()
