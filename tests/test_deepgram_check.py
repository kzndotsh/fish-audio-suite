"""``scripts/deepgram_check.py``: the parts that do not need a mic or a Deepgram key."""

from __future__ import annotations

import asyncio

import pytest
from deepgram_check import main, run


def test_the_script_explains_itself_without_opening_the_mic_or_the_network(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "--turns" in capsys.readouterr().out


def test_a_missing_key_is_reported_before_anything_is_opened(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert asyncio.run(run(1, None)) == 2
    assert "DEEPGRAM_API_KEY" in capsys.readouterr().err
