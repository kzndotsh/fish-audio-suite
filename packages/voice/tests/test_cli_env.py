from __future__ import annotations

import threading
from pathlib import Path

import pytest

from fish_audio_suite_voice import cli as voice_cli
from fish_audio_suite_voice.cli import apply_cli_env_files, cfg, request_quit


def test_env_file_fills_missing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_API_KEY", raising=False)
    path = tmp_path / "voice.env"
    path.write_text(
        "FISH_VOICE_ID=vid-from-file\nexport FISH_API_KEY='key-from-file'\n", encoding="utf-8"
    )
    missing = tmp_path / "missing.env"
    loaded = apply_cli_env_files([path], required=True)
    assert path in loaded
    assert not missing.exists()
    assert cfg()["fish_voice_id"] == "vid-from-file"
    assert cfg()["fish_api_key"] == "key-from-file"


def test_process_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_VOICE_ID", "from-shell")
    path = tmp_path / "voice.env"
    path.write_text("FISH_VOICE_ID=from-file\n", encoding="utf-8")
    apply_cli_env_files([path], required=True)
    assert cfg()["fish_voice_id"] == "from-shell"


def test_first_file_wins_later_fills_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_ID", raising=False)
    monkeypatch.delenv("FISH_LLM_MODEL", raising=False)
    first = tmp_path / "a.env"
    first.write_text("FISH_VOICE_ID=a\n", encoding="utf-8")
    second = tmp_path / "b.env"
    second.write_text("FISH_VOICE_ID=b\nFISH_LLM_MODEL=openai/gpt-4o-mini\n", encoding="utf-8")
    apply_cli_env_files([first, second], required=True)
    c = cfg()
    assert c["fish_voice_id"] == "a"
    assert c["llm_model"] == "openai/gpt-4o-mini"


def test_request_quit_sets_mic_and_turn_cancel() -> None:
    stop = threading.Event()
    turn = threading.Event()
    llm = threading.Event()
    voice_cli.STOP_RECORD = stop
    voice_cli.TURN.cancel = turn
    voice_cli.TURN.llm_cancel = llm
    try:
        request_quit()
        assert stop.is_set()
        assert turn.is_set()
        assert llm.is_set()
    finally:
        voice_cli.STOP_RECORD = threading.Event()
        voice_cli.TURN.cancel = None
        voice_cli.TURN.llm_cancel = None
