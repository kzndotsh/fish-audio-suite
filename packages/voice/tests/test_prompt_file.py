from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT
from fish_audio_suite_voice.cli import main
from fish_audio_suite_voice.config import VoiceCliConfig, load_config, system_prompt_from_file
from fish_audio_suite_voice.history import opening_history

CARD = "You are Mira, a dry-witted ship's engineer. Stay in character."


@pytest.fixture(autouse=True)
def _clean_prompt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISH_VOICE_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("FISH_VOICE_SYSTEM_PROMPT_FILE", raising=False)


def _card(tmp_path: Path, text: str = CARD) -> Path:
    path = tmp_path / "mira.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_character_file_comes_first_and_keeps_the_voice_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT_FILE", str(_card(tmp_path, f"\n{CARD}\n\n")))
    config = load_config()
    assert config.system_prompt == f"{CARD}\n\n{DEFAULT_SYSTEM_PROMPT}"
    assert config.pin_seed


def test_a_character_file_still_pins_the_cue_example() -> None:
    history, pinned = opening_history(f"{CARD}\n\n{DEFAULT_SYSTEM_PROMPT}", seed=True)
    assert pinned == 3
    assert history[1]["role"] == "user"


def test_an_inline_prompt_is_used_whole_and_gets_no_cue_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "Answer in one word.")
    config = load_config()
    assert config.system_prompt == "Answer in one word."
    assert not config.pin_seed
    assert opening_history(config.system_prompt, seed=config.pin_seed)[1] == 1


def test_a_blank_inline_prompt_still_means_no_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "")
    assert load_config().system_prompt == ""


def test_the_file_wins_over_the_inline_prompt_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT_FILE", str(_card(tmp_path)))
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "ignored")
    assert load_config().system_prompt.startswith(CARD)
    assert "FISH_VOICE_SYSTEM_PROMPT is ignored" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda tmp: tmp / "missing.md", "cannot read prompt file"),
        (lambda tmp: _card(tmp, "  \n"), "is empty"),
        (lambda tmp: _card(tmp, "x" * (64 * 1024 + 1)), "over 64 KiB"),
    ],
    ids=["missing", "empty", "too-big"],
)
def test_an_unusable_file_warns_and_the_default_prompt_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make: Callable[[Path], Path],
    message: str,
) -> None:
    path = make(tmp_path)
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT_FILE", str(path))
    assert load_config().system_prompt == DEFAULT_SYSTEM_PROMPT
    assert message in capsys.readouterr().err


def test_a_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "latin.md"
    path.write_bytes(b"caf\xe9")
    assert system_prompt_from_file(str(path), DEFAULT_SYSTEM_PROMPT) is None


def test_a_home_relative_path_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _card(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert system_prompt_from_file("~/mira.md", "RULES") == f"{CARD}\n\nRULES"


def test_the_cli_flag_loads_the_file_and_a_bad_path_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    async def fake_loop(config: VoiceCliConfig) -> int:
        seen.append(config.system_prompt)
        return 0

    monkeypatch.setattr("fish_audio_suite_voice.cli.run_loop", fake_loop)
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_VOICE_ID", "v")
    assert main(["--env-file", str(tmp_path / ".env"), "--prompt-file", str(_card(tmp_path))]) == 0
    assert len(seen) == 1
    assert seen[0].startswith(CARD)
    assert seen[0].endswith(DEFAULT_SYSTEM_PROMPT)
    assert (
        main(["--env-file", str(tmp_path / ".env"), "--prompt-file", str(tmp_path / "no.md")]) == 2
    )


def test_a_directory_or_a_fifo_is_refused_without_reading_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert system_prompt_from_file(str(tmp_path), "RULES") is None
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    assert system_prompt_from_file(str(fifo), "RULES") is None
    assert capsys.readouterr().err.count("is not a regular file") == 2


def test_only_one_byte_past_the_limit_is_read(tmp_path: Path) -> None:
    big = tmp_path / "big.md"
    big.write_bytes(b"a" * (10 * 1024 * 1024))
    assert system_prompt_from_file(str(big), "RULES") is None
