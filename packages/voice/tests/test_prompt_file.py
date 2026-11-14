from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest

from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT
from fish_audio_suite_voice.cli.main import main
from fish_audio_suite_voice.config.settings import (
    VoiceCliConfig,
    load_config,
    system_prompt_from_file,
)
from fish_audio_suite_voice.session.conversation import Conversation, HistoryView

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
    assert config.prompt.text == f"{CARD}\n\n{DEFAULT_SYSTEM_PROMPT}"
    assert config.prompt.pin_seed


def test_a_character_file_still_pins_the_cue_example() -> None:
    opened = Conversation.opening(f"{CARD}\n\n{DEFAULT_SYSTEM_PROMPT}", seed=True)
    assert [m["role"] for m in HistoryView(1).messages(opened)] == ["system", "user", "assistant"]


def test_an_inline_prompt_is_used_whole_and_gets_no_cue_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "Answer in one word.")
    config = load_config()
    assert config.prompt.text == "Answer in one word."
    assert not config.prompt.pin_seed
    opened = Conversation.opening(config.prompt.text, seed=config.prompt.pin_seed)
    assert HistoryView(1).messages(opened) == [{"role": "system", "content": "Answer in one word."}]


def test_a_blank_inline_prompt_still_means_no_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "")
    assert load_config().prompt.text == ""


def test_the_file_wins_over_the_inline_prompt_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT_FILE", str(_card(tmp_path)))
    monkeypatch.setenv("FISH_VOICE_SYSTEM_PROMPT", "ignored")
    assert load_config().prompt.text.startswith(CARD)
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
    assert load_config().prompt.text == DEFAULT_SYSTEM_PROMPT
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

    async def fake_loop(config: VoiceCliConfig, **_kwargs: object) -> int:
        seen.append(config.prompt.text)
        return 0

    monkeypatch.setattr("fish_audio_suite_voice.cli.main.run_loop", fake_loop)
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


class _CountingFile:
    """A binary file that counts the bytes read from it."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.bytes_read = 0

    def __enter__(self) -> _CountingFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self.bytes_read += len(data)
        return data


def test_only_one_byte_past_the_limit_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    big = tmp_path / "big.md"
    big.write_bytes(b"a" * (10 * 1024 * 1024))
    opened: list[_CountingFile] = []
    original = Path.open

    def counting_open(self: Path, mode: str = "r") -> _CountingFile:
        assert mode == "rb"
        counted = _CountingFile(original(self, "rb"))
        opened.append(counted)
        return counted

    monkeypatch.setattr(Path, "open", counting_open)
    assert system_prompt_from_file(str(big), "RULES") is None
    assert [file.bytes_read for file in opened] == [64 * 1024 + 1]
    assert "is over 64 KiB" in capsys.readouterr().err


def test_a_slot_in_the_file_puts_the_voice_rules_where_the_file_says(tmp_path: Path) -> None:
    path = _card(tmp_path, "{{default_prompt}}\n\nNow the character: Mira.")
    assert system_prompt_from_file(str(path), "RULES") == "RULES\n\nNow the character: Mira."
    middle = _card(tmp_path, "Mira is dry.\n{{ default_prompt }}\nShe never rambles.")
    assert (
        system_prompt_from_file(str(middle), "RULES") == "Mira is dry.\nRULES\nShe never rambles."
    )


def test_a_file_without_the_slot_still_gets_the_rules_after_it(tmp_path: Path) -> None:
    assert system_prompt_from_file(str(_card(tmp_path)), "RULES") == f"{CARD}\n\nRULES"


def test_the_slot_is_replaced_literally_and_every_time(tmp_path: Path) -> None:
    rules = r"Say \1 and \g<0> as written. {{default_prompt}} stays."
    path = _card(tmp_path, "A {{default_prompt}} B {{default_prompt}} C")
    assert system_prompt_from_file(str(path), rules) == f"A {rules} B {rules} C"


def test_a_file_that_is_only_the_slot_gives_the_default_prompt(tmp_path: Path) -> None:
    path = _card(tmp_path, "  {{default_prompt}}  \n")
    assert system_prompt_from_file(str(path), DEFAULT_SYSTEM_PROMPT) == DEFAULT_SYSTEM_PROMPT


def test_the_slot_works_through_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FISH_VOICE_SYSTEM_PROMPT_FILE", str(_card(tmp_path, "{{default_prompt}}\n\n" + CARD))
    )
    config = load_config()
    assert config.prompt.text == f"{DEFAULT_SYSTEM_PROMPT}\n\n{CARD}"
    assert config.prompt.pin_seed
