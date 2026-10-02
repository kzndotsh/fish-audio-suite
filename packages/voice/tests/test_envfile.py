from __future__ import annotations

import os
from pathlib import Path

import pytest

from fish_audio_suite_voice.envfile import apply_cli_env_files, load_dotenv


def _load(tmp_path: Path, text: str) -> None:
    path = tmp_path / "t.env"
    path.write_text(text, encoding="utf-8")
    assert load_dotenv(path)


def test_dotenv_quotes_comments_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ENVF_A", "ENVF_B", "ENVF_C", "ENVF_D", "ENVF_E"):
        monkeypatch.delenv(name, raising=False)
    _load(
        tmp_path,
        "# comment\n"
        "ENVF_A=plain value # trailing comment\n"
        'ENVF_B="quoted # not a comment"\n'
        "export ENVF_C='single quoted'\n"
        'ENVF_D="line one\nline two"\n'
        "\n"
        "ENVF_E=after\n",
    )
    assert os.environ["ENVF_A"] == "plain value"
    assert os.environ["ENVF_B"] == "quoted # not a comment"
    assert os.environ["ENVF_C"] == "single quoted"
    assert os.environ["ENVF_D"] == "line one\nline two"
    assert os.environ["ENVF_E"] == "after"


def test_dotenv_double_quote_escapes_and_bom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ENVF_F", raising=False)
    monkeypatch.delenv("ENVF_G", raising=False)
    path = tmp_path / "bom.env"
    path.write_bytes(b'\xef\xbb\xbfENVF_F="a\\nb \\"q\\""\nENVF_G=\'C:\\temp\\\'\n')
    assert load_dotenv(path)
    assert os.environ["ENVF_F"] == 'a\nb "q"'
    assert os.environ["ENVF_G"] == "C:\\temp\\"


def test_dotenv_never_overrides_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENVF_H", "from-shell")
    _load(tmp_path, "ENVF_H=from-file\n")
    assert os.environ["ENVF_H"] == "from-shell"


def test_apply_cli_env_files_warns_for_a_missing_required_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert apply_cli_env_files([tmp_path / "nope.env"], required=True) == []
    assert "not found" in capsys.readouterr().err
    assert apply_cli_env_files([tmp_path / "nope.env"], required=False) == []
    assert capsys.readouterr().err == ""


def test_dotenv_skips_a_value_with_a_nul_byte_and_keeps_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("ENVF_NUL_A", "ENVF_NUL_B", "ENVF_NUL_C"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "nul.env"
    path.write_text("ENVF_NUL_A=1\nENVF_NUL_B=bad\x00value\nENVF_NUL_C=3\n", encoding="utf-8")
    try:
        assert load_dotenv(path)
        assert os.environ["ENVF_NUL_A"] == "1"
        assert "ENVF_NUL_B" not in os.environ
        assert os.environ["ENVF_NUL_C"] == "3"
        assert "NUL" in capsys.readouterr().err
    finally:
        for name in ("ENVF_NUL_A", "ENVF_NUL_B", "ENVF_NUL_C"):
            os.environ.pop(name, None)
