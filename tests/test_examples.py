"""The examples: offline ones run in-process, live ones import, and the README lists them all."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIVE = ("speak_to_file", "stream_llm_reply", "custom_sink")


def test_clean_llm_text_cuts_what_it_cleaned(capsys: pytest.CaptureFixture[str]) -> None:
    importlib.import_module("clean_llm_text").main()
    lines = capsys.readouterr().out.splitlines()
    cleaned = ast.literal_eval(lines[0].removeprefix("cleaned: "))
    pieces = [ast.literal_eval(line.removeprefix("piece: ")) for line in lines[1:]]
    assert len(pieces) > 1
    assert "".join(pieces) == cleaned
    assert "think" not in cleaned
    assert "http" not in cleaned
    assert "**" not in cleaned
    assert "read the guide first." in cleaned


def test_captions_prints_both_formats(capsys: pytest.CaptureFixture[str]) -> None:
    importlib.import_module("captions").main()
    out = capsys.readouterr().out
    assert "00:00:00,000 --> 00:00:01,800" in out
    assert "WEBVTT" in out
    assert "00:00:00.000 --> 00:00:01.800" in out


def test_fish_errors_catches_each_status_by_type(capsys: pytest.CaptureFixture[str]) -> None:
    importlib.import_module("fish_errors").main()
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "402: check the key or the credit balance (Insufficient credits)",
        "429: wait 7.0s, then retry",
        "503: Fish is down, retryable=True",
        "404: Voice not found, retryable=False",
    ]


@pytest.mark.parametrize("name", LIVE)
def test_live_example_exits_naming_the_missing_key(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = importlib.import_module(name)
    # An empty value wins over the repository .env, because load_dotenv keeps set keys.
    monkeypatch.setenv("FISH_API_KEY", "")
    monkeypatch.setenv("FISH_VOICE_ID", "")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert "FISH_API_KEY" in str(caught.value)


def test_proxy_example_declares_its_own_dependency() -> None:
    source = (EXAMPLES / "openai_sdk_proxy.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert '# dependencies = ["openai>=1"]' in source
    assert "# requires-python = " in source


def test_every_example_has_main_and_a_docstring_that_says_how_to_run_it() -> None:
    for path in sorted(EXAMPLES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree) or ""
        assert "Run: " in doc, path.name
        assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), path
        assert 'if __name__ == "__main__":' in path.read_text(encoding="utf-8"), path.name


def test_readme_table_lists_exactly_the_example_files() -> None:
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\| \[(\w+\.py)\]", readme, flags=re.MULTILINE))
    assert listed == {p.name for p in EXAMPLES.glob("*.py")}
    needs = set(re.findall(r"^\| \[\w+\.py\]\(.*?\) \| .* \| (.+) \|$", readme, flags=re.MULTILINE))
    assert needs <= {"nothing", "FISH_API_KEY + FISH_VOICE_ID", "the proxy running"}
