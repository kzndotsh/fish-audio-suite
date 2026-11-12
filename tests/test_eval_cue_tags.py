"""``scripts/eval_cue_tags.py``: the counting, the prompt files, and a run against a fake model."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import eval_cue_tags
import pytest
from chat_fakes import install_chat

from fish_audio_suite_kit import DEFAULT_SYSTEM_PROMPT, ChatMessage


def test_cues_are_listed_as_written_and_as_fish_is_sent_them() -> None:
    reply = "[Happy] Hi! [laugh] Ha ha. [soft chuckle] Heh. [in a storytelling voice] So."
    assert eval_cue_tags.cues_in(reply) == [
        "happy",
        "laughing",
        "soft chuckle",
        "in a storytelling voice",
    ]
    assert eval_cue_tags.cues_in(reply, official=True) == ["happy", "laughing", "chuckling"]
    assert eval_cue_tags.cues_in("No cues at all.") == []


def test_one_cue_everywhere_scores_no_variety_and_an_even_spread_scores_full() -> None:
    flat = eval_cue_tags.cue_stats("flat", ["[happy] Hi. [happy] Yes.", "[happy] Sure."])
    assert (flat.cues, flat.distinct, flat.evenness) == (3, 1, 0.0)
    assert (flat.top_cue, flat.top_share, flat.top3_share) == ("happy", 1.0, 1.0)
    even = eval_cue_tags.cue_stats("even", ["[happy] Hi. [sad] Oh. [curious] Why? [calm] Fine."])
    assert even.distinct == 4
    assert even.evenness == pytest.approx(1.0)
    assert even.top_share == pytest.approx(0.25)
    assert even.top3_share == pytest.approx(0.75)


def test_off_list_dropped_sound_and_bare_sentence_shares_are_counted() -> None:
    stats = eval_cue_tags.cue_stats(
        "mixed",
        ["[happy] Hi! [smiling] Yes. [in a storytelling voice] So. [laughing] Ha ha. No cue here."],
        failed=2,
        cost=0.5,
    )
    assert stats.replies == 1
    assert stats.failed == 2
    assert stats.cost == 0.5
    assert stats.cues == 4
    assert stats.distinct == 4
    assert stats.distinct_sent == 2  # smiling becomes happy, the description is dropped
    assert stats.off_list_share == pytest.approx(0.5)
    assert stats.dropped_share == pytest.approx(0.25)
    assert stats.sound_share == pytest.approx(0.25)
    assert stats.sentences == 5
    assert stats.bare_sentence_share == pytest.approx(0.2)
    assert list(stats.counts) == ["happy", "smiling", "in a storytelling voice", "laughing"]


def test_no_replies_or_no_cues_gives_zeros_not_an_error() -> None:
    assert eval_cue_tags.cue_stats("none", []).evenness == 0.0
    plain = eval_cue_tags.cue_stats("plain", ["Hello there. How are you?"])
    assert (plain.cues, plain.top_cue, plain.top_share) == (0, "", 0.0)
    assert plain.bare_sentence_share == 1.0
    assert "none" in eval_cue_tags.table([plain])[2]
    assert eval_cue_tags.cue_lists([plain]) == ["- `plain`: no cues"]


def test_prompt_files_are_named_by_stem_after_the_default(tmp_path: Path) -> None:
    wide = tmp_path / "wide.txt"
    wide.write_text("  Use many cues.  \n", encoding="utf-8")
    other = tmp_path / "sub" / "wide.md"
    other.parent.mkdir()
    other.write_text("Use few cues.", encoding="utf-8")
    prompts = eval_cue_tags.read_system_prompts([wide, other])
    assert list(prompts) == ["default", "wide", "wide-2"]
    assert prompts["default"] == DEFAULT_SYSTEM_PROMPT
    assert prompts["wide"] == "Use many cues."
    assert list(eval_cue_tags.read_system_prompts([wide], default=False)) == ["wide"]
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        eval_cue_tags.read_system_prompts([empty])


def test_the_built_in_lines_are_distinct() -> None:
    assert len(set(eval_cue_tags.LINES)) == len(eval_cue_tags.LINES) >= 8


async def _answer(model: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
    """One mood under the default prompt, and a different one per line otherwise."""
    if model == "broken/model":
        raise RuntimeError("provider down")
    if messages[0]["content"] == DEFAULT_SYSTEM_PROMPT:
        yield "[happy] Sure thing."
        return
    moods = ("sad", "curious", "calm")
    yield f"[{moods[len(messages[-1]['content']) % 3]}] Sure thing."


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    install_chat(monkeypatch, _answer)


@pytest.mark.usefixtures("fake_backend")
def test_main_compares_a_prompt_file_with_the_default_and_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    wide = tmp_path / "wide.txt"
    wide.write_text("Use many cues.", encoding="utf-8")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    base = [
        str(wide),
        "-l",
        "a",
        "-l",
        "bb",
        "-l",
        "ccc",
        "-n",
        "2",
        "--model",
        "good/model",
        "--env-file",
        str(tmp_path / "none.env"),
    ]
    assert eval_cue_tags.main([*base, "--out", str(md), "--show"]) == 0
    out = capsys.readouterr().out
    assert "[wide] line 3/3, run 2/2: [sad] Sure thing." in out
    assert "| `default` | 6/6 | 1.0 | 1 | 1 | 0.00 | happy 100% |" in out
    assert "| `wide` | 6/6 | 1.0 | 3 | 3 | 1.00 |" in out
    assert "- `default`: happy 6" in out
    report = md.read_text(encoding="utf-8")
    assert report.startswith("# Cue variety: `good/model`")
    assert "| `wide` | 6/6 |" in report
    assert eval_cue_tags.main([*base, "--out", str(js)]) == 0
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["model"] == "good/model"
    assert [s["name"] for s in payload["stats"]] == ["default", "wide"]
    assert payload["stats"][1]["counts"] == {"sad": 2, "curious": 2, "calm": 2}
    assert len(payload["replies"]["wide"]) == 6
    assert "Sure thing." not in capsys.readouterr().out  # replies are shown only with --show


@pytest.mark.usefixtures("fake_backend")
def test_main_reports_what_is_missing_and_fails_when_no_request_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("OPENROUTER_API_KEY", "FISH_LLM_API_KEY", "FISH_LLM_PROVIDER", "FISH_LLM_BASE"):
        monkeypatch.delenv(name, raising=False)
    env = ["--env-file", str(tmp_path / "none.env"), "--model", "good/model", "-l", "a"]
    assert eval_cue_tags.main([*env, "--dry-run"]) == 0
    assert capsys.readouterr().out.strip() == "default"
    assert eval_cue_tags.main(env) == 2
    assert "No API key" in capsys.readouterr().err
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert eval_cue_tags.main([*env, "-n", "0"]) == 2
    assert eval_cue_tags.main([*env, "--no-default"]) == 2
    assert eval_cue_tags.main([*env, str(tmp_path / "missing.txt")]) == 2
    broken = [*env[:2], "--model", "broken/model", "-l", "a"]
    assert eval_cue_tags.main(broken) == 2  # every request failed
    assert "| `default` | 0/3 |" in capsys.readouterr().out


def test_the_seed_flag_pins_the_opening_example_under_a_prompt_file_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, int]] = []

    async def recording(_model: str, messages: list[ChatMessage]) -> AsyncIterator[str]:
        seen.append((messages[0]["content"], len(messages)))
        yield "[calm] Fine."

    install_chat(monkeypatch, recording)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    wide = tmp_path / "wide.txt"
    wide.write_text("Use many cues.", encoding="utf-8")
    args = [str(wide), "-l", "a", "-n", "1", "--model", "m", "--env-file", str(tmp_path / "x")]
    assert eval_cue_tags.main(args) == 0
    assert dict(seen) == {DEFAULT_SYSTEM_PROMPT: 4, "Use many cues.": 2}  # system, example, line
    seen.clear()
    assert eval_cue_tags.main([*args, "--seed"]) == 0
    assert dict(seen) == {DEFAULT_SYSTEM_PROMPT: 4, "Use many cues.": 4}
