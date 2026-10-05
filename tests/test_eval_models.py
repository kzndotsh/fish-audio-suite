"""The model evaluation script, tested without any network: a fake backend stands in."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import eval_models
import pytest
from loguru import logger

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice import llm
from fish_audio_suite_voice.debug import configure_voice_logging
from fish_audio_suite_voice.llm_tune import LlmTune


def test_voice_checks_flag_what_a_voice_cannot_say() -> None:
    good = eval_models.voice_checks("[happy] Hello there! It is good to hear you.")
    assert good.starts_with_cue
    assert not good.ends_with_cue
    assert not good.markdown
    assert not good.emoji
    assert good.words == 8
    assert good.sentences == 2
    bad = eval_models.voice_checks("**Sure!** Here is a list:\n- one\n- two 😀 [sigh]")
    assert not bad.starts_with_cue
    assert bad.ends_with_cue
    assert bad.markdown
    assert bad.emoji


def test_prompts_come_from_flags_and_a_dash_separated_file(tmp_path: Path) -> None:
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(
        "First line.\nStill the first.\n---\nSecond.\n-----\n\nThird.\n", encoding="utf-8"
    )
    assert eval_models.read_prompts(["Flag prompt"], prompts_file) == [
        "Flag prompt",
        "First line.\nStill the first.",
        "Second.",
        "Third.",
    ]
    assert eval_models.read_prompts([], None) == list(eval_models.PROMPTS)


def test_the_models_listed_at_the_top_of_the_script_are_not_empty() -> None:
    assert eval_models.MODELS, "MODELS at the top of scripts/eval_models.py must not be empty"
    assert all("/" in model for model in eval_models.MODELS)


class _FakeBackend:
    def __init__(self, model: str) -> None:
        self.model = model

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        cancel: asyncio.Event | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[str]:
        del cancel, trace_id
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        if self.model == "empty/model":
            return
        if self.model == "broken/model":
            raise RuntimeError("provider down")
        yield "[happy] "
        yield f"Hello from {self.model}!"
        # The real backend logs this line when a reply ends, with the cost OpenRouter reports.
        logger.debug(
            "llm.stream_end finish=stop in=262 out=26 cost=$0.00012 chunks=25 provider=DeepInfra model=x"
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def opened(
        tune: LlmTune, *, session_id: str | None = None
    ) -> AsyncGenerator[_FakeBackend, None]:
        del session_id
        yield _FakeBackend(tune.model)

    monkeypatch.setattr(eval_models, "open_chat_backend", opened)


@pytest.mark.usefixtures("fake_backend")
def test_every_model_and_prompt_runs_and_failures_do_not_stop_the_sweep() -> None:
    runs = asyncio.run(
        eval_models.evaluate(
            LlmTune(api_key="k"),
            ["good/model", "empty/model", "broken/model"],
            ["one", "two"],
            "system",
            runs=2,
            seed=None,
            warmup=True,
        )
    )
    assert len(runs) == 3 * 2 * 2
    good = [r for r in runs if r.model == "good/model"]
    assert all(not r.error and r.text == "[happy] Hello from good/model!" for r in good)
    assert all(r.checks is not None and r.checks.starts_with_cue for r in good)
    assert all("empty reply" in r.error for r in runs if r.model == "empty/model")
    assert all("provider down" in r.error for r in runs if r.model == "broken/model")


@pytest.mark.usefixtures("fake_backend")
def test_the_summary_counts_only_successful_runs() -> None:
    runs = asyncio.run(
        eval_models.evaluate(
            LlmTune(api_key="k"),
            ["good/model", "empty/model"],
            ["one"],
            "system",
            runs=3,
            seed=None,
            warmup=False,
        )
    )
    good = eval_models.summarize("good/model", [r for r in runs if r.model == "good/model"])
    assert (good.ok, good.failed) == (3, 0)
    assert good.starts_with_cue == 1.0
    assert good.ends_with_cue == 0.0
    empty = eval_models.summarize("empty/model", [r for r in runs if r.model == "empty/model"])
    assert (empty.ok, empty.failed) == (0, 3)
    assert empty.errors


@pytest.mark.usefixtures("fake_backend")
def test_main_writes_markdown_and_json_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    monkeypatch.setattr(eval_models, "MODELS", ("good/model", "empty/model", "good/model"))
    base = [
        "-p",
        "Hi there",
        "-n",
        "2",
        "--env-file",
        str(tmp_path / "none.env"),
    ]
    assert eval_models.main([*base, "--out", str(md)]) == 0
    assert eval_models.main([*base, "--out", str(js)]) == 0
    report = md.read_text(encoding="utf-8")
    assert "| `good/model` | 2/2 |" in report
    assert "$0.00012 | DeepInfra |" in report
    assert "Hello from good/model!" in report
    assert "empty reply" in report
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert {s["model"] for s in payload["summaries"]} == {"good/model", "empty/model"}
    assert len(payload["runs"]) == 4
    out = capsys.readouterr().out
    # Each reply is printed as it arrives, then the table.
    assert "[good/model] prompt 1/1, run 1/2" in out
    assert "$0.00012, via DeepInfra" in out
    assert (
        "Spent $0.00024 across 2 of 4 requests" in out
    )  # only the model that replied reports a cost
    assert "Hello from good/model!" in out
    assert "[empty/model] prompt 1/1, run 1/2: FAILED" in out
    assert eval_models.main([*base, "--quiet"]) == 0
    quiet = capsys.readouterr().out
    assert "Hello from good/model!" not in quiet
    assert "good/model" in quiet  # the table row


def test_main_needs_models_and_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "OPENROUTER_API_KEY",
        "FISH_LLM_API_KEY",
        "OPENAI_API_KEY",
        "FISH_LLM_PROVIDER",
        "FISH_LLM_BASE",
    ):
        monkeypatch.delenv(name, raising=False)
    nothing = ["--env-file", str(tmp_path / "none.env")]
    monkeypatch.setattr(eval_models, "MODELS", ())
    assert eval_models.main(nothing) == 2  # no models listed
    monkeypatch.setattr(eval_models, "MODELS", ("a/b",))
    assert eval_models.main(nothing) == 2  # a model, but no key
    assert eval_models.main(["--dry-run", *nothing]) == 0
    assert eval_models.main(["--runs", "0", *nothing]) == 2


def test_the_prompts_file_setting_at_the_top_replaces_the_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "prompts.txt").write_text("First.\n---\nSecond.\n", encoding="utf-8")
    monkeypatch.setattr(eval_models, "ROOT", tmp_path)
    monkeypatch.setattr(eval_models, "PROMPTS_FILE", "prompts.txt")
    assert eval_models.read_prompts([], None) == ["First.", "Second."]
    # A flag still wins over the file setting.
    assert eval_models.read_prompts(["Flag prompt"], None) == ["Flag prompt"]


def test_the_cost_parser_reads_the_line_the_backend_really_logs() -> None:
    """Guard against the backend changing its ``llm.stream_end`` line, which the script reads."""
    lines: list[str] = []
    configure_voice_logging(debug=True)
    logger.remove()
    logger.add(lambda message: lines.append(str(message.record["message"])), level="DEBUG")  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType]
    try:
        stats = getattr(llm, "_ChatStats")()  # noqa: B009 - a private type, built only for this guard
        stats.last_finish, stats.yielded, stats.last_provider = "stop", 25, "DeepInfra"
        stats.last_usage = {"prompt_tokens": 262, "completion_tokens": 26, "cost": 0.00009}
        getattr(llm, "_finish_llm")(stats, "some/model")  # noqa: B009
    finally:
        configure_voice_logging(debug=False)
    parsed = [u for u in map(eval_models.parse_stream_end, lines) if u is not None]
    assert len(parsed) == 1
    assert parsed[0].cost == pytest.approx(0.00009)
    assert parsed[0].provider == "DeepInfra"
    stats.last_usage = {"prompt_tokens": 1, "completion_tokens": 2}
    unpriced = eval_models.parse_stream_end(
        "llm.stream_end finish=stop in=1 out=2 chunks=3 provider=Local model=m"
    )
    assert unpriced is not None
    assert unpriced.cost is None
    assert eval_models.parse_stream_end("llm.request model=m msgs=4 nitro=False") is None
