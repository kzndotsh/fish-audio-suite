"""Choosing a chat provider, and what goes on the wire for Experiential."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from fish_audio_suite_voice.cli import llm_setting_names
from fish_audio_suite_voice.debug import configure_voice_logging
from fish_audio_suite_voice.llm import open_chat_backend
from fish_audio_suite_voice.llm_tune import (
    EXPERIENTIAL_API_BASE,
    OPENROUTER_API_BASE,
    LlmTune,
    provider_for_base,
)

_ENV = (
    "FISH_LLM_PROVIDER",
    "FISH_LLM_BASE",
    "OPENROUTER_BASE_URL",
    "FISH_LLM_BACKEND",
    "FISH_LLM_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "EXPLABS_API_KEY",
    "FISH_LLM_MODEL",
    "OPENROUTER_MODEL",
    "FISH_LLM_MODEL_OPENROUTER",
    "FISH_LLM_MODEL_EXPERIENTIAL",
    "FISH_LLM_REASONING_EFFORT",
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_the_default_is_openrouter_and_never_reads_the_experiential_key(
    env: pytest.MonkeyPatch,
) -> None:
    env.setenv("OPENROUTER_API_KEY", "or-key")
    env.setenv("EXPLABS_API_KEY", "xpl-key")
    tune = LlmTune.from_env()
    assert tune.provider == "openrouter"
    assert tune.backend == "openrouter"
    assert tune.base == OPENROUTER_API_BASE
    assert tune.api_key == "or-key"


def test_naming_experiential_picks_its_base_key_and_model(env: pytest.MonkeyPatch) -> None:
    env.setenv("FISH_LLM_PROVIDER", "Experiential")
    env.setenv("EXPLABS_API_KEY", "xpl-key")
    env.setenv("OPENROUTER_API_KEY", "or-key")
    env.setenv("OPENAI_API_KEY", "oa-key")
    env.setenv("FISH_LLM_MODEL", "shared-model")
    env.setenv("FISH_LLM_MODEL_EXPERIENTIAL", "glm-5.3-flash-abliterated")
    env.setenv("OPENROUTER_MODEL", "vendor/openrouter-only")
    tune = LlmTune.from_env()
    assert tune.provider == "experiential"
    assert tune.backend == "openai"
    assert tune.base == EXPERIENTIAL_API_BASE
    assert tune.api_key == "xpl-key"
    assert tune.model == "glm-5.3-flash-abliterated"


def test_each_providers_model_variable_lets_both_live_in_one_env(
    env: pytest.MonkeyPatch,
) -> None:
    env.setenv("FISH_LLM_MODEL_OPENROUTER", "vendor/or-model")
    env.setenv("FISH_LLM_MODEL_EXPERIENTIAL", "xp-model")
    env.setenv("FISH_LLM_MODEL", "fallback")
    assert LlmTune.from_env().model == "vendor/or-model"
    env.setenv("FISH_LLM_PROVIDER", "experiential")
    assert LlmTune.from_env().model == "xp-model"
    env.delenv("FISH_LLM_MODEL_EXPERIENTIAL")
    assert LlmTune.from_env().model == "fallback"


def test_the_experiential_model_ignores_the_openrouter_model_variable(
    env: pytest.MonkeyPatch,
) -> None:
    env.setenv("FISH_LLM_PROVIDER", "experiential")
    env.setenv("OPENROUTER_MODEL", "vendor/openrouter-only")
    assert LlmTune.from_env().model == ""


def test_a_base_on_the_experiential_host_is_experiential_without_naming_it(
    env: pytest.MonkeyPatch,
) -> None:
    env.setenv("FISH_LLM_BASE", "https://api.experientiallabs.ai/api/v1")
    env.setenv("EXPLABS_API_KEY", "xpl-key")
    env.setenv("OPENAI_API_KEY", "oa-key")
    tune = LlmTune.from_env()
    assert tune.provider == "experiential"
    assert tune.api_key == "xpl-key"


def test_a_key_never_goes_to_the_wrong_provider(env: pytest.MonkeyPatch) -> None:
    env.setenv("FISH_LLM_PROVIDER", "openrouter")
    env.setenv("FISH_LLM_BASE", "https://api.experientiallabs.ai/v1")
    env.setenv("OPENROUTER_API_KEY", "or-key")
    tune = LlmTune.from_env()
    assert tune.provider == "experiential"
    assert tune.api_key == ""


def test_the_old_openrouter_base_alias_cannot_redirect_a_named_provider(
    env: pytest.MonkeyPatch,
) -> None:
    env.setenv("FISH_LLM_PROVIDER", "experiential")
    env.setenv("OPENROUTER_BASE_URL", "https://proxy.example.test/v1")
    env.setenv("EXPLABS_API_KEY", "xpl-key")
    tune = LlmTune.from_env()
    assert tune.base == EXPERIENTIAL_API_BASE
    assert tune.api_key == "xpl-key"


def test_an_explicit_key_wins_and_a_blank_one_falls_back(env: pytest.MonkeyPatch) -> None:
    env.setenv("FISH_LLM_PROVIDER", "experiential")
    env.setenv("EXPLABS_API_KEY", "xpl-key")
    env.setenv("FISH_LLM_API_KEY", "")
    assert LlmTune.from_env().api_key == "xpl-key"
    env.setenv("FISH_LLM_API_KEY", "explicit")
    assert LlmTune.from_env().api_key == "explicit"


def test_any_other_server_uses_the_openai_key_and_is_custom(env: pytest.MonkeyPatch) -> None:
    env.setenv("FISH_LLM_BASE", "https://llm.example.test/v1")
    env.setenv("OPENAI_API_KEY", "oa-key")
    env.setenv("EXPLABS_API_KEY", "xpl-key")
    tune = LlmTune.from_env()
    assert tune.provider == "custom"
    assert tune.backend == "openai"
    assert tune.api_key == "oa-key"


def test_an_unknown_provider_warns_and_falls_back_while_custom_is_quiet(
    env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.setenv("FISH_LLM_PROVIDER", "nope")
    assert LlmTune.from_env().provider == "openrouter"
    assert "unknown FISH_LLM_PROVIDER" in capsys.readouterr().err
    env.setenv("FISH_LLM_PROVIDER", "custom")
    env.setenv("FISH_LLM_BASE", "https://llm.example.test/v1")
    assert LlmTune.from_env().provider == "custom"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://api.experientiallabs.ai/v1", "experiential"),
        ("https://experientiallabs.ai/v1", "experiential"),
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://experientiallabs.ai.evil.test/v1", None),
        ("https://evil.test/experientiallabs.ai", None),
        ("https://notexperientiallabs.ai/v1", None),
        ("http://[::1", None),
    ],
)
def test_the_provider_comes_from_the_host_only(base: str, expected: str | None) -> None:
    found = provider_for_base(base)
    assert (found.name if found else None) == expected


def test_reasoning_effort_is_validated(
    env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert LlmTune.from_env().reasoning_effort == ""
    env.setenv("FISH_LLM_REASONING_EFFORT", " LOW ")
    assert LlmTune.from_env().reasoning_effort == "low"
    env.setenv("FISH_LLM_REASONING_EFFORT", "turbo")
    assert LlmTune.from_env().reasoning_effort == ""
    assert "FISH_LLM_REASONING_EFFORT" in capsys.readouterr().err


def _experiential_tune(**kw: Any) -> LlmTune:
    fields: dict[str, Any] = {
        "backend": "openai",
        "base": EXPERIENTIAL_API_BASE,
        "api_key": "xpl_test",
        "model": "glm-5.3-flash-abliterated",
    }
    fields.update(kw)
    return LlmTune(**fields)


def _install(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real = httpx.AsyncClient

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("fish_audio_suite_voice.transports.httpx.AsyncClient", factory)


async def _collect(tune: LlmTune) -> str:
    async with open_chat_backend(tune) as backend:
        return "".join([tok async for tok in backend.stream([{"role": "user", "content": "hi"}])])


STREAM = (
    ": keep-alive\n\n"
    'data: {"choices":[{"delta":{"reasoning_content":"thinking it over"}}]}\n\n'
    ": keep-alive\n\n"
    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":" there"},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_the_request_matches_what_experiential_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=STREAM, headers={"content-type": "text/event-stream"})

    _install(monkeypatch, handler)
    spoken = asyncio.run(_collect(_experiential_tune(reasoning_effort="low", nitro=True)))
    assert spoken == "Hello there"
    request = seen[0]
    assert str(request.url) == "https://api.experientiallabs.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer xpl_test"
    body = json.loads(request.content)
    assert set(body) == {
        "messages",
        "model",
        "stream",
        "temperature",
        "max_tokens",
        "reasoning_effort",
    }
    assert body["reasoning_effort"] == "low"
    assert body["model"] == "glm-5.3-flash-abliterated"


def test_no_reasoning_field_is_sent_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text=STREAM)

    _install(monkeypatch, handler)
    asyncio.run(_collect(_experiential_tune()))
    assert "reasoning_effort" not in seen[0]


def test_a_spent_free_allowance_is_reported_with_its_code_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[int] = []
    error = {"error": {"code": "insufficient_quota", "message": "free_limit_reached"}}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json=error)

    _install(monkeypatch, handler)
    assert asyncio.run(_collect(_experiential_tune())) == ""
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "HTTP 429" in err
    assert "free_limit_reached" in err


def test_the_gateway_headers_reach_the_debug_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_voice_logging(debug=1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=STREAM,
            headers={
                "x-request-id": "req-123",
                "x-gateway-provider": "experiential_cloud",
                "x-gateway-zdr": "true",
            },
        )

    _install(monkeypatch, handler)
    asyncio.run(_collect(_experiential_tune()))
    err = capsys.readouterr().err
    assert "request=req-123" in err
    assert "via=experiential_cloud" in err
    assert "zdr=true" in err


def test_an_old_openrouter_backend_setting_does_not_follow_a_named_provider(
    env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.setenv("FISH_LLM_BACKEND", "openrouter")
    env.setenv("FISH_LLM_PROVIDER", "experiential")
    tune = LlmTune.from_env()
    assert tune.backend == "openai"
    assert tune.provider == "experiential"
    assert "does not fit experiential" in capsys.readouterr().err


def test_the_openrouter_backend_setting_still_holds_for_openrouter_and_custom_hosts(
    env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.setenv("FISH_LLM_BACKEND", "openrouter")
    assert LlmTune.from_env().backend == "openrouter"
    env.setenv("FISH_LLM_BASE", "https://proxy.example.test/v1")
    assert LlmTune.from_env().backend == "openrouter"
    assert capsys.readouterr().err == ""


def test_a_named_providers_key_is_not_picked_up_for_a_plain_http_base(
    env: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env.setenv("EXPLABS_API_KEY", "xpl_from_env")
    env.setenv("FISH_LLM_BASE", "http://api.experientiallabs.ai/v1")
    tune = LlmTune.from_env()
    assert tune.provider == "experiential"
    assert tune.api_key == ""
    assert "not https" in capsys.readouterr().err


def test_an_explicit_key_still_goes_to_a_plain_http_named_base(env: pytest.MonkeyPatch) -> None:
    env.setenv("EXPLABS_API_KEY", "xpl_from_env")
    env.setenv("FISH_LLM_API_KEY", "explicit")
    env.setenv("FISH_LLM_BASE", "http://api.experientiallabs.ai/v1")
    assert LlmTune.from_env().api_key == "explicit"


def test_a_custom_http_base_keeps_its_key(env: pytest.MonkeyPatch) -> None:
    env.setenv("OPENAI_API_KEY", "local-key")
    env.setenv("FISH_LLM_BASE", "http://127.0.0.1:11434/v1")
    assert LlmTune.from_env().api_key == "local-key"


def test_the_startup_blockers_name_the_selected_providers_variables(
    env: pytest.MonkeyPatch,
) -> None:
    env.setenv("FISH_LLM_PROVIDER", "experiential")
    assert llm_setting_names(LlmTune.from_env()) == (
        "EXPLABS_API_KEY",
        "FISH_LLM_MODEL_EXPERIENTIAL",
    )
    env.setenv("FISH_LLM_PROVIDER", "openrouter")
    assert llm_setting_names(LlmTune.from_env()) == (
        "OPENROUTER_API_KEY",
        "FISH_LLM_MODEL_OPENROUTER / OPENROUTER_MODEL",
    )
    env.setenv("FISH_LLM_PROVIDER", "custom")
    env.setenv("FISH_LLM_BASE", "http://127.0.0.1:11434/v1")
    assert llm_setting_names(LlmTune.from_env()) == ("OPENAI_API_KEY", "OPENROUTER_MODEL")
