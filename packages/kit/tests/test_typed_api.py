from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from importlib import metadata
from typing import Any, cast, get_args, get_type_hints

import pytest

import fish_audio_suite_kit as kit
from fish_audio_suite_kit import (
    FISH_TTS_MODEL_IDS,
    AudioFormat,
    ChatMessage,
    FishAuthError,
    FishErrorBody,
    FishHttpError,
    FishLatency,
    FishRateLimitError,
    FishTimeoutError,
    FishUpstreamError,
    SuiteDefaults,
    env_bool,
    known_latency,
    normalize_tts_model,
    parse_asr_body,
    retry_after_s,
)
from fish_audio_suite_kit._version import read_version

# --- errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (402, False), (404, False), (429, True), (500, True), (504, True)],
)
def test_only_429_and_5xx_are_retryable(status: int, retryable: bool) -> None:
    assert FishHttpError(status, "x").retryable is retryable


def test_every_fish_error_is_an_exception() -> None:
    for kind in (
        FishHttpError,
        FishAuthError,
        FishRateLimitError,
        FishUpstreamError,
        FishTimeoutError,
    ):
        assert issubclass(kind, FishHttpError)
        assert issubclass(kind, Exception)


def test_the_error_keeps_its_old_shape_and_gains_retry_after() -> None:
    error = FishHttpError(429, "slow \ud800 down", retry_after=3)
    assert (error.status, error.retry_after) == (429, 3.0)
    assert "\ud800" not in error.message
    assert str(error) == f"HTTP 429: {error.message}"
    assert FishHttpError(500, "x").retry_after is None


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_an_unusable_retry_after_is_stored_as_none(bad: float) -> None:
    assert FishHttpError(429, "x", retry_after=bad).retry_after is None


def test_a_rate_limit_error_carries_the_hint_and_catches_by_type() -> None:
    error = FishHttpError(429, "slow", retry_after=7)
    assert isinstance(error, FishRateLimitError)
    assert error.retry_after == 7.0
    with pytest.raises(FishRateLimitError):
        raise error


def test_the_factories_return_the_subclass_for_their_status() -> None:
    # The status and message of each are in test_http_errors.
    assert type(FishHttpError.for_unreachable()) is FishUpstreamError
    assert type(FishHttpError.for_non_json()) is FishUpstreamError
    assert type(FishHttpError.for_non_object()) is FishUpstreamError
    assert type(FishHttpError.for_timeout()) is FishTimeoutError
    # Calling a factory on a subclass still gives the fixed class.
    assert type(FishAuthError.for_unreachable()) is FishUpstreamError


def test_parse_asr_body_raises_the_typed_non_object_error() -> None:
    with pytest.raises(FishUpstreamError):
        parse_asr_body(["not", "an", "object"])
    with pytest.raises(FishUpstreamError):
        parse_asr_body({"text": 5})


# --- error body and JSON shapes --------------------------------------------------


def test_fish_error_body_is_frozen_and_cleans_its_message() -> None:
    body = FishErrorBody.of(cast(Any, "402"), "no \ud800 credits")
    assert body.status == 402
    assert "\ud800" not in body.message
    with pytest.raises(dataclasses.FrozenInstanceError):
        body.status = 500  # type: ignore[misc]


def test_parse_asr_body_returns_the_typed_body_and_text() -> None:
    data, text = parse_asr_body(
        {"text": "hi", "segments": [{"text": "hi", "start": 0.0, "end": 0.4}], "duration": 0.4}
    )
    assert text == "hi"
    assert data.get("segments", [{}])[0].get("end") == 0.4


# --- literals and narrowing -----------------------------------------------------


def test_every_latency_mode_reads_back_from_its_name() -> None:
    modes = get_args(FishLatency)
    for mode in modes:
        fallback = next(other for other in modes if other != mode)
        assert known_latency(f" {mode.upper()} ", fallback) == mode


def test_every_catalog_model_is_recognised_whatever_its_case() -> None:
    assert FISH_TTS_MODEL_IDS
    for model in FISH_TTS_MODEL_IDS:
        assert normalize_tts_model(f" {model.upper()} ") == model


def test_a_chat_message_has_only_a_role_and_content() -> None:
    hints = get_type_hints(ChatMessage)
    assert set(hints) == {"role", "content"}
    assert hints["content"] is str


def test_the_defaults_use_the_literal_values_and_stay_light() -> None:
    defaults = SuiteDefaults()
    assert defaults.latency in get_args(FishLatency)
    assert defaults.audio_format in get_args(AudioFormat)
    assert not hasattr(defaults, "__dict__")


# --- retry_after_s ----------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "7"}, 7.0),
        ({"retry-after": " 2.5 "}, 2.5),
        ({"RETRY-AFTER": "0"}, 0.0),
        ({"Retry-After": "86400"}, 86400.0),
        ({"Retry-After": "-1"}, None),
        ({"Retry-After": "nan"}, None),
        ({"Retry-After": "inf"}, None),
        ({"Retry-After": "1e3"}, None),
        ({"Retry-After": "86401"}, None),
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"Retry-After": ""}, None),
        ({"Retry-After": "\N{FULLWIDTH DIGIT SEVEN}"}, None),
        ({"Other": "7"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_retry_after_s_accepts_only_a_sane_wait(
    headers: dict[str, str] | None, expected: float | None
) -> None:
    assert retry_after_s(headers) == expected


# --- call shapes, __all__ and version ----------------------------------------------------


def test_env_bool_takes_its_default_by_keyword_only() -> None:
    assert env_bool("FISH_TEST_SURELY_UNSET", default=True) is True
    with pytest.raises(TypeError):
        cast(Any, env_bool)("FISH_TEST_SURELY_UNSET", True)


def _kit_modules() -> list[str]:
    return sorted(
        f"{kit.__name__}.{info.name}"
        for info in pkgutil.iter_modules(kit.__path__)
        if info.name != "__init__"
    )


@pytest.mark.parametrize("name", _kit_modules())
def test_every_kit_module_has_an_explicit_all_that_resolves(name: str) -> None:
    module = importlib.import_module(name)
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{name} has no __all__"
    assert len(exported) == len(set(exported)), f"{name}.__all__ repeats a name"
    missing = [item for item in exported if not hasattr(module, item)]
    assert missing == [], f"{name}.__all__ names things it does not define: {missing}"


def test_the_root_all_resolves() -> None:
    assert [item for item in kit.__all__ if not hasattr(kit, item)] == []


def test_version_matches_the_installed_distribution() -> None:
    assert kit.__version__ == metadata.version("fish-audio-suite-kit")


def test_version_falls_back_when_the_package_is_not_installed() -> None:
    def missing(_name: str) -> str:
        raise metadata.PackageNotFoundError

    assert read_version(missing) == "0+unknown"
    assert read_version(lambda name: f"9.9.9+{name}") == "9.9.9+fish-audio-suite-kit"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, FishAuthError),
        (402, FishAuthError),
        (403, FishAuthError),
        (429, FishRateLimitError),
        (500, FishUpstreamError),
        (502, FishUpstreamError),
        (503, FishUpstreamError),
        (504, FishTimeoutError),
    ],
)
def test_a_plain_fish_http_error_becomes_the_subclass_for_its_status(
    status: int, expected: type[FishHttpError]
) -> None:
    error = FishHttpError(status, "boom", retry_after=2)
    assert type(error) is expected
    assert error.status == status
    assert error.message == "boom"


def test_other_statuses_stay_plain_and_subclass_calls_are_unchanged() -> None:
    assert type(FishHttpError(404, "missing")) is FishHttpError
    assert type(FishAuthError(404, "odd")) is FishAuthError


def test_a_fish_error_survives_copy_and_pickle_with_its_class_and_retry_hint() -> None:
    import copy
    import pickle

    original = FishRateLimitError(429, "slow down", retry_after=7)
    roundtrip = pickle.loads(pickle.dumps(original))  # noqa: S301 - our own object
    for clone in (copy.copy(original), copy.deepcopy(original), roundtrip):
        assert type(clone) is FishRateLimitError
        assert clone.status == 429
        assert clone.message == "slow down"
        assert clone.retry_after == 7.0
    plain = pickle.loads(pickle.dumps(FishHttpError(404, "missing")))  # noqa: S301 - our own object
    assert type(plain) is FishHttpError
    assert plain.retry_after is None


def test_a_boolean_is_not_a_retry_after_hint() -> None:
    assert FishHttpError(429, "x", retry_after=True).retry_after is None
    assert FishHttpError(429, "x", retry_after=0).retry_after == 0.0
