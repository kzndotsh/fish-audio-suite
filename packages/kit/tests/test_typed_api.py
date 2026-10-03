from __future__ import annotations

import asyncio
import dataclasses
import importlib
import inspect
import pkgutil
import warnings
from importlib import metadata
from typing import Any, cast, get_args, get_type_hints

import pytest

import fish_audio_suite_kit as kit
from fish_audio_suite_kit import (
    FISH_LATENCIES,
    FISH_TTS_MODEL_IDS,
    AsrFormat,
    AudioFormat,
    ChatMessage,
    FishAudioSuiteError,
    FishAuthError,
    FishErrorBody,
    FishHttpError,
    FishLatency,
    FishRateLimitError,
    FishTimeoutError,
    FishUpstreamError,
    LatencySnapshot,
    SuiteDefaults,
    TtsModel,
    catalog_tts_model,
    ensure_trace_headers,
    env_bool,
    fish_backoff_s,
    fish_sleep_before_retry,
    fish_transport_error,
    known_asr_format,
    known_audio_format,
    known_latency,
    parse_asr_body,
    retry_after_seconds,
)
from fish_audio_suite_kit._deprecation import deprecated
from fish_audio_suite_kit._version import read_version

# --- errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, FishAuthError),
        (402, FishAuthError),
        (403, FishAuthError),
        (429, FishRateLimitError),
        (500, FishUpstreamError),
        (502, FishUpstreamError),
        (503, FishUpstreamError),
        (504, FishTimeoutError),
        (400, FishHttpError),
        (404, FishHttpError),
        (422, FishHttpError),
    ],
)
def test_from_status_picks_the_matching_class(status: int, kind: type[FishHttpError]) -> None:
    error = FishHttpError.from_status(status, "boom")
    assert type(error) is kind
    assert error.status == status
    assert error.message == "boom"


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (402, False), (404, False), (429, True), (500, True), (504, True)],
)
def test_only_429_and_5xx_are_retryable(status: int, retryable: bool) -> None:
    assert FishHttpError.from_status(status, "x").retryable is retryable


def test_every_fish_error_is_a_suite_error_and_an_exception() -> None:
    for kind in (
        FishHttpError,
        FishAuthError,
        FishRateLimitError,
        FishUpstreamError,
        FishTimeoutError,
    ):
        assert issubclass(kind, FishHttpError)
        assert issubclass(kind, FishAudioSuiteError)
        assert issubclass(kind, Exception)
    assert not issubclass(FishAudioSuiteError, FishHttpError)


def test_the_error_keeps_its_old_shape_and_gains_retry_after() -> None:
    error = FishHttpError(429, "slow \ud800 down", retry_after=3)
    assert (error.status, error.retry_after) == (429, 3.0)
    assert "\ud800" not in error.message
    assert str(error) == f"HTTP 429: {error.message}"
    assert FishHttpError(500, "x").retry_after is None


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_an_unusable_retry_after_is_stored_as_none(bad: float) -> None:
    assert FishHttpError(429, "x", retry_after=bad).retry_after is None
    assert FishHttpError.from_status(429, "x", retry_after=bad).retry_after is None


def test_a_rate_limit_error_carries_the_hint_and_catches_by_type() -> None:
    error = FishHttpError.from_status(429, "slow", retry_after=7)
    assert isinstance(error, FishRateLimitError)
    assert error.retry_after == 7.0
    with pytest.raises(FishRateLimitError):
        raise error
    with pytest.raises(FishAudioSuiteError):
        raise FishHttpError.from_status(401, "no key")


def test_the_factories_return_the_right_subclass_with_the_old_message() -> None:
    assert type(FishHttpError.unreachable()) is FishUpstreamError
    assert type(FishHttpError.non_json()) is FishUpstreamError
    assert type(FishHttpError.non_object()) is FishUpstreamError
    assert type(FishHttpError.timed_out()) is FishTimeoutError
    assert (FishHttpError.unreachable().status, FishHttpError.unreachable().message) == (
        502,
        "Fish upstream unreachable",
    )
    assert FishHttpError.timed_out().status == 504
    assert FishHttpError.non_json().message == "Fish returned a non-JSON body"
    assert FishHttpError.non_object().message == "Fish returned a non-object body"
    # Calling a factory on a subclass still gives the fixed class.
    assert type(FishAuthError.unreachable()) is FishUpstreamError


def test_parse_asr_body_raises_the_typed_non_object_error() -> None:
    with pytest.raises(FishUpstreamError):
        parse_asr_body(["not", "an", "object"])
    with pytest.raises(FishUpstreamError):
        parse_asr_body({"text": 5})


# --- error body and JSON shapes --------------------------------------------------


def test_fish_error_body_is_frozen_and_converts_to_the_wire_dict() -> None:
    body = FishErrorBody.of(cast(Any, "402"), "no \ud800 credits")
    assert body.status == 402
    assert "\ud800" not in body.message
    assert body.as_dict() == {"message": body.message, "status": 402}
    with pytest.raises(dataclasses.FrozenInstanceError):
        body.status = 500  # type: ignore[misc]


def test_fish_error_body_still_reads_like_the_old_dict() -> None:
    body = FishErrorBody(401, "Invalid Token")
    assert body["status"] == 401
    assert body["message"] == "Invalid Token"
    with pytest.raises(KeyError):
        cast(Any, body)["other"]


def test_parse_asr_body_returns_the_typed_body_and_text() -> None:
    data, text = parse_asr_body(
        {"text": "hi", "segments": [{"text": "hi", "start": 0.0, "end": 0.4}], "duration": 0.4}
    )
    assert text == "hi"
    assert data.get("segments", [{}])[0].get("end") == 0.4


# --- literals and narrowing -----------------------------------------------------


def test_the_literal_types_list_the_closed_value_sets() -> None:
    assert set(get_args(FishLatency)) == FISH_LATENCIES == {"low", "balanced", "normal"}
    assert set(get_args(AudioFormat)) == {"wav", "pcm", "mp3", "opus"}
    assert set(get_args(AsrFormat)) == {"json", "text", "verbose_json", "srt", "vtt"}
    # Every catalog model but the passthrough preview ids is a TtsModel.
    assert set(get_args(TtsModel)) == set(FISH_TTS_MODEL_IDS) - {"drama-3-preview"}
    assert get_type_hints(ChatMessage) == {
        "role": get_type_hints(ChatMessage)["role"],
        "content": str,
    }
    assert set(get_args(get_type_hints(ChatMessage)["role"])) == {"system", "user", "assistant"}


@pytest.mark.parametrize("name", get_args(AudioFormat))
def test_known_audio_format_accepts_each_format_in_any_case(name: str) -> None:
    assert known_audio_format(f" {name.upper()} ", "mp3") == name


def test_known_audio_format_and_asr_format_fall_back_to_the_default() -> None:
    assert known_audio_format("flac", "wav") == "wav"
    assert known_audio_format("", "pcm") == "pcm"
    assert known_asr_format("VTT", "json") == "vtt"
    assert known_asr_format("xml", "text") == "text"


def test_known_latency_returns_a_literal() -> None:
    assert known_latency("LOW", "normal") == "low"
    assert known_latency("turbo", "balanced") == "balanced"


def test_catalog_tts_model_narrows_only_the_catalog() -> None:
    assert catalog_tts_model(" S2.1-Pro ") == "s2.1-pro"
    assert catalog_tts_model("s1") == "s1"
    assert catalog_tts_model("drama-3-preview") is None
    assert catalog_tts_model("mystery") is None


def test_the_defaults_use_the_literal_values_and_stay_light() -> None:
    defaults = SuiteDefaults()
    assert defaults.latency in get_args(FishLatency)
    assert defaults.audio_format in get_args(AudioFormat)
    assert not hasattr(defaults, "__dict__")


def test_a_latency_snapshot_is_still_built_by_position() -> None:
    snapshot = LatencySnapshot(1.0, 2.0, 3.0, 4.0, 5.0, "trace-1")
    assert (snapshot.asr_ms, snapshot.voice_to_voice, snapshot.trace_id) == (1.0, 5.0, "trace-1")
    assert snapshot.first_audio is None
    assert dataclasses.replace(snapshot, first_audio=9.0).first_audio == 9.0


# --- retry_after_seconds ----------------------------------------------------------


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
def test_retry_after_seconds_accepts_only_a_sane_wait(
    headers: dict[str, str] | None, expected: float | None
) -> None:
    assert retry_after_seconds(headers) == expected


# --- deprecation helper --------------------------------------------------------------


def test_deprecated_warns_at_the_caller_and_keeps_the_behavior() -> None:
    @deprecated("new_name", "0.1.0")
    def old_name(value: int) -> int:
        """Doubles it."""
        return value * 2

    with pytest.warns(
        DeprecationWarning, match=r"old_name is deprecated since 0\.1\.0; use new_name"
    ) as caught:
        assert old_name(4) == 8
    assert caught[0].filename == __file__
    assert old_name.__name__ == "old_name"
    assert old_name.__doc__ == "Doubles it."


def test_deprecated_keeps_a_coroutine_function_a_coroutine_function() -> None:
    @deprecated("new_name", "0.1.0")
    async def old_name() -> str:
        return "done"

    assert inspect.iscoroutinefunction(old_name)
    with pytest.warns(DeprecationWarning, match="old_name"):
        assert asyncio.run(old_name()) == "done"


def test_the_kit_does_not_call_its_own_deprecated_names() -> None:
    # The suite turns warnings into errors, so any internal use would raise here.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fish_backoff_s(1, rng=None) > 0
        assert fish_transport_error(OSError("x"), timed_out=False)[0] == 502
        assert "traceparent" in ensure_trace_headers({})
        assert parse_asr_body({"text": "x"})[1] == "x"
        assert asyncio.run(fish_sleep_before_retry(kit.FISH_RETRY_ATTEMPTS - 1)) is False


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
    assert type(FishHttpError.from_status(401, "k")) is FishAuthError
