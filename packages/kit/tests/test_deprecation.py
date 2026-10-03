from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, cast

import pytest

import fish_audio_suite_kit as kit
from fish_audio_suite_kit import FishHttpError, deprecated_fields, env_renamed


@deprecated_fields("0.2.0", secs="seconds", tag="label")
@dataclasses.dataclass(frozen=True, slots=True)
class Wait:
    seconds: float = 0.0
    label: str = ""


@deprecated_fields("0.2.0", old_count="count")
@dataclasses.dataclass
class Plain:
    count: int = 0


# --- deprecated_fields ---------------------------------------------------------------


@pytest.mark.parametrize(("old", "new", "value"), [("secs", "seconds", 2.5), ("tag", "label", "x")])
def test_an_old_keyword_sets_the_new_field_and_warns(old: str, new: str, value: object) -> None:
    kwargs: dict[str, Any] = {old: value}
    with pytest.warns(
        DeprecationWarning, match=rf"Wait\({old}=\.\.\.\) is deprecated since 0\.2\.0; use {new}"
    ) as caught:
        built = Wait(**kwargs)
    assert getattr(built, new) == value
    assert caught[0].filename == __file__


@pytest.mark.parametrize(("old", "new"), [("secs", "seconds"), ("tag", "label")])
def test_an_old_attribute_reads_the_new_field_and_warns(old: str, new: str) -> None:
    built = Wait(3.0, "y")
    with pytest.warns(
        DeprecationWarning, match=rf"Wait\.{old} is deprecated since 0\.2\.0; use {new}"
    ) as caught:
        assert getattr(built, old) == getattr(built, new)
    assert caught[0].filename == __file__


def test_passing_an_old_and_a_new_name_together_is_a_type_error() -> None:
    kwargs: dict[str, Any] = {"secs": 1.0, "seconds": 2.0}
    with pytest.raises(TypeError, match="'secs' and its new name 'seconds'"):
        Wait(**kwargs)


def test_an_old_keyword_for_a_field_given_by_position_is_a_type_error() -> None:
    kwargs: dict[str, Any] = {"secs": 1.0}
    with pytest.raises(TypeError), pytest.warns(DeprecationWarning, match="secs"):
        Wait(2.0, **kwargs)


def test_positional_construction_is_unchanged() -> None:
    assert Wait(1.5, "z") == Wait(seconds=1.5, label="z")
    assert [f.name for f in dataclasses.fields(Wait)] == ["seconds", "label"]
    assert list(inspect.signature(Wait).parameters) == ["seconds", "label"]


def test_the_class_stays_frozen_and_slotted() -> None:
    built = Wait(1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cast(Any, built).seconds = 2.0
    assert not hasattr(built, "__dict__")
    assert copy.deepcopy(built) == built
    assert copy.copy(built) == built


def test_replace_needs_the_new_name() -> None:
    # replace() fills every field in by its new name, so an old keyword arrives
    # next to its new one and cannot be told apart from passing both.
    kwargs: dict[str, Any] = {"secs": 4.0}
    with pytest.raises(TypeError, match="replace"):
        dataclasses.replace(Wait(1.0, "k"), **kwargs)


def test_new_names_do_not_warn() -> None:
    assert Wait(seconds=1.0, label="q").seconds == 1.0
    assert dataclasses.replace(Wait(), seconds=2.0).seconds == 2.0


def test_a_plain_dataclass_works_too() -> None:
    kwargs: dict[str, Any] = {"old_count": 3}
    with pytest.warns(DeprecationWarning, match="old_count"):
        built = Plain(**kwargs)
    assert built.count == 3


# --- env_renamed -----------------------------------------------------------------------


def _collect() -> tuple[list[str], Callable[[str], None]]:
    seen: list[str] = []
    return seen, seen.append


def test_env_renamed_prefers_the_new_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISH_TEST_NEW", "a")
    monkeypatch.setenv("FISH_TEST_OLD", "b")
    seen, warn = _collect()
    assert env_renamed("FISH_TEST_NEW", "FISH_TEST_OLD", warn=warn) == "FISH_TEST_NEW"
    assert seen == []


def test_env_renamed_falls_back_to_the_old_name_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FISH_TEST_NEW", "  ")
    monkeypatch.setenv("FISH_TEST_OLD", "b")
    seen, warn = _collect()
    assert env_renamed("FISH_TEST_NEW", "FISH_TEST_OLD", warn=warn) == "FISH_TEST_OLD"
    assert seen == ["FISH_TEST_OLD is deprecated; use FISH_TEST_NEW"]


@pytest.mark.parametrize("old_value", [None, "", " "])
def test_env_renamed_returns_the_new_name_when_neither_is_set(
    monkeypatch: pytest.MonkeyPatch, old_value: str | None
) -> None:
    monkeypatch.delenv("FISH_TEST_NEW", raising=False)
    if old_value is None:
        monkeypatch.delenv("FISH_TEST_OLD", raising=False)
    else:
        monkeypatch.setenv("FISH_TEST_OLD", old_value)
    seen, warn = _collect()
    assert env_renamed("FISH_TEST_NEW", "FISH_TEST_OLD", warn=warn) == "FISH_TEST_NEW"
    assert seen == []


# --- deprecated kit names --------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("CLOUD_CHUNK_HI", "CHUNK_LENGTH_CLOUD_HI"),
        ("SELF_HOST_CHUNK_HI", "CHUNK_LENGTH_SELF_HOSTED_HI"),
        ("MIN_CHUNK_LO", "MIN_CHUNK_LENGTH_LO"),
        ("MIN_CHUNK_HI", "MIN_CHUNK_LENGTH_HI"),
        ("UNIT_LO", "UNIT_INTERVAL_LO"),
        ("UNIT_HI", "UNIT_INTERVAL_HI"),
        ("OpenAIError", "OpenAIErrorDetail"),
    ],
)
def test_a_renamed_constant_or_class_still_resolves_and_warns(old: str, new: str) -> None:
    with pytest.warns(
        DeprecationWarning, match=rf"{old} is deprecated since 0\.2\.0; use {new}\."
    ) as caught:
        value = getattr(kit, old)
    assert value is getattr(kit, new)
    assert caught[0].filename == __file__
    assert old not in kit.__all__
    assert new in kit.__all__


def test_a_renamed_constant_still_imports_from_the_root() -> None:
    with pytest.warns(DeprecationWarning, match="CLOUD_CHUNK_HI"):
        from fish_audio_suite_kit import CLOUD_CHUNK_HI
    assert CLOUD_CHUNK_HI == kit.CHUNK_LENGTH_CLOUD_HI == 300


def test_an_unknown_root_name_is_still_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'NOT_A_KIT_NAME'"):
        _ = getattr(kit, "NOT_A_KIT_NAME")  # noqa: B009


class _TimeoutError(Exception):
    pass


_FUNCTION_ALIASES: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = [
    ("skip_empty_delta", "is_empty_delta", (" ",), {}),
    ("known_tts_model", "normalize_tts_model", (" S2-Pro ",), {}),
    ("same_utterance", "is_same_utterance", ("Hi there.", "hi there"), {}),
    ("hold_tts", "tts_hold_at", ("Hello [ha",), {"line_start": True, "sentence_start": True}),
    ("number_or", "parse_number", ("16000.0", 0, int), {}),
    ("clamp_num", "clamp_number", ("9", 0, 5, 1, int), {}),
    ("retry_after_seconds", "retry_after_s", ({"Retry-After": "7"},), {}),
    ("fish_transport_error", "describe_transport_error", (None,), {"timed_out": True}),
    ("fish_request_error", "describe_request_error", (_TimeoutError(), _TimeoutError), {}),
]


@pytest.mark.parametrize(("old", "new", "args", "kwargs"), _FUNCTION_ALIASES)
def test_a_renamed_function_still_works_and_warns(
    old: str, new: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    old_func: Callable[..., object] = getattr(kit, old)
    new_func: Callable[..., object] = getattr(kit, new)
    with pytest.warns(
        DeprecationWarning, match=rf"{old} is deprecated since 0\.2\.0; use {new}\."
    ) as caught:
        result = old_func(*args, **kwargs)
    assert result == new_func(*args, **kwargs)
    assert caught[0].filename == __file__
    assert inspect.signature(old_func) == inspect.signature(new_func)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("unreachable", "for_unreachable"),
        ("timed_out", "for_timeout"),
        ("non_json", "for_non_json"),
        ("non_object", "for_non_object"),
    ],
)
def test_a_renamed_error_factory_still_works_and_warns(old: str, new: str) -> None:
    old_factory: Callable[[], FishHttpError] = getattr(FishHttpError, old)
    built_new: FishHttpError = getattr(FishHttpError, new)()
    with pytest.warns(
        DeprecationWarning, match=rf"{old} is deprecated since 0\.2\.0; use FishHttpError\.{new}"
    ):
        built = old_factory()
    assert type(built) is type(built_new)
    assert (built.status, built.message) == (built_new.status, built_new.message)
