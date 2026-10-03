from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, cast

import pytest

from fish_audio_suite_kit import deprecated_fields, env_renamed


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
