"""Read typed settings from environment variables, falling back on bad values."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from typing import Any, cast

from fish_audio_suite_kit.defaults import strip_base

__all__ = [
    "clamp_number",
    "env_base",
    "env_bool",
    "env_float",
    "env_int",
    "env_off",
    "env_text",
    "env_token",
    "parse_number",
]


def env_base(name: str, default: str) -> str:
    """Read a base URL from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Used when the variable is missing, blank, or empty after cleaning.

    Returns
    -------
    str
        The value, or the default, with ``strip_base`` applied.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return strip_base(default)
    return strip_base(raw) or strip_base(default)


def _env_word(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower()


_OFF_WORDS = frozenset({"0", "false", "no", "off"})


_ON_WORDS = frozenset({"1", "true", "yes", "on"})


def env_off(name: str) -> bool:
    """Return whether the variable is explicitly switched off.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    bool
        True only for ``0``, ``false``, ``no`` or ``off`` in any case. A missing or
        blank variable is not off.
    """
    return _env_word(name) in _OFF_WORDS


def env_bool(name: str, *, default: bool = False) -> bool:
    """Read an on/off flag from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool, optional
        Returned when the variable is missing or blank. Keyword-only, so a call
        reads ``env_bool("X", default=True)``.

    Returns
    -------
    bool
        True for ``1``, ``true``, ``yes`` or ``on`` in any case, False for any
        other non-blank value, and ``default`` when the variable is unset or blank.
    """
    word = _env_word(name)
    if not word:
        return default
    return word in _ON_WORDS


def env_token(name: str, default: str) -> str:
    """Read a single-word setting from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Returned when the variable is missing or blank.

    Returns
    -------
    str
        The stripped value, or ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    return text or default


def env_text(name: str, default: str = "") -> str:
    """Read free text from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str, optional
        Returned, stripped, when the variable is missing.

    Returns
    -------
    str
        The stripped value. A blank value stays blank and does not fall back to
        ``default``, so an empty key can be told apart from an unset one.
    """
    return os.environ.get(name, default).strip()


def _whole_int(value: Any) -> int:
    # int("16000.0") raises, so a decimal string fell back to another rate
    # and the speaker played the buffer at the wrong speed.
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        parsed = float(text)
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError
        return int(parsed)
    return int(value)


def _parsed[T: int | float](value: Any, parse: Callable[[Any], T]) -> T:
    if parse is int:
        return cast(T, _whole_int(value))
    return parse(value)


def parse_number[T: int | float](value: Any, default: T, parse: Callable[[Any], T]) -> T:
    """Parse a number from untrusted input.

    Parameters
    ----------
    value : Any
        A number or a numeric string, for example from a JSON request.
    default : int or float
        Returned for anything that does not parse.
    parse : Callable
        ``int`` or ``float``. With ``int``, a whole decimal such as ``"16000.0"``
        is accepted.

    Returns
    -------
    int or float
        The parsed value, or ``default`` for junk, a non-finite float, or a JSON
        boolean (``True`` would otherwise parse as 1).
    """
    # bool is an int subclass. True would parse as 1.
    if isinstance(value, bool):
        return default
    try:
        parsed = _parsed(value, parse)
    except (TypeError, ValueError, OverflowError):
        return default
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return default
    return parsed


# Plain ASCII decimals only. int() and float() also accept other scripts' digits
# ("\uff11\uff10") and underscores ("1_000"), so a lookalike value would otherwise
# pass for a limit setting.
_ASCII_NUMBER_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _env_num[T: int | float](name: str, default: T, parse: Callable[[str], T]) -> T:
    raw = os.environ.get(name, "").strip()
    if not raw or not _ASCII_NUMBER_RE.fullmatch(raw):
        return default
    return parse_number(raw, default, parse)


def env_int(name: str, default: int) -> int:
    """Read an integer from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Returned when the variable is missing, blank, not a number, or written with
        anything but plain ASCII digits (fullwidth digits, underscores and hex
        are rejected, so a lookalike cannot pass for a limit).

    Returns
    -------
    int
        The parsed value. A whole decimal such as ``16000.0`` is accepted.

    Examples
    --------
    >>> env_int("FISH_DOCTEST_SURELY_UNSET", 7)
    7
    """
    return _env_num(name, default, int)


def env_float(name: str, default: float) -> float:
    """Read a float from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Returned when the variable is missing, blank, not finite, not a number, or
        written with anything but plain ASCII digits.

    Returns
    -------
    float
        The parsed value, or ``default``.
    """
    return _env_num(name, default, float)


def clamp_number[T: int | float](
    value: Any,
    lo: T,
    hi: T,
    default: T,
    parse: Callable[[Any], T],
) -> T:
    """Parse a Fish numeric knob and keep it inside its documented range.

    Parameters
    ----------
    value : Any
        A number or numeric string from a caller.
    lo : int or float
        Lowest allowed value.
    hi : int or float
        Highest allowed value.
    default : int or float
        Returned when ``value`` does not parse, is non-finite, or is a boolean.
    parse : Callable
        ``int`` or ``float``.

    Returns
    -------
    int or float
        The parsed value clamped to ``[lo, hi]``, or ``default``.
    """
    # bool is an int subclass. True would clamp to 1.
    if isinstance(value, bool):
        return default
    try:
        n = _parsed(value, parse)
    except (TypeError, ValueError, OverflowError):
        n = default
    if isinstance(n, float) and not math.isfinite(n):
        return default
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n
