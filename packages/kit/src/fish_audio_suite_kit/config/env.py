"""Read typed settings from environment variables, falling back on bad values."""

from __future__ import annotations

import os
import re
from collections.abc import Callable

from fish_audio_suite_kit.config._values import parse_flag, parse_number
from fish_audio_suite_kit.http.urls import strip_base

__all__ = [
    "env_base",
    "env_bool",
    "env_float",
    "env_int",
    "env_text",
    "env_token",
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
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return parse_flag(raw) is True


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
