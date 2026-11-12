"""Read a number or an on/off word from untrusted input: a request body or a setting."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Final, cast

__all__ = [
    "clamp_number",
    "parse_flag",
    "parse_number",
]


def _whole_int(value: Any) -> int:
    # int("16000.0") raises. A whole decimal is accepted, so a sample rate
    # written that way is not swapped for a default rate.
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


_ON_WORDS: Final = frozenset({"1", "true", "yes", "on"})
_OFF_WORDS: Final = frozenset({"0", "false", "no", "off"})
# A strict switch takes only the two spellings a JSON client writes for a boolean.
_STRICT_ON_WORDS: Final = frozenset({"1", "true"})
_STRICT_OFF_WORDS: Final = frozenset({"0", "false"})


def parse_flag(word: str, *, strict: bool = False) -> bool | None:
    """Read an on/off word.

    Parameters
    ----------
    word : str
        A setting or a request field written as text. Compared after strip and
        lowercase.
    strict : bool, optional
        Accept only ``true``, ``1``, ``false`` and ``0``. Default False, which also
        accepts ``yes``, ``on``, ``no`` and ``off``.

    Returns
    -------
    bool or None
        True for an on word, False for an off word, and None for anything else, so the
        caller decides between its default and an error.

    Examples
    --------
    >>> parse_flag(" Yes "), parse_flag("off"), parse_flag("maybe")
    (True, False, None)
    >>> parse_flag("yes", strict=True) is None
    True
    """
    text = word.strip().lower()
    if text in (_STRICT_ON_WORDS if strict else _ON_WORDS):
        return True
    if text in (_STRICT_OFF_WORDS if strict else _OFF_WORDS):
        return False
    return None
