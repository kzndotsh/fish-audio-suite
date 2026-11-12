"""Read one setting from the environment, and hold a tune built in code to the same bounds.

Every voice setting except ``FISH_VOICE_DEBUG`` is read through one of these functions, so
each read names its variable, its default and its range or choices in one call.
``FISH_VOICE_DEBUG`` is read in ``debug``, which this module warns through, so it cannot
come back here. A setting a tune also checks
when it is built in code is declared once as an ``IntSetting``, ``FloatSetting`` or
``ChoiceSetting``: its ``read`` is the environment's call and its ``check`` the tune's, so
the two take the same bounds. The policy is voice's: a value
that is not a number, not a known word, or out of range warns on stderr and the default is
used. A call that passes ``quiet=True`` keeps the policy the shared ``FISH_*`` settings
have in the proxy too: a number that does not parse falls back without a warning and one
out of range is clamped. The word sets and number parsing are kit's (``parse_flag``,
``parse_number``, ``env_float``, ``env_int``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from fish_audio_suite_kit import env_float, env_int, parse_flag, parse_number
from fish_audio_suite_voice.debug import warn

__all__ = [
    "ChoiceSetting",
    "FloatSetting",
    "IntSetting",
    "check_range",
    "read_choice",
    "read_first",
    "read_flag",
    "read_float",
    "read_int",
    "read_raw",
    "read_text",
    "read_verbatim",
    "read_word",
]

# The warning most settings give. ``read_choice`` fills these fields: ``name``, ``raw`` (as
# set, stripped), ``value`` (lowercased), ``default``, ``choices`` (in order), ``sorted``
# and ``one_of`` (``a, b or c``).
IGNORED: Final = "fish-voice: ignoring {name}={value!r}, using {default}"


def read_verbatim(name: str) -> str | None:
    """Read a variable exactly as it is set.

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    str or None
        The value with its spaces, or None when it is unset. Set but empty is ``""``.
    """
    return os.environ.get(name)


def read_raw(name: str) -> str | None:
    """Read a variable, treating blank as unset.

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    str or None
        The stripped value, or None when it is unset or only whitespace.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def read_text(name: str, default: str = "") -> str:
    """Read free text. A variable that is set but blank reads as ``""``.

    Parameters
    ----------
    name : str
        Variable name.
    default : str, optional
        Used only when the variable is not set at all.

    Returns
    -------
    str
        The stripped value or ``default``.
    """
    value = os.environ.get(name)
    return default if value is None else value.strip()


def read_word(name: str, default: str) -> str:
    """Read a single word, such as a model id or a sink name. Blank reads as ``default``.

    Parameters
    ----------
    name : str
        Variable name.
    default : str
        Used when the variable is unset or blank.

    Returns
    -------
    str
        The stripped value or ``default``. It is not checked against a list.
    """
    raw = read_raw(name)
    return default if raw is None else raw


def read_first(*names: str) -> str:
    """Read the first of several variables that is set and not blank.

    Parameters
    ----------
    *names : str
        Variable names, first match wins.

    Returns
    -------
    str
        The stripped value, or ``""`` when none is set. A blank one is skipped, so an
        empty key copied from ``.env.example`` does not hide a real one further on.
    """
    for name in names:
        raw = read_raw(name)
        if raw is not None:
            return raw
    return ""


def read_flag(name: str, *, default: bool = False) -> bool:
    """Read an on/off switch. An unknown word warns and returns ``default``.

    Parameters
    ----------
    name : str
        Variable name.
    default : bool, optional
        Used when the variable is unset, blank, or not a known word.

    Returns
    -------
    bool
        ``1/true/yes/on`` is True and ``0/false/no/off`` is False, in any case.
    """
    raw = read_raw(name)
    if raw is None:
        return default
    value = parse_flag(raw)
    if value is None:
        warn(IGNORED.format(name=name, value=raw, default=default))
        return default
    return value


def read_int(
    name: str,
    default: int,
    *,
    lo: int | None = None,
    hi: int | None = None,
    quiet: bool = False,
) -> int:
    """Read a whole number.

    Parameters
    ----------
    name : str
        Variable name.
    default : int
        Used when the variable is unset, blank, not a whole number, or out of range.
    lo : int or None, optional
        Smallest accepted value.
    hi : int or None, optional
        Largest accepted value.
    quiet : bool, optional
        Kit's policy instead of the warning: plain ASCII digits only (``16000.0`` passes),
        anything else is ``default`` without a warning, and a value out of range is
        clamped to it.

    Returns
    -------
    int
        The parsed value, or ``default`` (with a warning unless ``quiet``).
    """
    if quiet:
        return _clamped(env_int(name, default), lo, hi)
    raw = read_raw(name)
    if raw is None:
        return default
    try:
        # int(), not kit's parse_number: a frame count written as 3.0 is a typo here.
        value = int(raw)
    except ValueError:
        warn(IGNORED.format(name=name, value=raw, default=default))
        return default
    if not _inside(value, lo, hi):
        warn(IGNORED.format(name=name, value=raw, default=default))
        return default
    return value


def read_float(
    name: str,
    default: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    positive: bool = False,
    quiet: bool = False,
) -> float:
    """Read a number.

    Parameters
    ----------
    name : str
        Variable name.
    default : float
        Used when the variable is unset, blank, not a finite number, or out of range.
    lo : float or None, optional
        Smallest accepted value.
    hi : float or None, optional
        Largest accepted value.
    positive : bool, optional
        Reject zero and below.
    quiet : bool, optional
        Kit's policy instead of the warning: plain ASCII digits only, anything else is
        ``default`` without a warning, and a value out of range is clamped to it.

    Returns
    -------
    float
        The parsed value, or ``default`` (with a warning unless ``quiet``).
    """
    if quiet:
        return _clamped(env_float(name, default), lo, hi)
    raw = read_raw(name)
    if raw is None:
        return default
    value = parse_number(raw, math.nan, float)
    if math.isnan(value) or (positive and value <= 0) or not _inside(value, lo, hi):
        warn(IGNORED.format(name=name, value=raw, default=default))
        return default
    return value


def read_choice[T: str](
    name: str,
    default: T,
    choices: tuple[T, ...],
    *,
    aliases: Mapping[str, T] | None = None,
    quiet: bool = False,
    warning: str = IGNORED,
) -> T:
    """Read one word of a fixed list, in any case.

    Parameters
    ----------
    name : str
        Variable name.
    default : str
        Used when the variable is unset, blank, or not one of ``choices``.
    choices : tuple of str
        The accepted words, lowercase, in the order the docs list them.
    aliases : Mapping or None, optional
        More accepted words and the choice each one stands for, such as ``off``.
    quiet : bool, optional
        Use ``default`` for an unknown word without a warning.
    warning : str, optional
        The warning for an unknown word, with the fields ``IGNORED`` describes.

    Returns
    -------
    str
        The lowercased choice, or ``default``.
    """
    raw = read_raw(name)
    if raw is None:
        return default
    value = raw.lower()
    for choice in choices:
        if choice == value:
            return choice
    if aliases is not None and value in aliases:
        return aliases[value]
    if not quiet:
        ordered = list(choices)
        warn(
            warning.format(
                name=name,
                raw=raw,
                value=value,
                default=default,
                choices=", ".join(ordered),
                sorted=", ".join(sorted(ordered)),
                one_of=f"{', '.join(ordered[:-1])} or {ordered[-1]}",
            )
        )
    return default


def check_range(
    where: str,
    value: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
    positive: bool = False,
) -> None:
    """Reject a value a tune is given in code that its ``from_env`` would not accept.

    Parameters
    ----------
    where : str
        The field, as ``Tune.field``, for the message.
    value : float
        The value given.
    lo : float or None, optional
        Smallest accepted value.
    hi : float or None, optional
        Largest accepted value.
    positive : bool, optional
        Reject zero and below.

    Raises
    ------
    ValueError
        When ``value`` is out of range or, for a float, not finite.
    """
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{where} must be a finite number, not {value}"
        raise ValueError(msg)
    if positive and value <= 0:
        msg = f"{where} must be above 0, not {value}"
        raise ValueError(msg)
    if not _inside(value, lo, hi):
        if lo is not None and hi is not None:
            msg = f"{where} must be from {lo} to {hi}, not {value}"
        elif lo is not None:
            msg = f"{where} must be at least {lo}, not {value}"
        else:
            msg = f"{where} must be at most {hi}, not {value}"
        raise ValueError(msg)


def _inside(value: float, lo: float | None, hi: float | None) -> bool:
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def _clamped[T: int | float](value: T, lo: T | None, hi: T | None) -> T:
    if lo is not None and value < lo:
        return lo
    if hi is not None and value > hi:
        return hi
    return value


@dataclass(frozen=True, slots=True)
class IntSetting:
    """A whole-number setting: its variable, its default and its range, stated once.

    Attributes
    ----------
    name : str
        The environment variable.
    default : int
        What an unset or unusable value reads as.
    lo, hi : int or None
        The smallest and largest value taken. None for no bound.
    """

    name: str
    default: int
    lo: int | None = None
    hi: int | None = None

    def read(self) -> int:
        """Read the variable, as ``read_int`` does.

        Returns
        -------
        int
            The value, or ``default`` with a warning when it is unusable.
        """
        return read_int(self.name, self.default, lo=self.lo, hi=self.hi)

    def check(self, where: str, value: int) -> None:
        """Hold a value given in code to the same range, as ``check_range`` does.

        Parameters
        ----------
        where : str
            The field, as ``Tune.field``, for the message.
        value : int
            The value given.
        """
        check_range(where, value, lo=self.lo, hi=self.hi)


@dataclass(frozen=True, slots=True)
class FloatSetting:
    """A number setting: its variable, its default and its range, stated once.

    Attributes
    ----------
    name : str
        The environment variable.
    default : float
        What an unset or unusable value reads as.
    lo, hi : float or None
        The smallest and largest value taken. None for no bound.
    positive : bool
        Zero and below are refused.
    """

    name: str
    default: float
    lo: float | None = None
    hi: float | None = None
    positive: bool = False

    def read(self) -> float:
        """Read the variable, as ``read_float`` does.

        Returns
        -------
        float
            The value, or ``default`` with a warning when it is unusable.
        """
        return read_float(self.name, self.default, lo=self.lo, hi=self.hi, positive=self.positive)

    def check(self, where: str, value: float) -> None:
        """Hold a value given in code to the same range, as ``check_range`` does.

        Parameters
        ----------
        where : str
            The field, as ``Tune.field``, for the message.
        value : float
            The value given.
        """
        check_range(where, value, lo=self.lo, hi=self.hi, positive=self.positive)


@dataclass(frozen=True, slots=True)
class ChoiceSetting[T: str]:
    """A setting that is one word of a fixed list: its variable, default and words, stated once.

    Attributes
    ----------
    name : str
        The environment variable.
    default : str
        What an unset or unknown word reads as.
    choices : tuple of str
        The accepted words, lowercase, in the order the docs list them.
    aliases : Mapping or None
        More accepted words and the choice each stands for, such as ``off``.
    warning : str
        The warning for an unknown word, with the fields ``read_choice`` fills.
    """

    name: str
    default: T
    choices: tuple[T, ...]
    aliases: Mapping[str, T] | None = None
    warning: str = IGNORED

    def read(self) -> T:
        """Read the variable, as ``read_choice`` does.

        Returns
        -------
        str
            The lowercased choice, or ``default``, with a warning for an unknown word.
        """
        return read_choice(
            self.name, self.default, self.choices, aliases=self.aliases, warning=self.warning
        )

    def check(self, where: str, value: str) -> None:
        """Refuse a value given in code that no word, alias or the default reads as.

        Parameters
        ----------
        where : str
            The field, as ``Tune.field``, for the message.
        value : str
            The value given.

        Raises
        ------
        ValueError
            When ``value`` is none of them. The message lists the words, and says ``or
            empty`` when an empty value is taken but is not one of them.
        """
        taken = {*self.choices, *(self.aliases or {}).values(), self.default}
        if value in taken:
            return
        empty = " or empty" if "" in taken and "" not in self.choices else ""
        msg = f"{where} must be one of {', '.join(self.choices)}{empty}, not {value!r}"
        raise ValueError(msg)
