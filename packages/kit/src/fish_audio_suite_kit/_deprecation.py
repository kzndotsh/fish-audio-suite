"""Deprecation warnings for names kept only for older callers."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, cast

__all__ = ["deprecated", "deprecated_fields"]


def deprecated[F: Callable[..., Any]](replacement: str, since: str) -> Callable[[F], F]:
    """Mark a function as deprecated without changing what it does.

    Parameters
    ----------
    replacement : str
        What to call instead, shown in the warning.
    since : str
        The kit version that deprecated the name.

    Returns
    -------
    Callable
        A decorator. The wrapped function emits ``DeprecationWarning`` on every
        call, attributed to the caller (``stacklevel=2``), then runs unchanged.
        A coroutine function stays a coroutine function.

    Notes
    -----
    ``warnings.deprecated`` only exists from Python 3.13 and the kit supports 3.12,
    so this is a small local version of it.

    Examples
    --------
    >>> @deprecated("new_name", "0.1.0")
    ... def old_name() -> int:
    ...     return 1
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     old_name()
    1
    >>> caught[0].category.__name__
    'DeprecationWarning'
    """

    def decorate(func: F) -> F:
        message = f"{func.__name__} is deprecated since {since}; use {replacement}."
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorate


def deprecated_fields[C: type](since: str, **renamed: str) -> Callable[[C], C]:
    """Keep the old keyword and attribute names of renamed dataclass fields working.

    Parameters
    ----------
    since : str
        The kit version that renamed the fields.
    **renamed : str
        Old name mapped to the new field name, for example ``ttfa="first_audio_ms"``.

    Returns
    -------
    Callable
        A class decorator. Put it above ``@dataclass``. It wraps ``__init__`` so an
        old keyword is passed on as the new field, with a ``DeprecationWarning``
        attributed to the caller, and adds a read-only property per old name that
        warns and returns the new field. Positional construction is unchanged, and
        a frozen or slotted dataclass works too, since only the class is changed.

    Notes
    -----
    Passing an old name and its new name together raises ``TypeError``, the same as
    passing one field twice. ``dataclasses.replace`` fills in every field by its new
    name, so it takes only new names: an old one raises that ``TypeError``. Type checkers do not see the old names; they exist only
    so code written against the old ones keeps running for one more minor release.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @deprecated_fields("0.2.0", secs="seconds")
    ... @dataclass(frozen=True, slots=True)
    ... class Wait:
    ...     seconds: float = 0.0
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     Wait(secs=2.0).secs
    2.0
    >>> [str(item.message) for item in caught]  # doctest: +NORMALIZE_WHITESPACE
    ['Wait(secs=...) is deprecated since 0.2.0; use seconds.',
     'Wait.secs is deprecated since 0.2.0; use seconds.']
    """

    def decorate(cls: C) -> C:
        init = cls.__init__
        owner = cls.__name__

        @functools.wraps(init)
        def __init__(self: object, *args: Any, **kwargs: Any) -> None:  # noqa: N807
            for old, new in renamed.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    msg = (
                        f"{owner}() got both {old!r} and its new name {new!r}"
                        " (dataclasses.replace takes only the new name)"
                    )
                    raise TypeError(msg)
                warnings.warn(
                    f"{owner}({old}=...) is deprecated since {since}; use {new}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
            init(self, *args, **kwargs)

        type.__setattr__(cls, "__init__", __init__)
        for old, new in renamed.items():
            type.__setattr__(cls, old, _renamed_property(owner, old, new, since))
        return cls

    return decorate


def _renamed_property(owner: str, old: str, new: str, since: str) -> property:
    message = f"{owner}.{old} is deprecated since {since}; use {new}."

    def read(self: object) -> Any:
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(self, new)

    return property(read, doc=f"Deprecated since {since}. Read ``{new}`` instead.")
