"""Deprecation warnings for names kept only for older callers."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, cast

__all__ = ["deprecated"]


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
