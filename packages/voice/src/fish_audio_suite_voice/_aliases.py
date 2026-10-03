"""Old names of renamed classes, still importable for one minor release."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

__all__ = ["resolve_alias"]


def resolve_alias(
    module: str,
    name: str,
    aliases: Mapping[str, tuple[str, str]],
    namespace: Mapping[str, Any],
) -> object:
    """Return the object a renamed module attribute now lives under, with a warning.

    Parameters
    ----------
    module : str
        ``__name__`` of the module whose ``__getattr__`` is asking.
    name : str
        The attribute that normal lookup did not find.
    aliases : Mapping
        Old name mapped to ``(new name, version that renamed it)``.
    namespace : Mapping
        The module's ``globals()``, which holds the new name.

    Returns
    -------
    object
        The object now exported under the new name. A ``DeprecationWarning``
        naming the replacement is attributed to the code that used the old name.

    Raises
    ------
    AttributeError
        ``name`` is not a deprecated alias either.
    """
    alias = aliases.get(name)
    if alias is None:
        msg = f"module {module!r} has no attribute {name!r}"
        raise AttributeError(msg)
    new, since = alias
    # 3: this function, the module's __getattr__, then the caller.
    warnings.warn(
        f"{name} is deprecated since {since}; use {new}.", DeprecationWarning, stacklevel=3
    )
    return namespace[new]
