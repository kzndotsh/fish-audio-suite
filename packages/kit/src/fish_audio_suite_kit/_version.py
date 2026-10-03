"""The installed kit version."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

__all__ = ["read_version"]

_DIST = "fish-audio-suite-kit"
_UNKNOWN = "0+unknown"


def read_version(lookup: Callable[[str], str] = version) -> str:
    """Return the installed version of the kit distribution.

    Parameters
    ----------
    lookup : Callable, optional
        Maps a distribution name to its version. Defaults to
        ``importlib.metadata.version``. Tests pass a stand-in.

    Returns
    -------
    str
        The version, or ``"0+unknown"`` when the package is not installed, for
        example when it runs from a bare source checkout.
    """
    try:
        return lookup(_DIST)
    except PackageNotFoundError:
        return _UNKNOWN
