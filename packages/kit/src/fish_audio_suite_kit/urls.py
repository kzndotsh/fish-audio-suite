"""Read a Fish base URL: clean it, tell the cloud host from a self-hosted one, and spot cleartext."""

from __future__ import annotations

import ipaddress
import re
from typing import Final
from urllib.parse import urlsplit

from fish_audio_suite_kit._charsets import cut_at_control

__all__ = [
    "is_insecure_fish_base",
    "strip_base",
]

_CLOUD_HOST: Final = "api.fish.audio"
_SCHEME_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def strip_base(url: str) -> str:
    """Drop surrounding space and trailing slashes so a joined path is not ``//``.

    Parameters
    ----------
    url : str
        A base URL. Anything from the first control character on is cut off, since
        a newline in a base makes ``httpx`` reject the URL.

    Returns
    -------
    str
        The URL without surrounding space or trailing slashes.
    """
    return cut_at_control(url.strip()).strip().rstrip("/")


def is_cloud_base(fish_base: str) -> bool:
    """Return whether a Fish base points at the cloud API.

    Parameters
    ----------
    fish_base : str
        Fish origin, with or without a scheme.

    Returns
    -------
    bool
        True when the parsed hostname is ``api.fish.audio`` or a subdomain of it. A
        path, a port or a look-alike domain does not count.
    """
    text = fish_base.strip()
    # Only a leading scheme or "//" makes the host parse as a netloc. A "//" later
    # in a path ("api.fish.audio/v1//edge") does not.
    if not (_SCHEME_RE.match(text) or text.startswith("//")):
        text = f"//{text}"
    try:
        host = (urlsplit(text).hostname or "").lower()
    except ValueError:
        return False
    return host == _CLOUD_HOST or host.endswith(f".{_CLOUD_HOST}")


def is_insecure_fish_base(base: str) -> bool:
    """Return whether a Fish base URL would send the API key in cleartext.

    Parameters
    ----------
    base : str
        A ``FISH_BASE`` value such as ``http://10.0.0.5:8080``.

    Returns
    -------
    bool
        True when the scheme is ``http`` and the host is not loopback
        (``localhost``, ``*.localhost``, ``127.0.0.0/8`` or ``::1``). False for
        https, for a base with no scheme or host, and for one that does not parse.

    Notes
    -----
    This only reports. LAN self-hosting over http is legitimate, so callers warn
    and carry on.

    Examples
    --------
    >>> is_insecure_fish_base("http://10.0.0.5:8080")
    True
    >>> is_insecure_fish_base("http://localhost:8080")
    False
    >>> is_insecure_fish_base("https://api.fish.audio")
    False
    """
    try:
        parts = urlsplit(base.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    if parts.scheme.lower() != "http" or not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True
