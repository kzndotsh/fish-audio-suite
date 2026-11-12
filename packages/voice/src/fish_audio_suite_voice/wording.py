"""Two small text helpers for what the terminal shows: an error's detail, and a short model id."""

from __future__ import annotations

__all__ = [
    "short_model",
    "with_detail",
]


def with_detail(message: str, exc: BaseException) -> str:
    """Add an error's own text to a fixed message, for the local terminal.

    Parameters
    ----------
    message : str
        The fixed message, for example the kit's ``Fish upstream unreachable``.
    exc : BaseException
        The error it stands for.

    Returns
    -------
    str
        ``message: text`` when the error has text of its own, otherwise
        ``message``. Kit keeps client-facing messages free of exception text;
        voice is a local tool, so the user sees the detail here.
    """
    detail = str(exc).strip()
    return f"{message}: {detail}" if detail and detail != message else message


def short_model(model: object) -> str:
    """Return a model id without its vendor prefix, for display only.

    Parameters
    ----------
    model : object
        An id such as ``vendor/name-7b``. ``None`` or blank gives ``""``.

    Returns
    -------
    str
        The part after the last ``/``. Any ``:suffix`` stays. Use the full id
        wherever two ids are compared or sent to the API.
    """
    return str(model or "").strip().rsplit("/", 1)[-1]
