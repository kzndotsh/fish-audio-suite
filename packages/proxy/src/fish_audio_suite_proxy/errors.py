"""The one error channel: every refusal is a ``ProxyError``, shown as an OpenAI error envelope."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fastapi import Request
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import OpenAIErrorBody, OpenAIErrorDetail, utf8_text

__all__ = ["FishProviderError", "ProxyError", "proxy_error_response", "unsupported_format"]

_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


class ProxyError(Exception):
    """A request the proxy refuses. The app's handler turns it into the response.

    Attributes
    ----------
    status : int
        HTTP status of the response.
    message : str
        Shown in ``error.message``.
    """

    def __init__(self, status: int, message: str) -> None:
        """Store the status and message.

        Parameters
        ----------
        status : int
            HTTP status of the response.
        message : str
            Client-facing reason.
        """
        self.status: int = int(status)
        self.message: str = message
        super().__init__(message)

    def response(self) -> JSONResponse:
        """Return the OpenAI error envelope for this refusal.

        Returns
        -------
        JSONResponse
            ``{error: {code, message, type}}`` with the error's status. The type
            follows the status, as OpenAI's own errors do.
        """
        return _envelope(self.status, self.message)


class FishProviderError(ProxyError):
    """Fish answered with an error. The client sees it as a provider error.

    Attributes
    ----------
    status : int
        Fish's HTTP status, or the status its error body names.
    message : str
        Fish's message.
    metadata : Mapping of str to str
        Fish's own error ``code`` (as ``provider_code``) and ``request_id``, when
        it sent them.
    """

    def __init__(
        self, status: int, message: str, *, metadata: Mapping[str, str] | None = None
    ) -> None:
        """Store Fish's status, message and ids.

        Parameters
        ----------
        status : int
            HTTP status of the response.
        message : str
            Fish's message.
        metadata : Mapping or None, optional
            Fish's ``provider_code`` and ``request_id``. ``error.code`` stays the
            HTTP status.
        """
        super().__init__(status, message)
        self.metadata: Mapping[str, str] = dict(metadata or {})

    def response(self) -> JSONResponse:
        """Return the envelope with ``type`` ``provider_error`` and Fish's metadata.

        Returns
        -------
        JSONResponse
            ``metadata.provider_name`` is ``fish-audio``, beside any ids Fish sent.
        """
        return _envelope(self.status, self.message, metadata=self.metadata)


def _openai_error_type(status: int) -> str:
    named = _ERROR_TYPES.get(status)
    if named is not None:
        return named
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


def _envelope(
    status: int, message: str, *, metadata: Mapping[str, str] | None = None
) -> JSONResponse:
    # Metadata marks a Fish error. Local validation keeps the OpenAI type for its status.
    err: OpenAIErrorDetail = {
        "code": int(status),
        "message": utf8_text(str(message)),
        "type": "provider_error" if metadata is not None else _openai_error_type(status),
    }
    if metadata is not None:
        err["metadata"] = {**metadata, "provider_name": "fish-audio"}
    body: OpenAIErrorBody = {"error": err}
    return JSONResponse(body, status_code=status)


def unsupported_format(name: str, allowed: Iterable[str]) -> ProxyError:
    """Build the 400 for a ``response_format`` the route cannot produce.

    Parameters
    ----------
    name : str
        The format the client named, already lowercased. Only its first 32
        characters are shown.
    allowed : Iterable of str
        The formats the route accepts, listed in the message.

    Returns
    -------
    ProxyError
        Status 400. Returning another format than the one asked for would break
        the client's decoder, so the request is refused instead.
    """
    return ProxyError(
        400, f"unsupported response_format {name[:32]!r}; use one of: {', '.join(allowed)}"
    )


def proxy_error_response(_request: Request, exc: Exception) -> JSONResponse:
    """Exception handler that turns a ``ProxyError`` into its error response.

    Parameters
    ----------
    _request : Request
        Unused. Part of the Starlette handler signature.
    exc : Exception
        A ``ProxyError``. Any other exception is a 500.

    Returns
    -------
    JSONResponse
        OpenAI envelope with the error's status.
    """
    if isinstance(exc, ProxyError):
        return exc.response()
    return _envelope(500, "internal error")
