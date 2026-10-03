"""OpenAI error envelope for local and Fish failures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import (
    FishErrorBody,
    FishHttpError,
    OpenAIErrorBody,
    OpenAIErrorDetail,
    parse_fish_error,
    utf8_text,
)

__all__ = [
    "ProxyError",
    "json_error",
    "json_from_call_failure",
    "openai_error_body",
    "provider_json_error",
    "provider_json_from_raw",
    "proxy_error_response",
    "read_json_object",
]

_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


class ProxyError(Exception):
    """A request the proxy refuses. The app maps it to an OpenAI error envelope.

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


def _openai_error_type(status: int) -> str:
    named = _ERROR_TYPES.get(status)
    if named is not None:
        return named
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


def openai_error_body(
    status: int,
    message: str,
    *,
    provider: bool = False,
    metadata: Mapping[str, str] | None = None,
) -> OpenAIErrorBody:
    """OpenAI ``{error: {code, message, type}}`` object.

    Parameters
    ----------
    status : int
        HTTP status stored as ``code``.
    message : str
        Client-facing message.
    provider : bool, optional
        When True, ``type`` is ``provider_error`` and metadata names Fish.
        Local validation keeps the OpenAI type for that status.
    metadata : Mapping or None, optional
        More provider details, such as Fish's ``provider_code`` and
        ``request_id``, added to ``metadata`` beside ``provider_name``. Used
        only when ``provider`` is True.

    Returns
    -------
    OpenAIErrorBody
        The envelope, not a response.
    """
    err: OpenAIErrorDetail = {
        "code": int(status),
        "message": utf8_text(str(message)),
        "type": "provider_error" if provider else _openai_error_type(status),
    }
    if provider:
        err["metadata"] = {**(metadata or {}), "provider_name": "fish-audio"}
    return {"error": err}


def json_error(status: int, message: str) -> JSONResponse:
    """Local validation or proxy failure. Not marked as a Fish provider error.

    Parameters
    ----------
    status : int
        HTTP status of the response.
    message : str
        Shown in ``error.message``.

    Returns
    -------
    JSONResponse
        OpenAI envelope with the matching status code.
    """
    return JSONResponse(openai_error_body(status, message), status_code=status)


def json_from_call_failure(err: FishHttpError) -> JSONResponse:
    """Turn a Fish call that gave no usable answer into the proxy's own error.

    Parameters
    ----------
    err : FishHttpError
        An error built by the kit: unreachable, timed out, or a reply body
        that is not JSON or not an object.

    Returns
    -------
    JSONResponse
        OpenAI envelope with the error's status and fixed message. It is not
        marked as a Fish provider error, because Fish never answered.
    """
    return json_error(err.status, err.message)


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
        return json_error(exc.status, exc.message)
    return json_error(500, "internal error")


async def read_json_object(request: Request) -> dict[str, Any] | JSONResponse:
    """Parse a JSON object body, or an OpenAI 400 response.

    Parameters
    ----------
    request : Request
        Incoming FastAPI request.

    Returns
    -------
    dict or JSONResponse
        The object when the body is a JSON object. A 400 response when the
        body is not JSON or is a JSON array or scalar. Callers must check
        the type before treating the result as fields.
    """
    try:
        # utf-8-sig drops a leading BOM. json.loads rejects that byte and the
        # request would 400 before Fish saw the text.
        raw = await request.body()
        parsed = json.loads(raw.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return json_error(400, "invalid JSON body")
    if not isinstance(parsed, dict):
        return json_error(400, "JSON body must be an object")
    return parsed


def provider_json_from_raw(status: int, raw: Any) -> JSONResponse:
    """Turn a Fish error body into an OpenAI provider error.

    Parameters
    ----------
    status : int
        Upstream HTTP status. ``parse_fish_error`` may replace it from the body.
    raw : Any
        Fish bytes, text, or an already-decoded object.

    Returns
    -------
    JSONResponse
        ``type`` is ``provider_error`` and ``metadata.provider_name`` is
        ``fish-audio``. The response status is the parsed Fish status.
    """
    return provider_json_error(parse_fish_error(status, raw))


def provider_json_error(
    detail: FishErrorBody, *, metadata: Mapping[str, str] | None = None
) -> JSONResponse:
    """Turn an already parsed Fish error into an OpenAI provider error.

    Parameters
    ----------
    detail : FishErrorBody
        The status and message from ``parse_fish_error``.
    metadata : Mapping or None, optional
        Fish's own error ``code`` (as ``provider_code``) and ``request_id``,
        when it sent them. ``error.code`` stays the HTTP status.

    Returns
    -------
    JSONResponse
        ``type`` is ``provider_error`` and ``metadata.provider_name`` is
        ``fish-audio``. The response status is ``detail.status``.
    """
    return JSONResponse(
        openai_error_body(detail.status, detail.message, provider=True, metadata=metadata),
        status_code=detail.status,
    )
