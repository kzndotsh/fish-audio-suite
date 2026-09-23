"""OpenAI error envelope for local and Fish failures."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from fish_audio_suite_kit import parse_fish_error, utf8_text

_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def _openai_error_type(status: int) -> str:
    named = _ERROR_TYPES.get(status)
    if named is not None:
        return named
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


def openai_error_body(status: int, message: str, *, provider: bool = False) -> dict[str, Any]:
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

    Returns
    -------
    dict
        The envelope, not a response.
    """
    err: dict[str, Any] = {
        "code": int(status),
        "message": utf8_text(str(message)),
        "type": "provider_error" if provider else _openai_error_type(status),
    }
    if provider:
        err["metadata"] = {"provider_name": "fish-audio"}
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
        parsed = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return json_error(400, "invalid JSON body")
    if not isinstance(parsed, dict):
        return json_error(400, "JSON body must be an object")
    return parsed


def json_from_upstream(status: int, raw: Any) -> JSONResponse:
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
    detail = parse_fish_error(status, raw)
    code = int(detail["status"])
    return JSONResponse(
        openai_error_body(code, str(detail["message"]), provider=True),
        status_code=code,
    )
