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
    err: dict[str, Any] = {
        "code": int(status),
        "message": utf8_text(str(message)),
        "type": "provider_error" if provider else _openai_error_type(status),
    }
    if provider:
        err["metadata"] = {"provider_name": "fish-audio"}
    return {"error": err}


def json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(openai_error_body(status, message), status_code=status)


async def read_json_object(request: Request) -> dict[str, Any] | JSONResponse:
    try:
        parsed = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return json_error(400, "invalid JSON body")
    if not isinstance(parsed, dict):
        return json_error(400, "JSON body must be an object")
    return parsed


def json_from_upstream(status: int, raw: Any) -> JSONResponse:
    detail = parse_fish_error(status, raw)
    code = int(detail["status"])
    return JSONResponse(
        openai_error_body(code, str(detail["message"]), provider=True),
        status_code=code,
    )
