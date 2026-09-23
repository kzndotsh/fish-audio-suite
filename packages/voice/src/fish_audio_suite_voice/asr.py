"""Fish ASR over httpx. Retries 429 and 5xx."""

from __future__ import annotations

import json
from typing import Any

import httpx

from fish_audio_suite_kit import (
    FISH_ASR_PATH,
    FISH_RETRY_ATTEMPTS,
    FishHttpError,
    SuiteDefaults,
    bearer,
    fish_non_json,
    fish_request_error,
    fish_retry_pause,
    fish_unreachable,
    parse_asr_body,
    parse_fish_error,
    scrub_asr,
    should_retry_fish_status,
    strip_base,
)
from fish_audio_suite_voice.debug import debug, header_meta, public_meta

_ASR_TIMEOUT_S = 60.0


async def _pause_or_raise(
    attempt: int,
    error: FishHttpError,
    cause: BaseException | None,
) -> None:
    if not await fish_retry_pause(attempt):
        return
    if cause is None:
        raise error
    raise error from cause


async def _post_fish(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    last_error: FishHttpError | None = None
    for attempt in range(FISH_RETRY_ATTEMPTS):
        try:
            response = await client.post(url, **kwargs)
        except httpx.RequestError as exc:
            status, message = fish_request_error(exc, httpx.TimeoutException)
            last_error = FishHttpError(status, message)
            await _pause_or_raise(attempt, last_error, exc)
            continue
        if response.status_code >= 400:
            detail = parse_fish_error(response.status_code, response.text)
            last_error = FishHttpError(int(detail["status"]), str(detail["message"]))
            if should_retry_fish_status(last_error.status):
                await _pause_or_raise(attempt, last_error, None)
                continue
            raise last_error
        return response
    status, message = fish_unreachable()
    raise last_error or FishHttpError(status, message)


async def fish_asr(
    audio_wav: bytes,
    api_key: str,
    *,
    base: str,
    language: str = "",
    extra_headers: dict[str, str] | None = None,
) -> str:
    headers = {
        "Authorization": bearer(api_key),
        "model": SuiteDefaults().asr_model,
        **(extra_headers or {}),
    }
    files = {"audio": ("utterance.wav", audio_wav, "audio/wav")}
    data: dict[str, str] = {}
    if language:
        data["language"] = language
    async with httpx.AsyncClient(timeout=_ASR_TIMEOUT_S) as client:
        response = await _post_fish(
            client,
            f"{strip_base(base)}{FISH_ASR_PATH}",
            headers=headers,
            files=files,
            data=data or None,
        )
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        status, message = fish_non_json()
        raise FishHttpError(status, message) from exc
    if isinstance(body, dict):
        debug(
            "fish.asr status={} language_sent={!r} headers={} meta={}",
            response.status_code,
            language or "auto",
            header_meta(dict(response.headers)),
            public_meta(body),
        )
    raw_text = parse_asr_body(body)[1]
    return scrub_asr(raw_text.strip())
