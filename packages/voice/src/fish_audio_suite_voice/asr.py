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
    is_asr_hallucination,
    is_caption_watermark,
    parse_asr_body,
    parse_fish_error,
    scrub_asr,
    should_retry_fish_status,
    strip_base,
    without_watermark_segments,
)
from fish_audio_suite_voice.debug import debug, header_meta, public_meta
from fish_audio_suite_voice.signals import STOP_RECORD

_ASR_TIMEOUT_S = 60.0
_ASR_CONNECT_S = 10.0


async def _pause_or_raise(
    attempt: int,
    error: FishHttpError,
    cause: BaseException | None,
) -> bool:
    """Return whether quit landed during the backoff and the next post must not run."""
    if not await fish_retry_pause(attempt):
        # The pause slept. Quit during that wait must not post the next try.
        return STOP_RECORD.is_set()
    if cause is None:
        raise error
    raise error from cause


async def _post_fish(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response | None:
    """Post once per attempt. None means the duplex quit flag stopped the retries."""
    last_error: FishHttpError | None = None
    for attempt in range(FISH_RETRY_ATTEMPTS):
        if STOP_RECORD.is_set():
            return None
        try:
            response = await client.post(url, **kwargs)
        except httpx.RequestError as exc:
            status, message = fish_request_error(exc, httpx.TimeoutException)
            last_error = FishHttpError(status, message)
            if await _pause_or_raise(attempt, last_error, exc):
                return None
            continue
        if response.status_code >= 400:
            detail = parse_fish_error(response.status_code, response.text)
            last_error = FishHttpError(int(detail["status"]), str(detail["message"]))
            if should_retry_fish_status(last_error.status):
                if await _pause_or_raise(attempt, last_error, None):
                    return None
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
    """Transcribe one WAV with Fish ASR, retrying 429 and 5xx.

    Parameters
    ----------
    audio_wav : bytes
        A mono WAV of one utterance.
    api_key : str
        Fish key. Sent as ``Bearer``.
    base : str
        Fish origin without a trailing slash.
    language : str, optional
        Hint. Empty omits the field. Fish may still return ``zh`` on noise.
    extra_headers : dict or None, optional
        Usually a ``traceparent`` shared with the following TTS turn.

    Returns
    -------
    str
        Scrubbed transcript.

    Raises
    ------
    FishHttpError
        Non-retryable status, exhausted retries, or a body that is not a
        JSON object with a string ``text``.

    Notes
    -----
    Connect times out in 10 seconds. Read, write, and pool wait 60 seconds,
    so a dead host fails before the read budget. ``STOP_RECORD`` during a
    retry pause returns an empty string instead of posting again.
    """
    headers = {
        "Authorization": bearer(api_key),
        "model": SuiteDefaults().asr_model,
        **(extra_headers or {}),
    }
    files = {"audio": ("utterance.wav", audio_wav, "audio/wav")}
    data: dict[str, str] = {}
    # A newline in the language value starts another multipart part.
    lang = _form_language(language)
    if lang:
        data["language"] = lang
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(_ASR_TIMEOUT_S, connect=_ASR_CONNECT_S),
    ) as client:
        response = await _post_fish(
            client,
            f"{strip_base(base)}{FISH_ASR_PATH}",
            headers=headers,
            files=files,
            data=data or None,
        )
        if response is None:
            return ""
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
    data, raw_text = parse_asr_body(body)
    text = scrub_asr(raw_text.strip())
    # Duplex only hears this string. A blank field or a caption watermark
    # still drops words that Fish put in segments.
    if is_asr_hallucination(text):
        alt = _segment_text(data)
        if alt and not is_asr_hallucination(alt):
            return alt
    # The whole transcript is real speech plus a watermark sentence. The
    # hallucination check keeps that text, and the next turn answers it.
    trimmed = without_watermark_segments(text, data.get("segments"))
    if trimmed != text and trimmed and not is_asr_hallucination(trimmed):
        return trimmed
    return text


def _form_language(language: str) -> str:
    text = language.strip()
    cut = next((index for index, ch in enumerate(text) if ord(ch) < 32), None)
    if cut is not None:
        text = text[:cut].strip()
    return text


def _segment_text(data: dict[str, Any]) -> str:
    segments = data.get("segments")
    if not isinstance(segments, list):
        return ""
    parts: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        raw = seg.get("text")
        if not isinstance(raw, str):
            continue
        piece = scrub_asr(raw.strip())
        # A watermark segment beside real speech would be spoken with it.
        if piece and not is_caption_watermark(piece):
            parts.append(piece)
    return " ".join(parts)
