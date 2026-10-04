"""Fish ASR over httpx. Retries 429 and 5xx."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any, Final, cast

import httpx

from fish_audio_suite_kit import (
    FISH_ASR_PATH,
    FISH_RETRY_ATTEMPTS,
    AsrBody,
    FishHttpError,
    SuiteDefaults,
    asr_language_hint,
    bearer,
    describe_request_error,
    fish_attempt_exhausted,
    fish_backoff_s,
    is_asr_hallucination,
    is_caption_watermark,
    parse_asr_body,
    parse_fish_error,
    retry_after_s,
    scrub_asr,
    strip_base,
    without_watermark_segments,
)
from fish_audio_suite_voice.debug import debug, with_detail
from fish_audio_suite_voice.pause import sleep_unless
from fish_audio_suite_voice.tune import HTTP_KEEPALIVE_S
from fish_audio_suite_voice.ws_tap import public_meta

__all__ = [
    "asr_client",
    "fish_asr",
]

_ASR_TIMEOUT_S: Final = 60.0
_ASR_CONNECT_S: Final = 10.0
_REQUEST_ID_MAX: Final = 128


def _request_id(headers: Mapping[str, str], body: object) -> str:
    """Return Fish's request id from ``x-request-id`` or the body, or ``""``.

    transcribe-1-pro sends it on success and on errors. Fish support asks for it.
    """
    from_body = body.get("request_id") if isinstance(body, dict) else None
    for value in (headers.get("x-request-id"), from_body):
        if isinstance(value, str) and (line := value.strip()):
            return line.splitlines()[0][:_REQUEST_ID_MAX]
    return ""


def _json_or_none(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


async def _pause_or_raise(
    attempt: int,
    error: FishHttpError,
    cause: BaseException | None,
    cancel: threading.Event | None,
    retry_after: float | None = None,
) -> bool:
    """Return whether quit landed during the backoff and the next post must not run."""
    if fish_attempt_exhausted(attempt):
        if cause is None:
            raise error
        raise error from cause
    wait = fish_backoff_s(attempt, retry_after=retry_after)
    # Quit during the wait must not post the next try.
    return await sleep_unless(wait, cancel.is_set if cancel is not None else _never)


def _never() -> bool:
    return False


async def _post_fish(
    client: httpx.AsyncClient,
    url: str,
    cancel: threading.Event | None = None,
    **kwargs: Any,
) -> httpx.Response | None:
    """Post once per attempt. None means the quit flag stopped the retries."""
    last_error: FishHttpError | None = None
    for attempt in range(FISH_RETRY_ATTEMPTS):
        if cancel is not None and cancel.is_set():
            return None
        try:
            response = await client.post(url, **kwargs)
        except httpx.RequestError as exc:
            status, message = describe_request_error(exc, httpx.TimeoutException)
            last_error = FishHttpError.from_status(status, with_detail(message, exc))
            if await _pause_or_raise(attempt, last_error, exc, cancel):
                return None
            continue
        if response.status_code >= 400:
            detail = parse_fish_error(response.status_code, response.text)
            request_id = _request_id(response.headers, _json_or_none(response.text))
            message = (
                f"{detail.message} (request_id {request_id})" if request_id else detail.message
            )
            last_error = FishHttpError.from_status(
                detail.status,
                message,
                retry_after=retry_after_s(response.headers),
            )
            if last_error.retryable:
                if await _pause_or_raise(attempt, last_error, None, cancel, last_error.retry_after):
                    return None
                continue
            raise last_error
        return response
    raise last_error or FishHttpError.for_unreachable()


def asr_client() -> httpx.AsyncClient:
    """Build an ``httpx`` client with the ASR timeouts.

    Returns
    -------
    httpx.AsyncClient
        Reusable for a whole session so each utterance skips the TLS
        handshake. Use it as an async context manager, or close it with
        ``aclose``.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(_ASR_TIMEOUT_S, connect=_ASR_CONNECT_S),
        limits=httpx.Limits(keepalive_expiry=HTTP_KEEPALIVE_S),
    )


async def fish_asr(
    audio_wav: bytes,
    api_key: str,
    *,
    base: str,
    language: str = "",
    model: str = "",
    extra_headers: dict[str, str] | None = None,
    cancel: threading.Event | None = None,
    client: httpx.AsyncClient | None = None,
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
        Hint, reduced to its ISO 639-1 code (``en-US`` is sent as ``en``).
        Empty, or a value with no such code, omits the field so Fish detects
        the language. Fish may still return ``zh`` on noise.
    model : str, optional
        Fish ASR model id. Empty uses the ``SuiteDefaults`` model.
    extra_headers : dict or None, optional
        Usually a ``traceparent`` shared with the following TTS turn.
    cancel : threading.Event or None, optional
        When set during a request or a retry pause, return an empty string
        instead of posting again.
    client : httpx.AsyncClient or None, optional
        A session-scoped client from ``asr_client``. A new one is opened for
        this call when omitted.

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
    so a dead host fails before the read budget. ``cancel`` during a retry
    pause returns an empty string instead of posting again. A 429 or 5xx waits
    with jitter and honours ``Retry-After``.
    """
    headers = {
        "Authorization": bearer(api_key),
        "model": model or SuiteDefaults().asr_model,
        **(extra_headers or {}),
    }
    files = {"audio": ("utterance.wav", audio_wav, "audio/wav")}
    form: dict[str, str] = {}
    # A newline in the language value starts another multipart part, and Fish
    # may answer 400 to a hint that is not a two-letter code.
    lang = asr_language_hint(_form_language(language))
    if lang:
        form["language"] = lang
    async with AsyncExitStack() as stack:
        http = client or await stack.enter_async_context(asr_client())
        response = await _post_fish(
            http,
            f"{strip_base(base)}{FISH_ASR_PATH}",
            cancel,
            headers=headers,
            files=files,
            data=form or None,
        )
        if response is None:
            return ""
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise FishHttpError.for_non_json() from exc
    if isinstance(body, dict):
        meta = public_meta(body)
        debug(
            "asr.done status={} audio={}s lang={} sent={} chars={} trace={} request_id={}",
            response.status_code,
            meta.get("duration"),
            meta.get("language_code"),
            lang or "auto",
            meta.get("text_chars"),
            response.headers.get("x-fish-trace-id", "")[:12],
            _request_id(response.headers, body) or "-",
        )
    parsed, raw_text = parse_asr_body(body)
    text = scrub_asr(raw_text.strip(), strip_cues=True)
    # Duplex only hears this string. A blank field or a caption watermark
    # still drops words that Fish put in segments.
    if is_asr_hallucination(text):
        alt = _segment_text(parsed)
        if alt and not is_asr_hallucination(alt):
            return alt
    # The whole transcript is real speech plus a watermark sentence. The
    # hallucination check keeps that text, and the next turn answers it.
    trimmed = without_watermark_segments(text, parsed.get("segments"), strip_cues=True)
    if trimmed != text and trimmed and not is_asr_hallucination(trimmed):
        return trimmed
    return text


def _form_language(language: str) -> str:
    text = language.strip()
    cut = next((index for index, ch in enumerate(text) if ord(ch) < 32), None)
    if cut is not None:
        text = text[:cut].strip()
    return text


def _segment_text(body: AsrBody) -> str:
    # Fish may send any shape. The kit only checks "text", so each piece is
    # narrowed here before it is read.
    segments: object = body.get("segments")
    if not isinstance(segments, list):
        return ""
    parts: list[str] = []
    for seg in cast(list[object], segments):
        if not isinstance(seg, dict):
            continue
        raw = cast(dict[str, object], seg).get("text")
        if not isinstance(raw, str):
            continue
        piece = scrub_asr(raw.strip(), strip_cues=True)
        # A watermark segment beside real speech would be spoken with it.
        if piece and not is_caption_watermark(piece):
            parts.append(piece)
    return " ".join(parts)
