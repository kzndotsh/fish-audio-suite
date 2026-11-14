"""Fish ASR over httpx, and the recogniser the batch path hears a clip with.

``fish_asr`` is the request: it retries 429, 5xx and network errors and returns the cleaned
text. ``transcribe_clip`` is the recogniser: it times the request, shares its trace id with the
reply, and turns every failure into the reason there is no turn.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Final, NoReturn

import httpx

from fish_audio_suite_kit import (
    FISH_ASR_PATH,
    FishAuthError,
    FishHttpError,
    SuiteDefaults,
    asr_language_hint,
    asr_tone,
    bearer,
    clean_transcript,
    cut_at_control,
    elapsed_ms,
    fish_request_id,
    make_traceparent,
    parse_asr_body,
    parse_fish_error,
    retry_after_s,
    strip_base,
    trace_id_of,
)
from fish_audio_suite_voice.asr.result import NoTurn, Transcript
from fish_audio_suite_voice.cli.wording import transport_failure
from fish_audio_suite_voice.config.settings import VoiceCliConfig
from fish_audio_suite_voice.logutil.debug import debug, warn
from fish_audio_suite_voice.logutil.meta import public_meta
from fish_audio_suite_voice.session.cancel import never
from fish_audio_suite_voice.session.pause import Retry, fish_retries

__all__ = [
    "AsrResult",
    "asr_client",
    "fish_asr",
    "transcribe_clip",
]

_ASR_TIMEOUT_S: Final = 60.0
_ASR_CONNECT_S: Final = 10.0
# Idle seconds the session's ASR connection stays open. httpx's default of 5 s is shorter
# than the gap between two turns, so each turn would pay a new TLS handshake.
_ASR_KEEPALIVE_S: Final = 120.0


@dataclass(frozen=True, slots=True)
class AsrResult:
    """One transcribed clip.

    Attributes
    ----------
    text : str
        The scrubbed transcript, with Fish's annotations removed. This is what the turn is
        judged and answered on.
    tone : tuple of str
        The annotations Fish heard in the voice, such as ``("laughter", "tired")``. Empty
        unless ``fish_asr`` was asked to keep them.
    """

    text: str
    tone: tuple[str, ...] = ()


async def _post_fish(
    client: httpx.AsyncClient,
    url: str,
    cancel: threading.Event | None = None,
    **kwargs: Any,
) -> httpx.Response | None:
    """Post once per attempt. None means the quit flag stopped the retries."""

    async def attempt(_number: int) -> httpx.Response | Retry[httpx.Response]:
        try:
            response = await client.post(url, **kwargs)
        except httpx.RequestError as exc:
            return Retry(_raises(transport_failure(exc), exc))
        if response.status_code >= 400:
            detail = parse_fish_error(
                response.status_code, response.text, response.headers, header_first=True
            )
            message = (
                f"{detail.message} (request_id {detail.request_id})"
                if detail.request_id
                else detail.message
            )
            error = FishHttpError(
                detail.status,
                message,
                retry_after=retry_after_s(response.headers),
            )
            if not error.retryable:
                raise error
            return Retry(_raises(error), error.retry_after)
        return response

    return await fish_retries(attempt, cancel.is_set if cancel is not None else never)


def _raises(error: FishHttpError, cause: BaseException | None = None) -> Callable[[], NoReturn]:
    """Return what raises ``error`` (from ``cause``) when no attempt is left."""

    def give_up() -> NoReturn:
        if cause is None:
            raise error
        raise error from cause

    return give_up


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
        limits=httpx.Limits(keepalive_expiry=_ASR_KEEPALIVE_S),
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
    hear_tone: bool = False,
) -> AsrResult:
    """Transcribe one WAV with Fish ASR, retrying 429, 5xx and network errors.

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
        When set before an attempt or during a retry pause, return an empty
        ``AsrResult`` instead of posting again. A request already sent is not cut short.
    client : httpx.AsyncClient or None, optional
        A session-scoped client from ``asr_client``. A new one is opened for
        this call when omitted.
    hear_tone : bool, optional
        Also return the annotations Fish put in the transcript (``[laughter]``). The text
        is scrubbed of them either way.

    Returns
    -------
    AsrResult
        The scrubbed transcript, and its tone when ``hear_tone`` is set.

    Raises
    ------
    FishHttpError
        Non-retryable status, exhausted retries, or a body that is not a
        JSON object with a string ``text``.

    Notes
    -----
    Connect times out in 10 seconds. Read, write, and pool wait 60 seconds,
    so a dead host fails before the read budget. A 429 or 5xx waits with
    jitter and honours ``Retry-After``.
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
    lang = asr_language_hint(cut_at_control(language.strip()).strip())
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
            return AsrResult("")
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
            fish_request_id(body, response.headers, header_first=True) or "-",
        )
    parsed, raw_text = parse_asr_body(body)
    tone = asr_tone(raw_text) if hear_tone else ()
    # The loop answers only this text: it falls back to the segments when the
    # transcript is blank or a watermark, and drops watermark segments beside speech.
    clean = clean_transcript(
        raw_text.strip(),
        parsed,
        strip_speakers=True,
        strip_cues=True,
    )
    return AsrResult(clean.text, tone)


async def transcribe_clip(
    wav: bytes,
    config: VoiceCliConfig,
    *,
    cancel: threading.Event,
    client: httpx.AsyncClient,
) -> Transcript | NoTurn:
    """Hear one clip with Fish ASR.

    Parameters
    ----------
    wav : bytes
        One utterance.
    config : VoiceCliConfig
        The key, base, language hint, model and whether to keep the tone.
    cancel : threading.Event
        The session's quit flag. Once it is set the clip is ``stopped``, whatever the request
        did, and no failure is reported.
    client : httpx.AsyncClient
        The session's ASR client.

    Returns
    -------
    Transcript or NoTurn
        The words, timed from the start of the request, with the trace id the reply's TTS
        turn shares. Otherwise ``fatal`` when Fish refuses the key, ``again`` for any other
        Fish or network failure, which is worth another listen, or ``stopped`` on quit.

    Raises
    ------
    Exception
        Anything that is not a Fish or network failure is a bug and propagates, instead of
        the loop listening again in silence.
    """
    started = time.perf_counter()
    parent = make_traceparent()
    try:
        heard = await fish_asr(
            wav,
            config.fish_api_key,
            base=config.fish_base,
            language=config.fish_asr_language,
            model=config.asr_model,
            extra_headers={"traceparent": parent},
            cancel=cancel,
            client=client,
            hear_tone=config.hear_tone,
        )
    except FishHttpError as e:
        if cancel.is_set():
            return NoTurn("stopped")
        warn(f"[asr] {e.status} {e.message}")
        return NoTurn("fatal" if isinstance(e, FishAuthError) else "again")
    except (httpx.HTTPError, OSError) as e:
        if cancel.is_set():
            return NoTurn("stopped")
        warn(f"[asr] {e}")
        return NoTurn("again")
    if cancel.is_set():
        return NoTurn("stopped")
    return Transcript(
        heard.text,
        tone=heard.tone,
        asr_ms=elapsed_ms(started),
        started=started,
        trace_id=trace_id_of(parent),
    )
