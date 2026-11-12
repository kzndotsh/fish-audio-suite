"""One transcription end to end: the OpenAI upload, the Fish ASR call, and the reply."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, get_args

import httpx
from fastapi import Request
from fastapi.responses import PlainTextResponse
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException

from fish_audio_suite_kit import (
    PRO_ASR_MODEL,
    AsrBody,
    AsrFormat,
    CaptionCue,
    CleanTranscript,
    FishHttpError,
    asr_language_hint,
    clean_transcript,
    cut_at_control,
    fish_request_id,
    format_as_srt,
    format_as_vtt,
    parse_asr_body,
    parse_flag,
    trace_id_of,
    utf8_text,
)
from fish_audio_suite_proxy.errors import ProxyError, unsupported_format
from fish_audio_suite_proxy.models import resolve_asr_model
from fish_audio_suite_proxy.phrases import AsrScrub, caption_cues, json_number, word_cues
from fish_audio_suite_proxy.request_body import decode_audio_b64, read_json_object
from fish_audio_suite_proxy.settings import ProxySettings
from fish_audio_suite_proxy.upstream import traced_model_headers

__all__ = ["fish_asr_request", "read_asr_request", "transcription_reply"]

log: Final[logging.Logger] = logging.getLogger("fish-audio-suite-proxy")

ASR_FORMATS: tuple[AsrFormat, ...] = get_args(AsrFormat)
_ASR_FORMAT_BY_NAME: dict[str, AsrFormat] = {name: name for name in ASR_FORMATS}
_DEFAULT_FORMAT: Final[AsrFormat] = "json"
_DIARIZE_VALUES: Final = ("auto", "true", "false")
# No speaker count needs more digits, and the cap keeps int() off a huge string.
_MAX_COUNT_DIGITS: Final = 6
_MAX_FILENAME_CHARS: Final = 255
_TIMED_FORMATS: Final = frozenset({"verbose_json", "vtt", "srt"})

type FieldReader = Callable[[str], object]
"""Reads one named field of the request: ``dict.get`` for JSON, ``FormData.get`` for a form."""


@dataclass(frozen=True, slots=True)
class ProAsrOptions:
    """Fields only ``transcribe-1-pro`` accepts, after validation.

    Attributes
    ----------
    diarize : str or None
        ``auto``, ``true`` or ``false``, or None to let Fish decide.
    num_speakers : int or None
        Exact speaker count hint, at least 1.
    min_speakers : int or None
        Lower speaker count hint, at least 1.
    max_speakers : int or None
        Upper speaker count hint, at least ``min_speakers``.
    tag_audio_events : bool or None
        Whether Fish adds emotion and event cues such as ``[laughter]``.
    """

    diarize: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    tag_audio_events: bool | None = None

    def form_fields(self) -> dict[str, str]:
        """Return the set fields as Fish multipart values.

        Returns
        -------
        dict
            Field name to value. Booleans are exactly ``true`` or ``false``.
            A field left as None is not included.
        """
        fields: dict[str, str] = {}
        if self.diarize is not None:
            fields["diarize"] = self.diarize
        for name, count in (
            ("num_speakers", self.num_speakers),
            ("min_speakers", self.min_speakers),
            ("max_speakers", self.max_speakers),
        ):
            if count is not None:
                fields[name] = str(count)
        if self.tag_audio_events is not None:
            fields["tag_audio_events"] = "true" if self.tag_audio_events else "false"
        return fields


@dataclass(frozen=True, slots=True)
class InboundAsr:
    """A transcription request after parsing, from a multipart form or JSON.

    Attributes
    ----------
    audio : bytes
        The audio to transcribe.
    filename : str
        Name sent to Fish with the upload.
    content_type : str
        Media type sent to Fish with the upload.
    model : str or None
        The client's model, or None when it sent none.
    language : str
        Language hint, or an empty string for none.
    response_format : AsrFormat
        The reply format. ``json`` when the client named none.
    granularities : tuple of str
        ``timestamp_granularities`` values.
    pro : ProAsrOptions
        Validated ``transcribe-1-pro`` fields. Sent to Fish only when the
        resolved model is ``transcribe-1-pro``.
    """

    audio: bytes
    filename: str
    content_type: str
    model: str | None
    language: str
    response_format: AsrFormat
    granularities: tuple[str, ...]
    pro: ProAsrOptions


def _one_line(value: str, default: str) -> str:
    # A control character in a multipart value or header would start another one.
    return cut_at_control(value.strip()).strip() or default


def _text(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _read_diarize(value: object) -> str | None:
    if _blank(value):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    word = value.strip().lower() if isinstance(value, str) else ""
    if word not in _DIARIZE_VALUES:
        raise ProxyError(400, "diarize must be auto, true or false")
    return word


def _read_count(name: str, value: object) -> int | None:
    if _blank(value):
        return None
    count: int | None = None
    if isinstance(value, int) and not isinstance(value, bool):
        count = value
    elif isinstance(value, str):
        digits = value.strip()
        if digits.isascii() and digits.isdigit() and len(digits) <= _MAX_COUNT_DIGITS:
            count = int(digits)
    if count is None or count < 1:
        raise ProxyError(400, f"{name} must be a whole number of at least 1")
    return count


def _read_switch(name: str, value: object) -> bool | None:
    if _blank(value):
        return None
    if isinstance(value, bool):
        return value
    switch = parse_flag(value, strict=True) if isinstance(value, str) else None
    if switch is None:
        raise ProxyError(400, f"{name} must be true or false")
    return switch


def _pro_options(read: FieldReader) -> ProAsrOptions:
    # 400 for a value Fish would refuse, so the client learns it before the upload.
    options = ProAsrOptions(
        diarize=_read_diarize(read("diarize")),
        num_speakers=_read_count("num_speakers", read("num_speakers")),
        min_speakers=_read_count("min_speakers", read("min_speakers")),
        max_speakers=_read_count("max_speakers", read("max_speakers")),
        tag_audio_events=_read_switch("tag_audio_events", read("tag_audio_events")),
    )
    low, high = options.min_speakers, options.max_speakers
    if options.num_speakers is not None and (low is not None or high is not None):
        raise ProxyError(400, "num_speakers cannot be combined with min_speakers or max_speakers")
    if low is not None and high is not None and low > high:
        raise ProxyError(400, "min_speakers cannot be more than max_speakers")
    counted = options.num_speakers is not None or low is not None or high is not None
    if counted and options.diarize == "false":
        raise ProxyError(400, "speaker counts cannot be sent with diarize=false")
    return options


def _response_format(value: object) -> AsrFormat:
    # A newline is not part of the name, and a blank or missing one is json.
    token = _one_line(_text(value, "").lower(), _DEFAULT_FORMAT)
    fmt = _ASR_FORMAT_BY_NAME.get(token)
    if fmt is None:
        raise unsupported_format(token, ASR_FORMATS)
    return fmt


def _granularities(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(item.strip() for item in values if isinstance(item, str) and item.strip())


def _inbound(
    read: FieldReader,
    *,
    audio: bytes,
    filename: str,
    content_type: str,
    granularities: Iterable[object],
) -> InboundAsr:
    model = read("model")
    language = _text(read("language"), "")
    pro = _pro_options(read)
    return InboundAsr(
        audio=audio,
        filename=filename,
        content_type=content_type,
        model=model if isinstance(model, str) else None,
        language=language,
        # Read after the pro fields, so a bad pro field is the error a client sees first.
        response_format=_response_format(read("response_format")),
        granularities=_granularities(granularities),
        pro=pro,
    )


def _listed(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items: list[Any] = value
        return items
    return []


def _asr_from_json(parsed: dict[str, Any]) -> InboundAsr:
    inner = parsed.get("input_audio")
    audio_obj: Mapping[str, Any] = inner if isinstance(inner, dict) else {}
    fmt = _one_line(_text(audio_obj.get("format"), "wav"), "wav")
    audio = decode_audio_b64(audio_obj.get("data"), field="input_audio")
    return _inbound(
        parsed.get,
        audio=audio,
        filename=f"utterance.{fmt}",
        content_type=f"audio/{fmt}",
        granularities=_listed(parsed.get("timestamp_granularities")),
    )


def _filename(raw: str | None) -> str:
    # The name becomes a multipart header value. Keep the last path part and
    # cap the length so a client cannot inflate the upstream request.
    name = _one_line((raw or "").replace("\\", "/").rsplit("/", 1)[-1], "")
    return name[:_MAX_FILENAME_CHARS] or "audio.webm"


def _asr_from_form(form: FormData, audio: bytes, upload: UploadFile) -> InboundAsr:
    return _inbound(
        form.get,
        audio=audio,
        filename=_filename(upload.filename),
        content_type=_one_line(upload.content_type or "", "application/octet-stream"),
        granularities=[
            *form.getlist("timestamp_granularities"),
            *form.getlist("timestamp_granularities[]"),
        ],
    )


_FORM_ERRORS = {
    400: "invalid multipart form body",
    413: "form upload is too large",
    422: "form fields could not be read",
}


async def read_asr_request(request: Request) -> InboundAsr:
    """Read multipart ``file`` or JSON ``input_audio`` into one upload.

    Parameters
    ----------
    request : Request
        Transcription request. JSON is chosen when ``Content-Type`` contains
        ``application/json``.

    Returns
    -------
    InboundAsr
        Audio plus the fields that affect the Fish form and the reply.

    Raises
    ------
    ProxyError
        400 for a body that is not JSON or a form, empty or undecodable audio, a
        ``transcribe-1-pro`` field Fish would refuse (a ``diarize`` other than
        auto, true or false, a speaker count below 1 or in a contradicting
        combination, a ``tag_audio_events`` that is not a boolean), or a
        ``response_format`` not in ``ASR_FORMATS``. The form parser's own status
        (such as 413) with a fixed message, because its text can name internals.
    """
    if "application/json" in request.headers.get("content-type", ""):
        return _asr_from_json(await read_json_object(request))
    try:
        # The context closes the form and every upload on all paths. A form that
        # is only awaited leaves its SpooledTemporaryFile open until garbage
        # collection. Everything the result needs is copied out before it closes.
        async with request.form() as form:
            upload = form.get("file")
            audio = await upload.read() if isinstance(upload, UploadFile) else b""
            if not isinstance(upload, UploadFile) or not audio:
                raise ProxyError(400, "empty audio upload")
            return _asr_from_form(form, audio, upload)
    except HTTPException as exc:
        raise ProxyError(
            exc.status_code, _FORM_ERRORS.get(exc.status_code, "invalid form body")
        ) from exc


@dataclass(frozen=True, slots=True)
class FishAsrRequest:
    """The Fish ``/v1/asr`` call for one transcription.

    Attributes
    ----------
    model : str
        The resolved Fish ASR model, sent as the ``model`` header.
    headers : dict
        The model and trace headers.
    files : dict
        The multipart audio part: field name to (filename, bytes, media type).
    form : dict
        The other multipart fields.
    language : str
        The language hint sent to Fish, or ``""`` when none was sent.
    """

    model: str
    headers: dict[str, str]
    files: dict[str, tuple[str, bytes, str]]
    form: dict[str, str]
    language: str


def fish_asr_request(
    inbound: InboundAsr, settings: ProxySettings, incoming: Mapping[str, str]
) -> FishAsrRequest:
    """Build the Fish ASR multipart call.

    Parameters
    ----------
    inbound : InboundAsr
        The parsed upload.
    settings : ProxySettings
        Supplies the default model and the language hint when the client omitted one.
    incoming : Mapping
        Client headers. A valid trace is forwarded; otherwise one is minted.

    Returns
    -------
    FishAsrRequest
        Timestamps are asked for when the reply format is ``verbose_json``,
        ``srt`` or ``vtt``, or any granularity is set. The language is reduced
        to its ISO 639-1 code (``en-US`` becomes ``en``) and left out when
        neither the client nor ``FISH_ASR_LANGUAGE`` gives one Fish accepts, such
        as ``English``; Fish then detects it. The ``transcribe-1-pro`` fields are
        sent only to that model, and dropping them for another model is logged.

    Notes
    -----
    Fish may still label noise as ``zh`` when language is omitted.
    """
    model = resolve_asr_model(inbound.model, settings.defaults.asr_model)
    timed = inbound.response_format in _TIMED_FORMATS or bool(inbound.granularities)
    form = {"ignore_timestamps": "false" if timed else "true"}
    # A newline in a form value would start another part, so each hint is cut
    # at its first control character.
    lang = asr_language_hint(_one_line(inbound.language, "")) or asr_language_hint(
        _one_line(settings.defaults.asr_language, "")
    )
    if lang:
        form["language"] = lang
    pro_fields = inbound.pro.form_fields()
    if model == PRO_ASR_MODEL:
        form.update(pro_fields)
    elif pro_fields:
        log.info("asr pro-only fields not sent to model=%s", model)
    return FishAsrRequest(
        model=model,
        headers=traced_model_headers(model, incoming),
        files={"audio": (inbound.filename, inbound.audio, inbound.content_type)},
        form=form,
        language=lang,
    )


def _language(data: AsrBody, hint: str | None) -> str | None:
    # OpenAI names the language in lowercase ("english"). Fish sends the name
    # ("English") and the ISO code ("en"); the name is preferred.
    name = _text(data.get("language"), "")
    code = _text(data.get("language_code"), "")
    return name.lower() or code or hint


def _transcript(
    data: AsrBody, transcript: str, scrub: AsrScrub
) -> tuple[CleanTranscript, list[CaptionCue]]:
    """Return the cleaned transcript and its caption cues; no cues for a hallucination."""
    fallback: list[CaptionCue] = []

    def phrase_text() -> str:
        # Fish can put real words only in segments. They are grouped into phrases,
        # so the text matches the captions; a watermark phrase is already left out.
        fallback.extend(caption_cues(data, "", scrub))
        return " ".join(cue.text for cue in fallback).strip()

    clean = clean_transcript(
        transcript,
        data,
        strip_speakers=scrub.speakers,
        strip_cues=scrub.cues,
        segment_text=phrase_text,
    )
    if clean.source == "hallucination":
        return clean, []
    cues = fallback if clean.source == "segments" else caption_cues(data, clean.text, scrub)
    return clean, cues


def _cue_rows(cues: list[CaptionCue]) -> list[dict[str, Any]]:
    return [
        {"id": index, "text": cue.text, "start": cue.start, "end": cue.end}
        for index, cue in enumerate(cues)
    ]


def _word_rows(data: AsrBody, scrub: AsrScrub) -> list[dict[str, Any]]:
    # Fish /v1/asr sends no ``words`` field: its ``segments`` hold one word each,
    # so they are the word list. A ``words`` array in the body (a self-hosted
    # server that sends one) wins, scrubbed the same way as the transcript.
    raw: object = dict(data).get("words")
    if not isinstance(raw, list):
        return [
            {"word": cue.text, "start": cue.start, "end": cue.end} for cue in word_cues(data, scrub)
        ]
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_word = _text(item.get("word"), "") or _text(item.get("text"), "")
        word = scrub(raw_word).strip()
        start = json_number(item.get("start"))
        end = json_number(item.get("end"))
        if word and start is not None and end is not None:
            rows.append({"word": word, "start": start, "end": end})
    return rows


def _verbose_body(
    text: str,
    cues: list[CaptionCue],
    data: AsrBody,
    *,
    language: str | None,
    granularities: Sequence[str],
    scrub: AsrScrub,
) -> dict[str, Any]:
    spoken_lang = _language(data, language)
    body: dict[str, Any] = {
        "task": "transcribe",
        # A surrogate in the language tag makes the JSON response fail to encode,
        # so the client never receives the transcript.
        "language": utf8_text(spoken_lang) if spoken_lang is not None else None,
        "duration": json_number(data.get("duration")),
        "text": text,
        "segments": _cue_rows(cues),
    }
    if any(g.strip().lower() == "word" for g in granularities) and (
        words := _word_rows(data, scrub)
    ):
        body["words"] = words
    return body


def transcription_body(
    fmt: AsrFormat,
    text: str,
    cues: list[CaptionCue],
    data: AsrBody,
    *,
    language: str | None,
    granularities: Sequence[str],
    scrub: AsrScrub,
) -> PlainTextResponse | dict[str, Any]:
    """Shape a cleaned transcript as JSON, verbose JSON, plain text, SRT, or VTT.

    Parameters
    ----------
    fmt : AsrFormat
        The reply format. ``srt`` and ``vtt`` return caption files.
    text : str
        The cleaned transcript.
    cues : list of CaptionCue
        Phrase cues from ``caption_cues``, returned as ``segments`` with an ``id``.
    data : AsrBody
        Decoded Fish JSON, used for ``duration`` (seconds), the language and words.
        The ``verbose_json`` language is Fish's ``language`` name in lowercase, as
        OpenAI sends it (``english``), else ``language_code``, else ``language``.
        With ``word`` granularity, ``verbose_json`` adds ``words``: one row per Fish
        word segment, or the body's own ``words`` array when it has one.
    language : str or None
        The language hint sent to Fish, used when Fish names none.
    granularities : sequence of str
        ``timestamp_granularities`` values.
    scrub : AsrScrub
        What to strip from each word row.

    Returns
    -------
    PlainTextResponse or dict
        Caption text, a verbose object, or ``{"text": ...}``. A word that is
        empty after scrubbing is left out.
    """
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(format_as_srt(cues), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(format_as_vtt(cues), media_type="text/vtt")
    if fmt == "verbose_json":
        return _verbose_body(
            text, cues, data, language=language, granularities=granularities, scrub=scrub
        )
    return {"text": text}


def transcription_reply(
    inbound: InboundAsr,
    fish: FishAsrRequest,
    upstream: httpx.Response,
    settings: ProxySettings,
) -> PlainTextResponse | dict[str, Any]:
    """Turn Fish's ASR answer into the OpenAI reply the client asked for.

    Parameters
    ----------
    inbound : InboundAsr
        The parsed request: its reply format and granularities.
    fish : FishAsrRequest
        The call that was sent: its model, trace and language hint.
    upstream : httpx.Response
        Fish's 2xx answer.
    settings : ProxySettings
        ``asr_strip_speakers`` and ``asr_strip_cues``, applied to the transcript,
        the captions and every word.

    Returns
    -------
    PlainTextResponse or dict
        ``{"text": ...}``, plain text, an SRT or VTT caption file, or the
        ``verbose_json`` object, as ``transcription_body`` shapes it. A transcript
        that is a hallucination (silence, a caption watermark) is empty, unless
        the segments hold real words.

    Raises
    ------
    ProxyError
        502 when Fish's body is not JSON or not a transcript.
    """
    try:
        raw = upstream.json()
    # ValueError covers bad JSON, bad UTF-8, and an integer past Python's digit limit.
    except (ValueError, RecursionError) as exc:
        err = FishHttpError.for_non_json()
        raise ProxyError(err.status, err.message) from exc
    try:
        data, transcript = parse_asr_body(raw)
    except FishHttpError as exc:
        raise ProxyError(exc.status, exc.message) from exc
    scrub = AsrScrub(speakers=settings.asr_strip_speakers, cues=settings.asr_strip_cues)
    clean, cues = _transcript(data, transcript, scrub)
    lang = _language(data, fish.language)
    log.info(
        "asr model=%s lang=%s chars=%d trace=%s request_id=%s",
        fish.model,
        lang,
        len(clean.scrubbed),
        trace_id_of(fish.headers["traceparent"]),
        fish_request_id(data, upstream.headers) or "-",
    )
    text = clean.text
    if clean.source == "hallucination":
        log.info("asr drop hallucination lang=%r chars=%d", lang, len(clean.scrubbed))
        text = ""
    return transcription_body(
        inbound.response_format,
        text,
        cues,
        data,
        language=fish.language or None,
        granularities=inbound.granularities,
        scrub=scrub,
    )
