"""Fish ASR request and OpenAI transcription response."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException

from fish_audio_suite_kit import (
    CaptionCue,
    SuiteDefaults,
    format_as_srt,
    format_as_vtt,
    number_or,
    scrub_asr,
)
from fish_audio_suite_proxy.errors import json_error, read_json_object
from fish_audio_suite_proxy.speech import ClipError, decode_audio_b64


@dataclass(frozen=True)
class _InboundAsr:
    audio: bytes
    filename: str
    content_type: str
    model: str | None
    language: str
    response_format: str
    granularities: list[str]


def _granularity_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            out.append(text)
    return out


def _asr_from_json(parsed: dict[str, Any]) -> _InboundAsr | JSONResponse:
    inner = parsed.get("input_audio")
    audio_obj = inner if isinstance(inner, dict) else {}
    data = audio_obj.get("data")
    fmt = _form_text(audio_obj.get("format"), "wav").strip() or "wav"
    try:
        audio = decode_audio_b64(data)
    except ClipError as exc:
        return json_error(400, exc.message)
    model = parsed.get("model")
    return _InboundAsr(
        audio,
        f"utterance.{fmt}",
        f"audio/{fmt}",
        model if isinstance(model, str) else None,
        _form_text(parsed.get("language"), ""),
        _form_text(parsed.get("response_format"), "json"),
        _granularity_list(parsed.get("timestamp_granularities")),
    )


def _form_text(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _asr_from_form(form: FormData, audio: bytes, upload: UploadFile) -> _InboundAsr:
    model = form.get("model")
    return _InboundAsr(
        audio,
        upload.filename or "audio.webm",
        upload.content_type or "application/octet-stream",
        model if isinstance(model, str) else None,
        _form_text(form.get("language"), ""),
        _form_text(form.get("response_format"), "json"),
        form_strings(form, "timestamp_granularities", "timestamp_granularities[]"),
    )


def _empty_upload() -> JSONResponse:
    return json_error(400, "empty audio upload")


async def read_asr(request: Request) -> _InboundAsr | JSONResponse:
    """Read multipart ``file`` or JSON ``input_audio`` into one upload.

    Parameters
    ----------
    request : Request
        Transcription request. JSON is chosen when ``Content-Type`` contains
        ``application/json``.

    Returns
    -------
    _InboundAsr or JSONResponse
        Audio plus the fields that affect the Fish form, or a 400 response.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        parsed = await read_json_object(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        return _asr_from_json(parsed)

    try:
        form = await request.form()
    except HTTPException as exc:
        return json_error(exc.status_code, str(exc.detail))
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        return _empty_upload()
    audio = await upload.read()
    if not audio:
        return _empty_upload()
    return _asr_from_form(form, audio, upload)


def form_strings(form: FormData, *names: str) -> list[str]:
    """Collect repeated form fields, including the ``[]`` OpenAI spelling.

    Parameters
    ----------
    form : FormData
        Multipart body.
    *names : str
        Field names. Both ``timestamp_granularities`` and
        ``timestamp_granularities[]`` are read by callers.

    Returns
    -------
    list of str
        One string per submitted value. Missing names contribute nothing.
    """
    out: list[str] = []
    for name in names:
        out.extend(
            item.strip() for item in form.getlist(name) if isinstance(item, str) and item.strip()
        )
    return out


def _seconds(value: Any) -> float:
    return number_or(value or 0, 0.0, float)


def _duration_s(value: Any) -> float:
    """Fish `duration` is a number. A string is not a caption length."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return 0.0


def _segment_cue(seg: Any, *, strip_speakers: bool) -> CaptionCue | None:
    if not isinstance(seg, dict):
        return None
    raw_text = seg.get("text", "")
    if not isinstance(raw_text, str):
        return None
    body = scrub_asr(raw_text, strip_speakers=strip_speakers)
    if not body:
        return None
    return CaptionCue(_seconds(seg.get("start", 0)), _seconds(seg.get("end", 0)), body)


def caption_cues(data: dict[str, Any], text: str, *, strip_speakers: bool) -> list[CaptionCue]:
    """Build timed cues from Fish segments, or one cue for the whole transcript.

    Parameters
    ----------
    data : dict
        Decoded Fish ASR JSON.
    text : str
        Scrubbed full transcript, used when ``segments`` is missing or empty.
    strip_speakers : bool
        Drop speaker labels inside each segment.

    Returns
    -------
    list of CaptionCue
        Segment cues when any segment has text. Otherwise one cue from 0 to
        ``duration`` covering ``text``, or an empty list when ``text`` is empty.
    """
    cues: list[CaptionCue] = []
    raw_segments = data.get("segments") or []
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            cue = _segment_cue(seg, strip_speakers=strip_speakers)
            if cue is not None:
                cues.append(cue)
    if cues:
        return cues
    if not text:
        return []
    return [CaptionCue(0.0, _duration_s(data.get("duration")), text)]


def _cue_rows(cues: list[CaptionCue], key: str) -> list[dict[str, Any]]:
    return [{key: cue.text, "start": cue.start, "end": cue.end} for cue in cues]


def _plain_transcript(
    fmt: str,
    text: str,
    cues: list[CaptionCue],
) -> PlainTextResponse | None:
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(format_as_srt(cues), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(format_as_vtt(cues), media_type="text/vtt")
    return None


def _json_number(value: Any) -> Any:
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return None
    return value


def _json_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _verbose_body(
    text: str,
    cues: list[CaptionCue],
    data: dict[str, Any],
    *,
    language: str | None,
    granularities: list[str],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "task": "transcribe",
        "language": (
            _json_text(data.get("language_code")) or _json_text(data.get("language")) or language
        ),
        "duration": _json_number(data.get("duration")),
        "text": text,
        "segments": _cue_rows(cues, "text"),
    }
    if any(g.strip().lower() == "word" for g in granularities):
        body["words"] = _cue_rows(cues, "word")
    return body


def transcription_body(
    fmt: str,
    text: str,
    cues: list[CaptionCue],
    data: dict[str, Any],
    *,
    language: str | None,
    granularities: list[str],
) -> PlainTextResponse | dict[str, Any]:
    """Shape the Fish ASR result as JSON, verbose JSON, SRT, or VTT.

    Parameters
    ----------
    fmt : str
        ``response_format``. ``srt`` and ``vtt`` return caption files.
    text : str
        Scrubbed transcript.
    cues : list of CaptionCue
        Timed phrases. Word rows are added only for ``verbose_json`` when a
        granularity is ``word``.
    data : dict
        Decoded Fish JSON, used for duration and language.
    language : str or None
        Client or env language hint.
    granularities : list of str
        ``timestamp_granularities`` values.

    Returns
    -------
    PlainTextResponse or dict
        Caption text, a verbose object, or ``{"text": ...}``.
    """
    plain = _plain_transcript(fmt, text, cues)
    if plain is not None:
        return plain
    if fmt == "verbose_json":
        return _verbose_body(
            text,
            cues,
            data,
            language=language,
            granularities=granularities,
        )
    return {"text": text}


_TIMED_FORMATS = frozenset({"verbose_json", "vtt", "srt"})


def asr_upload(
    inbound: _InboundAsr,
    defaults: SuiteDefaults,
    fmt: str,
    granularities: list[str],
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str], str]:
    """Build the Fish ASR multipart body.

    Parameters
    ----------
    inbound : _InboundAsr
        Parsed upload.
    defaults : SuiteDefaults
        Supplies the language hint when the client omitted one.
    fmt : str
        Response format. ``verbose_json``, ``srt``, and ``vtt`` ask Fish for
        timestamps.
    granularities : list of str
        Any non-empty list also asks for timestamps.

    Returns
    -------
    tuple
        httpx ``files``, form fields, and the language sent upstream. The
        language is ``""`` when both the client and ``FISH_ASR_LANGUAGE``
        are blank, and that key is then left out of the form.

    Notes
    -----
    Fish may still label noise as ``zh`` when language is omitted.
    """
    want_ts = fmt in _TIMED_FORMATS or bool(granularities)
    form = {"ignore_timestamps": "false" if want_ts else "true"}
    lang = (inbound.language or defaults.asr_language or "").strip()
    if lang:
        form["language"] = lang
    files = {"audio": (inbound.filename, inbound.audio, inbound.content_type)}
    return files, form, lang
