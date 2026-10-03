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
    is_caption_watermark,
    number_or,
    scrub_asr,
    utf8_text,
)
from fish_audio_suite_proxy.errors import ProxyError, json_error, read_json_object
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
    fmt = _single_line(_form_text(audio_obj.get("format"), "wav"), "wav")
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


ASR_FORMATS = ("json", "text", "verbose_json", "srt", "vtt")


def asr_response_format(raw: str) -> str:
    """Return one response-format token. A newline is not part of the name.

    Parameters
    ----------
    raw : str
        The client's ``response_format``.

    Returns
    -------
    str
        The lowercase token, cut at the first control character.

    Raises
    ------
    ProxyError
        400 when the token is not one of ``ASR_FORMATS``. The reply is never
        sent in a different format than the one asked for.
    """
    fmt = _single_line(raw.lower(), "json")
    if fmt not in ASR_FORMATS:
        allowed = ", ".join(ASR_FORMATS)
        raise ProxyError(400, f"unsupported response_format {fmt[:32]!r}; use one of: {allowed}")
    return fmt


def _single_line(value: str, default: str) -> str:
    # A control character in a multipart Content-Type becomes another header.
    text = value.strip()
    cut = next((i for i, ch in enumerate(text) if ord(ch) < 32), None)
    if cut is not None:
        text = text[:cut].strip()
    return text or default


def _asr_from_form(form: FormData, audio: bytes, upload: UploadFile) -> _InboundAsr:
    model = form.get("model")
    return _InboundAsr(
        audio,
        _filename(upload.filename),
        _single_line(upload.content_type or "", "application/octet-stream"),
        model if isinstance(model, str) else None,
        _form_text(form.get("language"), ""),
        _form_text(form.get("response_format"), "json"),
        form_strings(form, "timestamp_granularities", "timestamp_granularities[]"),
    )


_MAX_FILENAME_CHARS = 255


def _filename(raw: str | None) -> str:
    # The name becomes a multipart header value. Keep the last path part and
    # cap the length so a client cannot inflate the upstream request.
    name = _single_line((raw or "").replace("\\", "/").rsplit("/", 1)[-1], "")
    return name[:_MAX_FILENAME_CHARS] or "audio.webm"


def _empty_upload() -> JSONResponse:
    return json_error(400, "empty audio upload")


_FORM_ERRORS = {
    400: "invalid multipart form body",
    413: "form upload is too large",
    422: "form fields could not be read",
}


def _form_error_message(status: int) -> str:
    # The parser's own text can name internals, so the client gets a fixed message.
    return _FORM_ERRORS.get(status, "invalid form body")


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
        # The context closes the form and every upload on all paths. A form that
        # is only awaited leaves its SpooledTemporaryFile open until garbage
        # collection. Everything the result needs is copied out before it closes.
        async with request.form() as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                return _empty_upload()
            audio = await upload.read()
            if not audio:
                return _empty_upload()
            return _asr_from_form(form, audio, upload)
    except HTTPException as exc:
        return json_error(exc.status_code, _form_error_message(exc.status_code))


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
    """Fish `duration` is a number. A string or boolean is not a caption length."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _segment_cue(seg: Any, *, strip_speakers: bool, strip_cues: bool) -> CaptionCue | None:
    if not isinstance(seg, dict):
        return None
    raw_text = seg.get("text", "")
    if not isinstance(raw_text, str):
        return None
    body = scrub_asr(raw_text, strip_speakers=strip_speakers, strip_cues=strip_cues)
    if not body:
        return None
    start = max(0.0, _seconds(seg.get("start", 0)))
    end = max(start, _seconds(seg.get("end", 0)))
    return CaptionCue(start, end, body)


def caption_cues(
    data: dict[str, Any],
    text: str,
    *,
    strip_speakers: bool,
    strip_cues: bool = False,
) -> list[CaptionCue]:
    """Build timed cues from Fish segments, or one cue for the whole transcript.

    Parameters
    ----------
    data : dict
        Decoded Fish ASR JSON.
    text : str
        Scrubbed full transcript, used when ``segments`` is missing or empty.
    strip_speakers : bool
        Drop speaker labels inside each segment.
    strip_cues : bool, optional
        Drop ``[cue]`` annotations inside each segment. Default False.

    Returns
    -------
    list of CaptionCue
        Segment cues when any segment has text. A known caption watermark
        segment is omitted. Otherwise one cue from 0 to ``duration`` covering
        ``text``, or an empty list when ``text`` is empty.
    """
    cues: list[CaptionCue] = []
    raw_segments = data.get("segments") or []
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            cue = _segment_cue(seg, strip_speakers=strip_speakers, strip_cues=strip_cues)
            if cue is not None and not is_caption_watermark(cue.text):
                cues.append(cue)
    if cues:
        return cues
    if not text:
        return []
    return [CaptionCue(0.0, max(0.0, _duration_s(data.get("duration"))), text)]


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


def _json_number(value: Any) -> int | float | None:
    # bool is an int subclass. True is not a duration.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _json_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _word_rows(
    data: dict[str, Any],
    *,
    strip_speakers: bool,
    strip_cues: bool,
) -> list[dict[str, Any]]:
    # Fish segments are phrases, not words, so they are never relabeled as
    # words. Only real word timings from Fish are passed through, scrubbed the
    # same way as the transcript so the flags hold for every field.
    raw = data.get("words")
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw = _json_text(item.get("word")) or _json_text(item.get("text")) or ""
        word = scrub_asr(raw, strip_speakers=strip_speakers, strip_cues=strip_cues).strip()
        start = _json_number(item.get("start"))
        end = _json_number(item.get("end"))
        if word and start is not None and end is not None:
            rows.append({"word": word, "start": start, "end": end})
    return rows


def _verbose_body(
    text: str,
    cues: list[CaptionCue],
    data: dict[str, Any],
    *,
    language: str | None,
    granularities: list[str],
    strip_speakers: bool,
    strip_cues: bool,
) -> dict[str, Any]:
    raw_lang = _json_text(data.get("language_code")) or _json_text(data.get("language")) or language
    # A surrogate in the language tag makes the JSON response fail to encode,
    # so the client never receives the transcript.
    spoken_lang = utf8_text(raw_lang) if isinstance(raw_lang, str) else raw_lang
    body: dict[str, Any] = {
        "task": "transcribe",
        "language": spoken_lang,
        "duration": _json_number(data.get("duration")),
        "text": text,
        "segments": _cue_rows(cues, "text"),
    }
    wants_words = any(g.strip().lower() == "word" for g in granularities)
    words = (
        _word_rows(data, strip_speakers=strip_speakers, strip_cues=strip_cues)
        if wants_words
        else None
    )
    if words:
        body["words"] = words
    return body


def transcription_body(
    fmt: str,
    text: str,
    cues: list[CaptionCue],
    data: dict[str, Any],
    *,
    language: str | None,
    granularities: list[str],
    strip_speakers: bool = False,
    strip_cues: bool = False,
) -> PlainTextResponse | dict[str, Any]:
    """Shape the Fish ASR result as JSON, verbose JSON, SRT, or VTT.

    Parameters
    ----------
    fmt : str
        ``response_format``. ``srt`` and ``vtt`` return caption files.
    text : str
        Scrubbed transcript.
    cues : list of CaptionCue
        Timed phrases returned as ``segments``.
    data : dict
        Decoded Fish JSON, used for duration and language. A ``words`` array is
        returned for ``verbose_json`` with ``word`` granularity only when Fish
        sent word timings. Segments are never relabeled as words.
    language : str or None
        Client or env language hint.
    granularities : list of str
        ``timestamp_granularities`` values.
    strip_speakers : bool, optional
        Drop speaker labels from each word row. Default False.
    strip_cues : bool, optional
        Drop ``[cue]`` annotations from each word row. Default False.

    Returns
    -------
    PlainTextResponse or dict
        Caption text, a verbose object, or ``{"text": ...}``. A word that is
        empty after scrubbing is left out.
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
            strip_speakers=strip_speakers,
            strip_cues=strip_cues,
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
    # A newline in a form value starts another part. "en\r\n..." was sent
    # to Fish as a second field, and the language itself was only "en".
    lang = _single_line(inbound.language or defaults.asr_language or "", "")
    if lang:
        form["language"] = lang
    files = {"audio": (inbound.filename, inbound.audio, inbound.content_type)}
    return files, form, lang
