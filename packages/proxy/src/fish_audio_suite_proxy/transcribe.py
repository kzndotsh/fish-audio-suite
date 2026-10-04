"""Fish ASR request and OpenAI transcription response."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, get_args

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException

from fish_audio_suite_kit import (
    AsrBody,
    AsrFormat,
    CaptionCue,
    SuiteDefaults,
    asr_language_hint,
    format_as_srt,
    format_as_vtt,
    scrub_asr,
    utf8_text,
)
from fish_audio_suite_proxy.errors import ProxyError, json_error, read_json_object
from fish_audio_suite_proxy.fields import AudioDecodeError, decode_audio_b64
from fish_audio_suite_proxy.phrases import word_cues

__all__ = [
    "ASR_FORMATS",
    "PRO_ASR_MODEL",
    "InboundAsr",
    "ProAsrOptions",
    "asr_upload",
    "form_strings",
    "read_asr_format",
    "read_asr_request",
    "transcription_body",
]


PRO_ASR_MODEL: Final = "transcribe-1-pro"
_DIARIZE_VALUES: Final = ("auto", "true", "false")
_MAX_COUNT_DIGITS: Final = 6
_TRUE_WORDS: Final = frozenset({"true", "1"})
_FALSE_WORDS: Final = frozenset({"false", "0"})


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
    word = value.strip().lower() if isinstance(value, str) else ""
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ProxyError(400, f"{name} must be true or false")


def _pro_options(read: Callable[[str], object]) -> ProAsrOptions:
    """Validate the ``transcribe-1-pro`` fields of a request.

    Raises
    ------
    ProxyError
        400 for a value Fish would refuse: a ``diarize`` other than auto, true
        or false, a speaker count below 1, ``num_speakers`` with
        ``min_speakers`` or ``max_speakers``, ``min_speakers`` above
        ``max_speakers``, a count with ``diarize=false``, or a
        ``tag_audio_events`` that is not a boolean.
    """
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
    response_format : str
        The client's requested response format, not yet validated.
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
    response_format: str
    granularities: tuple[str, ...]
    pro: ProAsrOptions = ProAsrOptions()


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


def _asr_from_json(parsed: dict[str, Any]) -> InboundAsr | JSONResponse:
    inner = parsed.get("input_audio")
    audio_obj = inner if isinstance(inner, dict) else {}
    data = audio_obj.get("data")
    fmt = _single_line(_form_text(audio_obj.get("format"), "wav"), "wav")
    try:
        audio = decode_audio_b64(data, field="input_audio")
    except AudioDecodeError as exc:
        return json_error(400, exc.message)
    model = parsed.get("model")
    return InboundAsr(
        audio,
        f"utterance.{fmt}",
        f"audio/{fmt}",
        model if isinstance(model, str) else None,
        _form_text(parsed.get("language"), ""),
        _form_text(parsed.get("response_format"), "json"),
        tuple(_granularity_list(parsed.get("timestamp_granularities"))),
        _pro_options(parsed.get),
    )


def _form_text(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


ASR_FORMATS: tuple[AsrFormat, ...] = get_args(AsrFormat)
_ASR_FORMAT_BY_NAME: dict[str, AsrFormat] = {name: name for name in ASR_FORMATS}


def read_asr_format(raw: str) -> AsrFormat:
    """Return one response-format token. A newline is not part of the name.

    Parameters
    ----------
    raw : str
        The client's ``response_format``.

    Returns
    -------
    AsrFormat
        The lowercase token, cut at the first control character.

    Raises
    ------
    ProxyError
        400 when the token is not one of ``ASR_FORMATS``. The reply is never
        sent in a different format than the one asked for.
    """
    token = _single_line(raw.lower(), "json")
    fmt = _ASR_FORMAT_BY_NAME.get(token)
    if fmt is None:
        allowed = ", ".join(ASR_FORMATS)
        raise ProxyError(400, f"unsupported response_format {token[:32]!r}; use one of: {allowed}")
    return fmt


def _single_line(value: str, default: str) -> str:
    # A control character in a multipart Content-Type becomes another header.
    text = value.strip()
    cut = next((i for i, ch in enumerate(text) if ord(ch) < 32), None)
    if cut is not None:
        text = text[:cut].strip()
    return text or default


def _asr_from_form(form: FormData, audio: bytes, upload: UploadFile) -> InboundAsr:
    model = form.get("model")
    return InboundAsr(
        audio,
        _filename(upload.filename),
        _single_line(upload.content_type or "", "application/octet-stream"),
        model if isinstance(model, str) else None,
        _form_text(form.get("language"), ""),
        _form_text(form.get("response_format"), "json"),
        tuple(form_strings(form, "timestamp_granularities", "timestamp_granularities[]")),
        _pro_options(form.get),
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


async def read_asr_request(request: Request) -> InboundAsr | JSONResponse:
    """Read multipart ``file`` or JSON ``input_audio`` into one upload.

    Parameters
    ----------
    request : Request
        Transcription request. JSON is chosen when ``Content-Type`` contains
        ``application/json``.

    Returns
    -------
    InboundAsr or JSONResponse
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


def _cue_rows(cues: list[CaptionCue]) -> list[dict[str, Any]]:
    return [
        {"id": index, "text": cue.text, "start": cue.start, "end": cue.end}
        for index, cue in enumerate(cues)
    ]


def _plain_transcript(
    fmt: AsrFormat,
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
    data: AsrBody,
    *,
    strip_speakers: bool,
    strip_cues: bool,
) -> list[dict[str, Any]]:
    # Fish /v1/asr sends no ``words`` field: its ``segments`` hold one word each,
    # so they are the word list. A ``words`` array in the body (a server that sends
    # one) wins, scrubbed the same way as the transcript.
    raw: object = dict(data).get("words")
    if not isinstance(raw, list):
        return [
            {"word": cue.text, "start": cue.start, "end": cue.end}
            for cue in word_cues(data, strip_speakers=strip_speakers, strip_cues=strip_cues)
        ]
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_word = _json_text(item.get("word")) or _json_text(item.get("text")) or ""
        word = scrub_asr(raw_word, strip_speakers=strip_speakers, strip_cues=strip_cues).strip()
        start = _json_number(item.get("start"))
        end = _json_number(item.get("end"))
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
    strip_speakers: bool,
    strip_cues: bool,
) -> dict[str, Any]:
    # OpenAI names the language in lowercase ("english"). Fish sends the name
    # ("English") and the ISO code ("en"); the name is preferred.
    name = _json_text(data.get("language"))
    raw_lang = (name.lower() if name else None) or _json_text(data.get("language_code")) or language
    # A surrogate in the language tag makes the JSON response fail to encode,
    # so the client never receives the transcript.
    spoken_lang = utf8_text(raw_lang) if isinstance(raw_lang, str) else raw_lang
    body: dict[str, Any] = {
        "task": "transcribe",
        "language": spoken_lang,
        "duration": _json_number(data.get("duration")),
        "text": text,
        "segments": _cue_rows(cues),
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
    fmt: AsrFormat,
    text: str,
    cues: list[CaptionCue],
    data: AsrBody,
    *,
    language: str | None,
    granularities: Sequence[str],
    strip_speakers: bool = False,
    strip_cues: bool = False,
) -> PlainTextResponse | dict[str, Any]:
    """Shape the Fish ASR result as JSON, verbose JSON, SRT, or VTT.

    Parameters
    ----------
    fmt : AsrFormat
        ``response_format``. ``srt`` and ``vtt`` return caption files.
    text : str
        Scrubbed transcript.
    cues : list of CaptionCue
        Phrase cues from ``caption_cues``, returned as ``segments`` with an ``id``.
    data : AsrBody
        Decoded Fish JSON, used for ``duration`` (seconds) and language. The
        ``verbose_json`` language is Fish's ``language`` name in lowercase, as
        OpenAI sends it (``english``), else ``language_code``, else
        ``language`` below. With
        ``word`` granularity, ``verbose_json`` adds ``words``: one row per Fish
        word segment, or the body's own ``words`` array when it has one.
    language : str or None
        The language hint sent to Fish, used when Fish names none.
    granularities : sequence of str
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
    inbound: InboundAsr,
    defaults: SuiteDefaults,
    fmt: AsrFormat,
    granularities: Sequence[str],
    *,
    model: str = "",
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str], str]:
    """Build the Fish ASR multipart body.

    Parameters
    ----------
    inbound : InboundAsr
        Parsed upload.
    defaults : SuiteDefaults
        Supplies the language hint when the client omitted one.
    fmt : AsrFormat
        Response format. ``verbose_json``, ``srt``, and ``vtt`` ask Fish for
        timestamps.
    granularities : sequence of str
        Any non-empty list also asks for timestamps.
    model : str, optional
        The resolved Fish model. The ``inbound.pro`` fields are added only for
        ``transcribe-1-pro``; ``transcribe-1`` does not take them.

    Returns
    -------
    tuple
        httpx ``files``, form fields, and the language sent upstream. The
        language is reduced to its ISO 639-1 code (``en-US`` becomes ``en``)
        and is ``""`` when neither the client nor ``FISH_ASR_LANGUAGE`` gives
        one Fish accepts, such as ``English``. That key is then left out of the
        form and Fish detects the language.

    Notes
    -----
    Fish may still label noise as ``zh`` when language is omitted.
    """
    want_ts = fmt in _TIMED_FORMATS or bool(granularities)
    form = {"ignore_timestamps": "false" if want_ts else "true"}
    # A newline in a form value starts another part. "en\r\n..." was sent
    # to Fish as a second field, and the language itself was only "en".
    lang = asr_language_hint(_single_line(inbound.language, "")) or asr_language_hint(
        _single_line(defaults.asr_language, "")
    )
    if lang:
        form["language"] = lang
    if model == PRO_ASR_MODEL:
        form.update(inbound.pro.form_fields())
    files = {"audio": (inbound.filename, inbound.audio, inbound.content_type)}
    return files, form, lang
