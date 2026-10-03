"""Shared Fish TTS/ASR knobs. Literals live on SuiteDefaults."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, cast
from urllib.parse import urlsplit

from fish_audio_suite_kit._deprecation import deprecated, deprecated_fields
from fish_audio_suite_kit.literals import AsrFormat, AudioFormat, FishLatency, TtsModel

__all__ = [
    "CHUNK_LENGTH_CLOUD_HI",
    "CHUNK_LENGTH_LO",
    "CHUNK_LENGTH_SELF_HOSTED_HI",
    "DEFAULT_SEED_EXCHANGE",
    "DEFAULT_SYSTEM_PROMPT",
    "FISH_LATENCIES",
    "FISH_TTS_MODEL_IDS",
    "MIN_CHUNK_LENGTH_HI",
    "MIN_CHUNK_LENGTH_LO",
    "MS_PER_S",
    "TTS_SPEED_HI",
    "TTS_SPEED_LO",
    "UNIT_INTERVAL_HI",
    "UNIT_INTERVAL_LO",
    "LatencySnapshot",
    "SuiteDefaults",
    "catalog_tts_model",
    "chunk_length_hi",
    "clamp_num",
    "clamp_number",
    "elapsed_ms",
    "env_base",
    "env_bool",
    "env_float",
    "env_int",
    "env_off",
    "env_renamed",
    "env_text",
    "env_token",
    "is_insecure_fish_base",
    "known_asr_format",
    "known_audio_format",
    "known_latency",
    "known_mp3_bitrate",
    "known_opus_bitrate",
    "known_tts_model",
    "normalize_tts_model",
    "number_or",
    "parse_number",
    "strip_base",
]

DEFAULT_SYSTEM_PROMPT: Final = (
    "You are a voice assistant. Your reply is spoken aloud by a text-to-speech voice. "
    "English by default. Speak the user's language if they switch. "
    "Say only words that should be heard: no markdown, bullets, emoji, or URLs. "
    "Keep replies short and conversational, and do not repeat the user's sentence back. "
    "Square-bracket cues are silent stage directions for the voice and are never spoken. "
    "Never mention, describe or explain a cue, and never treat one as something the user "
    "asked for. "
    "Use [laughing] for a laugh and [break] for a pause, and a whisper only if the user asks "
    "you to whisper. "
    "Start every reply with a cue for the opening mood, such as [happy], [curious], [calm] or "
    "[excited]. The voice holds a cue until the next one, so change the cue whenever the "
    "feeling shifts: a joke landing, a sad turn, a surprise, a pause. A reply of two or more "
    "sentences usually has at least two cues, one for each shift in feeling. "
)

# One opening exchange that shows several cues in a reply. In a test on two
# models a conversation drifted to one cue per reply even though the system
# prompt asks for more, apparently because the model copies its own earlier
# replies. With this pinned, the default model went from 1.1 to 2.2 cues per
# reply and Claude Haiku 4.5 from 1.0 to 1.9.
DEFAULT_SEED_EXCHANGE: Final[tuple[tuple[str, str], ...]] = (
    (
        "Hi there!",
        "[happy] Hey, it's so good to hear you! [curious] What are we getting into today?",
    ),
)


_OPUS_AUTO: Final = -1000


@dataclass(frozen=True, slots=True)
class SuiteDefaults:
    """Shared Fish TTS and ASR knobs. Callers still clamp before a request.

    Notes
    -----
    ``chunk_length`` starts at 200. Cloud accepts 100-300; a self-hosted
    base accepts up to 1000 (see ``chunk_length_hi``). ``opus_bitrate``
    of -1000 asks Fish to pick the rate. ``tts_partial_chars`` is how far
    ``next_tts_cut`` will flush without a sentence end. ``asr_language``
    is empty unless the caller or env sets a hint.
    """

    tts_model: str = "s2.1-pro"
    asr_model: str = "transcribe-1"
    asr_language: str = ""
    latency: FishLatency = "normal"
    chunk_length: int = 200
    min_chunk_length: int = 50
    audio_format: AudioFormat = "mp3"
    mp3_bitrate: int = 128
    opus_bitrate: int = _OPUS_AUTO
    opus_sample_rate: int = 48000
    speed: float = 1.0
    volume: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    max_new_tokens: int = 1024
    early_stop_threshold: float = 1.0
    normalize: bool = True
    normalize_loudness: bool = True
    condition_on_previous_chunks: bool = True
    tts_partial_chars: int = 40
    sample_rate: int = 44100
    fish_base: str = "https://api.fish.audio"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@deprecated_fields(
    "0.2.0",
    llm_ttft="llm_first_token_ms",
    llm_ttfs="tts_first_text_ms",
    ttfa="tts_first_audio_ms",
    voice_to_voice="voice_to_voice_ms",
    first_audio="first_audio_ms",
)
@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """One cascade turn. Times are milliseconds. Never store utterance text.

    Attributes
    ----------
    asr_ms : float or None
        From sending the finished utterance to Fish ASR until its transcript is back.
    llm_first_token_ms : float or None
        From the LLM request until its first token.
    tts_first_text_ms : float or None
        From the start of the TTS turn until the first text is sent to Fish. When
        the reply streams, this includes waiting for the model's first piece.
    tts_first_audio_ms : float or None
        From the start of the TTS turn until the first audio chunk arrives from Fish.
    voice_to_voice_ms : float or None
        From sending the utterance to ASR until the TTS turn ends, after playback
        finishes or is cut off.
    trace_id : str or None
        W3C trace id of the turn, when one was made.
    first_audio_ms : float or None
        From sending the utterance to ASR until the first Fish audio chunk goes to
        the speaker. This is the wait the user hears after they stop talking.

    Notes
    -----
    The fields keep their order, so a snapshot built by position keeps its
    meaning; a new field goes last and is not keyword-only. The 0.1 names
    (``llm_ttft``, ``llm_ttfs``, ``ttfa``, ``voice_to_voice``, ``first_audio``)
    still work as keywords and attributes with a ``DeprecationWarning``.
    """

    asr_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_text_ms: float | None = None
    tts_first_audio_ms: float | None = None
    voice_to_voice_ms: float | None = None
    trace_id: str | None = None
    # Last, and deliberately not keyword-only: a caller that builds a snapshot by
    # position keeps its meaning, and a new field must never shift the old ones.
    first_audio_ms: float | None = None

    def log_line(self) -> str:
        """One stdout timing line. Missing times are omitted. No utterance text.

        Returns
        -------
        str
            ``[timing asr=…ms … trace=…]``. Each key is its field name without
            ``_ms``. ``trace`` appears only when set.

        Examples
        --------
        >>> LatencySnapshot(asr_ms=40, tts_first_audio_ms=12.4).log_line()
        '[timing asr=40ms tts_first_audio=12ms]'
        """
        parts = [
            _timing_field("asr", self.asr_ms),
            _timing_field("llm_first_token", self.llm_first_token_ms),
            _timing_field("tts_first_text", self.tts_first_text_ms),
            _timing_field("tts_first_audio", self.tts_first_audio_ms),
            _timing_field("first_audio", self.first_audio_ms),
            _timing_field("voice_to_voice", self.voice_to_voice_ms),
        ]
        if self.trace_id:
            parts.append(f"trace={self.trace_id}")
        shown = [part for part in parts if part]
        return "[timing " + " ".join(shown) + "]"


def _timing_field(name: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"{name}={value:.0f}ms"


MS_PER_S: Final = 1000


def elapsed_ms(started: float) -> float:
    """Milliseconds since a ``time.perf_counter`` reading.

    Parameters
    ----------
    started : float
        Value previously returned by ``time.perf_counter``.

    Returns
    -------
    float
        Elapsed milliseconds. Not rounded.
    """
    return (time.perf_counter() - started) * MS_PER_S


def strip_base(url: str) -> str:
    """Drop surrounding space and trailing slashes so a joined path is not ``//``.

    Parameters
    ----------
    url : str
        A base URL. Anything from the first control character on is cut off, since
        a newline in a base makes ``httpx`` reject the URL.

    Returns
    -------
    str
        The URL without surrounding space or trailing slashes.
    """
    text = url.strip()
    # A newline in the base makes httpx raise InvalidURL when the client
    # is built, so the turn never starts.
    cut = next((index for index, ch in enumerate(text) if ord(ch) < 32), None)
    if cut is not None:
        text = text[:cut].strip()
    return text.rstrip("/")


def env_base(name: str, default: str) -> str:
    """Read a base URL from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Used when the variable is missing, blank, or empty after cleaning.

    Returns
    -------
    str
        The value, or the default, with ``strip_base`` applied.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return strip_base(default)
    return strip_base(raw) or strip_base(default)


def _env_word(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower()


_OFF_WORDS = frozenset({"0", "false", "no", "off"})
_ON_WORDS = frozenset({"1", "true", "yes", "on"})


def env_off(name: str) -> bool:
    """Return whether the variable is explicitly switched off.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    bool
        True only for ``0``, ``false``, ``no`` or ``off`` in any case. A missing or
        blank variable is not off.
    """
    return _env_word(name) in _OFF_WORDS


def env_bool(name: str, *, default: bool = False) -> bool:
    """Read an on/off flag from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool, optional
        Returned when the variable is missing or blank. Keyword-only, so a call
        reads ``env_bool("X", default=True)``.

    Returns
    -------
    bool
        True for ``1``, ``true``, ``yes`` or ``on`` in any case, False for any
        other non-blank value, and ``default`` when the variable is unset or blank.
    """
    word = _env_word(name)
    if not word:
        return default
    return word in _ON_WORDS


def env_token(name: str, default: str) -> str:
    """Read a single-word setting from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Returned when the variable is missing or blank.

    Returns
    -------
    str
        The stripped value, or ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    return text or default


def env_text(name: str, default: str = "") -> str:
    """Read free text from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str, optional
        Returned, stripped, when the variable is missing.

    Returns
    -------
    str
        The stripped value. A blank value stays blank and does not fall back to
        ``default``, so an empty key can be told apart from an unset one.
    """
    return os.environ.get(name, default).strip()


def env_renamed(new: str, old: str, *, warn: Callable[[str], None]) -> str:
    """Pick which of a renamed variable's two names to read.

    Parameters
    ----------
    new : str
        The current environment variable name.
    old : str
        The name it replaced, still honored for one minor release.
    warn : Callable
        Called with ``"<old> is deprecated; use <new>"`` when the old name is the
        one in use, so each package reports it its own way (a log line, a printed
        warning).

    Returns
    -------
    str
        ``new`` when it is set to a non-blank value, else ``old`` when that is set
        to a non-blank value, else ``new``. Pass the result to an ``env_*`` reader.

    Examples
    --------
    >>> env_renamed("FISH_DOCTEST_NEW_UNSET", "FISH_DOCTEST_OLD_UNSET", warn=print)
    'FISH_DOCTEST_NEW_UNSET'
    """
    if os.environ.get(new, "").strip():
        return new
    if os.environ.get(old, "").strip():
        warn(f"{old} is deprecated; use {new}")
        return old
    return new


def _whole_int(value: Any) -> int:
    # int("16000.0") raises, so a decimal string fell back to another rate
    # and the speaker played the buffer at the wrong speed.
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        parsed = float(text)
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError
        return int(parsed)
    return int(value)


def _parsed[T: int | float](value: Any, parse: Callable[[Any], T]) -> T:
    if parse is int:
        return cast(T, _whole_int(value))
    return parse(value)


def parse_number[T: int | float](value: Any, default: T, parse: Callable[[Any], T]) -> T:
    """Parse a number from untrusted input.

    Parameters
    ----------
    value : Any
        A number or a numeric string, for example from a JSON request.
    default : int or float
        Returned for anything that does not parse.
    parse : Callable
        ``int`` or ``float``. With ``int``, a whole decimal such as ``"16000.0"``
        is accepted.

    Returns
    -------
    int or float
        The parsed value, or ``default`` for junk, a non-finite float, or a JSON
        boolean (``True`` would otherwise parse as 1).
    """
    # bool is an int subclass. True would parse as 1.
    if isinstance(value, bool):
        return default
    try:
        parsed = _parsed(value, parse)
    except (TypeError, ValueError, OverflowError):
        return default
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return default
    return parsed


# Plain ASCII decimals only. int() and float() also accept other scripts' digits
# ("\uff11\uff10") and underscores ("1_000"), so a lookalike value would otherwise
# pass for a limit setting.
_ASCII_NUMBER_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _env_num[T: int | float](name: str, default: T, parse: Callable[[str], T]) -> T:
    raw = os.environ.get(name, "").strip()
    if not raw or not _ASCII_NUMBER_RE.fullmatch(raw):
        return default
    return parse_number(raw, default, parse)


def env_int(name: str, default: int) -> int:
    """Read an integer from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Returned when the variable is missing, blank, not a number, or written with
        anything but plain ASCII digits (fullwidth digits, underscores and hex
        are rejected, so a lookalike cannot pass for a limit).

    Returns
    -------
    int
        The parsed value. A whole decimal such as ``16000.0`` is accepted.

    Examples
    --------
    >>> env_int("FISH_DOCTEST_SURELY_UNSET", 7)
    7
    """
    return _env_num(name, default, int)


def env_float(name: str, default: float) -> float:
    """Read a float from the process environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Returned when the variable is missing, blank, not finite, not a number, or
        written with anything but plain ASCII digits.

    Returns
    -------
    float
        The parsed value, or ``default``.
    """
    return _env_num(name, default, float)


CHUNK_LENGTH_LO: Final = 100
CHUNK_LENGTH_CLOUD_HI: Final = 300
CHUNK_LENGTH_SELF_HOSTED_HI: Final = 1000
MIN_CHUNK_LENGTH_LO: Final = 0
MIN_CHUNK_LENGTH_HI: Final = 100
TTS_SPEED_LO: Final = 0.5
TTS_SPEED_HI: Final = 2.0
UNIT_INTERVAL_LO: Final = 0.0
UNIT_INTERVAL_HI: Final = 1.0


_CLOUD_HOST: Final = "api.fish.audio"


_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _is_cloud_base(fish_base: str) -> bool:
    text = fish_base.strip()
    # Only a leading scheme or "//" makes the host parse as a netloc. A "//" later
    # in a path ("api.fish.audio/v1//edge") does not.
    if not (_SCHEME_RE.match(text) or text.startswith("//")):
        text = f"//{text}"
    try:
        host = (urlsplit(text).hostname or "").lower()
    except ValueError:
        return False
    return host == _CLOUD_HOST or host.endswith(f".{_CLOUD_HOST}")


def is_insecure_fish_base(base: str) -> bool:
    """Return whether a Fish base URL would send the API key in cleartext.

    Parameters
    ----------
    base : str
        A ``FISH_BASE`` value such as ``http://10.0.0.5:8080``.

    Returns
    -------
    bool
        True when the scheme is ``http`` and the host is not loopback
        (``localhost``, ``*.localhost``, ``127.0.0.0/8`` or ``::1``). False for
        https, for a base with no scheme or host, and for one that does not parse.

    Notes
    -----
    This only reports. LAN self-hosting over http is legitimate, so callers warn
    and carry on.

    Examples
    --------
    >>> is_insecure_fish_base("http://10.0.0.5:8080")
    True
    >>> is_insecure_fish_base("http://localhost:8080")
    False
    >>> is_insecure_fish_base("https://api.fish.audio")
    False
    """
    try:
        parts = urlsplit(base.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return False
    if parts.scheme.lower() != "http" or not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def chunk_length_hi(fish_base: str, *, self_hosted: bool | None = None) -> int:
    """Return the highest ``chunk_length`` the Fish base accepts.

    Parameters
    ----------
    fish_base : str
        Fish origin. The cloud host is matched on the parsed hostname, so a
        path, a port, or a look-alike domain does not count as cloud.
    self_hosted : bool or None, optional
        Overrides detection. True always allows the self-hosted cap, which is
        how a reverse proxy in front of a self-hosted server opts in. False
        always applies the cloud cap. Default None detects from ``fish_base``.

    Returns
    -------
    int
        300 for the cloud API, 1000 for self-hosted fish-speech.
    """
    hosted = (not _is_cloud_base(fish_base)) if self_hosted is None else self_hosted
    return CHUNK_LENGTH_SELF_HOSTED_HI if hosted else CHUNK_LENGTH_CLOUD_HI


def clamp_number[T: int | float](
    value: Any,
    lo: T,
    hi: T,
    default: T,
    parse: Callable[[Any], T],
) -> T:
    """Parse a Fish numeric knob and keep it inside its documented range.

    Parameters
    ----------
    value : Any
        A number or numeric string from a caller.
    lo : int or float
        Lowest allowed value.
    hi : int or float
        Highest allowed value.
    default : int or float
        Returned when ``value`` does not parse, is non-finite, or is a boolean.
    parse : Callable
        ``int`` or ``float``.

    Returns
    -------
    int or float
        The parsed value clamped to ``[lo, hi]``, or ``default``.
    """
    # bool is an int subclass. True would clamp to 1.
    if isinstance(value, bool):
        return default
    try:
        n = _parsed(value, parse)
    except (TypeError, ValueError, OverflowError):
        n = default
    if isinstance(n, float) and not math.isfinite(n):
        return default
    if n < lo:
        return lo
    if n > hi:
        return hi
    return n


FISH_TTS_MODEL_IDS: Final = (
    "s2.1-pro",
    "s2.1-pro-free",
    "s2-pro",
    "s1",
    "drama-3-preview",
)
# Each table maps a lowercase name to the Literal it stands for, so a lookup
# narrows the type without a cast.
_LATENCY_BY_NAME: Final[dict[str, FishLatency]] = {
    "low": "low",
    "balanced": "balanced",
    "normal": "normal",
}
FISH_LATENCIES: Final = frozenset(_LATENCY_BY_NAME)
_AUDIO_FORMAT_BY_NAME: Final[dict[str, AudioFormat]] = {
    "wav": "wav",
    "pcm": "pcm",
    "mp3": "mp3",
    "opus": "opus",
}
_ASR_FORMAT_BY_NAME: Final[dict[str, AsrFormat]] = {
    "json": "json",
    "text": "text",
    "verbose_json": "verbose_json",
    "srt": "srt",
    "vtt": "vtt",
}
_TTS_MODEL_BY_NAME: Final[dict[str, TtsModel]] = {
    "s1": "s1",
    "s2-pro": "s2-pro",
    "s2.1-pro": "s2.1-pro",
    "s2.1-pro-free": "s2.1-pro-free",
    "drama-3-preview": "drama-3-preview",
}


def normalize_tts_model(name: str) -> str:
    """Catalog ids are lowercase. Any other single-token id is returned stripped.

    Parameters
    ----------
    name : str
        A model id from a caller or the environment.

    Returns
    -------
    str
        The lowercase catalog id, the stripped id when it is a single printable
        token that Fish may know (a model newer than the catalog), or the
        default model when ``name`` is blank or could split a header. The result is
        ``str`` and not ``TtsModel`` because other ids pass through; use
        ``catalog_tts_model`` to narrow.
    """
    text = name.strip()
    # The id is a request header. A newline would split that header, and a
    # non-ASCII character makes the client refuse to send it.
    if text and all(32 < ord(ch) < 127 and not ch.isspace() for ch in text):
        key = text.lower()
        if key in FISH_TTS_MODEL_IDS:
            return key
        return text
    return SuiteDefaults().tts_model


def catalog_tts_model(name: str) -> TtsModel | None:
    """Narrow a model id to the catalog ``TtsModel`` set.

    Parameters
    ----------
    name : str
        A model id. Compared after strip and lowercase.

    Returns
    -------
    TtsModel or None
        The catalog id, or None for any other id.

    Examples
    --------
    >>> catalog_tts_model(" S2-Pro ")
    's2-pro'
    >>> catalog_tts_model("Drama-3-Preview")
    'drama-3-preview'
    >>> catalog_tts_model("mystery") is None
    True
    """
    return _TTS_MODEL_BY_NAME.get(name.strip().lower())


def known_latency(name: str, default: FishLatency) -> FishLatency:
    """Accept ``low``, ``balanced``, or ``normal``. Anything else keeps ``default``.

    Parameters
    ----------
    name : str
        Caller or env latency. Compared after strip and lowercase.
    default : FishLatency
        Value returned when ``name`` is not one of the three Fish modes.

    Returns
    -------
    FishLatency
        A known latency, or ``default``.

    Examples
    --------
    >>> known_latency(" Balanced ", "normal")
    'balanced'
    >>> known_latency("turbo", "normal")
    'normal'
    """
    return _LATENCY_BY_NAME.get(name.strip().lower(), default)


def known_audio_format(name: str, default: AudioFormat) -> AudioFormat:
    """Accept ``wav``, ``pcm``, ``mp3`` or ``opus``. Anything else keeps ``default``.

    Parameters
    ----------
    name : str
        A format name. Compared after strip and lowercase.
    default : AudioFormat
        Value returned when ``name`` is not a Fish audio format.

    Returns
    -------
    AudioFormat
        A known format, or ``default``.

    Examples
    --------
    >>> known_audio_format("PCM", "mp3")
    'pcm'
    >>> known_audio_format("flac", "mp3")
    'mp3'
    """
    return _AUDIO_FORMAT_BY_NAME.get(name.strip().lower(), default)


def known_asr_format(name: str, default: AsrFormat) -> AsrFormat:
    """Accept a transcription response format. Anything else keeps ``default``.

    Parameters
    ----------
    name : str
        ``json``, ``text``, ``verbose_json``, ``srt`` or ``vtt``. Compared after
        strip and lowercase.
    default : AsrFormat
        Value returned when ``name`` is not one of them.

    Returns
    -------
    AsrFormat
        A known format, or ``default``.
    """
    return _ASR_FORMAT_BY_NAME.get(name.strip().lower(), default)


_MP3_BITRATES: dict[int, Literal[64, 128, 192]] = {64: 64, 192: 192}


def known_mp3_bitrate(rate: int) -> Literal[64, 128, 192]:
    """Keep 64 and 192. Every other rate snaps to 128.

    Parameters
    ----------
    rate : int
        Requested MP3 bitrate.

    Returns
    -------
    Literal[64, 128, 192]
        A Fish-accepted MP3 bitrate.
    """
    return _MP3_BITRATES.get(rate, 128)


_OPUS_BITRATES = frozenset({_OPUS_AUTO, 24000, 32000, 48000, 64000})


def known_opus_bitrate(rate: int) -> int:
    """Keep -1000, 24000, 32000, 48000, or 64000. Anything else snaps to -1000.

    Parameters
    ----------
    rate : int
        Requested Opus bitrate. ``-1000`` means Fish chooses.

    Returns
    -------
    int
        A documented Opus bitrate, or ``-1000``.
    """
    if rate in _OPUS_BITRATES:
        return rate
    return _OPUS_AUTO


# --- Deprecated names -------------------------------------------------------------------


@deprecated("parse_number", "0.2.0")
def number_or[T: int | float](value: Any, default: T, parse: Callable[[Any], T]) -> T:
    """Call ``parse_number``. Deprecated since 0.2.0."""
    return parse_number(value, default, parse)


@deprecated("clamp_number", "0.2.0")
def clamp_num[T: int | float](
    value: Any,
    lo: T,
    hi: T,
    default: T,
    parse: Callable[[Any], T],
) -> T:
    """Call ``clamp_number``. Deprecated since 0.2.0."""
    return clamp_number(value, lo, hi, default, parse)


@deprecated("normalize_tts_model", "0.2.0")
def known_tts_model(name: str) -> str:
    """Call ``normalize_tts_model``. Deprecated since 0.2.0."""
    return normalize_tts_model(name)
