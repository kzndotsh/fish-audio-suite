"""Shared Fish TTS/ASR knobs. Literals live on SuiteDefaults."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlsplit

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
    "TTS_SPEED_HI",
    "TTS_SPEED_LO",
    "UNIT_INTERVAL_HI",
    "UNIT_INTERVAL_LO",
    "SuiteDefaults",
    "catalog_tts_model",
    "chunk_length_hi",
    "is_insecure_fish_base",
    "known_asr_format",
    "known_audio_format",
    "known_latency",
    "known_mp3_bitrate",
    "known_opus_bitrate",
    "normalize_tts_model",
    "strip_base",
]

DEFAULT_SYSTEM_PROMPT: Final = (
    "You are a voice assistant. Your reply is spoken aloud by a text-to-speech voice. "
    "English by default. Speak the user's language if they switch. "
    "Say only words that should be heard: no markdown, bullets, emoji, or URLs. "
    "Talk the way a person does in a live conversation, not the way people write: contractions, "
    "short sentences, and fragments are fine. Most replies are one to three sentences, and a few "
    "words is right when that is all the moment needs. Answer or react to what was just said "
    "first, and do not repeat the user's sentence back. Ask a question only now and then, never "
    "at the end of every reply. Vary how you open and close, and do not reuse a stock phrase "
    "from an earlier reply. A natural 'um', 'well' or trailing off is fine, at most once in a "
    "reply and not in every reply. Match the user's energy: slower and softer when they are, "
    "livelier when they are. Write numbers and symbols out as words. "
    "Square-bracket cues are silent stage directions for the voice and are never spoken. "
    "Never mention, describe or explain a cue, and never treat one as something the user "
    "asked for. "
    "Every sentence starts with its own cue, and the cue is one word from this list, never a "
    "description of a face, a voice or an action: happy, sad, angry, excited, calm, nervous, "
    "confident, surprised, delighted, scared, worried, frustrated, empathetic, embarrassed, "
    "proud, relaxed, grateful, curious, sarcastic, hopeful, disappointed, determined. "
    "Repeat the same cue to keep a feeling going, and change the cue whenever the feeling "
    "shifts: a joke landing, a sad turn, a surprise. A cue always comes before the words it "
    "colors, never after them: do not end a reply with a cue. "
    "Sounds are cues too: [laughing], [chuckling], [sighing], [gasping], [clear throat]. "
    "Follow a sound with the words of it so the voice has something to say ([laughing] "
    "Ha ha, [sighing] Oh well), use them sparingly, and never the same one twice in a row. "
    "[break] is a short pause and [long-break] a longer one: use them only in the middle of "
    "a sentence, never beside a period, question mark or exclamation mark, and most replies "
    "need none. A whisper only if the user asks you to whisper. "
    "Write for the ear: no dashes, semicolons or parentheses, just commas and periods. "
)

# One opening exchange that shows several cues in a reply. In a test on two
# models a conversation drifted to one cue per reply even though the system
# prompt asks for more, apparently because the model copies its own earlier
# replies. With this pinned, the default model went from 1.1 to 2.2 cues per
# reply and Claude Haiku 4.5 from 1.0 to 1.9.
DEFAULT_SEED_EXCHANGE: Final[tuple[tuple[str, str], ...]] = (
    (
        "Hi there!",
        "[happy] Oh, hey, it's so good to hear you. [calm] I'm all yours.",
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
    asr_model: str = "transcribe-1-pro"
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
