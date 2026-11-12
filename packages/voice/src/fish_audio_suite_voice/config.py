"""Voice CLI settings from the process environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, overload
from urllib.parse import urlsplit

from fish_audio_suite_kit import (
    DEFAULT_SYSTEM_PROMPT,
    CueStyle,
    SuiteDefaults,
    env_base,
    env_text,
    env_token,
    is_insecure_fish_base,
    normalize_tts_model,
)
from fish_audio_suite_voice.debug import warn
from fish_audio_suite_voice.llm_tune import DEFAULT_HISTORY_TURNS, LlmTune
from fish_audio_suite_voice.playback_kind import DEFAULT_PLAYBACK, playback_key
from fish_audio_suite_voice.tts_tune import TtsTune
from fish_audio_suite_voice.tune import (
    AecTune,
    BargeTune,
    ListenTune,
    SttTune,
    TurnTune,
    read_flag,
    read_float,
    read_int,
    read_text,
)

# A fresh utterance cannot end sooner: the turn needs about 1.2 s of silence plus
# the minimum voiced time. A faster clip is mostly pre-roll or leftover audio.
DEFAULT_REPEAT_WINDOW_S: Final = 1.5
_OUTPUT_LATENCIES: Final = ("auto", "low", "high")
_TTS_CUE_MODES: Final = ("official", "free")

__all__ = [
    "SystemPrompt",
    "VoiceCliConfig",
    "fish_problem",
    "load_config",
    "log_dir_from_env",
    "prompt_from_file",
    "system_prompt_from_file",
    "warn_if_insecure_base",
]


@dataclass(frozen=True, slots=True)
class VoiceCliConfig:
    """Duplex settings read from the process environment. No YAML.

    Attributes
    ----------
    tts : TtsTune
        What Fish is asked to speak with, and the fade and level applied before the sink.
    llm : LlmTune
        Chat backend and request settings.
    listen : ListenTune
        The voice detector, its strictness, and the limits of one utterance.
    barge : BargeTune
        Barge-in gate and the pauses around a reply.
    aec : AecTune
        Mic cleanup: echo cancellation, automatic gain and noise suppression.
    stt : SttTune
        Which speech recognition hears the user.
    turn : TurnTune
        The end-of-turn check on the Fish path. Off by default.
    hear_tone : bool
        Keep the tone Fish ASR hears in the voice (``[laughter]``) and give the model a short
        note about it for that turn. Experimental, and off by default.
    history_turns : int
        User and assistant pairs kept in the chat history.
    repeat_window_s : float
        A transcript equal to the previous line is dropped only when it ends
        within this many seconds of the mic opening. 0 never drops a repeat.
    mood_lead : bool
        Rewrite a sentence-leading mood word into a ``[cue]``.
    tts_cues : str
        ``"official"`` keeps only Fish's own cues and maps or removes the rest (a voice may act
        out ``[smiling]`` as a hum). ``"free"`` sends every cue the model writes as written.
    stream_tts : bool
        Speak the reply while the model is still writing it. Off by default.
    drop_narration : bool
        Drop stage-direction lines such as ``She smiles.`` before TTS.
    output_latency : str
        The sound card's playback buffer: ``"low"`` or ``"high"`` (bigger, so a busy moment
        does not make it run dry and click). ``"auto"`` is ``"high"`` in the full-screen app,
        which is busy drawing, and ``"low"`` otherwise.
    full_screen : bool
        Whether the session runs in the full-screen app, which decides ``"auto"`` above.
        Set by the CLI, not read from the environment.

    Notes
    -----
    ``fish_api_key`` and ``fish_voice_id`` are empty unless the env sets them.
    The tune objects are read once here and passed down, so nothing below
    this layer reads the environment. Process env wins over ``--env-file``.
    """

    fish_api_key: str = field(repr=False)
    fish_base: str
    fish_voice_id: str
    fish_asr_language: str
    tts_model: str
    playback: str
    system_prompt: str
    device: str | None
    output_latency: str = "auto"
    asr_model: str = "transcribe-1-pro"
    pin_seed: bool = True
    tts: TtsTune = field(default_factory=TtsTune)
    llm: LlmTune = field(default_factory=LlmTune)
    listen: ListenTune = field(default_factory=ListenTune)
    barge: BargeTune = field(default_factory=BargeTune)
    aec: AecTune = field(default_factory=AecTune)
    stt: SttTune = field(default_factory=SttTune)
    turn: TurnTune = field(default_factory=TurnTune)
    hear_tone: bool = False
    history_turns: int = DEFAULT_HISTORY_TURNS
    repeat_window_s: float = DEFAULT_REPEAT_WINDOW_S
    mood_lead: bool = False
    tts_cues: str = "official"
    drop_narration: bool = False
    stream_tts: bool = False
    full_screen: bool = False

    @property
    def cue_style(self) -> CueStyle:
        """Return how cues are shaped for ``tts_model``, from ``tts_cues`` and ``mood_lead``.

        The speaker and the reply both take it from here, so they shape alike.
        """
        return CueStyle.for_model(
            self.tts_model, official=self.tts_cues == "official", lead=self.mood_lead
        )

    @property
    def sink_latency(self) -> str:
        """Return the buffer size the playback stream asks for: ``low`` or ``high``."""
        if self.output_latency != "auto":
            return self.output_latency
        # The screen is busy drawing, so a bigger buffer rides it out.
        return "high" if self.full_screen else "low"


def fish_problem(c: VoiceCliConfig) -> str | None:
    """Say which Fish setting a session cannot start without.

    Parameters
    ----------
    c : VoiceCliConfig
        The settings.

    Returns
    -------
    str or None
        ``FISH_API_KEY missing`` or ``FISH_VOICE_ID missing``, or None when both are set.
    """
    if not c.fish_api_key:
        return "FISH_API_KEY missing"
    if not c.fish_voice_id:
        return "FISH_VOICE_ID missing"
    return None


def _asr_model(default: str) -> str:
    # Voice only runs the two native ids. Anything else would 4xx on every turn.
    chosen = env_token("FISH_ASR_MODEL", default).lower()
    return chosen if chosen in {"transcribe-1", "transcribe-1-pro"} else default


def _base_host(base: str) -> str:
    # Only the hostname is safe to print. A base can carry user:password@ or a
    # query string with a credential.
    try:
        return urlsplit(base).hostname or "unknown"
    except ValueError:
        return "unknown"


def warn_if_insecure_base(c: VoiceCliConfig) -> bool:
    """Warn when an API key would travel over plain http to a non-loopback host.

    Parameters
    ----------
    c : VoiceCliConfig
        The settings built by ``load_config``.

    Returns
    -------
    bool
        True when at least one warning was printed. Self-hosting over http on a
        LAN is legitimate, so this only warns and never stops the run.
    """
    warned = False
    if c.fish_api_key and is_insecure_fish_base(c.fish_base):
        warn(
            f"[fish] FISH_BASE host {_base_host(c.fish_base)} is plain http, so FISH_API_KEY is sent "
            "unencrypted. Use https unless this host is on a network you trust."
        )
        warned = True
    if c.llm.api_key and is_insecure_fish_base(c.llm.base):
        warn(
            f"[llm] FISH_LLM_BASE host {_base_host(c.llm.base)} is plain http, so the LLM key is sent "
            "unencrypted. Use https unless this host is on a network you trust."
        )
        warned = True
    return warned


def log_dir_from_env() -> str:
    """Read ``FISH_VOICE_LOG_DIR``: a folder to write a complete log of each run to, or empty."""
    return read_text("FISH_VOICE_LOG_DIR")


def _tts_cues() -> str:
    """Read ``FISH_TTS_CUES``: ``official`` or ``free``."""
    value = read_text("FISH_TTS_CUES", "official").lower() or "official"
    if value not in _TTS_CUE_MODES:
        warn(f"fish-voice: ignoring FISH_TTS_CUES={value!r}, use official or free")
        return "official"
    return value


def _output_latency() -> str:
    """Read ``FISH_VOICE_OUTPUT_LATENCY``: ``auto``, ``low`` or ``high``."""
    value = read_text("FISH_VOICE_OUTPUT_LATENCY", "auto").lower() or "auto"
    if value not in _OUTPUT_LATENCIES:
        warn(f"fish-voice: ignoring FISH_VOICE_OUTPUT_LATENCY={value!r}, use auto, low or high")
        return "auto"
    return value


def load_config() -> VoiceCliConfig:
    """Read ``VoiceCliConfig`` from the current process environment.

    Returns
    -------
    VoiceCliConfig
        Clamped numeric knobs. A missing key keeps the ``SuiteDefaults`` value.
        Call this after dotenv loading; an earlier call will not see those keys.

    Notes
    -----
    ``FISH_API_KEY`` uses ``env_text``, so a blank value stays blank instead
    of falling back to a default key. There is no default voice id.
    """
    d = SuiteDefaults()
    prompt = _system_prompt()
    return VoiceCliConfig(
        fish_api_key=env_text("FISH_API_KEY"),
        fish_base=env_base("FISH_BASE", d.fish_base),
        fish_voice_id=env_text("FISH_VOICE_ID"),
        fish_asr_language=env_text("FISH_ASR_LANGUAGE", d.asr_language),
        tts_model=normalize_tts_model(env_token("FISH_TTS_MODEL", d.tts_model)),
        tts=TtsTune.from_env(),
        playback=playback_key(
            env_token(
                "FISH_VOICE_PLAYBACK",
                DEFAULT_PLAYBACK,
            )
        ),
        system_prompt=prompt.text,
        pin_seed=prompt.pin_seed,
        device=os.environ.get("FISH_VOICE_DEVICE"),
        output_latency=_output_latency(),
        asr_model=_asr_model(d.asr_model),
        llm=LlmTune.from_env(),
        listen=ListenTune.from_env(),
        barge=BargeTune.from_env(),
        aec=AecTune.from_env(),
        stt=SttTune.from_env(),
        turn=TurnTune.from_env(),
        hear_tone=read_flag("FISH_VOICE_HEAR_TONE", default=False),
        history_turns=read_int(
            "FISH_VOICE_HISTORY_TURNS",
            DEFAULT_HISTORY_TURNS,
            lo=1,
        ),
        repeat_window_s=read_float(
            "FISH_VOICE_REPEAT_WINDOW",
            DEFAULT_REPEAT_WINDOW_S,
            lo=0.0,
        ),
        mood_lead=read_flag("FISH_TTS_MOOD_LEAD", default=False),
        tts_cues=_tts_cues(),
        drop_narration=read_flag(
            "FISH_TTS_DROP_NARRATION",
            default=False,
        ),
        stream_tts=read_flag(
            "FISH_VOICE_STREAM_TTS",
            default=False,
        ),
    )


@dataclass(frozen=True, slots=True)
class SystemPrompt:
    """The session's system prompt, and whether the opening exchange is pinned after it.

    Attributes
    ----------
    text : str
        The system prompt.
    pin_seed : bool
        Pin the opening exchange that shows several cues. It is pinned for the default
        prompt and for a character file, which keeps the voice rules. A prompt written
        out in full is left alone. This is the one place that decides it.
    """

    text: str
    pin_seed: bool

    @classmethod
    def written(cls, text: str) -> SystemPrompt:
        """Take a prompt written out in full, such as ``FISH_VOICE_SYSTEM_PROMPT``.

        Parameters
        ----------
        text : str
            The whole system prompt.

        Returns
        -------
        SystemPrompt
            Pinned only when it is the default prompt.
        """
        return cls(text, pin_seed=text == DEFAULT_SYSTEM_PROMPT)


# A character card is a few KB. This stops a wrong path (a log, a binary) from
# becoming the system prompt.
_PROMPT_FILE_MAX_BYTES: Final = 64 * 1024
# Where a character file asks for the default voice rules. ``{{default_prompt}}``, with
# optional spaces inside the braces.
_DEFAULT_SLOT: Final = re.compile(r"\{\{\s*default_prompt\s*\}\}")


def system_prompt_from_file(path: str, default: str) -> str | None:
    """Read a character file and combine it with the voice rules.

    Parameters
    ----------
    path : str
        File with the character or scene, in UTF-8. ``~`` is expanded and a
        relative path is read from the working directory.
    default : str
        The voice rules (cue tags, short spoken replies).

    Returns
    -------
    str or None
        When the file holds ``{{default_prompt}}``, the file text with every
        such slot replaced by ``default``, so the file decides where the voice
        rules go. Otherwise the file text, a blank line, then ``default``. None,
        after a warning, when the file is missing, too large, not UTF-8 or
        empty, so the caller can fall back to the default prompt.
    """
    file = Path(path).expanduser()
    if file.exists() and not file.is_file():
        # A directory, a FIFO or a device would block or never end when read.
        warn(f"fish-voice: prompt file {file} is not a regular file, ignoring it")
        return None
    try:
        # One byte past the limit is enough to tell a file is too big, and a
        # huge file is never read into memory.
        with file.open("rb") as handle:
            raw = handle.read(_PROMPT_FILE_MAX_BYTES + 1)
        if len(raw) > _PROMPT_FILE_MAX_BYTES:
            warn(
                f"fish-voice: prompt file {file} is over {_PROMPT_FILE_MAX_BYTES // 1024} KiB, ignoring it"
            )
            return None
        card = raw.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        warn(f"fish-voice: cannot read prompt file {file}: {exc}")
        return None
    if not card:
        warn(f"fish-voice: prompt file {file} is empty, ignoring it")
        return None
    if _DEFAULT_SLOT.search(card):
        # A function, not a string: a backslash in ``default`` is not an escape.
        return _DEFAULT_SLOT.sub(lambda _: default, card)
    return f"{card}\n\n{default}"


@overload
def prompt_from_file(path: str, *, required: Literal[True]) -> SystemPrompt | None: ...
@overload
def prompt_from_file(path: str, *, required: Literal[False]) -> SystemPrompt: ...
def prompt_from_file(path: str, *, required: bool) -> SystemPrompt | None:
    """Build the system prompt from a character file, with the voice rules.

    Parameters
    ----------
    path : str
        The file, as ``system_prompt_from_file`` reads it.
    required : bool
        What a file that cannot be used means. True (``--prompt-file``) returns None, so
        the caller can stop. False (``FISH_VOICE_SYSTEM_PROMPT_FILE``) falls back to the
        default prompt. Either way a warning says why first.

    Returns
    -------
    SystemPrompt or None
        The file and the voice rules, with the opening exchange pinned.
    """
    composed = system_prompt_from_file(path, DEFAULT_SYSTEM_PROMPT)
    if composed is not None:
        return SystemPrompt(composed, pin_seed=True)
    return None if required else SystemPrompt.written(DEFAULT_SYSTEM_PROMPT)


def _system_prompt() -> SystemPrompt:
    file = os.environ.get("FISH_VOICE_SYSTEM_PROMPT_FILE", "").strip()
    inline = os.environ.get("FISH_VOICE_SYSTEM_PROMPT")
    if file:
        if inline is not None:
            warn(
                "fish-voice: FISH_VOICE_SYSTEM_PROMPT_FILE is set, so FISH_VOICE_SYSTEM_PROMPT is ignored"
            )
        return prompt_from_file(file, required=False)
    # A prompt set but blank means "no system prompt", so unlike the other keys
    # a blank value still counts here.
    return SystemPrompt.written(DEFAULT_SYSTEM_PROMPT if inline is None else inline)
