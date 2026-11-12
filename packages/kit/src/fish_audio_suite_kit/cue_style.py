"""How a reply's cues are shaped for one TTS model, and the one call that shapes it.

A caller builds one ``CueStyle`` from the model and its settings, then shapes every reply,
whole or streamed, with it. The order of the steps and which flags each one takes live
here, so the proxy, the voice app and a stream cannot run them differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fish_audio_suite_kit.catalog import takes_directions
from fish_audio_suite_kit.cues import normalize_cues
from fish_audio_suite_kit.dialogue import extract_quoted_speech
from fish_audio_suite_kit.scrub_tts import scrub_tts
from fish_audio_suite_kit.stream_holds import tts_hold_at

__all__ = [
    "CueMode",
    "CueStyle",
]

CueMode = Literal["official", "free", "directed"]


@dataclass(frozen=True, slots=True)
class CueStyle:
    """How cues are shaped before Fish reads a reply.

    Attributes
    ----------
    mode : {"official", "free", "directed"}
        ``official`` keeps only Fish's own cues, mapping any other cue to the nearest
        official one or removing it (``[smiling]`` becomes ``[happy]``). ``free`` keeps
        every cue as written, lowercased. ``directed`` is for a model directed in plain
        language (``drama-3``): directions and paired tags such as ``<whisper>`` go through
        as written. Default ``free``.
    lead : bool
        Rewrite a sentence-leading mood word (``Excited, hello``) into a cue. Default
        False, because the word also opens ordinary sentences (``Curious, isn't it?``).
        A directed reply is never rewritten, but a stream still holds an unfinished mood
        word back while this is on.

    Examples
    --------
    >>> style = CueStyle.for_model("s2.1-pro", official=True, lead=True)
    >>> style.shape("**Excited,** hi there. [smiling] Bye.")
    '[excited] hi there. [happy] Bye.'
    >>> CueStyle.for_model("drama-3", official=True, lead=False).mode
    'directed'
    """

    mode: CueMode = "free"
    lead: bool = False

    @classmethod
    def for_model(cls, model: str, *, official: bool, lead: bool) -> CueStyle:
        """Return the style a TTS model takes.

        Parameters
        ----------
        model : str
            Fish TTS model id, already resolved from any alias.
        official : bool
            Keep only Fish's own cues. Not used for a directed model.
        lead : bool
            Rewrite a sentence-leading mood word into a cue.

        Returns
        -------
        CueStyle
            ``directed`` when the model takes plain-language directions, otherwise
            ``official`` or ``free``.
        """
        if takes_directions(model):
            return cls("directed", lead=lead)
        return cls("official" if official else "free", lead=lead)

    @property
    def carries_mood(self) -> bool:
        """Whether a sentence sent without an emotion gets the last one (``MoodCarry``).

        A directed model reads each direction where it is written, so it gets none.
        """
        return self.mode != "directed"

    def shape(
        self,
        text: str,
        *,
        line_start: bool = True,
        continued: bool = False,
        before: str = "",
        after: str = "",
        sentence_start: bool = True,
        dialogue_only: bool = False,
    ) -> str:
        """Scrub ``text`` and shape its cues for this style: what Fish should read.

        Parameters
        ----------
        text : str
            A whole reply, or one stable chunk of a streamed one.
        line_start : bool, optional
            True when ``text`` begins a line. Default True.
        continued : bool, optional
            True when text was already sent before this chunk. Default False.
        before : str, optional
            Text already sent on this line before this chunk. Scrubbing reads its last
            character, and cue shaping reads it to tell whether a stop at the start of
            the chunk ends a sentence.
        after : str, optional
            Text still held after this chunk. Only its first character is read.
        sentence_start : bool, optional
            True when ``text`` starts a sentence, so a mood word there can be a lead.
            Default True. Only read when ``lead`` is on.
        dialogue_only : bool, optional
            Keep only quoted speech, dropping the narration around it. Default False.

        Returns
        -------
        str
            ``scrub_tts``, then ``extract_quoted_speech`` when ``dialogue_only``, then
            ``normalize_cues``, each with this style's flags. An edge space that scrub kept
            for a neighbour is still there.
        """
        directed = self.mode == "directed"
        cleaned = scrub_tts(
            text,
            line_start=line_start,
            continued=continued,
            before=before,
            after=after,
            directed=directed,
        )
        if dialogue_only:
            cleaned = extract_quoted_speech(cleaned)
        spoken = normalize_cues(
            cleaned,
            lead=self.lead,
            continued=not sentence_start,
            before=before,
            official=self.mode == "official",
            directed=directed,
        )
        # Scrub keeps one edge space for a neighbour, even one it decoded (``&#160;``),
        # and a rewritten lead would drop it.
        if cleaned[:1] == " " and spoken[:1] != " ":
            spoken = f" {spoken}"
        if cleaned[-1:] == " " and spoken[-1:] != " ":
            spoken = f"{spoken} "
        return spoken

    def hold_at(
        self, text: str, *, line_start: bool, sentence_start: bool, before: str = ""
    ) -> int:
        """Return where a stream must keep buffering before ``shape`` can take the text.

        Parameters
        ----------
        text : str
            Unshaped chunk, including any unfinished span.
        line_start : bool
            True when this chunk begins a line.
        sentence_start : bool
            True when a mood word here can become a lead cue.
        before : str, optional
            Shaped text already accepted on this line. Default empty.

        Returns
        -------
        int
            ``tts_hold_at`` with this style's ``lead``: the start of the unfinished span,
            or ``len(text)`` when the chunk is stable.
        """
        return tts_hold_at(
            text,
            line_start=line_start,
            sentence_start=sentence_start,
            before=before,
            lead=self.lead,
        )
