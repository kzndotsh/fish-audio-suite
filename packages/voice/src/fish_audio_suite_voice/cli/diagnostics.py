"""The full-screen app's Diagnostics screen (F12): what is for tuning, not for chatting.

Four tabs: the timings of the last turn and the recent ones, the log, the last reply as it
was sent to Fish with its cues, and what the session runs on. The chat screen shows none
of it. The app keeps the log while this screen is closed and hands it over when it opens.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from textual.app import ComposeResult, RenderResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.content import Content
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Log, Static, TabbedContent, TabPane

from fish_audio_suite_kit import split_cues
from fish_audio_suite_voice.cli.chat_screens import printable
from fish_audio_suite_voice.cli.session_view import SessionView, split_cells, turn_timings
from fish_audio_suite_voice.session.latency import LatencySnapshot, percentile, turn_summary

__all__ = [
    "LOG_LINES",
    "DiagnosticsScreen",
    "Timings",
    "cue_content",
]

LOG_LINES = 1000  # log lines kept, whether or not the screen is open
_NO_TURNS = "no turns yet"
_STAGE_TONES = {"asr": "$text-secondary", "llm": "$text-primary", "tts": "$text-accent"}
# Each stage has its own pattern as well as its own colour, so the bar reads without colour:
# dots in the top half, in every row, and in the bottom half.
_STAGE_DOTS = {"asr": "⠛", "llm": "⣿", "tts": "⣤"}


def cue_content(text: str) -> Content:
    """Return ``text`` as content with each ``[cue]`` styled and nothing read as markup.

    Parameters
    ----------
    text : str
        A reply.

    Returns
    -------
    Content
        The printable text, its cues in the accent colour.
    """
    parts: list[tuple[str, str] | str] = [
        (piece, "italic $text-accent") if is_cue else piece
        for piece, is_cue in split_cues(printable(text))
    ]
    return Content.assemble(*parts)


class Timings(Widget):
    """The last turn: the wait before you heard audio, and where it went.

    Three lines: the wait beside the whole turn's length, one bar split into recognising
    (asr), thinking (llm) and speaking (tts), and the number behind each part. The parts
    differ by pattern as well as colour, with a gap between them, and come in the same order
    as the legend, so the bar reads without colour. The numbers are always shown.
    """

    DEFAULT_CSS: ClassVar[str] = """
    Timings {
        height: 3;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:  # noqa: A002 - Textual's own name
        super().__init__(id=id)
        self._snapshot: LatencySnapshot | None = None

    def show(self, snapshot: LatencySnapshot | None) -> None:
        """Show a finished turn, or the empty state for ``None``."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> RenderResult:
        """Draw the wait, the split bar and the legend, fitted to the width."""
        snapshot = self._snapshot
        if snapshot is None:
            return Content.assemble((_NO_TURNS, "dim"))
        timings = turn_timings(snapshot)
        if timings is None:
            return Content(turn_summary(snapshot) or _NO_TURNS)
        width = max(1, self.size.width)
        wait = f"{timings.wait_ms / 1000:.2f}s"
        total = f"total {timings.total_ms / 1000:.2f}s" if timings.total_ms is not None else ""
        gap = " " * max(1, width - len(wait) - len(" wait") - len(total))
        head = Content.assemble((wait, "bold"), " wait", gap, (total, "dim"))
        gaps = max(0, len(timings.stages) - 1)  # a blank cell between parts marks each edge
        cells = split_cells([ms for _, ms in timings.stages], max(1, width - gaps))
        parts: list[tuple[str, str] | str] = []
        for position, ((name, _), count) in enumerate(zip(timings.stages, cells, strict=True)):
            if position and width > gaps:
                parts.append(" ")
            parts.append((_STAGE_DOTS[name] * count, _STAGE_TONES[name]))
        bar = Content.assemble(*parts)
        decimals = 2 if len(_legend(timings.stages, 2)) <= width else 1
        legend = Content.assemble(*_legend_parts(timings.stages, decimals))
        return Content("\n").join([head, bar, legend])


def _legend(stages: tuple[tuple[str, float], ...], decimals: int) -> str:
    return " ".join(f"{name} {ms / 1000:.{decimals}f}" for name, ms in stages)


def _legend_parts(
    stages: tuple[tuple[str, float], ...], decimals: int
) -> list[tuple[str, str] | str]:
    parts: list[tuple[str, str] | str] = []
    for position, (name, ms) in enumerate(stages):
        if position:
            parts.append(" ")
        parts.append((name, _STAGE_TONES[name]))
        parts.append(f" {ms / 1000:.{decimals}f}")
    return parts


def _seconds(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.2f} s"


def _recent_table(view: SessionView) -> Content:
    turns = view.recent_turns
    if not turns:
        return Content("")
    head = f"{'turn':>4}  {'heard':>7}  {'first word':>10}  {'first audio':>11}  {'total':>7}"
    rows = [Content.assemble((head, "dim"))]
    first = view.turns_timed - len(turns) + 1
    for offset, turn in reversed(list(enumerate(turns))):
        rows.append(
            Content(
                f"{first + offset:>4}  {_seconds(turn.asr_ms):>7}  "
                f"{_seconds(turn.llm_first_token_ms):>10}  {_seconds(turn.first_audio_ms):>11}  "
                f"{_seconds(turn.voice_to_voice_ms):>7}"
            )
        )
    waits = [turn.first_audio_ms for turn in turns if turn.first_audio_ms is not None]
    if waits:
        spread = (
            f"wait before audio: p50 {_seconds(percentile(waits, 0.5))}   "
            f"p95 {_seconds(percentile(waits, 0.95))}   "
            f"over {len(waits)} turn{'' if len(waits) == 1 else 's'}"
        )
        rows += [Content(""), Content.assemble((spread, "dim"))]
    return Content("\n").join(rows)


class DiagnosticsScreen(Screen[None]):
    """Timings, the log, the reply as Fish read it, and what the session runs on.

    Parameters
    ----------
    title : str
        The chat's title, for the header.
    info : str
        The models and voice the session uses.
    log : Iterable of str
        The log lines so far, oldest first, already made printable.
    view : SessionView
        The session as it is when the screen opens.
    sent : str
        The last reply as it was sent to Fish.
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "close", "Back to chat")]

    def __init__(
        self, *, title: str, info: str, log: Iterable[str], view: SessionView, sent: str
    ) -> None:
        super().__init__()
        self._title = title
        self._info = info
        self._backlog = list(log)
        self._latest = (view, sent)
        self._shown: tuple[SessionView, str] | None = None
        # Kept here, not queried, so a late update while the screen closes changes nothing.
        self._timings = Timings(id="latency")
        self._recent = Static(id="recent")
        self._log = Log(id="log", max_lines=LOG_LINES)
        self._sent = Static(id="sent")
        self._session = Static(id="session")

    def compose(self) -> ComposeResult:
        """Lay out the header, the four tabs and the keys."""
        with Horizontal(id="diag-head"):
            yield Static("Diagnostics", id="diag-title")
            yield Static(f"{printable(self._title)} · for tuning, not for chatting", id="diag-note")
            yield Static("F12 back to chat", id="diag-back")
        with TabbedContent():
            with TabPane("Timings", id="timings-tab"):
                yield self._timings
                yield self._recent
            with TabPane("Log", id="log-tab"):
                yield self._log
            with TabPane("Sent to Fish", id="sent-tab"):
                yield Static("the last reply as Fish read it, cues included", classes="dim")
                yield self._sent
            with TabPane("Session", id="session-tab"):
                yield self._session
        yield Footer()

    def on_mount(self) -> None:
        """Fill the log with what arrived before the screen opened, and draw the session."""
        self._log.write_lines(self._backlog)
        self._backlog = []
        self._draw()

    def show(self, view: SessionView, sent: str) -> None:
        """Draw the session as it is now.

        Parameters
        ----------
        view : SessionView
            The latest view.
        sent : str
            The last reply as it was sent to Fish.
        """
        self._latest = (view, sent)
        self._draw()

    def _draw(self) -> None:
        if self._shown == self._latest:
            return
        self._shown = view, sent = self._latest
        self._timings.show(view.last_turn)
        self._recent.update(_recent_table(view))
        self._sent.update(cue_content(sent) if sent else Content.assemble(("no reply yet", "dim")))
        sees = f"the model sees {view.sent_turns} of {view.chat_turns} turns"
        self._session.update(Content("\n").join([Content(self._info), Content(sees)]))

    def add_log(self, line: str) -> None:
        """Add a log line that arrived while the screen is open.

        Parameters
        ----------
        line : str
            The line, already made printable.
        """
        if self._log.is_mounted:
            self._log.write_line(line)
        else:
            self._backlog.append(line)  # opened but not drawn yet: keep the order

    def action_close(self) -> None:
        """Go back to the chat."""
        self.dismiss(None)
