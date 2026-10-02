"""Single-pass text rules. Each one does the work of a regex that would rescan a run.

Reply text is model output. A pattern that matches a run and then looks at what
follows it starts a fresh scan from every position in a long run of spaces, tabs,
marks, or pipes, which is quadratic. These functions walk the string once, so a
long run costs the same per character as any other text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

_SPACES = frozenset(" \t")
_STOPS = frozenset(".!?…。！？,;:，；：")
_MARKS = frozenset("*_`~")
_AFTER_WORD_PUNCT = frozenset(",.!?;:，！：")
_SEPARATOR_CHARS = frozenset(" \t|:-")


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _drop_space_runs_before(text: str, followers: frozenset[str]) -> str:
    # A run of spaces or tabs goes when the character after it is a follower. The
    # follower stays. Only the start of a run is a candidate, so a run is read once.
    out: list[str] = []
    n = len(text)
    last = 0
    i = 0
    while i < n:
        if text[i] in _SPACES:
            j = i + 1
            while j < n and text[j] in _SPACES:
                j += 1
            if j < n and text[j] in followers:
                out.append(text[last:i])
                last = j
                i = j + 1
                continue
            i = j
        else:
            i += 1
    out.append(text[last:])
    return "".join(out)


def collapse_space_before_stop(text: str) -> str:
    """Remove spaces and tabs that sit right before a sentence stop or comma.

    Parameters
    ----------
    text : str
        Text with a stop such as ``.``, ``,``, ``?`` or ``。`` after a gap.

    Returns
    -------
    str
        ``text`` with each such run removed and the stop kept.
    """
    return _drop_space_runs_before(text, _STOPS)


def strip_space_before_newline(text: str) -> str:
    """Remove spaces and tabs at the end of each line.

    Parameters
    ----------
    text : str
        Text that may have trailing spaces before a newline.

    Returns
    -------
    str
        ``text`` with each run before a newline removed. A final line with no
        newline keeps its trailing spaces.
    """
    return _drop_space_runs_before(text, frozenset("\n"))


def _per_line(text: str, rewrite: Callable[[str], str | None], *, line_start: bool) -> str:
    lines = text.split("\n")
    for index, line in enumerate(lines):
        # A chunk that continues a line has no row marker at index 0.
        if index == 0 and not line_start:
            continue
        replacement = rewrite(line)
        if replacement is not None:
            lines[index] = replacement
    return "\n".join(lines)


def _separator_line(line: str) -> str | None:
    if "|" in line and all(ch in _SEPARATOR_CHARS for ch in line):
        return ""
    return None


def blank_table_separators(text: str, *, line_start: bool) -> str:
    """Blank each markdown table separator line such as ``| --- | :-: |``.

    Parameters
    ----------
    text : str
        Text of one or more lines.
    line_start : bool
        False when the first line continues an earlier line, so it is left alone.

    Returns
    -------
    str
        ``text`` with each separator line emptied. A line needs a pipe, so a
        prose hyphen stays.
    """
    return _per_line(text, _separator_line, line_start=line_start)


def _table_cells(row: str) -> str:
    cells = [cell.strip() for cell in row.split("|")]
    return " ".join(cell for cell in cells if cell)


def _row_line(line: str) -> str | None:
    n = len(line)
    k = 0
    while k < n and line[k] in _SPACES:
        k += 1
    if k >= n or line[k] != "|":
        return None
    rest = line[k + 1 :]
    # Cells need a pipe with at least one character on each side of it.
    if "|" in rest[1:-1]:
        return _table_cells(rest)
    return None


def flatten_table_rows(text: str, *, line_start: bool) -> str:
    """Turn each markdown table row into its cells separated by single spaces.

    Parameters
    ----------
    text : str
        Text of one or more lines.
    line_start : bool
        False when the first line continues an earlier line, so it is left alone.

    Returns
    -------
    str
        ``text`` with each ``| a | b |`` row replaced by ``a b``.
    """
    return _per_line(text, _row_line, line_start=line_start)


def _wrap_end(text: str, start: int, run_end: int) -> int:
    """Return where the emphasis marks that begin at ``start`` end, or -1.

    The rules are the ones the old pattern spelled out as five alternatives,
    tried in order. ``_`` is both a mark and a word character, so a shorter run
    can end in front of it.
    """
    n = len(text)
    prev = text[start - 1] if start else ""
    after_word = bool(prev) and _is_word(prev)
    followed_by_word = run_end < n and _is_word(text[run_end])

    def last_end(low: int, *, word_after: bool) -> int:
        # Greedy, then give marks back one at a time. The lookahead at a shorter
        # end sees a mark: a word character only when that mark is ``_``.
        for end in range(run_end, low - 1, -1):
            at_word = followed_by_word if end == run_end else text[end] == "_"
            if at_word == word_after:
                return end
        return -1

    if after_word:
        # Two or more marks between word characters, then a word character.
        found = last_end(start + 2, word_after=True)
        if found >= 0:
            return found
    elif not prev or prev not in _MARKS:
        # An opener: no word or mark before it, a word character after.
        found = last_end(start + 1, word_after=True)
        if found >= 0:
            return found
    attached = (
        after_word
        or (start >= 2 and prev in _AFTER_WORD_PUNCT and _is_word(text[start - 2]))
        or prev == "]"
    )
    if attached:
        # A closer: after a word, after a word and a comma or stop, or after ``]``,
        # and not before a word character.
        return last_end(start + 1, word_after=False)
    return -1


def emphasis_runs(text: str) -> Iterator[tuple[int, int]]:
    """Yield the ``(start, end)`` of each run of emphasis marks that should be handled.

    Parameters
    ----------
    text : str
        Text that may hold ``*``, ``_``, backtick, or ``~`` marks.

    Yields
    ------
    tuple of int and int
        Slice bounds of a run, in order and without overlap.

    Notes
    -----
    A mark between word characters is an identifier or a product, and a mark spaced
    on both sides is an operator. A closer may follow a comma or a ``]``.
    """
    n = len(text)
    i = 0
    run_end = 0
    while i < n:
        if text[i] not in _MARKS:
            i += 1
            continue
        if i >= run_end:
            run_end = i + 1
            while run_end < n and text[run_end] in _MARKS:
                run_end += 1
        end = _wrap_end(text, i, run_end)
        if end >= 0:
            yield i, end
            i = end
        else:
            i += 1
