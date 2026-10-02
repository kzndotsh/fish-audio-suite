"""Estimate how much of a reply the listener actually heard.

History after a barge-in must hold only that prefix. The estimate is a rate
(about 16 characters a second at 1x speed) applied to the bytes that reached
the device, minus what was still buffered, cut back to a finished word.
"""

from __future__ import annotations

import re

from fish_audio_suite_kit import strip_cue_tags
from fish_audio_suite_voice.aec import SAMPLE_BYTES

# Conversational English at speed 1.0. Other languages and voices drift, so
# this is an estimate and the cut is always rounded down to a word.
CHARS_PER_S = 16
_MIN_SPEED = 0.1
# A model can still write [clear]. That is a throat-clear, not words, so it
# stays out of the next turn's history. Other cues stay, so the model keeps
# seeing the style it used.
_CLEAR_TAG_RE = re.compile(r"\[clear\]", re.IGNORECASE)


def _audible_text(text: str) -> str:
    return strip_cue_tags(text)


def _pcm_chars(bytes_played: int, sample_rate: int, speed: float, latency_s: float) -> int:
    # bytes_played counts bytes handed to the device. The last output_latency_s
    # of them are still in its buffer, so the listener has not heard them.
    secs = bytes_played / max(sample_rate * SAMPLE_BYTES, 1) - max(0.0, latency_s)
    return max(0, int(secs * CHARS_PER_S * max(speed, _MIN_SPEED)))


def _cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code < 0x3100
        or 0x3400 <= code < 0x4DC0
        or 0x4E00 <= code < 0xA000
        or 0xF900 <= code < 0xFB00
        or 0xAC00 <= code < 0xD7B0
    )


def _word_char(ch: str) -> bool:
    return ch.isalnum() and not _cjk_char(ch)


def _extends_tail(tail: str, nxt: str) -> bool:
    # "there" is finished when the next character is "." or a space.
    # "ther" is not finished when the next character is "e". A Cyrillic
    # letter continues the word too. "there.Friend" has finished "there":
    # the period is not part of the next word. Keeping only "Hello" made
    # the next turn say "there" again.
    if not tail or _cjk_char(nxt):
        return False
    last = tail[-1]
    if nxt.isalnum():
        if _word_char(last):
            return True
        # "3.14" is one number. "there.Friend" is two words.
        if last == "." and nxt.isdigit() and len(tail) > 1 and tail[-2].isdigit():
            return True
        return last in "'’-" and len(tail) > 1 and _word_char(tail[-2])
    return nxt in "'’-" and _word_char(last)


def _boundary_finished(prefix: str, nxt: str) -> bool:
    if not prefix or _extends_tail(prefix, nxt):
        return False
    # "3" before "." is the start of "3.14", not a finished word.
    return not (nxt == "." and prefix[-1].isdigit())


def _finished_prefix(text: str, nxt: str) -> str:
    if _boundary_finished(text, nxt):
        return text
    for index in range(len(text) - 1, 0, -1):
        if _boundary_finished(text[:index], text[index]):
            return text[:index]
    return ""


def _word_prefix(text: str, n: int) -> str:
    if n >= len(text):
        return text.strip()
    cut = text[:n]
    # A newline or a non-breaking space is a word boundary. Splitting on
    # ASCII space only kept "Hello there" and dropped "friend." once the
    # next line had started. "Hello\u00a0there" has no ASCII space, so a
    # cut in "there" forgot "Hello" and the next turn said it again.
    last = -1
    for index, ch in enumerate(cut):
        if ch.isspace():
            last = index
    if last < 0:
        # A cut with no space is the middle of the first English word. History
        # should not record that fragment. CJK has no spaces, including after
        # an English word ("Hello你好"). Dropping that cut forgot speech the
        # speaker had already played, so the next turn said it again.
        stripped = cut.strip()
        if any(_cjk_char(ch) for ch in stripped):
            return stripped
        # "Hello there" cut on the last letter of "Hello" has no space yet.
        # The next character is the space, so that word was played. Dropping
        # it made the next turn say "Hello" again.
        if stripped and not _extends_tail(stripped, text[n]):
            return stripped
        # "Hello.Friend" has no space yet. The cut is inside "Friend",
        # so the finished "Hello." was dropped and the next turn said it again.
        return _finished_prefix(stripped, text[n])
    head = cut[:last].strip()
    tail = cut[last + 1 :]
    if not tail:
        return head
    # "Hello there" has finished "there" when the next character is a space
    # or a period. Stopping at the space before it made the next turn say
    # "there" again. "Hello 你" has finished "你" even when "好" is still coming.
    if not _extends_tail(tail, text[n]):
        return f"{head} {tail}".strip()
    # "there.Frien" still extends into "d", but "there." was already played.
    # Returning only "Hello" made the next turn say "there" again.
    for index in range(len(tail) - 1, 0, -1):
        if not _extends_tail(tail[:index], tail[index]):
            return f"{head} {tail[:index]}".strip()
    extra = []
    for ch in tail:
        if not _cjk_char(ch):
            break
        extra.append(ch)
    if extra:
        return f"{head} {''.join(extra)}".strip()
    return head


def spoken_prefix(
    sent_text: str,
    *,
    bytes_played: int,
    sample_rate: int,
    audio_format: str,
    got_audio: bool,
    cancelled: bool,
    failed: bool = False,
    speed: float = 1.0,
    output_latency_s: float = 0.0,
) -> str:
    """Return the part of ``sent_text`` the listener heard.

    Parameters
    ----------
    sent_text : str
        Text sent to Fish for the turn, cues included.
    bytes_played : int
        Bytes the sink accepted.
    sample_rate : int
        PCM sample rate of ``bytes_played``.
    audio_format : str
        Fish output format. Only ``pcm`` can be cut by length.
    got_audio : bool
        True after the first Fish audio chunk.
    cancelled : bool
        True when barge-in or Ctrl+C stopped the turn.
    failed : bool, optional
        True when the socket dropped or Fish returned an error.
    speed : float, optional
        Prosody speed. Faster speech covers more text per second.
    output_latency_s : float, optional
        Device buffer in seconds. That much of ``bytes_played`` is not heard yet.

    Returns
    -------
    str
        The full text without ``[clear]`` when the turn finished. After a
        cancel or failure, a word-aligned prefix with every cue removed. Empty when nothing was
        played, or when the format cannot be cut by length, so history never
        records a reply the listener did not hear.
    """
    if not sent_text.strip() or not got_audio or bytes_played <= 0:
        # Fish can deliver a chunk the sink then drops, such as a dead mpv
        # pipe. History should not record a reply the speaker never played.
        return ""
    if not (cancelled or failed):
        return " ".join(_CLEAR_TAG_RE.sub(" ", sent_text).split())
    # Encoded audio has no fixed bytes-per-second, so a partial turn cannot be
    # cut. Recording the whole reply would tell the next turn it was heard.
    if audio_format != "pcm":
        return ""
    chars = _pcm_chars(bytes_played, sample_rate, speed, output_latency_s)
    if chars <= 0:
        return ""
    return _word_prefix(_audible_text(sent_text), chars)
