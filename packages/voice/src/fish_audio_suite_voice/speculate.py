"""Start writing the reply while the user may still be finishing, and keep it only if they were.

With ``FISH_VOICE_EAGER_EOT_THRESHOLD`` set, Deepgram Flux says when a turn is *probably* over
a little before it is sure. The model is asked for its reply then, in the background, and
nothing is shown or spoken. If Flux then confirms the same words, the reply that is already
under way is used, which saves the time between the two signals. If the user carried on, or
the confirmed words differ, it is thrown away and the reply is asked for as usual.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fish_audio_suite_kit import ChatMessage
from fish_audio_suite_voice.debug import debug
from fish_audio_suite_voice.llm import ChatBackend

__all__ = ["Speculation", "same_words"]

type _Item = tuple[str | None, BaseException | None]


def same_words(a: str, b: str) -> bool:
    """Say whether two transcripts are the same words, ignoring case, punctuation and spacing.

    Parameters
    ----------
    a : str
        One transcript.
    b : str
        Another.

    Returns
    -------
    bool
        True when they match.

    Examples
    --------
    >>> same_words("Hey,  there.", "hey, there.")
    True
    >>> same_words("hey there", "hey there you")
    False
    """
    words = _words(a)
    return bool(words) and words == _words(b)


def _words(text: str) -> list[str]:
    return "".join(c if c.isalnum() or c.isspace() else " " for c in text.casefold()).split()


class Speculation:
    """A reply being written ahead of time for ``text``.

    Parameters
    ----------
    backend : ChatBackend
        The model.
    messages : list of dict
        The chat so far, with ``text`` already added as the user's last line.
    text : str
        The words the reply answers.
    """

    def __init__(self, backend: ChatBackend, messages: list[ChatMessage], text: str) -> None:
        self.text: str = text
        self._stop = asyncio.Event()
        self._queue: asyncio.Queue[_Item] = asyncio.Queue()
        self._task = asyncio.create_task(self._fill(backend, messages))
        debug("llm.speculate start for {!r}", text)

    async def _fill(self, backend: ChatBackend, messages: list[ChatMessage]) -> None:
        try:
            async for token in backend.stream(messages, cancel=self._stop):
                self._queue.put_nowait((token, None))
        except asyncio.CancelledError:
            pass  # thrown away, or stopped by its own flag: the end of the stream either way
        except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001 - handed to whoever reads it
            self._queue.put_nowait((None, exc))
            return
        self._queue.put_nowait((None, None))

    def cancel(self) -> None:
        """Throw the reply away and stop asking the model for more."""
        self._stop.set()
        self._task.cancel()

    def matches(self, text: str) -> bool:
        """Say whether this reply answers ``text``."""
        return same_words(self.text, text)

    async def replay(self, cancel: asyncio.Event) -> AsyncIterator[str]:
        """Hand over the tokens written so far, then the rest as they arrive.

        Parameters
        ----------
        cancel : asyncio.Event
            Set to stop reading, as when the user interrupts.

        Yields
        ------
        str
            One piece of the reply.
        """
        try:
            while True:
                getter = asyncio.ensure_future(self._queue.get())
                stopper = asyncio.ensure_future(cancel.wait())
                done, pending = await asyncio.wait(
                    {getter, stopper}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if getter not in done:
                    return
                token, error = getter.result()
                if error is not None:
                    raise error
                if token is None:
                    return
                yield token
        finally:
            self.cancel()
