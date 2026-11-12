"""Stop flags, and telling our own cancellation and its tear-down noise from real errors."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any, Final, Protocol

from fish_audio_suite_voice.debug import debug

__all__ = [
    "StopFlag",
    "all_cancel_noise",
    "any_of",
    "first_real_error",
    "is_cancel_noise",
    "is_own_cancel",
    "never",
    "quiet_shutdown",
    "reap",
]


class StopFlag(Protocol):
    """Anything with ``is_set``, such as a ``threading.Event``, that ends a mic stream."""

    def is_set(self) -> bool:
        """Return whether the stream should stop."""
        ...


class _AnyOf:
    """A stop flag that is set when any of several are."""

    def __init__(self, flags: tuple[StopFlag | Callable[[], bool], ...]) -> None:
        self._flags = flags

    def is_set(self) -> bool:
        return any(flag() if callable(flag) else flag.is_set() for flag in self._flags)


def any_of(*flags: StopFlag | Callable[[], bool]) -> StopFlag:
    """Return a stop flag that is set when any of ``flags`` is.

    Parameters
    ----------
    *flags : StopFlag or Callable
        Flags, or functions that say whether to stop, such as the session's mute check.
        They are asked in order, and the first that says stop ends the asking.

    Returns
    -------
    StopFlag
        Set while any of them is. It reads them each time it is asked.
    """
    return _AnyOf(flags)


def never() -> bool:
    """Say no: the stand-in for a stop or mute check that is never set.

    Returns
    -------
    bool
        Always False.
    """
    return False


# Last resort when a teardown error is not an exception type we know. Matching
# on message text breaks when an SDK rewords it, so a hit is logged.
_CANCEL_NOISE: Final = (
    "athrow",
    "cancel scope",
    "generator didn't stop",
    "different task than it was entered",
)


def is_cancel_noise(exc: BaseException, *, cancelled: bool = False) -> bool:
    """Return whether ``exc`` is a barge-in or Ctrl+C tear-down, not a Fish error.

    Parameters
    ----------
    exc : BaseException
        An error from the websocket task or a task group.
    cancelled : bool, optional
        True when the caller's cancel flag is already set. Any ordinary error
        raised while the turn is being torn down is then expected noise.

    Returns
    -------
    bool
        True for ``CancelledError`` and ``GeneratorExit``, and for any error
        raised after cancel was requested. Without a cancel flag, a known
        anyio or asyncio message still matches, and the match is logged at
        debug so a reworded message shows up.
    """
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return True
    if cancelled and isinstance(exc, (Exception, BaseExceptionGroup)):
        return True
    msg = str(exc).lower()
    if any(phrase in msg for phrase in _CANCEL_NOISE):
        debug("cancel.noise matched by message: {}", msg)
        return True
    return False


def all_cancel_noise(exc: BaseException, *, cancelled: bool = False) -> bool:
    """Return whether ``exc``, and every error inside it when it is a group, is cancel noise.

    Parameters
    ----------
    exc : BaseException
        An error from a stream, possibly an exception group from a task group.
    cancelled : bool, optional
        Passed to ``is_cancel_noise`` for each error.

    Returns
    -------
    bool
        True for a non-empty group whose every leaf is noise, or a single error that is.
        A group that holds one real error is not noise: it must be reported, or the reply
        stops unexplained.
    """
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            all_cancel_noise(item, cancelled=cancelled) for item in exc.exceptions
        )
    return is_cancel_noise(exc, cancelled=cancelled)


def first_real_error(exc: BaseException) -> BaseException:
    """Return the first error in ``exc`` that is not cancel noise, unwrapping groups.

    Parameters
    ----------
    exc : BaseException
        An error from one Fish attempt, possibly an exception group.

    Returns
    -------
    BaseException
        The first leaf, depth first, that ``is_cancel_noise`` does not match. When every
        leaf is noise, the first leaf. ``exc`` itself when it is not a group.

    Notes
    -----
    The first entry of a group is often the cancel from a sibling task. Stopping there
    would hide a 503, and a turn with no audio would not be retried.
    """
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        for item in exc.exceptions:
            found = first_real_error(item)
            if not is_cancel_noise(found):
                return found
        return first_real_error(exc.exceptions[0])
    return exc


def is_own_cancel(flag: asyncio.Event | threading.Event | None) -> bool:
    """Return whether a ``CancelledError`` here comes from our own cancel flag.

    Parameters
    ----------
    flag : asyncio.Event or threading.Event or None
        The cancel flag the caller was given. ``None`` means the caller has no flag.

    Returns
    -------
    bool
        True when the flag is set and the running task has no pending
        cancellation request of its own. Only then may the caller turn a
        ``CancelledError`` into a normal early stop. Anything else, such as
        ``asyncio.timeout``, a task group or an outer ``task.cancel()``, must
        propagate.
    """
    if flag is None or not flag.is_set():
        return False
    task = asyncio.current_task()
    return task is None or task.cancelling() == 0


async def reap(task: asyncio.Task[Any], *, wait_s: float | None = None) -> None:
    """Wait for a task that was just cancelled and drop its outcome.

    Parameters
    ----------
    task : asyncio.Task
        A child task that ``task.cancel()`` was already called on.
    wait_s : float or None, optional
        Seconds to wait. None waits until the task finishes. A task that is
        still running after that is left for the loop shutdown.

    Notes
    -----
    Unlike ``suppress(BaseException)`` around ``await task``, this never
    swallows ``KeyboardInterrupt``, ``SystemExit`` or a cancellation of the
    caller. The child's own exception is read so asyncio does not report it
    as never retrieved.
    """
    done, _pending = await asyncio.wait({task}, timeout=wait_s)
    if task in done and not task.cancelled():
        task.exception()


async def quiet_shutdown(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel leftover tasks on a private loop before it is closed.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The loop ``run_isolated`` created. The current task is left running
        so this coroutine can finish the gather.
    """
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks(loop) if task is not current]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
