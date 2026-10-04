"""Tell our own cancellation and its tear-down noise from real errors."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Final

from fish_audio_suite_voice.debug import debug

__all__ = [
    "is_cancel_noise",
    "is_own_cancel",
    "quiet_shutdown",
    "reap",
]


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
