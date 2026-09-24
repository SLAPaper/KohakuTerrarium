"""Private in-process session-state owner.

One dedicated control task drains a stdlib queue of short synchronous
callbacks. Callers receive a Future for each accepted callback. This is
not a durable broker, revision log, or generic workflow runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import queue
from typing import Any, Callable, TypeVar

from kohakuterrarium.errors import ConflictError, InvalidRequestError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
_STOP = object()


class SessionStateOwner:
    """Serialize short synchronous session-state commits on one task."""

    def __init__(self, *, name: str = "session_state_owner") -> None:
        self._name = name
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._wakeup: asyncio.Event | None = None
        self._closing = False
        self._closed = False
        self._draining = False
        self._has_started = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def is_running(self) -> bool:
        return (
            self._has_started
            and not self._closing
            and not self._closed
            and self._task is not None
            and not self._task.done()
        )

    @property
    def is_closing(self) -> bool:
        return self._closing and not self._closed

    @property
    def has_started(self) -> bool:
        return self._has_started

    def start(self) -> None:
        """Idempotently start the owner on the running event loop."""
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = queue.SimpleQueue()
        self._wakeup = asyncio.Event()
        self._closing = False
        self._closed = False
        self._draining = False
        self._has_started = True
        self._task = loop.create_task(self._run(), name=self._name)
        self._task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """Settle work even when cancellation precedes the task's first step."""
        if task is not self._task:
            return
        self._closed = True
        self._closing = False
        error = asyncio.CancelledError() if task.cancelled() else task.exception()
        self._abort_queued(error or InvalidRequestError("session-state owner stopped"))

    def close_admission(self) -> None:
        """Reject later public submits without cancelling accepted work."""
        if self._closed or self._closing:
            return
        self._closing = True
        if self._task is None or self._task.done():
            self._closed = True
            return

    def submit(self, callback: Callable[[], T]) -> asyncio.Future[T]:
        """Accept a public state change while admission is open."""
        if self._closing:
            raise InvalidRequestError(
                "session-state owner is closed to public requests"
            )
        return self.submit_internal(callback)

    def submit_internal(self, callback: Callable[[], T]) -> asyncio.Future[T]:
        """Accept execution finalization until the final drain begins."""
        if inspect.iscoroutinefunction(callback) or inspect.isasyncgenfunction(
            callback
        ):
            raise InvalidRequestError(
                "session-state owner callback must be synchronous; "
                "await inside the callback is not allowed"
            )
        if self._closed or self._draining:
            raise InvalidRequestError("session-state owner is closed")
        if self._task is None or self._task.done() or self._loop is None:
            raise InvalidRequestError(
                "session-state owner is not running; call start() first"
            )
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ConflictError(
                "session-state owner requires its owning event loop"
            ) from exc
        if running_loop is not self._loop:
            raise ConflictError("session-state owner requires its owning event loop")
        if asyncio.current_task() is self._task:
            raise ConflictError(
                "reentrant session-state owner submit from the owner task "
                "would deadlock"
            )
        result: asyncio.Future[T] = self._loop.create_future()
        self._queue.put((callback, result))
        self._wake()
        return result

    async def close(self) -> None:
        """Close public admission, drain accepted work, and stop the task."""
        self.close_admission()
        task = self._task
        if task is not None and not task.done() and not self._draining:
            self._draining = True
            self._queue.put(_STOP)
            self._wake()
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            except Exception:
                logger.error("Session-state owner failed before close", exc_info=True)
        self._closed = True
        self._closing = False
        self._task = None
        self._wakeup = None

    def _wake(self) -> None:
        wakeup = self._wakeup
        loop = self._loop
        if wakeup is None or loop is None or loop.is_closed():
            return

        def set_wakeup() -> None:
            wakeup.set()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            set_wakeup()
        else:
            loop.call_soon_threadsafe(set_wakeup)

    async def _get_item(self) -> Any:
        assert self._wakeup is not None
        while True:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                pass
            self._wakeup.clear()
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                await self._wakeup.wait()

    def _execute(self, callback: Callable[[], Any], fut: asyncio.Future[Any]) -> None:
        try:
            result = callback()
        except asyncio.CancelledError:
            fut.cancel()
            return
        except Exception as exc:
            self._settle(fut, exc=exc)
            return
        if inspect.isawaitable(result):
            closer = getattr(result, "close", None)
            if callable(closer):
                closer()
            self._settle(
                fut,
                exc=InvalidRequestError(
                    "session-state owner callback must not return an awaitable"
                ),
            )
            return
        self._settle(fut, result=result)

    def _settle(
        self,
        fut: asyncio.Future[Any],
        *,
        result: Any = None,
        exc: BaseException | None = None,
    ) -> None:
        if fut.done():
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    def _abort_queued(self, exc: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                continue
            _callback, fut = item
            if isinstance(exc, asyncio.CancelledError):
                fut.cancel()
            else:
                self._settle(fut, exc=exc)

    async def _run(self) -> None:
        try:
            while True:
                item = await self._get_item()
                if item is _STOP:
                    self._drain_accepted()
                    return
                callback, fut = item
                self._execute(callback, fut)
        except asyncio.CancelledError:
            self._abort_queued(asyncio.CancelledError())
            raise
        except Exception as exc:
            self._abort_queued(exc)
            raise

    def _drain_accepted(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                continue
            callback, fut = item
            self._execute(callback, fut)
