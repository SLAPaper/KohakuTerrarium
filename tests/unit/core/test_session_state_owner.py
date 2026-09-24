"""Unit tests for the private in-process session-state owner."""

import asyncio
import inspect

import pytest

from kohakuterrarium.core.session_state_owner import SessionStateOwner
from kohakuterrarium.errors import ConflictError, InvalidRequestError


async def test_fifo_callbacks_run_on_owner_task():
    owner = SessionStateOwner()
    owner.start()
    order: list[int] = []
    tasks: list[asyncio.Task] = []
    try:

        def first():
            order.append(1)
            tasks.append(asyncio.current_task())
            return "a"

        def second():
            order.append(2)
            tasks.append(asyncio.current_task())
            return "b"

        first_future = owner.submit(first)
        second_future = owner.submit(second)
        assert await first_future == "a"
        assert await second_future == "b"
        assert order == [1, 2]
        assert tasks[0] is owner.task is tasks[1]
        assert tasks[0] is not asyncio.current_task()
    finally:
        await owner.close()


async def test_caller_cancel_does_not_cancel_accepted_commit():
    owner = SessionStateOwner()
    owner.start()
    ran: list[str] = []
    try:
        future = owner.submit(lambda: ran.append("done") or "committed")
        future.cancel()
        with pytest.raises(asyncio.CancelledError):
            await future
        await asyncio.sleep(0)
        assert ran == ["done"]
        assert owner.task is not None and not owner.task.done()
        assert await owner.submit(lambda: "next") == "next"
    finally:
        await owner.close()


async def test_callback_exception_settles_future_and_later_work_runs():
    owner = SessionStateOwner()
    owner.start()
    try:

        def boom():
            raise RuntimeError("commit failed")

        with pytest.raises(RuntimeError, match="commit failed"):
            await owner.submit(boom)
        assert await owner.submit(lambda: "next") == "next"
    finally:
        await owner.close()


async def test_reentrant_submit_from_owner_task_rejects():
    owner = SessionStateOwner()
    owner.start()
    try:

        def nested():
            with pytest.raises(ConflictError, match="(?i)reentr"):
                owner.submit(lambda: "inner")
            return "outer"

        assert await owner.submit(nested) == "outer"
    finally:
        await owner.close()


async def test_close_rejects_late_submit_and_is_idempotent():
    owner = SessionStateOwner()
    owner.start()
    await owner.close()
    await owner.close()
    with pytest.raises(InvalidRequestError, match="(?i)closed"):
        owner.submit(lambda: None)
    assert owner.task is None or owner.task.done()


async def test_close_resolves_pending_and_drains_accepted():
    owner = SessionStateOwner()
    owner.start()
    order: list[str] = []

    def first():
        order.append("first")
        return "first"

    def later():
        order.append("later")
        return "later"

    first_future = owner.submit(first)
    second_future = owner.submit(later)
    owner.close_admission()
    close_task = asyncio.create_task(owner.close())
    with pytest.raises(InvalidRequestError, match="(?i)closed"):
        owner.submit(lambda: "late")
    assert await first_future == "first"
    assert await second_future == "later"
    await close_task
    assert order == ["first", "later"]


async def test_start_is_idempotent_and_asserts_task_identity():
    owner = SessionStateOwner()
    owner.start()
    first = owner.task
    owner.start()
    try:
        assert owner.task is first
        assert owner.is_running
        assert await owner.submit(lambda: (asyncio.current_task(),)) == (first,)
    finally:
        await owner.close()
    assert not owner.is_running


async def test_reject_async_callback_and_awaitable_result():
    owner = SessionStateOwner()
    owner.start()
    try:

        async def async_cb():
            return 1

        with pytest.raises(InvalidRequestError, match="(?i)async|await"):
            owner.submit(async_cb)

        created = []

        def returns_coro():
            coro = async_cb()
            created.append(coro)
            return coro

        with pytest.raises(InvalidRequestError, match="(?i)await"):
            await owner.submit(returns_coro)
        assert created
        # Closing the coroutine in the owner must prevent "never awaited".
        loop = asyncio.get_running_loop()

        def returns_future():
            fut = loop.create_future()
            fut.set_result("x")
            return fut

        with pytest.raises(InvalidRequestError, match="(?i)await"):
            await owner.submit(returns_future)
    finally:
        await owner.close()
    await asyncio.sleep(0)


async def test_submit_before_start_rejected():
    owner = SessionStateOwner()
    with pytest.raises(InvalidRequestError, match="(?i)not running|start"):
        owner.submit(lambda: 1)


async def test_worker_failure_settles_pending_and_close_is_safe():
    owner = SessionStateOwner()
    owner.start()
    task = owner.task
    assert task is not None
    first = owner.submit(lambda: 1)
    second = owner.submit(lambda: 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(asyncio.CancelledError):
        await second
    await owner.close()
    with pytest.raises(InvalidRequestError, match="(?i)closed"):
        owner.submit(lambda: 3)


async def test_close_without_start_is_idempotent_and_restart_works():
    owner = SessionStateOwner()
    await owner.close()
    await owner.close()
    with pytest.raises(InvalidRequestError, match="(?i)closed"):
        owner.submit(lambda: None)
    owner.start()
    try:
        assert await owner.submit(lambda: "ok") == "ok"
    finally:
        await owner.close()


async def test_callback_cancelled_error_settles_current_and_later_work():
    owner = SessionStateOwner()
    owner.start()

    def cancelled():
        raise asyncio.CancelledError()

    current = owner.submit(cancelled)
    later = owner.submit(lambda: "later")
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(current), 1)
        assert await asyncio.wait_for(later, 1) == "later"
    finally:
        await owner.close()


async def test_close_after_worker_failure_settles_accepted_work():
    owner = SessionStateOwner()
    owner.start()
    owner._queue.put(("invalid-envelope",))
    pending = owner.submit(lambda: "must not run")
    with pytest.raises(ValueError):
        await asyncio.wait_for(pending, 1)
    await owner.close()
    assert not owner.is_running


async def test_public_close_keeps_internal_finalization_until_drain():
    owner = SessionStateOwner()
    owner.start()
    order = []
    first = owner.submit(lambda: order.append("public"))
    owner.close_admission()
    try:
        with pytest.raises(InvalidRequestError):
            owner.submit(lambda: order.append("rejected"))
        await owner.submit_internal(lambda: order.append("terminal"))
        await first
        assert order == ["public", "terminal"]
    finally:
        await owner.close()
    with pytest.raises(InvalidRequestError):
        owner.submit_internal(lambda: order.append("too late"))


def test_submit_signature_is_sync_returning_awaitable():
    owner = SessionStateOwner()
    assert not inspect.iscoroutinefunction(owner.submit)
    assert not inspect.iscoroutinefunction(owner.start)
    assert inspect.iscoroutinefunction(owner.close)
