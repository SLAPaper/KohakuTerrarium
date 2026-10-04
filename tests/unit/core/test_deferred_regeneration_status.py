"""Execution cancellation does not prove that its independent waiter was cancelled."""

import asyncio

import pytest

from kohakuterrarium.core.deferred_regeneration import (
    DeferredRegenerationStatus,
    run_deferred_regeneration,
)
from kohakuterrarium.errors import ConflictError


@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_cancelled_execution_marks_waiter_provenance_unknown(cancel_waiter):
    entered = asyncio.Event()
    execution = None

    async def operation(agent, **options):
        nonlocal execution
        execution = asyncio.current_task()
        entered.set()
        await asyncio.Event().wait()

    record = DeferredRegenerationStatus()
    waiter = asyncio.create_task(run_deferred_regeneration(operation, None, {}, record))
    await entered.wait()
    if cancel_waiter:
        waiter.cancel()
    execution.cancel()
    observed = (await asyncio.gather(waiter, return_exceptions=True))[0]
    assert isinstance(observed, asyncio.CancelledError)
    status = record.snapshot(waiter, "token")
    assert status["status"] == "cancelled"
    assert status["waiter_cancelled"] is None
    assert status["applied"] is None and status["receipt"] is None
    assert execution.done()


async def test_failure_without_applied_receipt_retains_unknown_application_state():
    async def operation(agent, **options):
        raise ConflictError("source changed before admission")

    record = DeferredRegenerationStatus()
    waiter = asyncio.create_task(run_deferred_regeneration(operation, None, {}, record))
    with pytest.raises(ConflictError, match="source changed"):
        await waiter
    status = record.snapshot(waiter, "token")
    assert status["status"] == "error"
    assert status["applied"] is None and status["receipt"] is None
    assert status["waiter_cancelled"] is False
    assert "source changed" in status["error"]
