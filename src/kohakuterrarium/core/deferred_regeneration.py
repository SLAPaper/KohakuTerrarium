"""Deferred request status separates settled operation results from waiter cancellation."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from kohakuterrarium.core.branch_selection import _join_accepted


@dataclass
class DeferredRegenerationStatus:
    status: str = "pending"
    error: str | None = None
    receipt: dict | None = None
    applied: bool | None = None
    waiter_cancelled: bool | None = False

    def snapshot(self, task, request_id):
        """Detach a token's status without inferring rollback from task cancellation."""
        value = {
            "request_id": request_id,
            "status": self.status,
            "error": self.error,
            "receipt": deepcopy(self.receipt),
            "applied": self.applied,
            "waiter_cancelled": self.waiter_cancelled,
        }
        if not task.done():
            value["status"] = "pending"
        elif task.cancelled() and self.status == "pending":
            value.update(status="cancelled", applied=False, waiter_cancelled=True)
        return value


async def run_deferred_regeneration(
    operation: Callable, agent: Any, options: dict, record: DeferredRegenerationStatus
):
    """Join accepted execution and retain its terminal result when the waiter cancels."""
    execution = asyncio.create_task(operation(agent, **options))
    try:
        return await _join_accepted(execution)
    except asyncio.CancelledError:
        record.waiter_cancelled = None if execution.cancelled() else True
        raise
    finally:
        if execution.cancelled():
            record.status = "cancelled"
        else:
            error = execution.exception()
            if error is None:
                result = execution.result()
                record.status = result["status"]
                record.receipt = deepcopy(result)
                record.applied = True
            else:
                record.status = getattr(error, "status", "error")
                record.error = str(error)
                receipt = getattr(error, "receipt", None)
                record.receipt = deepcopy(receipt) if receipt else None
                record.applied = getattr(error, "may_have_run", None)
