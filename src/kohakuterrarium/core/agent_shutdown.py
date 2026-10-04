"""One accepted agent shutdown per running lifetime."""

import asyncio

from kohakuterrarium.core.agent_state import close_owner_admission
from kohakuterrarium.core.branch_rerun import drain_history_reruns
from kohakuterrarium.core.branch_selection import (
    _join_accepted,
    drain_history_selections,
    mark_history_selection_stopping,
)
from kohakuterrarium.core.model_switch_runtime import (
    drain_model_switches,
    mark_model_switch_closing,
)
from kohakuterrarium.core.operation_admission import require_stop_available
from kohakuterrarium.errors import ConflictError


def _close_admission(agent):
    mark_model_switch_closing(agent)
    mark_history_selection_stopping(agent)
    close_owner_admission(agent)


async def _finish(agent):
    lock = getattr(agent, "_stop_lock", None)
    if lock is None:
        lock = agent._stop_lock = asyncio.Lock()
    async with lock:
        await drain_model_switches(agent)
        await drain_history_selections(agent)
        await drain_history_reruns(agent)
        await agent._stop_inner()
        agent._stopped = True


async def stop_agent(agent):
    """Join accepted cleanup before propagating waiter cancellation or its cause."""
    require_stop_available(agent)
    if getattr(agent, "_starting", False):
        raise ConflictError(
            "Cannot stop while the agent is starting; cancel or await start first"
        )
    pending = getattr(agent, "_shutdown_task", None)
    if pending is not None and not pending.done():
        await _join_accepted(pending)
        return
    if getattr(agent, "_stopped", False):
        return
    task = asyncio.create_task(_finish(agent))
    agent._shutdown_task = task
    _close_admission(agent)
    await _join_accepted(task)
