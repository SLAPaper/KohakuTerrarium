"""Reject an operation that would join or mutate its own active execution."""

import asyncio
from typing import Any

from kohakuterrarium.core.regeneration_reservation import require_regeneration_available
from kohakuterrarium.core.shutdown_callbacks import reject_shutdown_callback_wait
from kohakuterrarium.errors import ConflictError


def require_stop_available(agent: Any) -> None:
    """Validate public shutdown authority before changing admission or host state."""
    reject_own_turn(agent, "stop and wait")
    reject_shutdown_callback_wait(agent)
    if asyncio.current_task() is getattr(agent, "_input_driver_task", None):
        raise ConflictError("Cannot stop and wait from the input driver callback")
    require_regeneration_available(agent)
    admission = getattr(agent, "_stop_admission", None)
    if callable(admission):
        admission()


def reject_own_turn(agent: Any, operation: str) -> None:
    """Reject blocking control from the agent's turn, controller or shutdown task."""
    caller = asyncio.current_task()
    tasks = (
        getattr(agent, "_turn_lock_holder", None),
        getattr(agent, "_active_turn_task", None),
        getattr(agent, "_processing_task", None),
        getattr(agent, "_turn_finalization_task", None),
        getattr(agent, "_shutdown_task", None),
    )
    if caller is not None and any(caller is task for task in tasks):
        raise ConflictError(f"Cannot {operation} from the agent's own active callback")
