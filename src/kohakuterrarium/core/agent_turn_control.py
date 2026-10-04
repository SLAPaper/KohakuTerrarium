"""Interruptible turn execution and cancellation-safe finalization."""

import asyncio
from typing import Any

from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class SettledProcessingFailure(RuntimeError):
    """A controller failure whose output and finalization have already settled."""


def cancel_active_execution(agent: Any, *, already_requested: bool) -> None:
    """Cancel execution without cancelling the consumer or accepted finalization."""
    if already_requested or getattr(agent, "_turn_phase", None) == "finalizing":
        return
    processing = getattr(agent, "_processing_task", None)
    task = (
        processing
        if processing is not None and not processing.done()
        else getattr(agent, "_active_turn_task", None)
    )
    if task is not None and not task.done():
        task.cancel()


def withdraw_turn_events(controller: Any, events: list[Any]) -> None:
    """Remove only unconsumed occurrences from an interrupted turn."""
    identities = {id(event) for event in events}
    controller._pending_events[:] = [
        event for event in controller._pending_events if id(event) not in identities
    ]
    kept = []
    queue = controller._event_queue
    while not queue.empty():
        event = queue.get_nowait()
        queue.task_done()
        if id(event) not in identities:
            kept.append(event)
    for event in kept:
        queue.put_nowait(event)


async def settle_turn_for_stop(agent: Any) -> None:
    """Stop new consumer claims and settle active execution before releasing dependencies."""
    resume = getattr(agent, "_consumer_resume", None)
    if resume is not None:
        resume.clear()
    active = getattr(agent, "_active_turn_task", None)
    if active is None or active.done():
        return
    agent._interrupt_active_turn()
    done, _ = await asyncio.wait({active}, timeout=10)
    if not done:
        raise ConflictError(
            "Active turn has not settled; shutdown cannot release its dependencies"
        )
    if not active.cancelled():
        try:
            active.result()
        except SettledProcessingFailure as exc:
            logger.warning(
                "Stopping after a settled processing failure", error=str(exc)
            )


async def finalize_turn(
    agent: Any, event: Any, controller: Any, text: list[str]
) -> None:
    """Join one accepted finalization through its durable output and cleanup."""
    agent._turn_phase = "finalizing"
    task = asyncio.create_task(agent._finalize_processing(event, controller, text))
    agent._turn_finalization_task = task
    try:
        await _join_accepted(task)
    finally:
        if getattr(agent, "_turn_finalization_task", None) is task:
            agent._turn_finalization_task = None
