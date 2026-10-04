"""Narrow source reservation for a compound interrupt-and-rerun request."""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

from kohakuterrarium.errors import ConflictError


@dataclass
class RegenerationReservation:
    task: asyncio.Task
    ready: asyncio.Event
    dispatch_ready: asyncio.Event
    started: bool = False
    completion: asyncio.Future | None = None
    compact_controller: object | None = None


def require_regeneration_available(agent):
    """Refuse foreign mutation while the captured source is reserved."""
    claim = getattr(agent, "_regeneration_reservation", None)
    if claim is None:
        return
    try:
        caller = asyncio.current_task()
    except RuntimeError:
        caller = None
    if caller is not claim.task:
        raise ConflictError("Compound regeneration is awaiting completion")


@contextmanager
def reserve_regeneration(agent):
    """Install reservation before any await and release it on every settled exit."""
    if getattr(agent, "_regeneration_reservation", None) is not None:
        raise ConflictError("Compound regeneration is already pending")
    claim = RegenerationReservation(
        asyncio.current_task(), asyncio.Event(), asyncio.Event()
    )
    agent._regeneration_reservation = claim
    try:
        yield claim
    finally:
        if getattr(agent, "_regeneration_reservation", None) is claim:
            agent._regeneration_reservation = None
        claim.ready.set()
        claim.dispatch_ready.set()
        if claim.compact_controller is not None:
            asyncio.get_running_loop().call_soon(
                agent._maybe_trigger_compact, claim.compact_controller
            )


def defer_regeneration_compact(agent, controller):
    """Defer only the admitted rerun's automatic compaction until source release."""
    claim = getattr(agent, "_regeneration_reservation", None)
    run = getattr(agent, "_active_event_run", None) or []
    if claim is None or len(run) != 1 or run[0].future is not claim.completion:
        return False
    caller = asyncio.current_task()
    if caller is None or not any(
        caller is getattr(agent, name, None)
        for name in (
            "_active_turn_task",
            "_processing_task",
            "_turn_finalization_task",
        )
    ):
        return False
    if agent.controller is not controller:
        return False
    claim.compact_controller = controller
    return True


def reserved_regeneration(operation):
    """Wrap the compound operation in one noninheritable source claim."""

    @wraps(operation)
    async def accepted(agent, *args, **kwargs):
        with reserve_regeneration(agent):
            return await operation(agent, *args, **kwargs)

    return accepted


def bind_regeneration_completion(agent, completion):
    """Permit only the exact admitted rerun to run before reservation release."""
    claim = getattr(agent, "_regeneration_reservation", None)
    if claim is not None:
        claim.completion = completion
        claim.dispatch_ready.set()


async def wait_regeneration_admission(agent, run):
    """Park competing claimed input without blocking the admitted rerun itself."""
    while (claim := getattr(agent, "_regeneration_reservation", None)) is not None:
        if (
            len(run) == 1
            and run[0].future is claim.completion
            and claim.completion is not None
        ):
            claim.started = True
            return True
        if claim.started:
            await claim.ready.wait()
        else:
            await claim.dispatch_ready.wait()
            if getattr(agent, "_regeneration_reservation", None) is claim:
                return False
    return True
