"""Admit user input against an explicit selected history before enqueueing."""

import asyncio

from kohakuterrarium.core.branch_selection import (
    _select_branch,
    require_history_selection_available,
)
from kohakuterrarium.core.event_inbox import EventEnvelope
from kohakuterrarium.errors import ConflictError


def normalize_branch_view(value):
    """Detach positive turn and branch coordinates supplied by a transport."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("branch_view must be a turn-to-branch mapping")
    result = {}
    for key, branch in value.items():
        if isinstance(key, str) and key.isascii() and key.isdecimal():
            key = int(key)
        if type(key) is not int or key <= 0 or type(branch) is not int or branch <= 0:
            raise ValueError("branch_view requires positive integer coordinates")
        if key in result:
            raise ValueError("branch_view contains duplicate turn coordinates")
        result[key] = branch
    return result


async def continue_on_branch(agent, event, view):
    """Select and enqueue together, or preserve ordinary current-branch ingress."""
    require_history_selection_available(agent)
    current = dict(getattr(agent, "_parent_branch_path", ()) or ())
    current[getattr(agent, "_turn_index", 0)] = getattr(agent, "_branch_id", 0)
    if all(current.get(turn) == branch for turn, branch in view.items()):
        return await agent._process_event(event)
    if getattr(agent, "_paused", False):
        raise ConflictError("Cannot continue a selected branch while paused")
    event.context["await_turn"] = True
    future = asyncio.get_running_loop().create_future()
    envelope = EventEnvelope(event, future=future)
    await _select_branch(agent, view, continuation=envelope)
    outcome = await future
    return bool(outcome.was_primary)
