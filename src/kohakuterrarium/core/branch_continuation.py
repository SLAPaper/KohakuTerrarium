"""Admit user input against an explicit selected history before enqueueing."""

import asyncio
from copy import deepcopy

from kohakuterrarium.core.branch_selection import (
    _select_branch,
    require_history_selection_available,
)
from kohakuterrarium.core.event_inbox import EventEnvelope
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.session.history import resolve_branch_view_strict
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


def _resolve_on_worker(store, namespace, view):
    return resolve_branch_view_strict(store.get_events(namespace), view)


def _live_path(agent):
    current = dict(getattr(agent, "_parent_branch_path", ()) or ())
    turn = getattr(agent, "_turn_index", 0)
    branch = getattr(agent, "_branch_id", 0)
    if turn > 0 and branch > 0:
        current[turn] = branch
    return current


async def continue_on_branch(agent, event, view):
    """Select and enqueue together, or preserve ordinary current-branch ingress."""
    require_history_selection_available(agent)
    require_model_mutation_available(agent)
    view = deepcopy(view)
    current = _live_path(agent)
    store = getattr(agent, "session_store", None)
    if store is None:
        same = all(current.get(turn) == branch for turn, branch in view.items())
    else:
        controller = agent.controller
        conversation = controller.conversation
        owner = agent._session_state_owner
        provider = agent.llm
        namespace = agent.config.name
        resolved = await store.run(_resolve_on_worker, store, namespace, view)
        require_history_selection_available(agent)
        require_model_mutation_available(agent)
        if (
            agent.session_store is not store
            or agent.controller is not controller
            or controller.conversation is not conversation
            or agent._session_state_owner is not owner
            or agent.llm is not provider
            or agent.config.name != namespace
            or _live_path(agent) != current
        ):
            raise ConflictError("Continuation source changed during branch resolution")
        same = resolved == current
    if same:
        return await agent._process_event(event)
    if getattr(agent, "_paused", False):
        raise ConflictError("Cannot continue a selected branch while paused")
    event.context["await_turn"] = True
    future = asyncio.get_running_loop().create_future()
    envelope = EventEnvelope(event, future=future)
    await _select_branch(agent, view, continuation=envelope)
    outcome = await future
    return bool(outcome.was_primary)
