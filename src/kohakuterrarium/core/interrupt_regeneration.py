"""Source-pinned external interruption and deferred regeneration requests."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from kohakuterrarium.core.branch_edit_prepare import EditRequest
from kohakuterrarium.core.branch_rerun import request_rerun
from kohakuterrarium.core.branch_selection import _await_owner, _scope_id
from kohakuterrarium.core.message_locator import is_turn_user_message
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.operation_admission import reject_own_turn
from kohakuterrarium.core.regeneration_reservation import reserved_regeneration
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.raw_history import (
    UserMessageSelector,
    append_user_event_pair,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class RegenerationTurnError(ConflictError):
    """A requested rerun settled without completing successfully."""

    def __init__(self, status, receipt=None, *, may_have_run=None):
        self.status = status
        self.receipt = {**deepcopy(receipt or {}), "status": status}
        self.may_have_run = bool(receipt) if may_have_run is None else may_have_run
        super().__init__(f"Regeneration turn ended with status {status}")


def completed_regeneration_result(outcome, receipt):
    """Return settled success or raise an error retaining the admitted branch identity."""
    if outcome.status != "ok":
        raise RegenerationTurnError(outcome.status, receipt)
    return {**deepcopy(receipt), "status": "completed"}


@dataclass(frozen=True)
class RegenerationBinding:
    store: Any
    controller: Any
    conversation: Any
    scope: Any
    coordinates: tuple
    namespace: str
    tail_identity: Any
    tail_message: Any
    owner: Any
    provider: Any
    tail_content: Any


def _tail(agent):
    return next(
        (
            message
            for message in reversed(agent.controller.conversation.get_messages())
            if is_turn_user_message(message)
        ),
        None,
    )


def _capture(agent):
    require_model_mutation_available(agent)
    agent._ensure_rerun_available()
    if getattr(agent, "_model_switch_closing", False) or getattr(
        agent, "_starting", False
    ):
        raise ConflictError("Cannot request regeneration while starting or stopping")
    tail = _tail(agent)
    return RegenerationBinding(
        agent.session_store,
        agent.controller,
        agent.controller.conversation,
        getattr(agent, "_runtime_scope_id", None),
        (
            agent._turn_index,
            agent._branch_id,
            tuple(tuple(pair) for pair in agent._parent_branch_path),
        ),
        agent.config.name,
        deepcopy(getattr(tail, "metadata", None)),
        tail,
        getattr(agent, "_session_state_owner", None),
        agent.llm,
        deepcopy(getattr(tail, "content", None)),
    )


def _validate(agent, binding, *, check_admission=True):
    if check_admission:
        require_model_mutation_available(agent)
    agent._ensure_rerun_available()
    if (
        agent.session_store is not binding.store
        or agent.controller is not binding.controller
        or agent.controller.conversation is not binding.conversation
        or getattr(agent, "_session_state_owner", None) is not binding.owner
        or agent.llm is not binding.provider
        or getattr(binding.tail_message, "content", None) != binding.tail_content
        or getattr(agent, "_runtime_scope_id", None) != binding.scope
        or (
            agent._turn_index,
            agent._branch_id,
            tuple(tuple(pair) for pair in agent._parent_branch_path),
        )
        != binding.coordinates
        or agent.config.name != binding.namespace
        or getattr(agent, "_model_switch_closing", False)
    ):
        raise ConflictError("Regeneration source changed while interrupting")


def _target_in_store(store, namespace, coordinates):
    turn, branch, _ = coordinates
    candidates = [
        event
        for event in store.get_events(namespace)
        if event.get("type") == "user_message"
        and (event.get("turn_index"), event.get("branch_id")) == (turn, branch)
    ]
    if len(candidates) != 1:
        raise ConflictError("Interrupted turn has no unique persisted user target")
    return UserMessageSelector(candidates[0]["event_id"], turn, branch)


@reserved_regeneration
async def interrupt_and_regenerate(
    agent,
    *,
    timeout=10.0,
    turn_index=None,
    branch_view=None,
    request_id=None,
    target=None,
    binding=None,
):
    """Interrupt one captured turn and admit a rerun only after source validation."""
    reject_own_turn(agent, "interrupt and regenerate")
    branch_view = deepcopy(branch_view)
    binding = _capture(agent) if binding is None else binding
    gate = getattr(agent, "_interrupt_regeneration_lock", None)
    if gate is None:
        gate = agent._interrupt_regeneration_lock = asyncio.Lock()
    if gate.locked():
        raise ConflictError("Interrupt and regeneration is already pending")
    async with gate:
        _validate(agent, binding)
        if binding.scope is None:

            def initialize_scope():
                _validate(agent, binding, check_admission=False)
                return replace(binding, scope=_scope_id(agent))

            binding = await _await_owner(
                binding.owner.submit_internal(initialize_scope)
            )
        await agent.interrupt_and_wait(timeout=timeout)
        _validate(agent, binding)
        tail = _tail(agent)
        tail_coordinates = (
            (getattr(tail, "metadata", None) or {}).get("turn_index"),
            (getattr(tail, "metadata", None) or {}).get("branch_id"),
        )
        if (
            target is None
            and turn_index is None
            and (
                tail is not binding.tail_message
                or tail is None
                or binding.store is not None
                and tail_coordinates != binding.coordinates[:2]
            )
        ):
            if binding.store is None:
                raise ConflictError("Interrupted input has no retained user target")
            target = await binding.store.run(
                _target_in_store, binding.store, binding.namespace, binding.coordinates
            )
        elif (
            target is None
            and turn_index is None
            and getattr(tail, "metadata", None) != binding.tail_identity
        ):
            raise ConflictError("Regeneration user target changed while interrupting")
        _validate(agent, binding)
        edit = (
            EditRequest(
                turn_index=turn_index,
                branch_view=branch_view,
                target=target,
                regenerate=True,
            )
            if target is not None or turn_index is not None
            else None
        )
        outcome, receipt = await request_rerun(
            agent,
            append_user_event_pair,
            request_id=request_id,
            edit=edit,
            validate=lambda: _validate(agent, binding, check_admission=False),
            return_receipt=True,
        )
        return completed_regeneration_result(outcome, receipt)


def _request_status(task, request_id):
    if not task.done():
        return {"request_id": request_id, "status": "pending", "error": None}
    if task.cancelled():
        return {"request_id": request_id, "status": "cancelled", "error": None}
    error = task.exception()
    return {
        "request_id": request_id,
        "status": (
            getattr(error, "status", "error") if error is not None else "completed"
        ),
        "error": str(error) if error is not None else None,
    }


def get_regeneration_status(agent, request_id):
    """Return detached status for the retained latest deferred request."""
    task = getattr(agent, "_deferred_regeneration_task", None)
    if task is None or request_id != getattr(agent, "_deferred_regeneration_id", None):
        raise ConflictError("Regeneration request is unknown or no longer retained")
    return _request_status(task, request_id)


def _observe_request(agent, task, request_id):
    if task.cancelled():
        agent.output_router.notify_activity("regeneration_cancelled", request_id)
        return
    error = task.exception()
    if error is not None:
        logger.warning(
            "Deferred regeneration failed", error=str(error), request_id=request_id
        )
        status = _request_status(task, request_id)
        kind = (
            "regeneration_cancelled"
            if status["status"] == "interrupted"
            else "regeneration_failed"
        )
        agent.output_router.notify_activity(kind, str(error), metadata=status)
    else:
        agent.output_router.notify_activity(
            "regeneration_done", request_id, metadata=_request_status(task, request_id)
        )


def request_regeneration(
    agent,
    *,
    timeout=10.0,
    turn_index=None,
    branch_view=None,
    request_id=None,
    target=None,
):
    """Return a request token without waiting inside the requesting callback."""
    pending = getattr(agent, "_deferred_regeneration_task", None)
    previous_id = getattr(agent, "_deferred_regeneration_id", None)
    options = (timeout, turn_index, deepcopy(branch_view), target)
    if pending is not None and (
        not pending.done() or request_id is not None and request_id == previous_id
    ):
        if (
            options != getattr(agent, "_deferred_regeneration_options", None)
            or request_id is not None
            and request_id != previous_id
        ):
            raise ConflictError(
                "A different deferred regeneration request already exists"
            )
        return previous_id
    binding = _capture(agent)
    token = request_id or str(uuid4())
    task = asyncio.create_task(
        interrupt_and_regenerate(
            agent,
            timeout=timeout,
            turn_index=turn_index,
            branch_view=deepcopy(branch_view),
            request_id=token,
            target=target,
            binding=binding,
        )
    )
    agent._deferred_regeneration_task = task
    agent._deferred_regeneration_id = token
    agent._deferred_regeneration_options = options
    task.add_done_callback(lambda done: _observe_request(agent, done, token))
    agent.output_router.notify_activity("regeneration_requested", token)
    return token
