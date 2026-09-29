"""Persist and install selected branch context without regenerating a response."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from kohakuterrarium.core.branch_background import background_work_allows_selection
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.history_compatibility import validate_history_compatibility
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.runtime_context import (
    RuntimeContextAllocator,
    RuntimeContextPreparation,
)
from kohakuterrarium.core.runtime_context_replacement_prepare import (
    prepare_selected_branch_context,
)
from kohakuterrarium.errors import ConflictError, HistoryCompatibilityError
from kohakuterrarium.llm.generation import provider_source
from kohakuterrarium.llm.message import Message
from kohakuterrarium.session.history import select_live_event_ids
from kohakuterrarium.session.snapshot_coverage import (
    encode_snapshot,
    save_bound_snapshot,
)
from kohakuterrarium.utils.logging import get_logger
from kohakuterrarium.session.model_switch_history import validate_switch_history

logger = get_logger(__name__)

_IDLE = "Cannot select a branch while the agent is not locally idle"
_ACTIVE = "A branch selection is already in progress"
_MISSING = "Cannot select a branch without a session store"


@dataclass(frozen=True)
class _SelectionRequest:
    """Immutable facts captured before the first await. No store I/O."""

    namespace: str
    view: dict[int, int] | None
    store: Any
    prefix: str
    baseline: tuple[Message, ...]
    config: Any
    target_source: dict[str, Any]
    active_source: dict[str, Any]
    conversation_id: int
    turn_index: int
    branch_id: int
    parent_path: tuple[tuple[int, int], ...]
    agent_id: str
    runtime_scope_id: str


def require_history_selection_available(agent: Any) -> None:
    """Refuse history mutation while a selection is pending or durability is unknown."""
    if getattr(agent, "_history_selection_uncertain", False):
        raise ConflictError(
            "Branch selection durability is uncertain; resume before continuing"
        )
    if getattr(agent, "_history_selection_pending", False):
        raise ConflictError("Branch selection is awaiting persistence")


def mark_history_selection_stopping(agent: Any) -> None:
    """Refuse new selections. Parent stop calls this before any await."""
    agent._history_selection_closing = True


async def drain_history_selections(agent: Any) -> None:
    """Join an accepted selection. Parent stop calls this before owner drain."""
    task = getattr(agent, "_history_selection_task", None)
    if task is None or task.done():
        return
    try:
        await _join_accepted(task)
    except Exception:
        logger.warning("Accepted branch selection failed during stop", exc_info=True)


def _reject_not_idle(agent: Any) -> None:
    require_model_mutation_available(agent)
    if getattr(agent, "session_store", None) is None:
        raise ConflictError(_MISSING)
    if getattr(agent, "_history_selection_uncertain", False):
        raise ConflictError(
            "Branch selection durability is uncertain; resume before continuing"
        )
    if getattr(agent, "_history_selection_closing", False) or getattr(
        agent, "_stopped", False
    ):
        raise ConflictError(_IDLE)
    owner = getattr(agent, "_session_state_owner", None)
    if (
        getattr(agent, "_starting", False)
        or not getattr(agent, "_running", False)
        or owner is None
        or not getattr(owner, "is_running", False)
    ):
        raise ConflictError(_IDLE)
    stop_lock = getattr(agent, "_stop_lock", None)
    if stop_lock is not None and stop_lock.locked():
        raise ConflictError(_IDLE)
    if getattr(agent, "_model_switch_pending", False) or getattr(
        agent, "_model_switch_uncertain", False
    ):
        raise ConflictError(_IDLE)
    compact = getattr(agent, "compact_manager", None)
    if compact is not None and getattr(compact, "is_compacting", False):
        raise ConflictError(_IDLE)
    lock = getattr(agent, "_processing_lock", None)
    if lock is not None and lock.locked():
        raise ConflictError(_IDLE)
    if not background_work_allows_selection(agent):
        raise ConflictError(_IDLE)


def _scope_id(agent: Any) -> str:
    scope = getattr(agent, "_runtime_scope_id", None)
    if not isinstance(scope, str) or not scope.strip():
        scope = str(uuid4())
        agent._runtime_scope_id = scope
    return scope


def _capture(agent: Any, view: dict[int, int] | None) -> _SelectionRequest:
    """Snapshot live identity. Must not touch the store."""
    if not background_work_allows_selection(agent):
        raise ConflictError(_IDLE)
    conversation = agent.controller.conversation
    messages = list(conversation.get_messages())
    baseline = tuple(
        copy.deepcopy(message) for message in messages if message.role == "system"
    )
    output = getattr(agent, "_session_output", None)
    prefix = getattr(output, "_event_key_prefix", None) or agent.config.name
    tool_format = getattr(
        getattr(agent.controller, "config", None), "tool_format", None
    )
    source = provider_source(getattr(agent, "llm", None), tool_format)
    path = tuple(
        (int(turn), int(branch))
        for turn, branch in (getattr(agent, "_parent_branch_path", None) or ())
    )
    requested = copy.deepcopy(view)
    return _SelectionRequest(
        namespace=str(agent.config.name),
        view=requested,
        store=agent.session_store,
        prefix=str(prefix),
        baseline=baseline,
        config=copy.deepcopy(conversation.config),
        target_source=dict(source),
        active_source=dict(source),
        conversation_id=id(conversation),
        turn_index=int(getattr(agent, "_turn_index", 0) or 0),
        branch_id=int(getattr(agent, "_branch_id", 0) or 0),
        parent_path=path,
        agent_id=str(agent.config.name),
        runtime_scope_id=_scope_id(agent),
    )


async def select_branch(agent: Any, branch_view: dict[int, int] | None = None) -> int:
    """Select a persisted branch while locally idle. Returns the live message count."""
    return await _select_branch(agent, branch_view)


async def _select_branch(agent, branch_view, continuation=None):
    """Publish a selection and its optional continuation in one owner commit."""
    _reject_not_idle(agent)
    require_history_selection_available(agent)
    gate = getattr(agent, "_history_selection_request_lock", None)
    if gate is None:
        gate = agent._history_selection_request_lock = asyncio.Lock()
    async with gate:
        _reject_not_idle(agent)
        existing = getattr(agent, "_history_selection_task", None)
        if existing is not None and not existing.done():
            raise ConflictError(_ACTIVE)
        view = copy.deepcopy(branch_view)
        lock = getattr(agent, "_processing_lock", None)
        if lock is not None:
            await lock.acquire()
        agent._history_selection_pending = True
        ready = asyncio.Event() if continuation is not None else None
        if ready is not None:
            agent._history_rerun_ready = ready
        try:
            task = asyncio.create_task(
                _run_selection(agent, view, continuation),
                name=f"history_selection_{agent.config.name}",
            )
        except BaseException:
            agent._history_selection_pending = False
            if ready is not None:
                ready.set()
            if lock is not None and lock.locked():
                lock.release()
            raise
        agent._history_selection_task = task
    try:
        return await _join_accepted(task)
    finally:
        if getattr(agent, "_history_selection_task", None) is task and task.done():
            agent._history_selection_task = None
        agent._history_selection_pending = False
        if ready is not None:
            ready.set()
        if lock is not None and lock.locked():
            lock.release()


async def _join_accepted(task: asyncio.Task) -> Any:
    """Finish accepted work, then re-raise waiter cancellation with its cause."""
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            break
    try:
        result = task.result()
    except BaseException as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


async def _run_selection(
    agent: Any, view: dict[int, int] | None, continuation=None
) -> int:
    lock = getattr(agent, "_processing_lock", None)
    previous = getattr(agent, "_turn_lock_holder", None)
    if lock is not None:
        agent._turn_lock_holder = asyncio.current_task()
    try:
        request = await _await_owner(
            agent._session_state_owner.submit_internal(lambda: _capture(agent, view))
        )
        return await _select_locked(agent, request, continuation)
    finally:
        if (
            lock is not None
            and getattr(agent, "_turn_lock_holder", None) is asyncio.current_task()
        ):
            agent._turn_lock_holder = previous


async def _select_locked(
    agent: Any, request: _SelectionRequest, continuation=None
) -> int:
    prepared = await request.store.run(_prepare_on_worker, request)
    await _publish_on_owner(agent, request, prepared, continuation)
    return int(prepared["count"])


def _prepare_on_worker(request: _SelectionRequest) -> dict[str, Any]:
    """Prepare detached selected history and its source-bound snapshot prefix."""
    events = [
        copy.deepcopy(event)
        for event in request.store.get_events(request.prefix)
        if isinstance(event, dict)
    ]
    preparation = prepare_selected_branch_context(
        agent_id=request.agent_id,
        runtime_scope_id=request.runtime_scope_id,
        raw_events=events,
        branch_view=request.view,
        baseline_system_messages=request.baseline,
        require_complete_tool_pairs=True,
    )
    messages = tuple(preparation.provider_conversation)
    switches = [event for event in events if event.get("type") == "model_switch"]
    selected_ids = select_live_event_ids(
        events, branch_view=dict(preparation.branch_view)
    )
    selected_events = [
        event for event in events if event.get("event_id") in selected_ids
    ]
    prepared = {
        "preparation": preparation,
        "messages": messages,
        "switches": switches,
        "selected_events": selected_events,
        "full_events": events,
        "watermark": _watermark(request, events),
        "snapshot_envelope": (
            encode_snapshot(request.store, request.prefix, [])
            if hasattr(request.store, "_event_seq")
            else None
        ),
        "turn_index": preparation.turn_index,
        "branch_id": preparation.branch_id,
        "parent_path": tuple(preparation.parent_branch_path),
        "branch_view": tuple(preparation.branch_view or ()),
        "count": len(messages),
        "persisted": False,
        "published": False,
        "baseline_used": not any(
            e.get("type") == "system_prompt_set" for e in selected_events
        ),
    }
    _validate_prepared(prepared, request)
    prepared["candidate"] = _candidate_conversation(request, prepared)
    return prepared


def _recorded(message: Message) -> bool:
    metadata = getattr(message, "metadata", None)
    return isinstance(metadata, dict) and metadata.get("event_id") is not None


def _merge_system(
    replayed: tuple[Message, ...], baseline: tuple[Message, ...]
) -> tuple[Message, ...]:
    """Recorded system events win. Otherwise keep the detached live system objects."""
    recorded = [
        message
        for message in replayed
        if message.role == "system" and _recorded(message)
    ]
    body = [message for message in replayed if message.role != "system"]
    if recorded:
        return tuple(recorded + body)
    return tuple(list(baseline) + body)


def _watermark(request: _SelectionRequest, events: list[dict[str, Any]]) -> int:
    try:
        watermark = int(request.store.max_event_id(request.prefix))
    except (AttributeError, TypeError, ValueError):
        watermark = 0
    for event in events:
        event_id = event.get("event_id")
        if isinstance(event_id, int) and event_id > watermark:
            watermark = event_id
    return watermark


def _validate_prepared(prepared: dict[str, Any], request: _SelectionRequest) -> None:
    """Reject native or generation incompatibility before any save."""
    messages = prepared["messages"]
    try:
        validate_history_compatibility(
            messages,
            request.target_source,
            current_source=request.active_source,
        )
        validate_switch_history(
            messages,
            prepared["selected_events"],
            prepared["full_events"],
            request.target_source,
        )
    except HistoryCompatibilityError:
        raise
    except Exception as exc:
        raise HistoryCompatibilityError(str(exc)) from exc


async def _publish_on_owner(
    agent: Any, request: _SelectionRequest, prepared: dict[str, Any], continuation=None
) -> None:
    owner = agent._session_state_owner
    bundle = await _await_owner(
        owner.submit_internal(lambda: _allocate(agent, request, prepared))
    )
    try:
        await request.store.run(_persist_on_worker, request, prepared)
        prepared["persisted"] = True

        def publish():
            _bind(agent, request, prepared, bundle)
            if continuation is not None:
                agent._event_inbox.put_front([continuation])

        await _await_owner(owner.submit_internal(publish))
        prepared["published"] = True
    except BaseException:
        # A failed multi-key save can already be partial on disk. Fail closed.
        agent._history_selection_uncertain = True
        if getattr(agent, "_runtime_context_bundle", None) is bundle:
            agent._runtime_context_bundle = getattr(
                agent, "_installed_runtime_context_bundle", None
            )
        raise


def _allocate(agent: Any, request: _SelectionRequest, prepared: dict[str, Any]) -> Any:
    allocator = getattr(agent, "_runtime_context_allocator", None)
    if not isinstance(allocator, RuntimeContextAllocator):
        allocator = RuntimeContextAllocator(request.agent_id, request.runtime_scope_id)
        agent._runtime_context_allocator = allocator
    preparation = prepared["preparation"]
    if not isinstance(preparation, RuntimeContextPreparation):
        raise ConflictError("Branch selection prepared an invalid context")
    candidate = prepared["candidate"]
    return allocator.install(
        replace(preparation, provider_conversation=tuple(candidate.get_messages()))
    )


def _persist_on_worker(request: _SelectionRequest, prepared: dict[str, Any]) -> None:
    messages = [_message_dict(message) for message in prepared["messages"]]
    save_bound_snapshot(
        request.store, request.prefix, messages, prepared["snapshot_envelope"]
    )
    state = request.store.state
    state[f"{request.prefix}:snapshot_event_id"] = prepared["watermark"]
    view = prepared["branch_view"]
    state[f"{request.prefix}:snapshot_branch"] = {
        "turn_index": prepared["turn_index"],
        "branch_id": prepared["branch_id"],
        "parent_branch_path": [list(item) for item in prepared["parent_path"]],
        "branch_view": dict(view),
    }
    view_key = f"{request.prefix}:branch_view"
    if view_key in state:
        state[view_key] = {turn: branch for turn, branch in view}
    request.store.flush()


def _message_dict(message: Message) -> dict[str, Any]:
    payload = message.to_dict()
    metadata = getattr(message, "metadata", None)
    if metadata:
        payload["metadata"] = copy.deepcopy(metadata)
    return payload


def _bind(
    agent: Any, request: _SelectionRequest, prepared: dict[str, Any], bundle: Any
) -> None:
    conversation = getattr(getattr(agent, "controller", None), "conversation", None)
    store = getattr(agent, "session_store", None)
    if (
        conversation is None
        or id(conversation) != request.conversation_id
        or store is not request.store
    ):
        raise ConflictError("Branch selection identity changed before publish")
    if (
        int(getattr(agent, "_turn_index", 0) or 0) != request.turn_index
        or int(getattr(agent, "_branch_id", 0) or 0) != request.branch_id
    ):
        raise ConflictError("Branch selection coordinates changed before publish")
    candidate = prepared["candidate"]
    candidate.config = conversation.config
    if prepared["baseline_used"]:
        live_systems = iter(
            m for m in conversation.get_messages() if m.role == "system"
        )
        candidate._messages[:] = [
            next(live_systems) if m.role == "system" else m
            for m in candidate.get_messages()
        ]
        bundle = replace(bundle, provider_conversation=tuple(candidate.get_messages()))
    agent.controller.conversation = candidate
    agent._turn_index = prepared["turn_index"]
    agent._branch_id = prepared["branch_id"]
    agent._parent_branch_path = list(prepared["parent_path"])
    agent._branch_view = {turn: branch for turn, branch in prepared["branch_view"]}
    agent._runtime_context_bundle = bundle
    agent._installed_runtime_context_bundle = bundle
    agent._history_selection_uncertain = False


def _candidate_conversation(
    request: _SelectionRequest, prepared: dict[str, Any]
) -> Any:
    """Append detached messages with trimming disabled, then restore the config object."""
    candidate = Conversation(replace(request.config, max_messages=0))
    for message in prepared["messages"]:
        candidate.append_message(copy.deepcopy(message))
    candidate.config = request.config
    return candidate


async def _await_owner(future: asyncio.Future) -> Any:
    """Join an accepted owner callback across waiter cancellation."""
    cancellation = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            break
    try:
        result = future.result()
    except BaseException as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


__all__ = [
    "drain_history_selections",
    "mark_history_selection_stopping",
    "require_history_selection_available",
    "select_branch",
]
