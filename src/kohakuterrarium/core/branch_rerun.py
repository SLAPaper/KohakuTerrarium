"""Durable edit/regeneration admission and owner-owned rerun enqueue."""

import asyncio
from copy import deepcopy
from typing import Any, Callable

from kohakuterrarium.core.branch_edit_prepare import EditRequest, prepare_edit
from kohakuterrarium.core.branch_selection import _join_accepted, _scope_id
from kohakuterrarium.core.event_inbox import EventEnvelope, TurnOutcome
from kohakuterrarium.core.events import EventType, TriggerEvent
from kohakuterrarium.core.history_compatibility import validate_history_compatibility
from kohakuterrarium.core.message_locator import is_turn_user_message
from kohakuterrarium.core.operation_admission import reject_own_turn
from kohakuterrarium.core.runtime_context import (
    RuntimeContextAllocator,
    RuntimeContextPreparation,
    RuntimeContextPreparationMode,
)
from kohakuterrarium.core.regeneration_reservation import bind_regeneration_completion
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.llm.generation import provider_source
from kohakuterrarium.session.compact_coverage import coverage_metadata, exact_origins
from kohakuterrarium.session.event_origin import EVENT_ORIGIN
from kohakuterrarium.session.history import select_live_event_ids
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def _check_idle(agent: Any, *, rerun: bool = True) -> None:
    agent._ensure_history_mutation_idle()
    if rerun:
        agent._ensure_rerun_available()
    owner = getattr(agent, "_session_state_owner", None)
    if owner is None or not owner.is_running:
        raise ConflictError("Regeneration requires a running session-state owner")
    if getattr(agent, "_model_switch_closing", False) or getattr(
        agent, "_starting", False
    ):
        raise ConflictError("Cannot regenerate while starting or stopping")
    if getattr(agent, "_active_event_run", None):
        raise ConflictError("A turn has already claimed input")
    compact = getattr(agent, "compact_manager", None)
    if compact is not None and compact.is_compacting:
        raise ConflictError("Cannot regenerate during compaction")
    for source in (
        getattr(agent, "executor", None),
        getattr(agent, "subagent_manager", None),
    ):
        running = getattr(source, "get_running_jobs", None)
        pending = getattr(source, "get_pending_count", None)
        if (callable(running) and running()) or (callable(pending) and pending()):
            raise ConflictError("Cannot regenerate while background work is pending")


async def request_rerun(
    agent: Any,
    append_pair: Callable,
    *,
    request_id: str | None,
    edit: EditRequest | None = None,
    validate: Callable | None = None,
    return_outcome: bool = False,
    return_receipt: bool = False,
) -> bool | TurnOutcome | tuple[TurnOutcome, dict]:
    """Admit a history mutation, then await its turn outside mutation ownership."""
    edit = deepcopy(edit)
    admitted = await run_history_mutation(
        agent, lambda: _admit(agent, append_pair, request_id, edit, validate)
    )
    if admitted is not None:
        completion, receipt = admitted
        outcome = (
            await _join_accepted(completion)
            if return_receipt
            else await asyncio.shield(completion)
        )
        if return_outcome or return_receipt:
            outcome = (
                outcome
                if isinstance(outcome, TurnOutcome)
                else TurnOutcome(status="error")
            )
            return (outcome, receipt) if return_receipt else outcome
        return True
    outcome = TurnOutcome(status="rejected", was_primary=False)
    if return_receipt:
        return outcome, {}
    return outcome if return_outcome else False


async def run_history_mutation(agent: Any, operation: Callable, *, rerun: bool = True):
    """Reserve local history until an accepted mutation and persistence finish."""
    reject_own_turn(agent, "mutate history")
    previous = getattr(agent, "_branch_rerun_completion", None)
    if previous is not None and not previous.done():
        await asyncio.shield(previous)
    async with agent._get_message_mutation_lock():
        await agent._wait_for_history_finalization()
        _check_idle(agent, rerun=rerun)
        lock = agent._processing_lock
        await lock.acquire()
        ready = asyncio.Event()
        agent._history_rerun_ready = ready
        agent._history_rerun_pending = True
        try:
            task = asyncio.create_task(operation())
        except BaseException:
            agent._history_rerun_pending = False
            ready.set()
            lock.release()
            raise
        agent._history_rerun_task = task

        def release(_task):
            agent._history_rerun_pending = False
            ready.set()
            lock.release()

        task.add_done_callback(release)
        return await _join_accepted(task)


def _capture(agent: Any, *, editing: bool = False, validate=None) -> dict | None:
    if validate is not None:
        validate()
    conversation = agent.controller.conversation
    messages = tuple(conversation.get_messages())
    last_user = next(
        (
            i
            for i in reversed(range(len(messages)))
            if is_turn_user_message(messages[i])
        ),
        -1,
    )
    if last_user < 0 and not editing:
        return None
    prefix = (
        getattr(getattr(agent, "_session_output", None), "_event_key_prefix", None)
        or agent.config.name
    )
    source = provider_source(
        getattr(agent, "llm", None),
        getattr(getattr(agent.controller, "config", None), "tool_format", None),
    )
    return {
        "conversation": conversation,
        "messages": messages,
        "retained": deepcopy(messages[: last_user + 1]),
        "all_detached": deepcopy(messages),
        "config": deepcopy(conversation.config),
        "last_user": last_user,
        "coordinates": (
            agent._turn_index,
            agent._branch_id,
            tuple(agent._parent_branch_path),
        ),
        "store": agent.session_store,
        "prefix": prefix,
        "source": source,
        "validate": validate,
    }


def _prepare(store, prefix, retained, coordinates, source) -> tuple[int, Any]:
    validate_history_compatibility(retained, source, current_source=source)
    turn, branch, _ = coordinates
    if store is not None:
        branch = max(
            [branch]
            + [
                e.get("branch_id", 0)
                for e in store.get_events(prefix)
                if e.get("turn_index") == turn and isinstance(e.get("branch_id"), int)
            ]
        )
    return branch + 1, deepcopy(retained[-1].content)


def _check_capture(agent, capture) -> None:
    validate = capture.get("validate")
    if validate is not None:
        validate()
    conv = agent.controller.conversation
    if (
        conv is not capture["conversation"]
        or agent.session_store is not capture["store"]
        or tuple(map(id, conv.get_messages())) != tuple(map(id, capture["messages"]))
        or (agent._turn_index, agent._branch_id, tuple(agent._parent_branch_path))
        != capture["coordinates"]
    ):
        raise ConflictError("Regeneration context changed during admission")


def _allocate(agent, capture, branch, edit=None):
    _check_capture(agent, capture)
    scope = _scope_id(agent)
    allocator = getattr(agent, "_runtime_context_allocator", None)
    if allocator is None:
        allocator = RuntimeContextAllocator(agent.config.name, scope)
        agent._runtime_context_allocator = allocator
    turn, _, path = capture["coordinates"]
    messages = capture["messages"][: capture["last_user"] + 1]
    mode = RuntimeContextPreparationMode.EXACT_INSTALLED_LIVE_REUSE
    if edit is not None:
        turn, path = edit["turn"], edit["path"]
        messages = (
            edit["messages"]
            if edit["rebuilt"]
            else capture["messages"][: edit["index"]]
        )
        if edit["rebuilt"]:
            if edit["preserve_baseline"]:
                systems = iter(m for m in capture["messages"] if m.role == "system")
                candidate = edit["candidate"]
                candidate._messages[:] = [
                    next(systems) if m.role == "system" else m
                    for m in candidate.get_messages()
                ]
                messages = tuple(candidate.get_messages())
            mode = RuntimeContextPreparationMode.NORMALIZED_CANONICAL_REPLAY
    preparation = RuntimeContextPreparation(
        agent_id=agent.config.name,
        runtime_scope_id=scope,
        provider_conversation=messages,
        turn_index=turn,
        branch_id=branch,
        parent_branch_path=path,
        preparation_mode=mode,
    )
    return allocator.install(preparation)


def _commit(agent, capture, branch, request_id, bundle, user_event_id):
    _check_capture(agent, capture)
    completion = asyncio.get_running_loop().create_future()
    event = TriggerEvent(
        type=EventType.USER_INPUT,
        content="",
        context={"rerun": True, "edited": False, "request_id": request_id},
        stackable=False,
    )
    user = capture["messages"][capture["last_user"]]
    metadata = dict(user.metadata)
    metadata.update(turn_index=agent._turn_index, branch_id=branch)
    for key in ("event_id", "event_origin", "user_pair_id", "user_pair_origin"):
        metadata.pop(key, None)
    if user_event_id is not None:
        metadata.update(user_event_id)
    capture["conversation"].truncate_from(capture["last_user"] + 1)
    user.metadata = metadata
    agent._branch_id = branch
    agent._branch_view = dict((*capture["coordinates"][2], (agent._turn_index, branch)))
    bind_regeneration_completion(agent, completion)
    agent._event_inbox.put_front([EventEnvelope(event, future=completion)])
    agent._branch_rerun_completion = completion
    agent._runtime_context_bundle = bundle
    agent._installed_runtime_context_bundle = bundle
    return (completion,)


def _commit_edit(agent, capture, edit, request_id, bundle, user_event_id):
    _check_capture(agent, capture)
    event = TriggerEvent(
        type=EventType.USER_INPUT,
        content=edit["normalized"],
        context={
            "rerun": True,
            "edited": True,
            "request_id": request_id,
            "_history_user_identity": {
                "turn_index": edit["turn"],
                "branch_id": edit["branch"],
                **(user_event_id or {}),
            },
        },
        stackable=False,
    )
    if edit["rebuilt"]:
        candidate = edit["candidate"]
        candidate.config = capture["conversation"].config
        agent.controller.conversation = candidate
    else:
        capture["conversation"].truncate_from(edit["index"])
    agent._turn_index = edit["turn"]
    agent._branch_id = edit["branch"]
    agent._parent_branch_path = list(edit["path"])
    agent._branch_view = dict((*edit["path"], (edit["turn"], edit["branch"])))
    completion = asyncio.get_running_loop().create_future()
    bind_regeneration_completion(agent, completion)
    agent._event_inbox.put_front([EventEnvelope(event, future=completion)])
    agent._branch_rerun_completion = completion
    agent._runtime_context_bundle = bundle
    agent._installed_runtime_context_bundle = bundle
    return (completion,)


def _legacy_baseline(store, prefix, view, message, coverage):
    """Recover a same-store legacy interval from its selected durable summary."""
    events = store.get_events(prefix)
    live_ids = select_live_event_ids(events, branch_view=view)
    intervals = set()
    bounds = (
        coverage.get("replaced_from_event_id"),
        coverage.get("replaced_to_event_id"),
    )
    has_bounds = any(
        key in coverage for key in ("replaced_from_event_id", "replaced_to_event_id")
    )
    if has_bounds and not (
        all(type(value) is int for value in bounds) and 0 < bounds[0] <= bounds[1]
    ):
        raise ValueError("Invalid retained legacy interval")
    for event in events:
        if (
            event.get("type") not in ("compact_complete", "compact_replace")
            or event.get("event_id") not in live_ids
            or EVENT_ORIGIN in event
            or coverage_metadata(event) != coverage_metadata(coverage)
        ):
            continue
        summary = event.get("summary", "") or event.get("summary_text", "")
        rendered = f"[Previous context summary (compact round {event.get('round')})]\n\n{summary}"
        if message.content not in (summary, rendered):
            continue
        first, last = event.get("replaced_from_event_id"), event.get(
            "replaced_to_event_id"
        )
        if type(first) is int and type(last) is int and 0 < first <= last:
            if not has_bounds:
                intervals.add((first, last))
            elif first <= bounds[0] <= bounds[1] <= last:
                intervals.add(bounds)
    if len(intervals) != 1:
        raise ValueError("Retained legacy summary has no unique local baseline")
    first, last = intervals.pop()
    return {"replaced_from_event_id": first, "replaced_to_event_id": last}


def _persist_rerun(
    store, append_pair, prefix, content, turn, branch, path, retained, source_view
):
    """Persist the new user and the compact baselines retained by its branch."""
    baselines = []
    for message in retained:
        coverage = (message.metadata or {}).get("compact_coverage")
        if coverage is None:
            continue
        if message.role != "assistant" or not isinstance(message.content, str):
            raise ValueError("Invalid retained compact summary")
        origins = exact_origins(coverage)
        interval = (
            _legacy_baseline(store, prefix, source_view, message, coverage)
            if origins is None
            else {}
        )
        baselines.append(
            {
                **coverage_metadata(coverage),
                **interval,
                "summary": message.content,
                "compact_path": [*path, (turn, branch)],
            }
        )
    user_event_id = append_pair(
        store, prefix, {"content": content}, turn, branch, list(path), with_receipt=True
    )
    for payload in baselines:
        store.append_event(
            prefix,
            "compact_replace",
            payload,
            turn_index=turn,
            branch_id=branch,
            parent_branch_path=list(path),
        )
    return user_event_id


async def _admit(agent, append_pair, request_id, edit=None, validate=None):
    owner = agent._session_state_owner
    capture = await owner.submit_internal(
        lambda: _capture(agent, editing=edit is not None, validate=validate)
    )
    if capture is None:
        return None
    args = (
        capture["store"],
        capture["prefix"],
        capture["retained"],
        capture["coordinates"],
        capture["source"],
    )
    store = capture["store"]
    prepared_edit = None
    if edit is not None:
        edit_args = (
            store,
            capture["prefix"],
            capture["all_detached"],
            capture["coordinates"],
            capture["config"],
            capture["source"],
            edit,
        )
        prepared_edit = (
            await store.run(prepare_edit, *edit_args)
            if store is not None
            else prepare_edit(*edit_args)
        )
        if prepared_edit is None:
            return None
        branch, content = prepared_edit["branch"], prepared_edit["content"]
    else:
        branch, content = (
            await store.run(_prepare, *args) if store is not None else _prepare(*args)
        )
    bundle = await owner.submit_internal(
        lambda: _allocate(agent, capture, branch, prepared_edit)
    )
    user_event_id = None
    try:
        if store is not None:
            turn, old_branch, path = capture["coordinates"]
            source_view = dict((*path, (turn, old_branch)))
            if prepared_edit is not None:
                turn, path = prepared_edit["turn"], prepared_edit["path"]
                source_view = prepared_edit["source_view"]
            retained = (
                prepared_edit["messages"]
                if prepared_edit is not None
                else capture["retained"][:-1]
            )
            user_event_id = await store.run(
                _persist_rerun,
                store,
                append_pair,
                capture["prefix"],
                content,
                turn,
                branch,
                path,
                retained,
                source_view,
            )
        if prepared_edit is not None:
            receipt = await owner.submit_internal(
                lambda: _commit_edit(
                    agent, capture, prepared_edit, request_id, bundle, user_event_id
                )
            )
        else:
            receipt = await owner.submit_internal(
                lambda: _commit(
                    agent, capture, branch, request_id, bundle, user_event_id
                )
            )
        turn, _, path = capture["coordinates"]
        if prepared_edit is not None:
            turn, path = prepared_edit["turn"], prepared_edit["path"]
        return receipt[0], {
            "request_id": request_id,
            "turn_index": turn,
            "branch_id": branch,
            "parent_branch_path": [list(pair) for pair in path],
            "branch_view": {str(t): b for t, b in (*path, (turn, branch))},
        }
    except BaseException:
        agent._history_rerun_uncertain = True
        raise


async def drain_history_reruns(agent: Any) -> None:
    """Join accepted admission without waiting for its subsequent LLM turn."""
    task = getattr(agent, "_history_rerun_task", None)
    if task is None or task.done():
        return
    try:
        await _join_accepted(task)
    except Exception:
        logger.warning("Accepted regeneration failed during stop", exc_info=True)
