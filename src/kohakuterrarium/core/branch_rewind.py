"""Validated owner-owned rewind with best-effort affinity snapshot persistence."""

from dataclasses import replace
from typing import Any

from kohakuterrarium.core.branch_rerun import (
    _capture,
    _check_capture,
    run_history_mutation,
)
from kohakuterrarium.core.branch_selection import _scope_id
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.history_compatibility import validate_history_compatibility
from kohakuterrarium.core.runtime_context import (
    RuntimeContextAllocator,
    RuntimeContextPreparation,
    RuntimeContextPreparationMode,
)
from kohakuterrarium.session.history import select_live_event_ids
from kohakuterrarium.session.model_switch_history import validate_switch_history
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


async def rewind(agent: Any, index: int) -> None:
    """Validate a retained prefix, commit it, and attempt to save its snapshot."""
    await run_history_mutation(agent, lambda: _rewind(agent, index), rerun=False)


def _retained_evidence(messages, count, events):
    """Bound evidence by a complete ordered user/announcement correspondence."""
    anchors = []
    for index, message in enumerate(messages):
        if message.role == "user":
            anchors.append((index, "user_message", message.content, message.metadata))
        elif message.role == "assistant" and message.tool_calls:
            anchors.append(
                (index, "assistant_tool_calls", message.tool_calls, message.metadata)
            )
    raw = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("type") in ("user_message", "assistant_tool_calls")
    ]
    if len(raw) != len(anchors):
        return events
    cutoff = len(events)
    for (message_index, kind, value, metadata), (event_index, event) in zip(
        anchors, raw
    ):
        field = "content" if kind == "user_message" else "tool_calls"
        if event.get("type") != kind or event.get(field) != value:
            return events
        if any(
            metadata.get(key) is not None and metadata[key] != event.get(key)
            for key in ("event_id", "turn_index", "branch_id")
        ):
            return events
        if message_index >= count:
            cutoff = min(cutoff, event_index)
    return events[:cutoff]


def _prepare(store, prefix, messages, config, coordinates, source, index):
    preview = Conversation(replace(config, max_messages=0))
    for message in messages:
        preview.append_message(message)
    preview.truncate_from(index)
    retained = preview.get_messages()
    validate_history_compatibility(retained, source, current_source=source)
    if store is not None:
        events = list(store.get_events(prefix))
        turn, branch, path = coordinates
        view = dict(path)
        if turn > 0:
            view[turn] = branch
        ids = select_live_event_ids(events, branch_view=view)
        selected = [e for e in events if e.get("event_id") in ids]
        evidence = _retained_evidence(messages, len(retained), selected)
        validate_switch_history(retained, evidence, events, source)
    return len(retained), preview.snapshot_messages()


def _commit(agent, capture, count):
    _check_capture(agent, capture)
    scope = _scope_id(agent)
    allocator = getattr(agent, "_runtime_context_allocator", None)
    if allocator is None:
        allocator = RuntimeContextAllocator(agent.config.name, scope)
        agent._runtime_context_allocator = allocator
    turn, branch, path = capture["coordinates"]
    bundle = allocator.install(
        RuntimeContextPreparation(
            agent_id=agent.config.name,
            runtime_scope_id=scope,
            provider_conversation=capture["messages"][:count],
            turn_index=turn,
            branch_id=branch,
            parent_branch_path=path,
            preparation_mode=RuntimeContextPreparationMode.EXACT_INSTALLED_LIVE_REUSE,
        )
    )
    capture["conversation"].truncate_from(count)
    agent._runtime_context_bundle = bundle
    agent._installed_runtime_context_bundle = bundle


async def _rewind(agent, index):
    owner = agent._session_state_owner
    capture = await owner.submit_internal(lambda: _capture(agent, editing=True))
    store = capture["store"]
    args = (
        store,
        capture["prefix"],
        capture["all_detached"],
        capture["config"],
        capture["coordinates"],
        capture["source"],
        index,
    )
    count, snapshot = (
        await store.run(_prepare, *args) if store is not None else _prepare(*args)
    )
    await owner.submit_internal(lambda: _commit(agent, capture, count))
    if store is not None:
        try:
            await store.run(store.save_conversation, capture["prefix"], snapshot)
        except Exception:
            logger.warning("Failed to save conversation after rewind", exc_info=True)
