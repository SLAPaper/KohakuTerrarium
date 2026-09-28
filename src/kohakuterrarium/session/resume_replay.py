"""Cold full-history replay validates child evidence before resume normalization."""

from types import SimpleNamespace

from kohakuterrarium.core.background_origin import completion_identity
from kohakuterrarium.core.history_compatibility import _native_groups
from kohakuterrarium.core.selected_subagent_history import (
    project_selected_subagents,
    restore_text_subagent_feedback,
)
from kohakuterrarium.session.history import (
    normalize_resumable_events,
    replay_conversation,
    select_live_event_ids,
)
from kohakuterrarium.session.history_feedback import (
    ToolIdentityResolver,
    canonicalize_provider_tool_order,
    subagent_feedback_message,
)
from kohakuterrarium.session.history_support import (
    compact_replay_events,
    index_parent_paths,
    resolve_selected_branches,
)


def _proven_background_call(event):
    origin = event.get("origin")
    if (
        not isinstance(origin, dict)
        or event.get("execution_class") != "background"
        or event.get("background") is not True
    ):
        return False
    identity = completion_identity(
        SimpleNamespace(job_id=event.get("job_id"), context=event),
        origin.get("runtime_scope_id"),
    )
    return (
        identity is not None
        and identity[1] is not None
        and all(type(event.get(key)) is int for key in ("turn_index", "branch_id"))
        and isinstance(event.get("parent_branch_path"), (list, tuple))
        and all(
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(type(value) is int for value in pair)
            for pair in event["parent_branch_path"]
        )
        and all(
            event.get(key) == origin.get(key)
            for key in ("turn_index", "branch_id", "parent_branch_path")
        )
    )


def _interrupt_proven_children(selected, projected, text_results):
    finished = {
        event["job_id"] for event in selected if event.get("type") == "subagent_result"
    }
    interrupted = {
        event["job_id"]
        for event in selected
        if event.get("type") == "subagent_call"
        and event["job_id"] not in finished
        and _proven_background_call(event)
    }
    output = []
    for event in projected:
        output.append(event)
        if event.get("type") != "tool_call" or event.get("job_id") not in interrupted:
            continue
        terminal = {
            **event,
            "type": "tool_result",
            "output": "Interrupted by session resume",
            "error": "Interrupted by session resume",
            "interrupted": True,
            "_synthetic_resume": True,
        }
        for key in ("event_id", "origin", "execution_class", "background"):
            terminal.pop(key, None)
        output.append(terminal)
        if not event.get("tool_call_id"):
            text_results[event["job_id"]] = subagent_feedback_message(terminal)
    return output


def replay_resume_history(events, branch_view=None):
    """Rebuild selected child calls without treating synthetic results as evidence."""
    live_ids = select_live_event_ids(events, branch_view=branch_view)
    selected = [
        event
        for event in events
        if not isinstance(event.get("event_id"), int) or event["event_id"] in live_ids
    ]
    if not any(
        event.get("type") in ("subagent_call", "subagent_result") for event in selected
    ):
        ToolIdentityResolver(selected)
        compacts = [
            event
            for event in selected
            if event.get("type") in ("compact_replace", "compact_complete")
        ]
        view = resolve_selected_branches(
            selected, index_parent_paths(selected), branch_view
        )
        visible = compact_replay_events(selected, compacts, view)
        return replay_conversation(
            normalize_resumable_events(visible),
            branch_view=branch_view,
            include_metadata=True,
        )
    if any(
        "_synthetic_resume" in event and event["_synthetic_resume"] is not False
        for event in selected
    ):
        raise ValueError(
            "Persisted synthetic resume events cannot authorize child recovery"
        )
    projected, text_results = project_selected_subagents(selected, allow_pending=True)
    ToolIdentityResolver(projected)
    child_ids = {
        event.get("tool_call_id") or event["job_id"]
        for event in selected
        if event.get("type") == "subagent_call"
    }
    compacts = [
        event
        for event in selected
        if event.get("type") in ("compact_replace", "compact_complete")
    ]
    view = resolve_selected_branches(
        selected, index_parent_paths(selected), branch_view
    )
    projected = compact_replay_events(projected, compacts, view)
    projected = _interrupt_proven_children(selected, projected, text_results)
    normalized = canonicalize_provider_tool_order(
        normalize_resumable_events(projected, live_job_ids=child_ids)
    )
    messages = replay_conversation(
        normalized, branch_view=branch_view, include_metadata=True
    )
    messages = restore_text_subagent_feedback(messages, text_results)
    _native_groups(messages, allow_pending_tail=True)
    return messages
