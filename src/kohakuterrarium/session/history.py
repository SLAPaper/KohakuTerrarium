"""Normalize, branch-select, and replay persisted session events."""

import json
from typing import Any, Iterable

from kohakuterrarium.session.feedback_admission import project_feedback_events
from kohakuterrarium.session.history_replay import replay_selected_events
from kohakuterrarium.session.history_support import (
    InvalidBranchViewError,
    _clean_tool_name,
    _coerce_path,
    _covered_by_replace,
    _event_signature_value,
    _path_matches,
    _rule_priority,
    collect_branch_metadata,
    collect_user_groups,
    compact_path_from_event,
    dedupe_adjacent_duplicate_events,
    index_parent_paths,
    project_branch_metadata,
    resolve_branch_view_strict,
    resolve_selected_branches,
    select_live_event_ids,
)

__all__ = [
    "InvalidBranchViewError",
    "_clean_tool_name",
    "_coerce_path",
    "_covered_by_replace",
    "_event_signature_value",
    "_path_matches",
    "_rule_priority",
    "collect_branch_metadata",
    "collect_user_groups",
    "compact_path_from_event",
    "dedupe_adjacent_duplicate_events",
    "index_parent_paths",
    "normalize_resumable_events",
    "normalize_tool_call_events",
    "project_branch_metadata",
    "replay_conversation",
    "resolve_branch_view_strict",
    "resolve_selected_branches",
    "select_live_event_ids",
]


def replay_conversation(
    events: Iterable[dict[str, Any]],
    *,
    branch_view: dict[int, int] | None = None,
    include_metadata: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild an OpenAI-shape message list from the event log.

    Events are branch-filtered and emitted in provider-ready message order.
    Consecutive text chunks coalesce, tool announcements pair with tool results,
    system prompts remain ordered, and compaction ranges become one summary.
    Unknown observability events are ignored.
    """
    return replay_selected_events(
        events, branch_view=branch_view, include_metadata=include_metadata
    )


def _coerce_tool_args_to_json(args: Any) -> str:
    """Best-effort serialisation for ``assistant_tool_calls.arguments``.

    OpenAI-shaped tool-call arguments must remain JSON strings for replay and
    downstream sanitization.
    """
    if isinstance(args, str):
        return args
    if args is None:
        return "{}"
    try:
        return json.dumps(args)
    except (TypeError, ValueError):
        return "{}"


def _inject_synthetic_announcements(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert missing assistant tool-call announcements before consumption.

    Pending calls are grouped until a structural event requires them. Existing
    announcements suppress duplicates. Synthetic announcements intentionally omit
    event identifiers so replay preserves their inserted position.
    """
    events = project_feedback_events(events)
    pending: list[dict[str, Any]] = []
    announced_ids: set[str] = set()
    result: list[dict[str, Any]] = []

    def _build_announce(items: list[dict[str, Any]]) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        for tc in items:
            tool_calls.append(
                {
                    "id": _provider_call_id(tc),
                    "type": "function",
                    "function": {
                        "name": _clean_tool_name(tc),
                        "arguments": _coerce_tool_args_to_json(
                            tc.get("tool_call_arguments") or tc.get("args")
                        ),
                    },
                }
            )
        return {
            "type": "assistant_tool_calls",
            "tool_calls": tool_calls,
            "content": "",
            "ts": items[0].get("ts", 0),
            "_synthetic_announce": True,
        }

    def _flush_pending() -> None:
        if not pending:
            return
        result.append(_build_announce(pending))
        for tc in pending:
            cid = _provider_call_id(tc)
            if cid:
                announced_ids.add(cid)
        pending.clear()

    for evt in events:
        etype = evt.get("type", "")

        if etype == "tool_call":
            cid = _provider_call_id(evt)
            # Calls already announced must not be synthesized again.
            if cid and cid in announced_ids:
                result.append(evt)
                continue
            pending.append(evt)
            result.append(evt)
            continue

        if etype == "subagent_call":
            # Sub-agent dispatches can interleave with calls from the same assistant
            # message and therefore do not delimit the pending call group.
            result.append(evt)
            continue

        if etype == "assistant_tool_calls":
            # Explicit announcements remove matching calls from the pending group.
            for tc in evt.get("tool_calls") or []:
                tid = str(tc.get("id") or "")
                if tid:
                    announced_ids.add(tid)
            pending = [
                tc for tc in pending if _provider_call_id(tc) not in announced_ids
            ]
            result.append(evt)
            continue

        # Announcements must precede the structural event that consumes the calls.
        _flush_pending()
        result.append(evt)

    # An unfinished stream still needs a valid trailing assistant announcement.
    _flush_pending()
    return result


def _provider_call_id(event: dict[str, Any]) -> str:
    return str(
        event.get("tool_call_id") or event.get("call_id") or event.get("job_id") or ""
    )


def normalize_tool_call_events(
    events: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy events and restore missing assistant tool-call announcements.

    Unlike resume normalization, live history replay must not finalize an
    unfinished background job as interrupted. It only repairs the declaration
    required to pair persisted tool results with their assistant call.
    """
    return _inject_synthetic_announcements([dict(evt) for evt in events])


def _pair_undispatched_declarations(events):
    """Pair durable announcements lacking a subsequent dispatch with interrupted results."""
    output = []
    for index, event in enumerate(events):
        output.append(event)
        if event.get("type") != "assistant_tool_calls" or not event.get(
            "veto_group_id"
        ):
            continue
        following = []
        for candidate in events[index + 1 :]:
            if candidate.get("type") == "assistant_tool_calls":
                break
            following.append(candidate)
        for call in event.get("tool_calls", []):
            provider_id = call.get("id")
            if any(
                row.get("tool_call_id") == provider_id
                and row.get("type")
                in ("tool_call", "tool_result", "subagent_call", "subagent_result")
                for row in following
            ):
                continue
            output.append(
                {
                    "type": "tool_result",
                    "tool_call_id": provider_id,
                    "call_id": provider_id,
                    "name": call.get("function", {}).get("name", "tool"),
                    "output": "Interrupted before dispatch confirmation",
                    "error": "Interrupted before dispatch confirmation",
                    "interrupted": True,
                    "final_state": "interrupted",
                    "_synthetic_resume": True,
                    "turn_index": event.get("turn_index"),
                    "branch_id": event.get("branch_id"),
                    "parent_branch_path": event.get("parent_branch_path"),
                    "ts": event.get("ts", 0),
                }
            )
    return output


def normalize_resumable_events(
    events: list[dict[str, Any]],
    *,
    live_job_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Mark unfinished tool/sub-agent work as interrupted for history replay.

    Jobs listed in ``live_job_ids`` remain running; all other unfinished work is
    represented by synthetic interrupted results. Missing assistant tool-call
    announcements are also injected so replay preserves valid tool pairing.
    """
    normalized = [dict(evt) for evt in events]
    started_tools: dict[str, dict[str, Any]] = {}
    finished_tools: set[str] = set()
    started_subagents: dict[str, dict[str, Any]] = {}
    finished_subagents: set[str] = set()
    live = live_job_ids or set()

    for evt in normalized:
        etype = evt.get("type", "")
        if etype == "tool_call":
            job_id = evt.get("call_id") or evt.get("job_id") or ""
            if job_id:
                started_tools[str(job_id)] = evt
        elif etype == "tool_result":
            job_id = evt.get("call_id") or evt.get("job_id") or ""
            if job_id:
                finished_tools.add(str(job_id))
        elif etype == "subagent_call":
            job_id = evt.get("job_id") or ""
            if job_id:
                started_subagents[str(job_id)] = evt
        elif etype == "subagent_result":
            job_id = evt.get("job_id") or ""
            if job_id:
                finished_subagents.add(str(job_id))

    synthetic_events: list[dict[str, Any]] = []

    for job_id, start_evt in started_tools.items():
        if job_id in finished_tools or job_id in live:
            continue
        synthetic_events.append(
            {
                "type": "tool_result",
                "name": start_evt.get("name", "tool") or "tool",
                "call_id": job_id,
                "job_id": start_evt.get("job_id", "") or job_id,
                "args": start_evt.get("args", {}),
                "output": "",
                "error": "Interrupted by session resume",
                "interrupted": True,
                "final_state": "interrupted",
                "ts": start_evt.get("ts", 0),
                "_synthetic_resume": True,
            }
        )

    for job_id, start_evt in started_subagents.items():
        if job_id in finished_subagents or job_id in live:
            continue
        synthetic_events.append(
            {
                "type": "subagent_result",
                "name": start_evt.get("name", "subagent") or "subagent",
                "job_id": job_id,
                "task": start_evt.get("task", ""),
                "background": bool(start_evt.get("background", False)),
                "output": "",
                "error": "Interrupted by session resume",
                "success": False,
                "interrupted": True,
                "final_state": "interrupted",
                "ts": start_evt.get("ts", 0),
                "_synthetic_resume": True,
            }
        )

    full = _pair_undispatched_declarations(normalized) + synthetic_events
    return normalize_tool_call_events(full)
