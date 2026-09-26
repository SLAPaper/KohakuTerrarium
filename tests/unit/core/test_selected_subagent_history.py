"""Strict SELECT projection preserves child identities without legacy promotion."""

from copy import deepcopy

import pytest

from kohakuterrarium.core.runtime_context import RuntimeContextPreparationError
from kohakuterrarium.core.runtime_context_replacement_prepare import (
    prepare_selected_branch_context,
)


def _events(native=True):
    common = {
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "job_id": "agent_child_one",
        "name": "child",
    }
    if native:
        common["tool_call_id"] = "provider"
    return [
        {
            "type": "user_message",
            "event_id": 1,
            "turn_index": 1,
            "branch_id": 1,
            "content": "question",
        },
        {
            **common,
            "type": "subagent_call",
            "event_id": 2,
            "task": "work",
            "tool_call_arguments": '{ "task": "work", "run_in_background": true }',
        },
        {**common, "type": "subagent_result", "event_id": 3, "output": "child result"},
    ]


def _prepare(events):
    return prepare_selected_branch_context(
        agent_id="alice",
        runtime_scope_id="scope",
        raw_events=events,
        branch_view={1: 1},
        require_complete_tool_pairs=True,
    )


@pytest.mark.parametrize("native", [False, True])
def test_projection_preserves_mode_and_does_not_mutate_raw(native):
    events = _events(native)
    unchanged = deepcopy(events)
    messages = _prepare(events).provider_conversation
    assert events == unchanged
    if native:
        assert messages[1].tool_calls[0]["id"] == "provider"
        assert (
            messages[1].tool_calls[0]["function"]["arguments"]
            == events[1]["tool_call_arguments"]
        )
        assert messages[2].role == "tool"
        assert messages[2].tool_call_id == "provider"
        assert messages[2].content == "child result"
    else:
        assert not any(m.tool_calls or m.role == "tool" for m in messages)
        assert any(m.role == "user" and "child result" in m.content for m in messages)


def test_text_child_feedback_follows_remaining_native_results():
    events = _events(native=False)
    events.insert(
        1,
        {
            "type": "assistant_tool_calls",
            "event_id": 4,
            "turn_index": 1,
            "branch_id": 1,
            "tool_calls": [
                {
                    "id": "agent_child_one",
                    "type": "function",
                    "function": {"name": "child", "arguments": '{"task":"work"}'},
                },
                {
                    "id": "native",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
            ],
        },
    )
    events.insert(
        3,
        {
            "type": "tool_call",
            "event_id": 5,
            "turn_index": 1,
            "branch_id": 1,
            "call_id": "native",
            "tool_name": "read",
            "args": {},
        },
    )
    events.append(
        {
            "type": "tool_result",
            "event_id": 6,
            "turn_index": 1,
            "branch_id": 1,
            "call_id": "native",
            "tool_name": "read",
            "output": "native result",
        }
    )
    for event_id, event in enumerate(events, 1):
        event["event_id"] = event_id
    messages = _prepare(events).provider_conversation
    assert [m.role for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[1].tool_calls[0]["id"] == "native"
    assert messages[2].content == "native result"
    assert "child result" in messages[3].content


@pytest.mark.parametrize(
    "damage",
    [
        "pending",
        "duplicate_call",
        "duplicate_result",
        "orphan",
        "job",
        "provider",
        "name",
        "missing_name",
        "missing_args",
        "bad_args",
        "args_list",
        "legacy_task",
        "provider_bool",
        "lineage",
        "call_alias",
        "tool_alias",
        "argument_alias",
        "origin",
        "partial_parent",
    ],
)
def test_bad_child_evidence_rejected_before_sanitization(damage):
    events = _events(native=damage != "legacy_task")
    if damage == "pending":
        events.pop()
    elif damage == "duplicate_call":
        events.insert(2, deepcopy(events[1]))
    elif damage == "duplicate_result":
        events.append(deepcopy(events[-1]))
    elif damage == "orphan":
        events.pop(1)
    elif damage == "job":
        events[-1]["job_id"] = "foreign"
    elif damage == "provider":
        events[-1]["tool_call_id"] = "foreign"
    elif damage == "name":
        events[-1]["name"] = "other"
    elif damage == "missing_name":
        events[1].pop("name")
    elif damage == "missing_args":
        events[1].pop("tool_call_arguments")
    elif damage == "bad_args":
        events[1]["tool_call_arguments"] = "not json"
    elif damage == "args_list":
        events[1]["tool_call_arguments"] = "[]"
    elif damage == "legacy_task":
        events[1]["task"] = None
    elif damage == "provider_bool":
        events[1]["tool_call_id"] = True
    elif damage == "call_alias":
        events[1]["call_id"] = "foreign"
    elif damage == "tool_alias":
        events[-1]["tool_name"] = "foreign"
    elif damage == "argument_alias":
        events[1]["args"] = {"task": "forged"}
    elif damage == "origin":
        events[1]["origin"] = {"delivery_id": "scope:job"}
        events[-1]["origin"] = {"delivery_id": "other:job"}
    elif damage == "partial_parent":
        events[1].pop("parent_branch_path")
    else:
        events[-1]["parent_branch_path"] = [[1, 2]]
    with pytest.raises(RuntimeContextPreparationError):
        _prepare(events)
