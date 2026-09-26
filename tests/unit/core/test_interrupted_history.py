"""Interruption pairing uses occurrence identity and strict captured provenance."""

from types import SimpleNamespace

import pytest

from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.interrupted_history import append_pending_result
from kohakuterrarium.core.tool_origin import capture_tool_context


def _fixture():
    conversation = Conversation()
    agent = SimpleNamespace(
        controller=SimpleNamespace(conversation=conversation),
        _runtime_scope_id="scope",
        _turn_index=2,
        _branch_id=1,
        _parent_branch_path=[[1, 1]],
        _direct_job_meta={},
    )
    context = capture_tool_context(agent, "job", "direct", "provider")
    meta = {"name": "read", "tool_call_id": "provider", "completion_context": context}
    agent._direct_job_meta["job"] = meta
    conversation.append("user", "question", metadata={"turn_index": 2, "branch_id": 1})
    conversation.append(
        "assistant",
        "",
        tool_calls=[
            {
                "id": "provider",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
        ],
    )
    return agent, meta


@pytest.mark.parametrize(
    "damage",
    [
        "scope",
        "sibling",
        "ancestor",
        "later_occurrence",
        "bad_class",
        "bad_parent",
        "missing_turn",
        "missing_branch",
        "partial_origin",
        "orphan",
        "already_paired",
        "later_input",
    ],
)
def test_rejects_unrelated_or_completed_occurrence(damage):
    agent, meta = _fixture()
    conversation = agent.controller.conversation
    if damage == "scope":
        agent._runtime_scope_id = "other"
    elif damage == "sibling":
        agent._branch_id = 2
    elif damage == "ancestor":
        agent._turn_index, agent._parent_branch_path = 1, []
    elif damage == "later_occurrence":
        agent._turn_index, agent._parent_branch_path = 3, [[1, 1], [2, 1]]
        conversation.get_messages()[0].metadata["turn_index"] = 3
    elif damage == "bad_class":
        meta["completion_context"]["origin"]["execution_class"] = "background"
    elif damage == "bad_parent":
        meta["completion_context"]["origin"]["parent_branch_path"] = [[True, 1]]
    elif damage == "missing_turn":
        meta["completion_context"]["origin"].pop("turn_index")
    elif damage == "missing_branch":
        meta["completion_context"]["origin"].pop("branch_id")
    elif damage == "partial_origin":
        for key in ("turn_index", "branch_id", "parent_branch_path", "delivery_id"):
            meta["completion_context"]["origin"].pop(key)
    elif damage == "orphan":
        conversation.truncate_from(1)
    elif damage == "already_paired":
        conversation.append(
            "tool", "genuine result", tool_call_id="provider", name="read"
        )
    else:
        conversation.append("user", "later input")
    before = conversation.to_json()
    assert not append_pending_result(agent, "job", meta, "interrupted")
    assert conversation.to_json() == before


def test_valid_partial_group_retains_other_results_and_pairs_once():
    agent, meta = _fixture()
    conversation = agent.controller.conversation
    conversation.get_messages()[1].tool_calls.append(
        {
            "id": "other",
            "type": "function",
            "function": {"name": "write", "arguments": "{}"},
        }
    )
    conversation.append("tool", "other result", tool_call_id="other", name="write")
    assert append_pending_result(agent, "job", meta, "interrupted")
    assert not append_pending_result(agent, "job", meta, "duplicate")
    assert [m.content for m in conversation.get_messages() if m.role == "tool"] == [
        "other result",
        "interrupted",
    ]
