"""EDIT validates the retained raw prefix, including compact provenance."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kohakuterrarium.core.branch_edit_prepare import EditRequest, prepare_edit
from kohakuterrarium.core.conversation import ConversationConfig
from kohakuterrarium.llm.message import Message
from kohakuterrarium.session.raw_history import UserMessageSelector


def _history(child=True):
    common = {
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "job_id": "job",
        "name": "child",
    }
    call = {
        **common,
        "type": "subagent_call" if child else "tool_call",
        "event_id": 2,
        "task": "work",
        "tool_call_id": "provider",
        "tool_call_arguments": '{"task":"work"}',
    }
    result = {
        **common,
        "type": "subagent_result" if child else "tool_result",
        "event_id": 4,
        "output": "done",
        "tool_call_id": "provider",
    }
    if not child:
        call.update(call_id="provider", args={"task": "work"})
    return [
        {
            "type": "user_message",
            "event_id": 1,
            "turn_index": 1,
            "branch_id": 1,
            "content": "first",
        },
        call,
        {
            "type": "user_message",
            "event_id": 3,
            "turn_index": 2,
            "branch_id": 1,
            "parent_branch_path": [[1, 1]],
            "content": "second",
        },
        result,
        {
            "type": "user_message",
            "event_id": 5,
            "turn_index": 3,
            "branch_id": 1,
            "parent_branch_path": [[1, 1], [2, 1]],
            "content": "canonical third",
        },
    ]


def _prepare(events, request):
    messages = tuple(
        Message(role="user", content=text, metadata={"turn_index": i, "branch_id": 1})
        for i, text in enumerate(["first", "second", "stale third"], 1)
    )
    store = SimpleNamespace(get_events=lambda name: deepcopy(events))
    return prepare_edit(
        store,
        "alice",
        messages,
        (3, 1, ((1, 1), (2, 1))),
        ConversationConfig(),
        {},
        request,
    )


def test_raw_edit_canonicalizes_late_background_tool_result():
    prepared = _prepare(
        _history(child=False),
        EditRequest(target=UserMessageSelector(5, 3, 1), new_content="edited"),
    )
    messages = prepared["messages"]
    assert [m.role for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[2].content == "done"


def test_regenerate_target_uses_rebuilt_canonical_content():
    prepared = _prepare(_history(), EditRequest(message_idx=2, regenerate=True))
    assert prepared["content"] == "canonical third"
    assert any(m.content == "done" for m in prepared["messages"])


def test_compact_cannot_split_native_child_pair():
    events = _history()
    events.append(
        {
            "type": "compact_replace",
            "event_id": 6,
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 2,
            "summary": "split pair",
            "turn_index": 3,
            "branch_id": 1,
            "parent_branch_path": [[1, 1], [2, 1]],
        }
    )
    with pytest.raises(ValueError, match="orphan"):
        _prepare(
            events,
            EditRequest(target=UserMessageSelector(5, 3, 1), new_content="edited"),
        )


def test_invalid_branch_user_position_returns_no_edit():
    assert (
        _prepare(
            _history(),
            EditRequest(
                branch_view={1: 1, 2: 1, 3: 1}, user_position=99, new_content="wrong"
            ),
        )
        is None
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_compact_baseline_does_not_hide_corrupt_retained_provenance(corrupt):
    events = _history()
    if corrupt:
        events[1].pop("tool_call_arguments")
    events.append(
        {
            "type": "compact_replace",
            "event_id": 6,
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 5,
            "summary": "retained summary",
            "turn_index": 3,
            "branch_id": 1,
            "parent_branch_path": [[1, 1], [2, 1]],
        }
    )
    request = EditRequest(target=UserMessageSelector(5, 3, 1), new_content="edited")
    if corrupt:
        with pytest.raises(ValueError, match="recorded arguments"):
            _prepare(events, request)
    else:
        prepared = _prepare(events, request)
        assert any(m.content == "retained summary" for m in prepared["messages"])
        assert not any(m.tool_calls for m in prepared["messages"])
