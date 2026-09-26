import copy

import pytest

from kohakuterrarium.core.runtime_context import (
    RuntimeContextPreparationError,
    RuntimeContextPreparationMode,
)
from kohakuterrarium.core.runtime_context_replacement_prepare import (
    prepare_selected_branch_context,
)
from kohakuterrarium.llm.message import Message
from kohakuterrarium.session.history import InvalidBranchViewError


def event(event_id, type_, turn=None, branch=None, parent_path=None, **fields):
    item = {"event_id": event_id, "type": type_, **fields}
    if turn is not None:
        item.update(
            turn_index=turn,
            branch_id=branch,
            parent_branch_path=tuple(parent_path or ()),
        )
    return item


def fork_events():
    return [
        event(1, "system_prompt_set", content="rules"),
        event(2, "user_message", turn=1, branch=1, content="root"),
        event(3, "text_chunk", turn=1, branch=1, content="root answer"),
        event(
            4,
            "user_message",
            turn=2,
            branch=1,
            parent_path=[(1, 1)],
            content="sibling",
        ),
        event(
            5,
            "text_chunk",
            turn=2,
            branch=1,
            parent_path=[(1, 1)],
            content="sibling answer",
        ),
        event(
            6,
            "user_message",
            turn=2,
            branch=2,
            parent_path=[(1, 1)],
            content="chosen",
        ),
        event(
            7,
            "user_message",
            turn=3,
            branch=1,
            parent_path=[(1, 1), (2, 2)],
            content="tail",
        ),
        event(
            8,
            "text_chunk",
            turn=3,
            branch=1,
            parent_path=[(1, 1), (2, 2)],
            content="tail answer",
        ),
    ]


def prepare(events, **kwargs):
    values = {
        "agent_id": "agent",
        "runtime_scope_id": "session",
        "raw_events": events,
        "branch_view": {2: 2},
    }
    values.update(kwargs)
    return prepare_selected_branch_context(**values)


def test_partial_view_resolves_authoritative_path_and_excludes_sibling():
    result = prepare(fork_events())
    assert result.branch_view == ((1, 1), (2, 2), (3, 1))
    assert (result.turn_index, result.branch_id) == (3, 1)
    assert result.parent_branch_path == ((1, 1), (2, 2))
    assert "sibling" not in [m.content for m in result.provider_conversation]
    assert isinstance(result.provider_conversation, tuple)
    assert (
        result.preparation_mode
        is RuntimeContextPreparationMode.NORMALIZED_CANONICAL_REPLAY
    )


def test_selected_branch_ends_at_last_available_turn():
    events = [e for e in fork_events() if not (e.get("turn_index") == 3)]
    result = prepare_selected_branch_context(
        agent_id="agent",
        runtime_scope_id="session",
        raw_events=events,
        branch_view={2: 1},
    )
    assert (result.turn_index, result.branch_id) == (2, 1)
    assert result.branch_view == ((1, 1), (2, 1))


def test_tool_events_normalize_order_ids_and_trailing_text():
    events = [
        event(1, "user_message", 1, 1, content="q"),
        event(2, "tool_call", 1, 1, call_id="c1", name="lookup", arguments="{}"),
        event(3, "tool_result", 1, 1, call_id="c1", content="ok"),
        event(4, "text_chunk", 1, 1, content="after tool"),
    ]
    result = prepare_selected_branch_context(
        agent_id="a", runtime_scope_id="s", raw_events=events, branch_view={1: 1}
    )
    assert [m.role for m in result.provider_conversation] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.provider_conversation[1].tool_calls[0]["id"] == "c1"
    assert result.provider_conversation[2].tool_call_id == "c1"
    assert result.provider_conversation[-1].content == "after tool"


def test_system_replay_wins_and_baseline_identity_is_fallback_only():
    baseline = Message("system", "baseline", metadata={"trusted": True})
    result = prepare(fork_events(), baseline_system_messages=(baseline,))
    assert result.provider_conversation[0].content == "rules"
    assert result.provider_conversation[0] is not baseline
    no_system = [e for e in fork_events() if e["type"] != "system_prompt_set"]
    fallback = prepare_selected_branch_context(
        agent_id="a",
        runtime_scope_id="s",
        raw_events=no_system,
        branch_view={2: 2},
        baseline_system_messages=(baseline,),
    )
    assert fallback.provider_conversation[0] is baseline


def test_inputs_are_deeply_unchanged_and_metadata_restored():
    events = fork_events()
    before = copy.deepcopy(events)
    result = prepare(events)
    assert events == before
    assert all(isinstance(m.metadata, dict) for m in result.provider_conversation)


@pytest.mark.parametrize("bad", [None, {"bad": 1}, [object()]])
def test_invalid_raw_sources_are_wrapped(bad):
    with pytest.raises(RuntimeContextPreparationError):
        prepare_selected_branch_context(
            agent_id="a", runtime_scope_id="s", raw_events=bad, branch_view={}
        )


def test_invalid_branch_view_type_propagates():
    with pytest.raises(InvalidBranchViewError):
        prepare_selected_branch_context(
            agent_id="a",
            runtime_scope_id="s",
            raw_events=fork_events(),
            branch_view={"x": 1},
        )


def test_empty_replay_is_valid_zero_context_with_baseline():
    result = prepare_selected_branch_context(
        agent_id="a",
        runtime_scope_id="s",
        raw_events=[],
        branch_view={},
        baseline_system_messages=(Message("system", "rules"),),
    )
    assert result.provider_conversation and result.turn_index == result.branch_id == 0
    assert result.branch_view == ()


def test_invalid_baseline_is_wrapped():
    with pytest.raises(RuntimeContextPreparationError):
        prepare_selected_branch_context(
            agent_id="a",
            runtime_scope_id="s",
            raw_events=[],
            branch_view={},
            baseline_system_messages=(object(),),
        )


def test_payload_is_detached_tuple():
    result = prepare(fork_events())
    assert isinstance(result.provider_conversation, tuple)
    assert all(isinstance(m, Message) for m in result.provider_conversation)
    result.provider_conversation[0].metadata["x"] = 1
    assert "x" not in fork_events()[0]


def test_empty_events_without_baseline_are_zero_context():
    result = prepare_selected_branch_context(
        agent_id="a", runtime_scope_id="s", raw_events=[], branch_view={}
    )
    assert result.provider_conversation == ()
    assert result.parent_branch_path == ()


def test_raw_event_payload_metadata_contains_coordinates():
    result = prepare_selected_branch_context(
        agent_id="a", runtime_scope_id="s", raw_events=fork_events(), branch_view={2: 2}
    )
    assert any(
        m.metadata.get("turn_index") == 2
        for m in result.provider_conversation
        if m.role == "user"
    )


def test_sibling_assistant_tail_is_not_selected():
    events = fork_events() + [event(9, "text_chunk", 2, 1, content="branch one tail")]
    result = prepare_selected_branch_context(
        agent_id="a", runtime_scope_id="s", raw_events=events, branch_view={2: 2}
    )
    assert all(m.content != "branch one tail" for m in result.provider_conversation)


def test_parent_path_is_coherent_for_selected_branch():
    result = prepare_selected_branch_context(
        agent_id="a", runtime_scope_id="s", raw_events=fork_events(), branch_view={2: 2}
    )
    assert result.parent_branch_path[-1] == (2, 2)
    assert result.branch_view[-1] == (3, 1)


def test_selected_branch_keeps_its_tool_pair_and_excludes_sibling_tool_pair():
    events = [
        event(101, "user_message", 1, 1, content="Find the release notes"),
        event(
            102,
            "tool_call",
            1,
            1,
            call_id="call-selected-42",
            name="read_file",
            arguments={"path": "docs/release-notes.md"},
        ),
        event(
            103,
            "tool_result",
            1,
            1,
            call_id="call-selected-42",
            content="selected contents",
        ),
        event(104, "user_message", 2, 1, [(1, 1)], content="Sibling request"),
        event(
            105,
            "tool_call",
            2,
            1,
            [(1, 1)],
            call_id="call-sibling-99",
            name="read_file",
            arguments={"path": "docs/sibling.md"},
        ),
        event(
            106,
            "tool_result",
            2,
            1,
            [(1, 1)],
            call_id="call-sibling-99",
            content="sibling contents",
        ),
        event(107, "user_message", 2, 2, [(1, 1)], content="Chosen request"),
    ]
    result = prepare_selected_branch_context(
        agent_id="agent-42",
        runtime_scope_id="session-2025-01",
        raw_events=events,
        branch_view={2: 2},
    )
    assert any(
        m.tool_calls and m.tool_calls[0]["id"] == "call-selected-42"
        for m in result.provider_conversation
    )
    assert any(
        m.tool_call_id == "call-selected-42" for m in result.provider_conversation
    )
    assert all(
        m.tool_call_id != "call-sibling-99" for m in result.provider_conversation
    )
    assert all(
        not (m.tool_calls and m.tool_calls[0]["id"] == "call-sibling-99")
        for m in result.provider_conversation
    )


@pytest.mark.parametrize("bad_value", ["1", True, 0, -1])
@pytest.mark.parametrize(
    "event_type,field", [("user_message", "turn_index"), ("tool_call", "branch_id")]
)
def test_raw_user_and_tool_coordinates_require_positive_ints(
    event_type, field, bad_value
):
    coordinates = {"turn_index": 1, "branch_id": 1}
    coordinates[field] = bad_value
    events = [
        {"event_id": 201, "type": event_type, **coordinates, "content": "query"},
    ]
    with pytest.raises(RuntimeContextPreparationError):
        prepare_selected_branch_context(
            agent_id="agent-42",
            runtime_scope_id="session-2025-01",
            raw_events=events,
            branch_view={1: 1},
        )


def test_tool_call_ids_are_preserved_exactly():
    events = [
        event(1, "user_message", 1, 1, content="q"),
        event(2, "tool_call", 1, 1, call_id="call-007", name="x", arguments={"a": 1}),
    ]
    result = prepare_selected_branch_context(
        agent_id="a", runtime_scope_id="s", raw_events=events, branch_view={1: 1}
    )
    assert result.provider_conversation[1].tool_calls[0]["id"] == "call-007"


def test_selected_tail_includes_assistant_messages():
    result = prepare(fork_events())
    assert any(
        m.role == "assistant" and m.content == "tail answer"
        for m in result.provider_conversation
    )


def test_no_system_baseline_is_not_inserted_when_replayed():
    baseline = Message("system", "b")
    result = prepare(fork_events(), baseline_system_messages=(baseline,))
    assert sum(m.role == "system" for m in result.provider_conversation) == 1


def test_branch_view_mapping_is_accepted():
    result = prepare(fork_events(), branch_view={2: 2})
    assert result.branch_view == ((1, 1), (2, 2), (3, 1))


def test_context_has_canonical_mode():
    assert (
        prepare(fork_events()).preparation_mode.value == "normalized_canonical_replay"
    )


def test_metadata_is_not_shared_with_raw_event():
    events = fork_events()
    result = prepare(events)
    if result.provider_conversation:
        result.provider_conversation[0].metadata["mutated"] = True
    assert "mutated" not in events[0]


def test_result_path_is_tuple_of_pairs():
    result = prepare(fork_events())
    assert isinstance(result.parent_branch_path, tuple)
    assert all(isinstance(p, tuple) for p in result.parent_branch_path)


def test_raw_events_can_be_tuple():
    result = prepare_selected_branch_context(
        agent_id="a",
        runtime_scope_id="s",
        raw_events=tuple(fork_events()),
        branch_view={2: 2},
    )

    assert result.turn_index == 3


def _strict_prepare(events, branch_view=None):
    if not any(item.get("type") == "user_message" for item in events):
        events = [event(0, "user_message", 1, 1, content="q"), *events]
    return prepare_selected_branch_context(
        agent_id="a",
        runtime_scope_id="s",
        raw_events=events,
        branch_view=branch_view or {1: 1},
        require_complete_tool_pairs=True,
    )


def _tool_call(event_id, call_id, *, turn=1, branch=1, parent_path=None):
    return event(
        event_id,
        "tool_call",
        turn,
        branch,
        parent_path,
        call_id=call_id,
        name="lookup",
        arguments="{}",
    )


def _tool_result(event_id, call_id, *, turn=1, branch=1, parent_path=None):
    return event(
        event_id,
        "tool_result",
        turn,
        branch,
        parent_path,
        call_id=call_id,
        content="result",
    )


def _tool_announcement(event_id, *call_ids, turn=1, branch=1, parent_path=None):
    return event(
        event_id,
        "assistant_tool_calls",
        turn,
        branch,
        parent_path,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    )


def test_strict_selected_native_sequence_accepts_normalized_complete_pair():
    result = _strict_prepare(
        [
            event(1, "user_message", 1, 1, content="q"),
            _tool_call(2, "call-1"),
            _tool_result(3, "call-1"),
        ]
    )

    assert [message.role for message in result.provider_conversation] == [
        "user",
        "assistant",
        "tool",
    ]
    assert result.provider_conversation[1].tool_calls[0]["id"] == "call-1"
    assert result.provider_conversation[2].tool_call_id == "call-1"


@pytest.mark.parametrize(
    "events",
    [
        [event(1, "assistant_tool_calls", 1, 1, tool_calls=[{"id": "call-1"}])],
        [_tool_result(1, "call-1")],
        [_tool_call(1, "call-1"), _tool_result(2, "call-1"), _tool_result(3, "call-1")],
        [
            _tool_call(1, "call-1"),
            _tool_result(2, "call-1"),
            _tool_call(3, "call-1"),
            _tool_result(4, "call-1"),
        ],
        [_tool_call(1, "   "), _tool_result(2, "   ")],
        [_tool_call(1, "call-1"), _tool_result(2, "call-2")],
        [_tool_call(1, "call-1")],
    ],
    ids=[
        "orphan-announcement",
        "orphan-result",
        "duplicate-result",
        "duplicate-call-id",
        "blank-ids",
        "mismatched-ids",
        "pending-direct-call",
    ],
)
def test_strict_selected_native_sequence_rejects_malformed_history(events):
    with pytest.raises(RuntimeContextPreparationError, match="structured tool history"):
        _strict_prepare(events)


@pytest.mark.parametrize("call_ids", [("call-1",), ("call-1", "call-2")])
def test_strict_selected_native_sequence_accepts_explicit_announcement_and_dispatch(
    call_ids,
):
    result = _strict_prepare(
        [
            _tool_announcement(1, *call_ids),
            *[_tool_call(2 + index, call_id) for index, call_id in enumerate(call_ids)],
            *[
                _tool_result(2 + len(call_ids) + index, call_id)
                for index, call_id in enumerate(call_ids)
            ],
        ]
    )

    assert [call["id"] for call in result.provider_conversation[1].tool_calls] == list(
        call_ids
    )
    assert [
        message.tool_call_id for message in result.provider_conversation[2:]
    ] == list(call_ids)


@pytest.mark.parametrize("result_before_duplicate", [False, True])
def test_strict_selected_native_sequence_rejects_second_dispatch(
    result_before_duplicate,
):
    events = [
        _tool_announcement(1, "call-1"),
        _tool_call(2, "call-1"),
        *([_tool_result(3, "call-1")] if result_before_duplicate else []),
        _tool_call(4, "call-1"),
        *([] if result_before_duplicate else [_tool_result(5, "call-1")]),
    ]

    with pytest.raises(
        RuntimeContextPreparationError, match="duplicate dispatch.*call-1"
    ):
        _strict_prepare(events)


def test_strict_validation_ignores_duplicate_dispatch_in_excluded_sibling():
    events = [
        event(1, "user_message", 1, 1, content="root"),
        _tool_call(2, "selected"),
        _tool_result(3, "selected"),
        event(4, "user_message", 2, 1, [(1, 1)], content="bad sibling"),
        _tool_call(5, "selected", turn=2, branch=1, parent_path=[(1, 1)]),
        event(6, "user_message", 2, 2, [(1, 1)], content="chosen"),
    ]

    result = _strict_prepare(events, {2: 2})

    assert (
        sum(
            message.tool_call_id == "selected"
            for message in result.provider_conversation
        )
        == 1
    )


def test_strict_validation_ignores_malformed_excluded_sibling():
    events = [
        event(1, "user_message", 1, 1, content="root"),
        _tool_call(2, "selected"),
        _tool_result(3, "selected"),
        event(4, "user_message", 2, 1, [(1, 1)], content="bad sibling"),
        _tool_result(5, "orphan", turn=2, branch=1, parent_path=[(1, 1)]),
        event(6, "user_message", 2, 2, [(1, 1)], content="chosen"),
    ]

    result = _strict_prepare(events, {2: 2})

    assert any(
        message.tool_call_id == "selected" for message in result.provider_conversation
    )
    assert all(
        message.tool_call_id != "orphan" for message in result.provider_conversation
    )


def test_unselected_sibling_alias_collision_does_not_poison_selected_prepare():
    events = [
        event(1, "user_message", 1, 1, content="root"),
        event(
            2,
            "tool_call",
            1,
            1,
            call_id="shared_internal",
            job_id="shared_internal",
            tool_call_id="call_provider_a",
            name="web_search",
            args={"query": "selected"},
        ),
        event(
            3,
            "tool_result",
            1,
            1,
            call_id="shared_internal",
            name="web_search",
            output="selected-done",
        ),
        event(4, "user_message", 2, 1, [(1, 1)], content="bad sibling"),
        event(
            5,
            "tool_call",
            2,
            1,
            [(1, 1)],
            call_id="shared_internal",
            job_id="shared_internal",
            tool_call_id="call_provider_b",
            name="bash",
            args={"command": "ls"},
        ),
        event(6, "user_message", 2, 2, [(1, 1)], content="chosen"),
    ]

    result = _strict_prepare(events, {2: 2})
    announced = [
        call["id"]
        for message in result.provider_conversation
        if message.tool_calls
        for call in message.tool_calls
    ]
    result_ids = [
        message.tool_call_id
        for message in result.provider_conversation
        if message.role == "tool"
    ]
    assert announced == ["call_provider_a"]
    assert result_ids == ["call_provider_a"]


@pytest.mark.parametrize("result_ids", [("call-1", "call-2"), ("call-2", "call-1")])
def test_strict_selected_native_sequence_accepts_complete_multi_call_block(result_ids):
    events = [
        event(1, "user_message", 1, 1, content="q"),
        _tool_call(2, "call-1"),
        _tool_call(3, "call-2"),
        *[_tool_result(4 + index, call_id) for index, call_id in enumerate(result_ids)],
    ]

    result = _strict_prepare(events)

    assert [call["id"] for call in result.provider_conversation[1].tool_calls] == [
        "call-1",
        "call-2",
    ]
    assert [message.tool_call_id for message in result.provider_conversation[2:]] == [
        "call-1",
        "call-2",
    ]


@pytest.mark.parametrize(
    "result_ids",
    [("call-1",), ("call-1", "call-1"), ("call-1", "other")],
)
def test_strict_selected_native_sequence_rejects_incomplete_multi_call_block(
    result_ids,
):
    events = [
        _tool_call(1, "call-1"),
        _tool_call(2, "call-2"),
        *[_tool_result(3 + index, call_id) for index, call_id in enumerate(result_ids)],
    ]

    with pytest.raises(RuntimeContextPreparationError, match="structured tool history"):
        _strict_prepare(events)
