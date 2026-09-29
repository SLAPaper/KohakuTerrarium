"""Retained edit context uses only its selected raw provenance."""

import pytest

from kohakuterrarium.core.branch_edit_prepare import (
    EditRequest,
    _message_turn,
    prepare_edit,
)
from kohakuterrarium.core.conversation import ConversationConfig
from kohakuterrarium.llm.message import Message
from kohakuterrarium.session.event_origin import EVENT_COORDINATE
from kohakuterrarium.session.store import SessionStore


@pytest.mark.parametrize(
    "metadata",
    [
        {"turn_index": 6, "branch_id": 2, "event_origin": ["source", "alice", 40]},
        {"turn_index": 6, "branch_id": 2, "event_id": 99},
        {"turn_index": 6, "branch_id": 2},
        {},
    ],
)
def test_compacted_user_resolves_by_evidence_not_position(metadata):
    events = [
        {
            "type": "user_message",
            "content": "older",
            "turn_index": 1,
            "branch_id": 1,
            "event_id": 1,
        },
        {
            "type": "user_message",
            "content": "retained",
            "turn_index": 6,
            "branch_id": 2,
            "event_id": 99,
            EVENT_COORDINATE: ["source", "alice", 40],
        },
    ]
    message = Message(role="user", content="retained", metadata=metadata)
    assert _message_turn(message, events, persisted=True) == 6


@pytest.mark.parametrize(
    "metadata",
    [
        {"turn_index": True},
        {"turn_index": 0},
        {"turn_index": -1},
        {"turn_index": "6"},
        {"turn_index": 6, "branch_id": 9},
        {"turn_index": 6, "event_id": 123},
        {"turn_index": 6, "event_origin": ["foreign", "alice", 99]},
        {},
    ],
)
def test_missing_or_conflicting_user_evidence_never_guesses(metadata):
    events = [
        {
            "type": "user_message",
            "content": "same",
            "turn_index": turn,
            "branch_id": 1,
            "event_id": turn,
        }
        for turn in (1, 6)
    ]
    with pytest.raises(ValueError):
        _message_turn(
            Message(role="user", content="same", metadata=metadata),
            events,
            persisted=True,
        )


@pytest.mark.parametrize("locator", ["index", "turn"])
@pytest.mark.parametrize("duplicate_at", ["target", "future"])
def test_discarded_future_cannot_make_retained_origin_ambiguous(
    tmp_path, locator, duplicate_at
):
    store = SessionStore(str(tmp_path / "future.kohakutr"))
    source = {"backend_key": "new", "backend_type": "openai"}
    try:
        store.append_event(
            "alice",
            "model_switch",
            {
                "schema": 1,
                "classification": "portable",
                "from_source": source,
                "to_source": source,
            },
        )
        messages = []
        contents = (
            ("repeated", "repeated", "later")
            if duplicate_at == "target"
            else ("repeated", "edit here", "repeated")
        )
        for turn, content in enumerate(contents, 1):
            store.append_event(
                "alice",
                "user_message",
                {"content": content},
                turn_index=turn,
                branch_id=1,
                parent_branch_path=[(t, 1) for t in range(1, turn)],
            )
            messages.append(
                Message(
                    role="user",
                    content=content,
                    name="legacy",
                    metadata={"turn_index": turn, "branch_id": 1},
                )
            )
        request = EditRequest(
            message_idx=1,
            new_content="replacement",
            turn_index=2 if locator == "turn" else None,
        )
        prepared = prepare_edit(
            store,
            "alice",
            tuple(messages),
            (3, 1, ((1, 1), (2, 1))),
            ConversationConfig(),
            source,
            request,
        )
        assert [m.content for m in prepared["messages"]] == ["repeated"]
        assert prepared["path"] == ((1, 1),)
        assert prepared["turn"] == 2
        assert prepared["content"] == "replacement"
    finally:
        store.close()


def test_plain_edit_ignores_sibling_native_origin_before_switch(tmp_path):
    store = SessionStore(str(tmp_path / "origins.kohakutr"))
    source = {"backend_key": "new", "backend_type": "openai", "tool_format": "native"}
    calls = [
        {
            "id": "native",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
    ]
    try:
        for branch in (1, 2):
            if branch == 2:
                store.append_event(
                    "alice",
                    "model_switch",
                    {
                        "schema": 1,
                        "classification": "portable",
                        "from_source": {"backend_key": "old", "backend_type": "openai"},
                        "to_source": source,
                    },
                )
            store.append_event(
                "alice",
                "user_message",
                {"content": "question"},
                turn_index=1,
                branch_id=branch,
                parent_branch_path=[],
            )
            store.append_event(
                "alice",
                "assistant_tool_calls",
                {"tool_calls": calls},
                turn_index=1,
                branch_id=branch,
                parent_branch_path=[],
            )
            store.append_event(
                "alice",
                "tool_result",
                {"tool_call_id": "native", "output": "done", "name": "read"},
                turn_index=1,
                branch_id=branch,
                parent_branch_path=[],
            )
        store.append_event(
            "alice",
            "user_message",
            {"content": "next"},
            turn_index=2,
            branch_id=1,
            parent_branch_path=[(1, 2)],
        )
        messages = (
            Message(
                role="user",
                content="question",
                metadata={"turn_index": 1, "branch_id": 2},
            ),
            Message(role="assistant", content="", tool_calls=calls),
            Message(role="tool", content="done", tool_call_id="native", name="read"),
            Message(
                role="user", content="next", metadata={"turn_index": 2, "branch_id": 1}
            ),
        )
        prepared = prepare_edit(
            store,
            "alice",
            messages,
            (2, 1, ((1, 2),)),
            ConversationConfig(),
            source,
            EditRequest(turn_index=2, new_content="edited"),
        )
        assert prepared["path"] == ((1, 2),)
        assert prepared["content"] == "edited"
        assert prepared["messages"][1].tool_calls == calls
        assert prepared["messages"][2].content == "done"
    finally:
        store.close()
