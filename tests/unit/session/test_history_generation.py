"""RED provenance replay contracts; no conversion or compatibility fence."""

from uuid import uuid4

import pytest

from kohakuterrarium.session.history import replay_conversation


def origin(model, mode="native"):
    return {
        "schema": 1,
        "generation_id": str(uuid4()),
        "source": {
            "llm_identifier": None,
            "backend_key": None,
            "backend_type": None,
            "provider_name": None,
            "model": model,
            "tool_format": mode,
        },
    }


def event(index, kind, *, branch=1, **payload):
    return {
        "event_id": index,
        "type": kind,
        "turn_index": 1,
        "branch_id": branch,
        **payload,
    }


@pytest.mark.parametrize("include_metadata", [False, True])
def test_adjacent_text_then_zero_text_native_generation_never_coalesces(
    include_metadata,
):
    first, second = origin("old", "bracket"), origin("new")
    call = {
        "id": "native-1",
        "type": "function",
        "function": {"name": "inspect", "arguments": "{}"},
    }
    events = [
        event(1, "assistant_generation_start", **first),
        event(
            2,
            "text_chunk",
            content="earlier text",
            generation_id=first["generation_id"],
        ),
        event(3, "assistant_generation_start", **second),
        event(
            4,
            "assistant_tool_calls",
            content="",
            tool_calls=[call],
            generation_id=second["generation_id"],
        ),
        event(5, "tool_result", name="inspect", call_id="native-1", output="done"),
    ]
    messages = replay_conversation(events, include_metadata=include_metadata)
    assert [(m["role"], m["content"]) for m in messages] == [
        ("assistant", "earlier text"),
        ("assistant", ""),
        ("tool", "done"),
    ], "a zero-text response must not steal the preceding generation's text"
    assert "tool_calls" not in messages[0]
    assert messages[1]["tool_calls"] == [call]
    assert messages[2]["tool_call_id"] == "native-1"
    if include_metadata:
        assert [m["metadata"]["generation"] for m in messages[:2]] == [first, second]
    else:
        assert all("metadata" not in m for m in messages)


@pytest.mark.parametrize(
    "sibling", [{"schema": 99}, {"generation_id": "not-a-uuid"}, None]
)
def test_sibling_start_cannot_rebind_selected_branch_origin(sibling):
    selected, rejected = origin("selected"), origin("sibling")
    if sibling is not None:
        rejected.update(sibling)
    events = [
        event(1, "user_message", content="question"),
        event(2, "assistant_generation_start", **selected),
        event(3, "user_message", branch=2, content="sibling question"),
        event(4, "assistant_generation_start", branch=2, **rejected),
        event(
            5,
            "text_chunk",
            content="selected answer",
            generation_id=selected["generation_id"],
        ),
    ]
    messages = replay_conversation(events, branch_view={1: 1}, include_metadata=True)
    assert [m["content"] for m in messages] == ["question", "selected answer"]
    assert messages[1].get("metadata", {}).get("generation") == selected


@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("preceding_text", [False, True])
def test_linked_reasoning_only_materializes_one_response(
    include_metadata, preceding_text
):
    first, second = origin("first"), origin("reasoner")
    extras = {
        "reasoning_content": "private thought",
        "reasoning_summary": "summary",
        "reasoning_details": [{"type": "encrypted", "data": "opaque"}],
        "reasoning": {"signature": "signed"},
        "_kt_assistant_segments": [{"type": "reasoning", "text": "thought"}],
    }
    events = [event(1, "assistant_generation_start", **first)]
    if preceding_text:
        events.append(
            event(2, "text", content="prior", generation_id=first["generation_id"])
        )
    events.extend(
        [
            event(3, "assistant_generation_start", **second),
            event(
                4,
                "assistant_reasoning",
                generation_id=second["generation_id"],
                **extras,
            ),
            event(
                5,
                "assistant_reasoning",
                generation_id=second["generation_id"],
                **extras,
            ),
            event(6, "processing_end"),
        ]
    )
    messages = replay_conversation(events, include_metadata=include_metadata)
    assert len(messages) == 1 + preceding_text
    expected = {"role": "assistant", "content": "", **extras}
    if include_metadata:
        expected["metadata"] = {"generation": second}
    assert messages[-1] == expected
    if preceding_text:
        assert messages[0]["content"] == "prior"
        assert not set(extras).intersection(messages[0])


@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("link", [None, "missing", 42, [], {}])
def test_invalid_reasoning_link_cannot_steal_previous_message(include_metadata, link):
    known = origin("known")
    events = [
        event(1, "assistant_generation_start", **known),
        event(2, "text", content="prior", generation_id=known["generation_id"]),
        event(3, "assistant_reasoning", generation_id=link, reasoning_content="wrong"),
    ]
    messages = replay_conversation(events, include_metadata=include_metadata)
    assert len(messages) == 1
    assert "reasoning_content" not in messages[0]


@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("linked", [False, True])
def test_reasoning_attaches_to_existing_text_and_preserves_legacy(
    include_metadata, linked
):
    known = origin("known")
    link = {"generation_id": known["generation_id"]} if linked else {}
    events = [event(1, "assistant_generation_start", **known)] if linked else []
    events.extend(
        [
            event(2, "text", content="answer", **link),
            event(3, "assistant_reasoning", reasoning_content="thought", **link),
        ]
    )
    messages = replay_conversation(events, include_metadata=include_metadata)
    assert len(messages) == 1
    assert messages[0]["content"] == "answer"
    assert messages[0]["reasoning_content"] == "thought"
    assert replay_conversation([events[-1]], include_metadata=include_metadata) == []


@pytest.mark.parametrize("include_metadata", [False, True])
def test_bare_start_creates_no_message_and_legacy_remains_unknown(include_metadata):
    assert (
        replay_conversation(
            [event(1, "assistant_generation_start", **origin("unused"))],
            include_metadata=include_metadata,
        )
        == []
    )
    messages = replay_conversation(
        [event(1, "text_chunk", content="legacy")], include_metadata=True
    )
    assert messages == [{"role": "assistant", "content": "legacy"}]
