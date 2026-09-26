"""Strict selected-generation links before any normalization consumer."""

from copy import deepcopy

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.generation_validation import (
    validate_selected_generation_events,
)


def origin(generation_id="g1", **source):
    return {
        "event_id": 2,
        "type": "assistant_generation_start",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "schema": 1,
        "generation_id": generation_id,
        "source": {
            "llm_identifier": None,
            "backend_key": None,
            "backend_type": None,
            "provider_name": None,
            "model": "selected",
            "tool_format": "native",
            **source,
        },
    }


def linked(kind="text", generation_id="g1", **extra):
    return {
        "event_id": 3,
        "type": kind,
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
        "generation_id": generation_id,
        "content": "answer",
        **extra,
    }


def test_valid_selected_origin_and_link_passes_without_mutation():
    events = [origin(), linked()]
    before = deepcopy(events)
    assert validate_selected_generation_events(events) is None
    assert events == before


def test_blank_generation_id_rejected_without_uuid_shape_check():
    events = [origin("  ")]
    with pytest.raises(HistoryCompatibilityError, match="metadata"):
        validate_selected_generation_events(events)
    assert (
        validate_selected_generation_events(
            [origin("not-a-uuid"), linked("text", "not-a-uuid")]
        )
        is None
    )


@pytest.mark.parametrize(
    "broken",
    [
        {"schema": True},
        {"schema": 2},
        {"generation_id": None},
        {"source": {"secret": "leak"}},
        {"source": {"model": "  "}},
        {"source": "native"},
    ],
)
def test_malformed_origin_fails_closed(broken):
    start = origin()
    if "source" in broken and not isinstance(broken["source"], dict):
        start["source"] = broken["source"]
    elif "source" in broken:
        start["source"].update(broken["source"])
    else:
        start.update(broken)
    with pytest.raises(HistoryCompatibilityError):
        validate_selected_generation_events([start, linked()])


def test_duplicate_selected_origin_rejected():
    with pytest.raises(HistoryCompatibilityError, match="Duplicate"):
        validate_selected_generation_events([origin(), origin(generation_id="g1")])


@pytest.mark.parametrize(
    "kind",
    [
        "text",
        "text_chunk",
        "assistant_reasoning",
        "assistant_image",
        "assistant_tool_calls",
    ],
)
def test_missing_link_after_active_origin_rejected(kind):
    event = linked(kind)
    del event["generation_id"]
    with pytest.raises(HistoryCompatibilityError, match="Missing"):
        validate_selected_generation_events([origin(), event])


def test_legacy_assistant_without_origin_still_unlinked():
    event = linked()
    del event["generation_id"]
    assert validate_selected_generation_events([event]) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event.update(generation_id="missing"),
        lambda event: event.update(generation_id=42),
        lambda event: event.update(turn_index=9),
        lambda event: event.update(branch_id=2),
        lambda event: event.update(parent_branch_path=[(9, 1)]),
        lambda event: event.update(event_id=1),
    ],
)
def test_invalid_selected_link_rejected(mutate):
    event = linked()
    mutate(event)
    with pytest.raises(
        HistoryCompatibilityError, match="Invalid selected generation link"
    ):
        validate_selected_generation_events([origin(), event])


def test_user_boundary_clears_active_origin_before_legacy_text():
    events = [
        origin(),
        {
            "event_id": 3,
            "type": "user_message",
            "turn_index": 2,
            "branch_id": 1,
            "content": "next",
        },
        {
            "event_id": 4,
            "type": "text",
            "turn_index": 2,
            "branch_id": 1,
            "content": "legacy",
        },
    ]
    assert validate_selected_generation_events(events) is None
