"""Canonical compact rules retain holes and stream-order summary anchors."""

from copy import deepcopy
from uuid import uuid4

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.compact_coverage import exact_origins
from kohakuterrarium.session.event_origin import EVENT_ORIGIN
from kohakuterrarium.session.history_replay import replay_selected_events
from kohakuterrarium.session.history_support import compact_replay_events
from kohakuterrarium.session.resume_replay import replay_resume_history


def history():
    uid = str(uuid4())

    def origin(value):
        return [uid, "alice", value]

    rows = [
        {
            "event_id": 80,
            "type": "user_message",
            "content": "covered first",
            EVENT_ORIGIN: origin(1),
        },
        {
            "event_id": 10,
            "type": "user_message",
            "content": "uncovered hole",
            EVENT_ORIGIN: origin(2),
        },
        {
            "event_id": 30,
            "type": "text_chunk",
            "content": "covered last",
            EVENT_ORIGIN: origin(3),
        },
    ]
    compact = {
        "event_id": 2,
        "type": "compact_complete",
        "summary": "summary",
        "compact_coverage_version": 1,
        "coverage_mode": "exact",
        "replaced_event_origins": [origin(1), origin(3)],
        "summary_anchor_origin": origin(1),
        "replaced_from_event_id": 1,
        "replaced_to_event_id": 100,
    }
    return rows, compact


def test_nonmonotonic_coverage_preserves_hole_and_payload():
    rows, compact = history()
    events = [*rows, compact]
    before = deepcopy(events)
    assert replay_selected_events(events) == [
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "uncovered hole"},
    ]
    assert events == before
    summary = replay_selected_events(events, include_metadata=True)[0]
    assert (
        summary["metadata"]["compact_coverage"]["replaced_event_origins"]
        == compact["replaced_event_origins"]
    )


def test_summary_anchor_can_be_retained_user_outside_membership():
    rows, compact = history()
    compact["replaced_event_origins"] = [rows[2][EVENT_ORIGIN]]
    assert replay_selected_events([*rows, compact]) == [
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "covered first"},
        {"role": "user", "content": "uncovered hole"},
    ]


@pytest.mark.parametrize("anchor", ["missing", "dead_branch"])
def test_summary_uses_first_visible_member_when_anchor_is_unavailable(anchor):
    rows, compact = history()
    if anchor == "missing":
        compact["summary_anchor_origin"] = [str(uuid4()), "alice", 999]
    else:
        rows[0].update(turn_index=1, branch_id=1, parent_branch_path=[])
        rows[2].update(turn_index=1, branch_id=2, parent_branch_path=[])
    replay = replay_selected_events(
        [*rows, compact], branch_view={1: 2} if anchor == "dead_branch" else None
    )
    expected = [
        {"role": "assistant", "content": "summary"},
        {"role": "user", "content": "uncovered hole"},
    ]
    assert replay == (list(reversed(expected)) if anchor == "dead_branch" else expected)


@pytest.mark.parametrize("versioned", [False, True])
def test_imported_legacy_interval_cannot_erase_destination_coordinates(
    versioned, caplog
):
    rows, compact = history()
    for key in (
        "compact_coverage_version",
        "coverage_mode",
        "replaced_event_origins",
        "summary_anchor_origin",
    ):
        compact.pop(key)
    if versioned:
        compact.update(
            compact_coverage_version=1,
            coverage_mode="legacy_fallback",
            coverage_reason="missing_user_receipt",
        )
    compact[EVENT_ORIGIN] = [str(uuid4()), "alice", 100]
    assert replay_selected_events([*rows, compact]) == [
        {"role": "user", "content": "covered first"},
        {"role": "user", "content": "uncovered hole"},
        {"role": "assistant", "content": "covered last"},
    ]
    assert "Imported compact interval has no canonical coverage" in caplog.text


def test_empty_exact_rule_does_not_replace_previous_summary():
    rows, compact = history()
    empty = {
        **compact,
        "event_id": 100,
        "summary": "not a baseline",
        "replaced_event_origins": [],
        "summary_anchor_origin": None,
    }
    assert replay_selected_events([*rows, compact, empty])[0]["content"] == "summary"
    assert exact_origins(empty) == frozenset()


@pytest.mark.parametrize(
    "value", [None, {}, [None], [["bad", "alice", 1]], [[str(uuid4()), "alice", True]]]
)
def test_malformed_exact_membership_never_falls_back_to_envelope(value):
    rows, compact = history()
    compact["replaced_event_origins"] = value
    with pytest.raises(HistoryCompatibilityError):
        replay_selected_events([*rows, compact])


def test_duplicate_origin_rejected_and_latest_stream_rule_wins():
    rows, compact = history()
    latest = {**compact, "event_id": 1, "summary": "newest"}
    assert replay_selected_events([*rows, compact, latest])[0]["content"] == "newest"
    latest["replaced_event_origins"] = [rows[0][EVENT_ORIGIN]] * 2
    with pytest.raises(HistoryCompatibilityError):
        replay_selected_events([*rows, latest])


def test_ordinary_cold_replay_does_not_resurrect_covered_synthetic_calls():
    rows, compact = history()
    rows[0].update(type="tool_call", call_id="job", name="work", args={})
    rows[2].update(type="tool_result", call_id="job", name="work", output="result")
    messages = replay_resume_history([*rows, compact])
    assert [(message["role"], message["content"]) for message in messages] == [
        ("assistant", "summary"),
        ("user", "uncovered hole"),
    ]
    assert not any(message.get("tool_calls") for message in messages)


def test_native_mask_uses_membership_not_enclosing_interval():
    rows, compact = history()
    rows[0].update(type="tool_call", call_id="covered")
    rows[1].update(type="tool_result", call_id="hole")
    projected = compact_replay_events(rows, [compact], {})
    assert projected[0]["type"] == "text_chunk"
    assert projected[1] == rows[1]
