from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from kohakuterrarium.core.branch_rerun import _legacy_baseline, _persist_rerun
from kohakuterrarium.llm.message import Message
from kohakuterrarium.session.event_origin import (
    EVENT_ORIGIN,
    append_imported,
    store_uuid,
)
from kohakuterrarium.session.history import replay_conversation
from kohakuterrarium.session.raw_history import append_user_event_pair
from kohakuterrarium.session.store import SessionStore


@pytest.mark.parametrize("versioned", [False, True])
def test_replayed_repeated_legacy_summary_retains_its_specific_interval(
    store, versioned
):
    coverage = (
        {"compact_coverage_version": 1, "coverage_mode": "legacy_fallback"}
        if versioned
        else {}
    )
    ids = []
    for turn in (1, 2, 3):
        ids.append(
            append_user_event_pair(
                store,
                "alice",
                {"content": f"user{turn}"},
                turn,
                1,
                [(t, 1) for t in range(1, turn)],
            )
        )
        if turn >= 2:
            store.append_event(
                "alice",
                "compact_complete",
                {
                    **coverage,
                    "summary": "SAME SUMMARY",
                    "round": turn - 1,
                    "replaced_from_event_id": 1,
                    "replaced_to_event_id": ids[-2],
                    "compact_path": [(t, 1) for t in range(1, turn + 1)],
                },
                turn_index=turn,
                branch_id=1,
                parent_branch_path=[(t, 1) for t in range(1, turn)],
            )
    view = {1: 1, 2: 1, 3: 1}
    messages = replay_conversation(
        store.get_events("alice"), branch_view=view, include_metadata=True
    )
    summary = next(m for m in messages if m["content"] == "SAME SUMMARY")
    assert summary["metadata"]["compact_coverage"]["replaced_to_event_id"] == ids[1]
    retained = Message(
        role="assistant", content=summary["content"], metadata=summary["metadata"]
    )
    _persist_rerun(
        store,
        append_user_event_pair,
        "alice",
        "edited",
        3,
        2,
        [(1, 1), (2, 1)],
        [retained],
        view,
    )
    replayed = replay_conversation(
        store.get_events("alice"), branch_view={1: 1, 2: 1, 3: 2}, include_metadata=True
    )
    assert [(m["role"], m["content"]) for m in replayed] == [
        ("assistant", "SAME SUMMARY"),
        ("user", "edited"),
    ]


@pytest.fixture
def store(tmp_path: Path):
    s = SessionStore(tmp_path / "test_baseline.kohakutr")
    store_uuid(s)
    try:
        yield s
    finally:
        s.close()


def test_legacy_baseline_selected_branch_unaffected_by_other_branch_different_range(
    store,
):
    prefix = "alice"
    append_user_event_pair(store, prefix, {"content": "t0"}, 0, 0, [])
    append_user_event_pair(store, prefix, {"content": "t1-b1"}, 1, 1, [(0, 0)])
    store.append_event(
        prefix,
        "compact_complete",
        {
            "summary": "shared_summary",
            "round": 1,
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 5,
        },
        turn_index=1,
        branch_id=1,
        parent_branch_path=[(0, 0)],
    )

    append_user_event_pair(store, prefix, {"content": "t1-b2"}, 1, 2, [(0, 0)])
    store.append_event(
        prefix,
        "compact_complete",
        {
            "summary": "shared_summary",
            "round": 1,
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 20,
        },
        turn_index=1,
        branch_id=2,
        parent_branch_path=[(0, 0)],
    )

    cov = {"compact_coverage_version": 1, "coverage_mode": "legacy_fallback"}
    view_b1 = {0: 0, 1: 1}
    view_b2 = {0: 0, 1: 2}

    msg_raw = Message(role="assistant", content="shared_summary")
    res_b1 = _legacy_baseline(store, prefix, view_b1, msg_raw, cov)
    assert res_b1 == {"replaced_from_event_id": 1, "replaced_to_event_id": 5}

    rendered = "[Previous context summary (compact round 1)]\n\nshared_summary"
    msg_rend = Message(role="assistant", content=rendered)
    res_rend = _legacy_baseline(store, prefix, view_b1, msg_rend, cov)
    assert res_rend == {"replaced_from_event_id": 1, "replaced_to_event_id": 5}

    res_b2 = _legacy_baseline(store, prefix, view_b2, msg_raw, cov)
    assert res_b2 == {"replaced_from_event_id": 1, "replaced_to_event_id": 20}

    msg_raw.metadata = {"compact_coverage": cov}
    user_receipt = _persist_rerun(
        store,
        append_user_event_pair,
        prefix,
        "rerun_b1",
        turn=1,
        branch=3,
        path=[(0, 0)],
        retained=[msg_raw],
        source_view=view_b1,
    )
    assert isinstance(user_receipt, dict)
    events = store.get_events(prefix)
    b3_replaces = [
        e
        for e in events
        if e.get("type") == "compact_replace" and e.get("branch_id") == 3
    ]
    assert len(b3_replaces) == 1
    assert b3_replaces[0]["replaced_from_event_id"] == 1
    assert b3_replaces[0]["replaced_to_event_id"] == 5


def test_legacy_baseline_imported_event_origin_raises_value_error_without_upgrade(
    store,
):
    prefix = "alice"
    append_user_event_pair(store, prefix, {"content": "t0"}, 0, 0, [])
    append_user_event_pair(store, prefix, {"content": "t1-b1"}, 1, 1, [(0, 0)])

    imported_origin = [str(uuid4()), prefix, 99]
    append_imported(
        store,
        prefix,
        "compact_complete",
        {
            "summary": "imported_summary",
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 5,
            "turn_index": 1,
            "branch_id": 1,
            "parent_branch_path": [(0, 0)],
            EVENT_ORIGIN: imported_origin,
        },
    )

    cov = {"compact_coverage_version": 1, "coverage_mode": "legacy_fallback"}
    view = {0: 0, 1: 1}
    msg = Message(
        role="assistant",
        content="imported_summary",
        metadata={"compact_coverage": cov},
    )

    with pytest.raises(
        ValueError, match="Retained legacy summary has no unique local baseline"
    ):
        _legacy_baseline(store, prefix, view, msg, cov)

    events_before = deepcopy(store.get_events(prefix))
    with pytest.raises(
        ValueError, match="Retained legacy summary has no unique local baseline"
    ):
        _persist_rerun(
            store,
            append_user_event_pair,
            prefix,
            "rerun_attempt",
            turn=1,
            branch=2,
            path=[(0, 0)],
            retained=[msg],
            source_view=view,
        )

    events_after = store.get_events(prefix)
    assert events_after == events_before
    assert not any(e.get("coverage_mode") == "exact" for e in events_after)


def test_persist_rerun_unmatched_summary_raises_value_error_before_writes(
    store,
):
    prefix = "alice"
    append_user_event_pair(store, prefix, {"content": "t0"}, 0, 0, [])
    append_user_event_pair(store, prefix, {"content": "t1-b1"}, 1, 1, [(0, 0)])
    store.append_event(
        prefix,
        "compact_complete",
        {
            "summary": "known_summary",
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 5,
        },
        turn_index=1,
        branch_id=1,
        parent_branch_path=[(0, 0)],
    )

    cov = {"compact_coverage_version": 1, "coverage_mode": "legacy_fallback"}
    view = {0: 0, 1: 1}
    unmatched_msg = Message(
        role="assistant",
        content="unmatched_summary",
        metadata={"compact_coverage": cov},
    )

    events_before = deepcopy(store.get_events(prefix))
    with pytest.raises(
        ValueError, match="Retained legacy summary has no unique local baseline"
    ):
        _persist_rerun(
            store,
            append_user_event_pair,
            prefix,
            "rerun_prompt",
            turn=1,
            branch=2,
            path=[(0, 0)],
            retained=[unmatched_msg],
            source_view=view,
        )

    events_after = store.get_events(prefix)
    assert events_after == events_before
    assert not any(e.get("content") == "rerun_prompt" for e in events_after)


def test_persist_rerun_ambiguous_summary_in_selected_branch_raises_value_error_before_writes(
    store,
):
    prefix = "alice"
    append_user_event_pair(store, prefix, {"content": "t0"}, 0, 0, [])
    append_user_event_pair(store, prefix, {"content": "t1-b1"}, 1, 1, [(0, 0)])
    store.append_event(
        prefix,
        "compact_complete",
        {
            "summary": "ambiguous_summary",
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 5,
        },
        turn_index=1,
        branch_id=1,
        parent_branch_path=[(0, 0)],
    )
    store.append_event(
        prefix,
        "compact_complete",
        {
            "summary": "ambiguous_summary",
            "compact_coverage_version": 1,
            "coverage_mode": "legacy_fallback",
            "replaced_from_event_id": 1,
            "replaced_to_event_id": 10,
        },
        turn_index=1,
        branch_id=1,
        parent_branch_path=[(0, 0)],
    )

    cov = {"compact_coverage_version": 1, "coverage_mode": "legacy_fallback"}
    view = {0: 0, 1: 1}
    ambiguous_msg = Message(
        role="assistant",
        content="ambiguous_summary",
        metadata={"compact_coverage": cov},
    )

    events_before = deepcopy(store.get_events(prefix))
    with pytest.raises(
        ValueError, match="Retained legacy summary has no unique local baseline"
    ):
        _persist_rerun(
            store,
            append_user_event_pair,
            prefix,
            "rerun_prompt",
            turn=1,
            branch=2,
            path=[(0, 0)],
            retained=[ambiguous_msg],
            source_view=view,
        )

    events_after = store.get_events(prefix)
    assert events_after == events_before
    assert not any(e.get("content") == "rerun_prompt" for e in events_after)


def test_persist_rerun_exact_baseline_preserves_origins_anchor_and_branch_isolation(
    store,
):
    prefix = "alice"
    append_user_event_pair(store, prefix, {"content": "t0"}, 0, 0, [])
    append_user_event_pair(store, prefix, {"content": "t1-b1"}, 1, 1, [(0, 0)])

    origin_1 = [str(uuid4()), prefix, 1]
    origin_2 = [str(uuid4()), prefix, 2]
    exact_cov = {
        "compact_coverage_version": 1,
        "coverage_mode": "exact",
        "replaced_event_origins": [origin_1, origin_2],
        "summary_anchor_origin": origin_1,
        "coverage_reason": "auto_compact",
    }
    exact_msg = Message(
        role="assistant",
        content="exact_summary_content",
        metadata={"compact_coverage": exact_cov},
    )

    source_events_before = deepcopy(store.get_events(prefix))
    view = {0: 0, 1: 1}

    user_receipt = _persist_rerun(
        store,
        append_user_event_pair,
        prefix,
        "exact_rerun_prompt",
        turn=1,
        branch=2,
        path=[(0, 0)],
        retained=[exact_msg],
        source_view=view,
    )
    assert isinstance(user_receipt, dict)

    events_after = store.get_events(prefix)
    # Source events remain unmodified
    assert events_after[: len(source_events_before)] == source_events_before

    new_events = events_after[len(source_events_before) :]
    user_inputs = [e for e in new_events if e.get("type") == "user_input"]
    user_messages = [e for e in new_events if e.get("type") == "user_message"]
    compact_replaces = [e for e in new_events if e.get("type") == "compact_replace"]

    assert len(user_inputs) == 1
    assert len(user_messages) == 1
    assert len(compact_replaces) == 1

    cr = compact_replaces[0]
    assert cr["turn_index"] == 1
    assert cr["branch_id"] == 2
    assert cr["parent_branch_path"] == [[0, 0]]
    assert cr["compact_path"] == [[0, 0], [1, 2]]
    assert cr["summary"] == "exact_summary_content"
    assert cr["replaced_event_origins"] == [origin_1, origin_2]
    assert cr["summary_anchor_origin"] == origin_1
    assert cr["coverage_mode"] == "exact"
    assert cr["compact_coverage_version"] == 1
    assert "replaced_from_event_id" not in cr
    assert "replaced_to_event_id" not in cr

    # Verify no same-origin rewrite or duplicate event IDs
    all_event_ids = [e["event_id"] for e in events_after]
    assert len(all_event_ids) == len(set(all_event_ids))
