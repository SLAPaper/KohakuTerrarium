"""Feedback admissions validate terminal constituents before hiding source rows."""

from copy import deepcopy

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.feedback_admission import (
    persist_feedback,
    project_feedback_events,
)
from kohakuterrarium.session.event_origin import EVENT_ORIGIN, capture_events
from kohakuterrarium.session.history_support import compact_replay_events
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.raw_history import append_user_event_pair
from kohakuterrarium.session.history_support import dedupe_adjacent_duplicate_events


def test_invalid_distinct_sources_does_not_append_injected_constituents(tmp_path):
    store = SessionStore(tmp_path / "duplicate-input.kohakutr")
    origin = {
        "runtime_scope_id": "scope",
        "delivery_id": "scope:job",
        "execution_class": "background",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
    }
    try:
        store.submit(
            store.append_event,
            "alice",
            "tool_call",
            {"job_id": "job", "origin": origin},
        ).result()
        store.submit(capture_events, store).result()
        before = store.submit(store.get_events, "alice").result()
        dispatch = {"mode": "dispatch_ack", "job_id": "job", "origin": origin}
        with pytest.raises(HistoryCompatibilityError, match="distinct"):
            store.submit(
                persist_feedback,
                store,
                "alice",
                "bad",
                [
                    dispatch,
                    dispatch,
                    {
                        "mode": "injected_input",
                        "input_type": "user_input",
                        "payload": {"content": "must not persist"},
                    },
                ],
                1,
                1,
                [],
            ).result()
        store.submit(store.flush).result()
        assert store.submit(store.get_events, "alice").result() == before
    finally:
        store.close(update_status=False)


def test_compact_feedback_projection_rejects_conflicting_canonical_duplicates(tmp_path):
    store = SessionStore(tmp_path / "conflicting-source.kohakutr")
    origin = {
        "runtime_scope_id": "scope",
        "delivery_id": "scope:job",
        "execution_class": "background",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
    }
    try:
        store.submit(
            store.append_event,
            "alice",
            "tool_call",
            {"job_id": "job", "origin": origin, "args": "original"},
        ).result()
        store.submit(
            persist_feedback,
            store,
            "alice",
            "ack",
            [{"mode": "dispatch_ack", "job_id": "job", "origin": origin}],
            1,
            1,
            [],
        ).result()
        _, captured = store.submit(capture_events, store).result()
        rows = [
            {**row, EVENT_ORIGIN: list(ref)}
            for name, _, row, ref in captured
            if name == "alice"
        ]
        duplicate = {**deepcopy(rows[0]), "args": "conflicting"}
        with pytest.raises(HistoryCompatibilityError, match="Conflicting"):
            compact_replay_events([duplicate, *rows], [], {1: 1})
    finally:
        store.close(update_status=False)


def test_unproven_dispatch_projection_is_rejected(tmp_path):
    store = SessionStore(tmp_path / "bad-projection.kohakutr")
    try:
        store.submit(
            store.append_event,
            "alice",
            "background_dispatch_result",
            {"output": "invented terminal"},
        ).result()
        rows = store.submit(store.get_events, "alice").result()
        with pytest.raises(HistoryCompatibilityError):
            project_feedback_events(rows)
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("fault", ["origin", "scope", "duplicate", "delivery"])
def test_dispatch_admission_rejects_malformed_or_duplicate_proof(tmp_path, fault):
    store = SessionStore(tmp_path / "dispatch-proof.kohakutr")
    origin = {
        "runtime_scope_id": "scope",
        "delivery_id": "scope:job",
        "execution_class": "background",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
    }
    try:
        store.submit(
            store.append_event,
            "alice",
            "tool_call",
            {"job_id": "job", "origin": origin},
        ).result()
        store.submit(
            persist_feedback,
            store,
            "alice",
            "acknowledged",
            [{"mode": "dispatch_ack", "job_id": "job", "origin": origin}],
            1,
            1,
            [],
        ).result()
        rows = store.submit(store.get_events, "alice").result()
        if fault == "duplicate":
            payload = {
                key: deepcopy(rows[-1][key])
                for key in (
                    "content",
                    "deliveries",
                    "source_event_origins",
                    "message_kind",
                )
            }
            store.submit(
                store.append_event,
                "alice",
                "tool_feedback_admitted",
                payload,
                turn_index=1,
                branch_id=1,
                parent_branch_path=[],
            ).result()
            rows = store.submit(store.get_events, "alice").result()
        else:
            rows = [rows[-1]]
            proof = rows[0]["deliveries"][0]
            if fault == "delivery":
                rows[0]["deliveries"] = ["bad constituent"]
            elif fault == "origin":
                proof.pop("origin")
            else:
                proof["origin"].pop("runtime_scope_id")
        with pytest.raises(HistoryCompatibilityError):
            project_feedback_events(rows)
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize(
    "change", [None, "swap", "missing", "pair", "content", "branch"]
)
def test_input_pair_admission_preserves_both_origins(tmp_path, projected, change):
    store = SessionStore(tmp_path / "pair-feedback.kohakutr")
    try:
        payload = {"content": "same content"}
        identity = store.submit(
            append_user_event_pair,
            store,
            "alice",
            payload,
            2,
            1,
            [(1, 1)],
            with_receipt=True,
        ).result()
        store.submit(
            persist_feedback,
            store,
            "alice",
            "same content",
            [{"mode": "user_pair", **identity, "payload": payload}],
            2,
            1,
            [[1, 1]],
        ).result()
        rows = store.submit(store.get_events, "alice").result()
        if projected:
            rows = dedupe_adjacent_duplicate_events(project_feedback_events(rows))
        proof = rows[-1]["deliveries"][0]
        if change == "swap":
            proof["event_origin"], proof["user_pair_origin"] = (
                proof["user_pair_origin"],
                proof["event_origin"],
            )
        elif change == "missing":
            rows[-1]["source_event_origins"].remove(proof["user_pair_origin"])
        elif change == "pair":
            proof["user_pair_id"] = "foreign pair"
        elif change == "content":
            rows[0]["content"] = "other input"
        elif change == "branch":
            rows[0]["branch_id"] = 99
        if change:
            with pytest.raises(HistoryCompatibilityError):
                project_feedback_events(rows)
        else:
            result = dedupe_adjacent_duplicate_events(project_feedback_events(rows))
            assert len(result) == 3
            assert result[0]["feedback_source_type"] == "user_input"
            assert result[1]["feedback_source_type"] == "user_message"
            assert (
                result[0]["user_pair_id"]
                == result[1]["user_pair_id"]
                == identity["user_pair_id"]
            )
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("change", [None, "origin", "type", "duplicate", "provider"])
def test_admission_does_not_hide_conflicting_source(tmp_path, change, background):
    store = SessionStore(tmp_path / "feedback.kohakutr")
    origin = {
        "runtime_scope_id": "scope",
        "delivery_id": "scope:job",
        "execution_class": "background" if background else "direct",
        "turn_index": 1,
        "branch_id": 1,
        "parent_branch_path": [],
    }
    try:
        for kind in ("tool_call", "tool_result"):
            store.submit(
                store.append_event,
                "alice",
                kind,
                {
                    "job_id": "job",
                    "call_id": "job",
                    **({"tool_call_id": "provider"} if background else {}),
                    "origin": deepcopy(origin),
                    "output": "result",
                },
            ).result()
        store.submit(
            persist_feedback,
            store,
            "alice",
            "feedback",
            [
                {
                    "job_id": "job",
                    "origin": origin,
                    **(
                        {"mode": "background_terminal", "tool_call_id": "provider"}
                        if background
                        else {}
                    ),
                }
            ],
            1,
            1,
            [],
        ).result()
        rows = store.submit(store.get_events, "alice").result()
        before = deepcopy(rows)
        source = rows[1] if background else rows[0]
        if change == "origin":
            source["origin"]["delivery_id"] = "foreign"
        elif change == "type":
            source["type"] = "user_message"
        elif change == "provider":
            source["tool_call_id"] = "foreign-provider"
        elif change == "duplicate":
            rows[-1]["source_event_origins"].append(rows[-1]["source_event_origins"][0])
        if change:
            with pytest.raises(HistoryCompatibilityError):
                project_feedback_events(rows)
        else:
            projected = project_feedback_events(rows)
            assert [row["type"] for row in projected] == [
                "tool_call" if background else "feedback_source",
                "feedback_source",
                "tool_feedback_admitted",
            ]
            assert project_feedback_events(projected) == projected
            assert rows == before
    finally:
        store.close(update_status=False)
