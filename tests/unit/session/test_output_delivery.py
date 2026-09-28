"""Output receipt proof validation survives projection but rejects incomplete facts."""

from copy import deepcopy

import pytest

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.event_origin import capture_events
from kohakuterrarium.session.output_delivery import (
    persist_output_delivery,
    validate_output_source,
)
from kohakuterrarium.session.store import SessionStore


@pytest.mark.parametrize("generation", [{}, {"generation_id": "generation-one"}, None])
def test_incomplete_generation_is_a_compatibility_failure(tmp_path, generation):
    store = SessionStore(tmp_path / "invalid-generation.kohakutr")
    try:
        store.submit(
            store.append_event,
            "alice",
            "assistant_generation_start",
            {"generation_id": "generation-one", "schema": 1, "source": {}},
        ).result()
        with pytest.raises(HistoryCompatibilityError):
            store.submit(
                persist_output_delivery, store, "alice", generation, {}
            ).result()
        assert [
            row["type"] for row in store.submit(store.get_events, "alice").result()
        ] == ["assistant_generation_start"]
    finally:
        store.close(update_status=False)


@pytest.mark.parametrize("change", [None, "payload", "generation", "type", "output_id"])
def test_output_receipt_requires_complete_attempt_and_generation(tmp_path, change):
    store = SessionStore(tmp_path / "outputs.kohakutr")
    generation = {
        "schema": 1,
        "generation_id": "generation-one",
        "source": {"model": "scripted"},
    }
    payload = {
        "output_id": "output-one",
        "target": "notice",
        "content": "body",
        "success": True,
        "error": None,
        "completed_at": "2026-01-01T00:00:00",
        "feedback_line": "- [notice]: body",
    }
    try:
        store.submit(
            store.append_event,
            "alice",
            "assistant_generation_start",
            deepcopy(generation),
            turn_index=1,
            branch_id=1,
            parent_branch_path=[],
        ).result()
        proof = store.submit(
            persist_output_delivery, store, "alice", generation, payload
        ).result()
        _, captured = store.submit(capture_events, store).result()
        index = {ref: row for name, _, row, ref in captured if name == "alice"}
        if change == "payload":
            proof["payload"] = {}
        elif change == "generation":
            index[tuple(proof["payload"]["generation_origin"])]["source"] = {
                "model": "foreign"
            }
        elif change == "type":
            index[tuple(proof["event_origin"])]["type"] = "tool_result"
        elif change == "output_id":
            proof["payload"].pop("output_id")
        if change:
            with pytest.raises(HistoryCompatibilityError):
                validate_output_source(proof, index)
        else:
            assert validate_output_source(proof, index) == tuple(proof["event_origin"])
            row = index[tuple(proof["event_origin"])]
            row.update(type="feedback_source", feedback_source_type="output_delivery")
            assert validate_output_source(proof, index) == tuple(proof["event_origin"])
    finally:
        store.close(update_status=False)
