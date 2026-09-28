"""Canonical receipts for completed named-output attempts and their generation."""

from copy import deepcopy

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.compact_coverage import parse_origin
from kohakuterrarium.session.event_origin import capture_events


def validate_output_source(proof, index):
    """Resolve one output attempt without treating its generation as covered content."""
    ref = parse_origin(proof.get("event_origin"))
    row = index.get(ref)
    payload = proof.get("payload")
    if (
        not isinstance(payload, dict)
        or not {
            "output_id",
            "target",
            "content",
            "success",
            "error",
            "completed_at",
            "feedback_line",
            "generation_id",
            "generation_origin",
            "source",
        }.issubset(payload)
        or not isinstance(payload.get("output_id"), str)
        or not payload["output_id"]
        or type(payload.get("success")) is not bool
        or row is None
        or (
            row.get("feedback_source_type")
            if row.get("type") == "feedback_source"
            else row.get("type")
        )
        != "output_delivery"
    ):
        raise HistoryCompatibilityError(
            f"Output delivery receipt missing or wrong type: {row.get('type') if row else None}"
        )
    changed = [key for key, value in payload.items() if row.get(key) != value]
    if changed:
        raise HistoryCompatibilityError(f"Output delivery receipt conflicts: {changed}")
    generation = index.get(parse_origin(row.get("generation_origin")))
    if (
        generation is None
        or generation.get("type") != "assistant_generation_start"
        or generation.get("generation_id") != row.get("generation_id")
        or generation.get("source") != row.get("source")
    ):
        raise HistoryCompatibilityError("Output generation receipt conflicts")
    for key in ("turn_index", "branch_id", "parent_branch_path"):
        if generation.get(key) != row.get(key):
            raise HistoryCompatibilityError("Output delivery lineage conflicts")
    return ref


def persist_output_delivery(store, namespace, generation, payload):
    """Confirm a completed attempt against one worker-local generation record."""
    if (
        not isinstance(generation, dict)
        or generation.get("schema") != 1
        or not isinstance(generation.get("generation_id"), str)
        or not generation["generation_id"]
        or not isinstance(generation.get("source"), dict)
    ):
        raise HistoryCompatibilityError(
            "Output attempt has invalid generation evidence"
        )
    uid, captured = capture_events(store)
    matches = [
        (row, origin)
        for name, _, row, origin in captured
        if name == namespace
        and row.get("type") == "assistant_generation_start"
        and row.get("generation_id") == generation.get("generation_id")
    ]
    if len(matches) != 1 or any(
        matches[0][0].get(key) != value for key, value in generation.items()
    ):
        raise HistoryCompatibilityError(
            "Output attempt generation is missing or ambiguous"
        )
    source, ref = matches[0]
    payload = {
        **deepcopy(payload),
        "generation_id": generation["generation_id"],
        "source": deepcopy(generation["source"]),
        "generation_origin": list(ref),
    }
    _, eid = store.append_event(
        namespace,
        "output_delivery",
        deepcopy(payload),
        turn_index=source.get("turn_index"),
        branch_id=source.get("branch_id"),
        parent_branch_path=deepcopy(source.get("parent_branch_path", [])),
    )
    store.flush()
    return {
        "mode": "output_delivery",
        "event_origin": [uid, namespace, eid],
        "payload": payload,
    }
