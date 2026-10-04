"""Source-bound durable text segments and idempotent finalization receipts."""

import hashlib
import json
from copy import deepcopy
from uuid import UUID, uuid4

from kohakuterrarium.errors import HistoryCompatibilityError
from kohakuterrarium.session.event_origin import (
    capture_events,
    row_origin,
    store_uuid,
)

SCHEMA = 1


def _coordinates(turn, branch, path):
    if turn is None and branch is None and isinstance(path, (list, tuple)) and not path:
        return []
    if type(turn) is not int or turn <= 0 or type(branch) is not int or branch <= 0:
        raise HistoryCompatibilityError("Invalid open-text coordinates")
    if not isinstance(path, (list, tuple)):
        raise HistoryCompatibilityError("Invalid open-text parent path")
    result = []
    for pair in path:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise HistoryCompatibilityError("Invalid open-text parent path")
        parent, version = pair
        if (
            type(parent) is not int
            or type(version) is not int
            or not 0 < parent < turn
            or version <= 0
        ):
            raise HistoryCompatibilityError("Invalid open-text parent path")
        if result and parent <= result[-1][0]:
            raise HistoryCompatibilityError("Invalid open-text parent path order")
        result.append([parent, version])
    return result


def _declaration(
    store, namespace, generation, turn, branch, path, *, with_event_id=False
):
    if (
        not isinstance(generation, dict)
        or type(generation.get("schema")) is not int
        or generation["schema"] != 1
    ):
        raise HistoryCompatibilityError("Invalid open-text generation declaration")
    identity = generation.get("generation_id")
    if (
        not isinstance(identity, str)
        or not identity
        or not isinstance(generation.get("source"), dict)
    ):
        raise HistoryCompatibilityError("Invalid open-text generation declaration")
    _, records = capture_events(store, unique=False)
    candidates = [
        (row, origin)
        for name, _, row, origin in records
        if name == namespace
        and row.get("type") == "assistant_generation_start"
        and row.get("generation_id") == identity
    ]
    if len(candidates) != 1:
        raise HistoryCompatibilityError(
            "Missing or ambiguous open-text generation declaration"
        )
    row, origin = candidates[0]
    if (
        row.get("schema") != generation["schema"]
        or row.get("source") != generation["source"]
        or (row.get("turn_index"), row.get("branch_id")) != (turn, branch)
        or _coordinates(turn, branch, row.get("parent_branch_path", [])) != path
    ):
        raise HistoryCompatibilityError("Open-text generation declaration mismatch")
    return (list(origin), row["event_id"]) if with_event_id else list(origin)


def prove_generation_origin(
    store, namespace, generation, turn, branch, path, *, event_id=None
):
    """Prove canonical generation origin once on the confirmed worker and pin proof fields."""
    coord_path = _coordinates(turn, branch, path)
    origin, local_event_id = _declaration(
        store, namespace, generation, turn, branch, coord_path, with_event_id=True
    )
    if event_id is not None and local_event_id != event_id:
        raise HistoryCompatibilityError("Open-text generation declaration mismatch")
    return {
        "schema": 1,
        "generation_id": generation["generation_id"],
        "source": deepcopy(generation["source"]),
        "turn_index": turn,
        "branch_id": branch,
        "parent_branch_path": coord_path,
        "store_uuid": store_uuid(store),
        "event_id": local_event_id,
        "namespace": namespace,
        "declaration_origin": list(origin),
    }


def _verify_indexed_declaration(
    store,
    namespace,
    generation,
    turn,
    branch,
    path,
    declaration_origin,
    *,
    ephemeral_proof=None,
):
    """Verify one live segment against its indexed actual declaration rather than full scan."""
    if (
        not isinstance(generation, dict)
        or type(generation.get("schema")) is not int
        or generation["schema"] != 1
    ):
        raise HistoryCompatibilityError("Invalid open-text generation declaration")
    identity = generation.get("generation_id")
    if (
        not isinstance(identity, str)
        or not identity
        or not isinstance(generation.get("source"), dict)
    ):
        raise HistoryCompatibilityError("Invalid open-text generation declaration")
    if (
        not isinstance(declaration_origin, (list, tuple))
        or len(declaration_origin) != 3
        or not isinstance(declaration_origin[0], str)
        or not isinstance(declaration_origin[1], str)
        or type(declaration_origin[2]) is not int
        or declaration_origin[2] <= 0
    ):
        raise HistoryCompatibilityError("Invalid open-text declaration origin")

    current_store_uuid = store_uuid(store)
    if ephemeral_proof is not None:
        if (
            not isinstance(ephemeral_proof, dict)
            or ephemeral_proof.get("schema") != 1
            or ephemeral_proof.get("store_uuid") != current_store_uuid
            or ephemeral_proof.get("namespace") != namespace
            or ephemeral_proof.get("generation_id") != identity
            or ephemeral_proof.get("source") != generation["source"]
            or (ephemeral_proof.get("turn_index"), ephemeral_proof.get("branch_id"))
            != (turn, branch)
            or ephemeral_proof.get("parent_branch_path") != path
            or list(ephemeral_proof.get("declaration_origin", []))
            != list(declaration_origin)
        ):
            raise HistoryCompatibilityError("Invalid open-text ephemeral proof")

    if ephemeral_proof is None:
        proven = prove_generation_origin(
            store, namespace, generation, turn, branch, path
        )
        if proven["declaration_origin"] != list(declaration_origin):
            raise HistoryCompatibilityError("Open-text declaration origin mismatch")
        event_id = proven["event_id"]
    else:
        event_id = ephemeral_proof.get("event_id")
        if type(event_id) is not int or event_id <= 0:
            raise HistoryCompatibilityError(
                "Invalid open-text declaration event identity"
            )
    row = store.get_event_by_id(namespace, event_id)
    if row is None:
        raise HistoryCompatibilityError(
            "Missing or ambiguous open-text generation declaration"
        )
    if (
        row.get("type") != "assistant_generation_start"
        or row.get("schema") != generation["schema"]
        or row.get("generation_id") != identity
        or row.get("source") != generation["source"]
        or (row.get("turn_index"), row.get("branch_id")) != (turn, branch)
        or _coordinates(turn, branch, row.get("parent_branch_path", [])) != path
    ):
        raise HistoryCompatibilityError("Open-text generation declaration mismatch")

    actual_origin = list(row_origin(current_store_uuid, namespace, row))
    if actual_origin != list(declaration_origin):
        raise HistoryCompatibilityError("Open-text declaration origin mismatch")
    return actual_origin


def build_record(
    store,
    namespace,
    text,
    generation,
    turn_index,
    branch_id,
    parent_branch_path,
    segment_id=None,
    chunk_seq=0,
    *,
    declaration_origin=None,
    ephemeral_proof=None,
):
    """Bind one partial segment to its unique durable generation declaration."""
    if not isinstance(namespace, str) or not namespace or not isinstance(text, str):
        raise HistoryCompatibilityError("Invalid open-text namespace or content")
    if type(chunk_seq) is not int or chunk_seq < 0:
        raise HistoryCompatibilityError("Invalid open-text chunk sequence")
    path = _coordinates(turn_index, branch_id, parent_branch_path)

    origin = declaration_origin
    if (
        origin is None
        and ephemeral_proof is not None
        and isinstance(ephemeral_proof, dict)
    ):
        origin = ephemeral_proof.get("declaration_origin")

    if origin is not None:
        declaration = _verify_indexed_declaration(
            store,
            namespace,
            generation,
            turn_index,
            branch_id,
            path,
            origin,
            ephemeral_proof=ephemeral_proof,
        )
    else:
        declaration = _declaration(
            store, namespace, generation, turn_index, branch_id, path
        )

    identity = str(uuid4()) if segment_id is None else segment_id
    try:
        UUID(identity)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HistoryCompatibilityError("Invalid open-text segment identity") from exc
    return {
        "schema": SCHEMA,
        "segment_id": identity,
        "store_uuid": store_uuid(store),
        "namespace": namespace,
        "text": text,
        "generation": deepcopy(generation),
        "declaration_origin": declaration,
        "turn_index": turn_index,
        "branch_id": branch_id,
        "parent_branch_path": path,
        "chunk_seq": chunk_seq,
    }


def segment_proof(record):
    """Detach immutable source fields from the buffered text."""
    return {key: deepcopy(value) for key, value in record.items() if key != "text"}


def _digest(record):
    return hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def validate_record(store, namespace, record, *, ephemeral_proof=None):
    """Validate a slot without appending text or clearing recovery data."""
    if isinstance(record, str):
        raise HistoryCompatibilityError(
            "Legacy open-text slot has no durable source proof"
        )
    if (
        not isinstance(record, dict)
        or type(record.get("schema")) is not int
        or record["schema"] != SCHEMA
    ):
        raise HistoryCompatibilityError("Invalid open-text record schema")
    if record.get("store_uuid") != store_uuid(store):
        raise HistoryCompatibilityError("Open-text store UUID mismatch")
    if record.get("namespace") != namespace:
        raise HistoryCompatibilityError("Open-text namespace mismatch")
    expected = build_record(
        store,
        namespace,
        record.get("text"),
        record.get("generation"),
        record.get("turn_index"),
        record.get("branch_id"),
        record.get("parent_branch_path"),
        record.get("segment_id"),
        record.get("chunk_seq"),
        declaration_origin=(
            record.get("declaration_origin") if ephemeral_proof is not None else None
        ),
        ephemeral_proof=ephemeral_proof,
    )
    if expected != record:
        raise HistoryCompatibilityError("Open-text declaration proof mismatch")


def recover_record(
    store, namespace, record=None, *, finalize="recovered", ephemeral_proof=None
):
    """Persist or reuse one proven segment, then conditionally clear its durable slot."""
    slot_key = f"{namespace}:open_text"
    slot = deepcopy(store.state.get(slot_key))
    is_cold = record is None
    record = deepcopy(slot if is_cold else record)
    if record is None or record == "":
        return None
    validate_record(
        store, namespace, record, ephemeral_proof=None if is_cold else ephemeral_proof
    )
    proof = segment_proof(record)
    coordinates = {
        coord_key: deepcopy(record[coord_key])
        for coord_key in ("turn_index", "branch_id", "parent_branch_path")
    }
    payload = {
        "content": record["text"],
        "chunk_seq": record["chunk_seq"],
        "generation_id": record["generation"]["generation_id"],
        "segment_id": record["segment_id"],
        "segment_origin": proof,
    }
    if finalize is not None:
        payload["finalize"] = finalize
    found = [
        row
        for row in store.get_events(namespace)
        if row.get("segment_id") == record["segment_id"]
    ]
    if found:
        receipt = found[0]
        prefix_confirmed = (
            isinstance(receipt.get("content"), str)
            and receipt["content"].startswith(record["text"])
            and receipt.get("slot_receipt") == _digest(record)
        )
        required = {**payload, **coordinates}
        for field in ("content", "finalize"):
            required.pop(field, None)
        if (
            len(found) != 1
            or receipt.get("type") != "text_chunk"
            or any(receipt.get(field) != value for field, value in required.items())
            or receipt.get("content") != record["text"]
            and not prefix_confirmed
        ):
            raise HistoryCompatibilityError("Conflicting recovered text segment")
    else:
        if (
            isinstance(slot, dict)
            and segment_proof(slot) == proof
            and isinstance(slot.get("text"), str)
            and record["text"].startswith(slot["text"])
        ):
            payload["slot_receipt"] = _digest(slot)
        _, event_id = store.append_event(
            namespace, "text_chunk", payload, **coordinates
        )
        receipt = store.get_event_by_id(namespace, event_id)
    store.checkpoint(strict=True)
    current = store.state.get(slot_key)
    if (
        current == record
        or isinstance(current, dict)
        and segment_proof(current) == proof
        and receipt.get("slot_receipt") == _digest(current)
    ):
        del store.state[slot_key]
        store.checkpoint(strict=True)
    return deepcopy(receipt)
