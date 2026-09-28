"""Snapshot envelopes bind a message base to an exact namespace prefix."""

from copy import deepcopy

from kohakuterrarium.errors import ConflictError, HistoryCompatibilityError
from kohakuterrarium.session.event_origin import (
    capture_events,
    row_origin,
    store_uuid,
    validate_origin,
)

SNAPSHOT_VERSION = "_kt_snapshot_v1"


def encode_snapshot(store, name, messages):
    if not isinstance(messages, list):
        return messages
    return {
        SNAPSHOT_VERSION: 1,
        "store_uuid": store_uuid(store),
        "namespace": name,
        "coverage": {
            "mode": "physical_prefix",
            "next_sequence": store._event_seq.get(name, 0),
        },
        "messages": deepcopy(messages),
    }


def capture_snapshot_boundary(store, name):
    """Capture a detached source prefix before taking an owner snapshot."""
    return encode_snapshot(store, name, []) if hasattr(store, "_event_seq") else None


def require_snapshot_boundary(store, name, envelope):
    """Reject a raw event advance across an owner capture interval."""
    if envelope is not None and (
        envelope["store_uuid"] != store_uuid(store)
        or envelope["namespace"] != name
        or envelope["coverage"]["next_sequence"] != store._event_seq.get(name, 0)
    ):
        raise ConflictError("Snapshot source prefix advanced during owner capture")


def save_bound_snapshot(store, name, messages, envelope=None):
    """Save a runtime snapshot with its current worker-owned physical prefix."""
    if not hasattr(store, "_event_seq"):
        store.save_conversation(name, messages)
        return
    value = (
        encode_snapshot(store, name, messages)
        if envelope is None
        else deepcopy(envelope)
    )
    value["messages"] = deepcopy(messages)
    store.save_conversation(name, value)


def decode_snapshot(value):
    if not isinstance(value, dict) or SNAPSHOT_VERSION not in value:
        return None
    if (
        type(value[SNAPSHOT_VERSION]) is not int
        or value[SNAPSHOT_VERSION] != 1
        or not isinstance(value.get("messages"), list)
    ):
        raise HistoryCompatibilityError("Invalid snapshot envelope")
    return value["messages"]


def prefix_origins(envelope, uid, namespace, captured):
    decode_snapshot(envelope)
    if envelope.get("store_uuid") != uid or envelope.get("namespace") != namespace:
        raise HistoryCompatibilityError(
            "Snapshot source binding does not match its store"
        )
    coverage = envelope.get("coverage")
    if not isinstance(coverage, dict):
        raise HistoryCompatibilityError("Snapshot prefix coverage is missing")
    rows = [row for row in captured if row[0] == namespace]
    if coverage.get("mode") == "physical_prefix":
        cursor = coverage.get("next_sequence")
        if type(cursor) is not int or cursor < 0:
            raise HistoryCompatibilityError("Invalid snapshot physical prefix")
        return [
            origin for _, key, _, origin in rows if int(key.rsplit(":e", 1)[1]) < cursor
        ]
    if coverage.get("mode") == "canonical_prefix":
        values = coverage.get("origins")
        if not isinstance(values, list):
            raise HistoryCompatibilityError("Invalid canonical snapshot prefix")
        try:
            members = [validate_origin(value, namespace) for value in values]
        except (TypeError, ValueError) as exc:
            raise HistoryCompatibilityError("Invalid snapshot event origin") from exc
        if len(set(members)) != len(members):
            raise HistoryCompatibilityError("Duplicate snapshot prefix member")
        existing = {origin for _, _, _, origin in rows}
        if not set(members) <= existing:
            raise HistoryCompatibilityError("Snapshot prefix member is missing")
        return members
    raise HistoryCompatibilityError("Unsupported snapshot prefix coverage")


def partition_snapshot(store, namespace, events):
    value = store.conversation.get(namespace)
    if not isinstance(value, dict) or SNAPSHOT_VERSION not in value:
        return None
    uid, captured = capture_events(store)
    members = set(prefix_origins(value, uid, namespace, captured))
    prior, tail = [], []
    for event in events:
        target = prior if row_origin(uid, namespace, event) in members else tail
        target.append(event)
    return prior, tail


def _legacy_transfer_snapshot(store, uid, name, rows):
    watermark = store.state.get(f"{name}:snapshot_event_id")
    local = [(row, origin) for namespace, _, row, origin in rows if namespace == name]
    if (
        type(watermark) is not int
        or watermark < 0
        or any(origin != (uid, name, row["event_id"]) for row, origin in local)
        or any(
            left[0]["event_id"] >= right[0]["event_id"]
            for left, right in zip(local, local[1:])
        )
    ):
        raise HistoryCompatibilityError(
            "Legacy snapshot has no proven local transfer prefix"
        )
    try:
        messages = store.load_conversation(name)
    except (TypeError, ValueError) as exc:
        raise HistoryCompatibilityError("Invalid legacy snapshot messages") from exc
    if not isinstance(messages, list):
        raise HistoryCompatibilityError("Invalid legacy snapshot messages")
    return {
        SNAPSHOT_VERSION: 1,
        "store_uuid": uid,
        "namespace": name,
        "messages": deepcopy(messages),
        "coverage": {
            "mode": "canonical_prefix",
            "origins": [
                list(origin) for row, origin in local if row["event_id"] <= watermark
            ],
        },
    }


def capture_snapshot_transfer(store, owned_names=None):
    """Capture events and source-bound conversation values on one source worker."""
    batch = capture_events(store)
    uid, rows = batch
    namespaces = {name for name, _, _, _ in rows}
    namespaces.update(
        key.decode() if isinstance(key, bytes) else key
        for key in store.conversation.keys(limit=2**31 - 1)
    )
    if owned_names is not None:
        namespaces = set(owned_names)
    else:
        declared = store.meta.get("agents", [])
        if not isinstance(declared, list) or any(
            not isinstance(name, str) or not name for name in declared
        ):
            raise HistoryCompatibilityError("Invalid declared snapshot namespaces")
        namespaces.update(declared)
    snapshots = {}
    for name in namespaces:
        value = deepcopy(store.conversation.get(name))
        if value is not None:
            if decode_snapshot(value) is None:
                value = _legacy_transfer_snapshot(store, uid, name, rows)
            members = prefix_origins(value, uid, name, rows)
            value["coverage"] = {
                "mode": "canonical_prefix",
                "origins": [list(origin) for origin in members],
            }
        state = {
            suffix: deepcopy(store.state.get(f"{name}:{suffix}"))
            for suffix in ("snapshot_branch", "branch_view")
        }
        snapshots[name] = (value, state)
    return batch, snapshots


def install_snapshot_transfer(store, batch, snapshots, receipts):
    """Relocate snapshot-local selectors using confirmed immediate-source receipts."""
    source_uid, rows = batch
    destination_uid = store_uuid(store)
    source_index = {
        (name, row["event_id"]): (key, origin) for name, key, row, origin in rows
    }
    mapping, canonical, seen = {}, {}, set()
    for receipt in receipts:
        source, target, origin = (
            receipt["source"],
            receipt["destination"],
            tuple(receipt["origin"]),
        )
        if (
            source[0] != source_uid
            or target[0] != destination_uid
            or source[1] != target[1]
            or source_index.get((source[1], source[3])) != (source[2], origin)
            or (source[1], source[3]) in mapping
            or origin in seen
        ):
            raise HistoryCompatibilityError("Invalid snapshot transfer receipt")
        persisted = store.events.get(target[2])
        if (
            not isinstance(persisted, dict)
            or persisted.get("event_id") != target[3]
            or row_origin(destination_uid, target[1], persisted) != origin
        ):
            raise HistoryCompatibilityError(
                "Snapshot receipt destination does not match durable row"
            )
        mapping[source[1], source[3]] = target[3]
        canonical[origin] = target[3]
        seen.add(origin)
    if len(mapping) != len(rows):
        raise HistoryCompatibilityError("Incomplete snapshot transfer receipts")
    staged = {}
    for name, (value, state) in snapshots.items():
        value = deepcopy(value)
        if value is not None:
            if not {tuple(origin) for origin in value["coverage"]["origins"]} <= seen:
                raise HistoryCompatibilityError("Snapshot prefix absent from transfer")
            value["store_uuid"] = destination_uid
            for message in value["messages"]:
                if not isinstance(message, dict) or not isinstance(
                    message.get("metadata", {}), dict
                ):
                    raise HistoryCompatibilityError("Invalid snapshot message metadata")
                metadata = message.get("metadata", {})
                if "event_origin" in metadata:
                    try:
                        origin = validate_origin(metadata["event_origin"], name)
                    except (TypeError, ValueError) as exc:
                        raise HistoryCompatibilityError(
                            "Invalid snapshot selector origin"
                        ) from exc
                    if origin not in canonical:
                        if origin[0] == source_uid or origin in {
                            tuple(member) for member in value["coverage"]["origins"]
                        }:
                            raise HistoryCompatibilityError(
                                "Snapshot canonical selector absent from transfer"
                            )
                        metadata.pop("event_id", None)
                        continue
                    metadata["event_id"] = canonical[origin]
                    continue
                for key in ("event_id",):
                    if key in metadata:
                        pair = name, metadata[key]
                        if pair not in mapping:
                            raise HistoryCompatibilityError(
                                f"Snapshot selector {key}={metadata[key]!r} absent from namespace {name!r} transfer"
                            )
                        metadata[key] = mapping[pair]
        staged[name] = (value, state)
    for name, (value, state) in staged.items():
        if value is None:
            store.conversation.pop(name, None)
        else:
            store.conversation[name] = value
        store.state.pop(f"{name}:snapshot_event_id", None)
        for suffix, item in state.items():
            if item is None:
                store.state.pop(f"{name}:{suffix}", None)
            else:
                store.state[f"{name}:{suffix}"] = item
    store.flush()
