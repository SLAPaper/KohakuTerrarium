"""Durable event origins and retryable per-store transfer receipts."""

from contextvars import ContextVar
from copy import deepcopy
from uuid import UUID, uuid4

STORE_UUID = "_kt_store_uuid_v1"
EVENT_ORIGIN = "_kt_event_origin_v1"
EVENT_COORDINATE = "_kt_event_coordinate_v1"
_IMPORT = ContextVar("kt_event_origin_import", default=None)
_LIMIT = 2**31 - 1


def _uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Malformed event store UUID")
    return value


def validate_origin(value, namespace):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Malformed event origin")
    uid, name, event_id = value
    _uuid(uid)
    if name != namespace or not isinstance(name, str) or not name:
        raise ValueError("Event origin namespace mismatch")
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("Invalid origin event ID")
    return tuple(value)


def event_key_order(key):
    """Order event keys by namespace and numeric physical sequence."""
    key = key.decode("utf-8") if isinstance(key, bytes) else key
    parts = key.rsplit(":e", 1)
    if (
        len(parts) != 2
        or not parts[0]
        or not parts[1].isascii()
        or not parts[1].isdigit()
    ):
        raise ValueError(f"Invalid event key: {key}")
    return parts[0], int(parts[1])


def strict_rows(store):
    """Read every event namespace without skipping unreadable rows."""
    store.events.flush_cache()
    rows = []
    for raw_key in sorted(store.events.keys(limit=_LIMIT), key=event_key_order):
        key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
        parts = key.rsplit(":e", 1)
        if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
            raise ValueError(f"Invalid event key: {key}")
        row = deepcopy(store.events[raw_key])
        if (
            not isinstance(row, dict)
            or type(row.get("event_id")) is not int
            or row["event_id"] <= 0
        ):
            raise ValueError(f"Invalid event row: {key}")
        if not isinstance(row.get("type"), str):
            raise ValueError(f"Invalid event type: {key}")
        rows.append((parts[0], key, row))
    return rows


def store_uuid(store, rows=None):
    value = store.meta.get(STORE_UUID)
    if value is not None:
        value = _uuid(value)
        if not getattr(store, "_readonly", False):
            store.meta.flush_cache()
        return value
    if getattr(store, "_readonly", False):
        raise ValueError("Read-only source has no durable event identity")
    rows = strict_rows(store) if rows is None else rows
    if any(EVENT_ORIGIN in row for _, _, row in rows):
        raise ValueError("Event origins exist without a store identity")
    value = str(uuid4())
    store.meta[STORE_UUID] = value
    store.meta.flush_cache()
    store.flush()
    return value


def row_origin(uid, namespace, row):
    value = row.get(EVENT_ORIGIN, [uid, namespace, row["event_id"]])
    return validate_origin(value, namespace)


def annotate_coordinate(store, namespace, row):
    """Project a canonical read coordinate without changing a persisted native row."""
    uid = getattr(store, "meta", {}).get(STORE_UUID)
    if uid is None or not isinstance(row, dict):
        return row
    value = deepcopy(row)
    value[EVENT_COORDINATE] = deepcopy(
        row.get(EVENT_ORIGIN, [uid, namespace, row.get("event_id")])
    )
    return value


def _payload(row):
    return {
        key: value
        for key, value in row.items()
        if key not in ("event_id", EVENT_ORIGIN, EVENT_COORDINATE)
    }


def capture_events(store, *, unique=True):
    """Read every durable row and mint a store identity if needed.

    Copy and import keep ``unique=True`` so colliding origins cannot hop stores.
    Local persist and resume lookups may pass ``unique=False`` to continue a
    historically colliding session without rewriting its rows.
    """
    rows = strict_rows(store)
    uid = store_uuid(store, rows)
    seen, event_ids = set(), set()
    captured = []
    for name, key, row in rows:
        origin = row_origin(uid, name, row)
        coordinate = (name, row["event_id"])
        if unique and (origin in seen or coordinate in event_ids):
            raise ValueError("Duplicate stored event origin or namespace event ID")
        seen.add(origin)
        event_ids.add(coordinate)
        captured.append((name, key, row, origin))
    if not getattr(store, "_readonly", False):
        for name, key, row, _ in captured:
            store._global_event_id = max(store._global_event_id, row["event_id"])
            store._event_seq[name] = max(
                store._event_seq.get(name, 0), int(key.rsplit(":e", 1)[1]) + 1
            )
        store.flush()
    return uid, captured


def validate_append_origin(store, namespace, data):
    event_id = data.get("event_id")
    if (
        type(event_id) is int
        and 0 < event_id <= store._global_event_id
        and store.meta.get(STORE_UUID) is not None
    ):
        if any(row["event_id"] == event_id for _, _, row in strict_rows(store)):
            raise ValueError("Preset event ID already exists in this store")
    if EVENT_ORIGIN not in data:
        return False
    binding = _IMPORT.get()
    if binding != (id(store), namespace, id(data)):
        raise ValueError("Reserved event origin requires internal import")
    validate_origin(data[EVENT_ORIGIN], namespace)
    return True


def append_imported(store, namespace, event_type, data):
    """Append a validated imported row with provenance in the same event value."""
    store_uuid(store)
    token = _IMPORT.set((id(store), namespace, id(data)))
    try:
        return store.append_event(namespace, event_type, data)
    finally:
        _IMPORT.reset(token)


def append_mirrored(store, namespace, event_type, data):
    """Recognize a repeated mirror ID without creating a conflicting physical row."""
    data = deepcopy(data)
    data.pop(EVENT_COORDINATE, None)
    event_id = data.get("event_id")
    if type(event_id) is int and 0 < event_id <= store._global_event_id:
        matches = [
            (name, key, row)
            for name, key, row in strict_rows(store)
            if row["event_id"] == event_id
        ]
        local = [(key, row) for name, key, row in matches if name == namespace]
        if len(local) > 1 or (matches and not local):
            raise ValueError("Conflicting mirrored event ID")
        if local:
            key, row = local[0]
            expected = {**data, "type": event_type}
            recorded = dict(row)
            if "ts" not in expected:
                recorded.pop("ts", None)
            if recorded != expected:
                raise ValueError("Conflicting mirrored event ID")
            return key, event_id
    if EVENT_ORIGIN in data:
        return append_imported(store, namespace, event_type, data)
    return store.append_event(namespace, event_type, data)


def import_events(store, batch):
    if getattr(store, "_readonly", False):
        raise ValueError("Cannot import events into a read-only store")
    source_uid, incoming = batch
    _uuid(source_uid)
    uid, existing = capture_events(store)
    if source_uid == uid:
        raise ValueError("Event copy requires distinct store UUIDs")
    index = {}
    for name, key, row, origin in existing:
        index[origin] = (key, row)
        store._global_event_id = max(store._global_event_id, row["event_id"])
        store._event_seq[name] = max(
            store._event_seq.get(name, 0), int(key.rsplit(":e", 1)[1]) + 1
        )
    seen = set()
    for name, key, row, origin in incoming:
        if origin != row_origin(source_uid, name, row) or origin in seen:
            raise ValueError("Invalid or repeated incoming event origin")
        seen.add(origin)
        if origin in index and _payload(index[origin][1]) != _payload(row):
            raise ValueError("Conflicting payload for event origin")
    receipt = []
    for name, key, row, origin in incoming:
        inserted = origin not in index
        if inserted:
            payload = deepcopy(row)
            payload.pop("event_id")
            payload[EVENT_ORIGIN] = list(origin)
            new_key, new_id = append_imported(store, name, row["type"], payload)
            index[origin] = (new_key, {**payload, "event_id": new_id})
        new_key, destination = index[origin]
        receipt.append(
            {
                "source": (source_uid, name, key, row["event_id"]),
                "origin": origin,
                "destination": (uid, name, new_key, destination["event_id"]),
                "inserted": inserted,
            }
        )
    store.flush()
    return receipt
