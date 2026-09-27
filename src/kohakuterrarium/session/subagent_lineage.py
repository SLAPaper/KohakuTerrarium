"""Persistent child-namespace ownership and allocation high-water marks."""

import json
from copy import deepcopy
from uuid import uuid4

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

PREFIX = "child_lineage:"


def lineage_key(parent):
    return PREFIX + json.dumps(parent, ensure_ascii=False)


def read_lineage(store, parent):
    value = store.state.get(lineage_key(parent))
    if value is None:
        return None
    validate_lineage(parent, value)
    return deepcopy(value)


def validate_lineage(parent, value):
    """Validate the identity, phase, and namespace of a detached lineage record."""
    if (
        not isinstance(value, dict)
        or value.get("v") != 1
        or not isinstance(value.get("origin"), str)
        or not value.get("origin")
        or not isinstance(value.get("token"), str)
        or not value.get("token")
        or value.get("phase") not in {"active", "departed", "installing"}
        or not isinstance(value.get("next_runs"), dict)
        or not isinstance(value.get("ancestors"), dict)
    ):
        raise ConflictError("Invalid child namespace provenance")
    if any(
        not isinstance(key, str)
        or key.rsplit(":", 1)[0] != parent
        or type(count) is not int
        or count < 0
        for key, count in value["next_runs"].items()
    ):
        raise ConflictError("Invalid child run reservation")


def namespace_counters(store, parent):
    return {
        key: value
        for key, value in store._subagent_runs.items()
        if key.rsplit(":", 1)[0] == parent
    }


def write_lineage(store, parent, record):
    store.state[lineage_key(parent)] = deepcopy(record)
    store.state.flush_cache()


def active_lineage(store, parent):
    record = read_lineage(store, parent)
    if record is None:
        record = {
            "v": 1,
            "origin": str(uuid4()),
            "token": str(uuid4()),
            "phase": "active",
            "to": None,
            "next_runs": namespace_counters(store, parent),
            "ancestors": {},
        }
    if record.get("phase") != "active":
        raise ConflictError("Child namespace has departed or is incompletely installed")
    return record


def reserve_run(store, parent, name):
    if ":" in name:
        raise ConflictError("Persisted child names cannot contain ':'")
    record = active_lineage(store, parent)
    key = f"{parent}:{name}"
    run = max(store._subagent_runs.get(key, 0), record["next_runs"].get(key, 0))
    record["next_runs"][key] = run + 1
    write_lineage(store, parent, record)
    store._subagent_runs[key] = run + 1
    return run


def admit_saved_run(store, parent, name, run):
    if ":" in name:
        raise ConflictError("Persisted child names cannot contain ':'")
    record = active_lineage(store, parent)
    key = f"{parent}:{name}"
    high = max(
        run + 1, store._subagent_runs.get(key, 0), record["next_runs"].get(key, 0)
    )
    record["next_runs"][key] = high
    write_lineage(store, parent, record)
    store._subagent_runs[key] = high


def restore_run_reservations(store):
    for raw in store.state.keys(limit=2**31 - 1):
        key = raw.decode() if isinstance(raw, bytes) else raw
        if not key.startswith(PREFIX):
            continue
        try:
            record = read_lineage(store, json.loads(key[len(PREFIX) :]))
        except (ConflictError, ValueError, TypeError):
            logger.warning(
                "Invalid child lineage retained for offline inspection",
                key=key,
                exc_info=True,
            )
            continue
        for name, value in record["next_runs"].items():
            store._subagent_runs[name] = max(store._subagent_runs.get(name, 0), value)
