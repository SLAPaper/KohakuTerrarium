"""Proof-carrying child namespaces transferred between their store workers."""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.store import iter_kv_keys
from kohakuterrarium.session.subagent_lineage import (
    active_lineage,
    namespace_counters,
    read_lineage,
    validate_lineage,
    write_lineage,
)


def _snapshot(store, parent):
    rows = {}
    for raw in iter_kv_keys(store.subagents):
        key = raw.decode() if isinstance(raw, bytes) else raw
        parts = key.rsplit(":", 3)
        if len(parts) == 4 and parts[0] == parent:
            rows[key] = deepcopy(store.subagents[raw])
    return rows, namespace_counters(store, parent)


def capture_children(store, parent):
    """Capture the active parent's rows, reservations, and ownership receipts."""
    rows, counters = _snapshot(store, parent)
    record = read_lineage(store, parent)
    if rows or counters or record is not None:
        record = active_lineage(store, parent)
        record["next_runs"] = dict(counters)
        write_lineage(store, parent, record)
    return {"parent": parent, "rows": rows, "counters": counters, "lineage": record}


def validate_children(store, payload):
    """Accept only an empty target or an unchanged proven departed ancestor."""
    parent, source = payload["parent"], payload["lineage"]
    if source is not None:
        validate_lineage(parent, source)
        if source["phase"] != "active" or payload["counters"] != source["next_runs"]:
            raise ConflictError("Child source is not an active prepared namespace")
    elif payload["rows"] or payload["counters"]:
        raise ConflictError("Child source has no ownership identity")
    if any(
        not isinstance(key, str)
        or len(key.rsplit(":", 3)) != 4
        or key.rsplit(":", 3)[0] != parent
        for key in payload["rows"]
    ):
        raise ConflictError("Child payload contains rows outside its parent namespace")
    rows, counters = _snapshot(store, parent)
    destination = read_lineage(store, parent)
    if not rows and not counters and destination is None:
        return
    if (
        source is None
        or destination is None
        or destination.get("phase") != "departed"
        or destination["origin"] != source["origin"]
    ):
        raise ConflictError(f"Unproven child namespace collision: {parent}")
    ancestors = source.get("ancestors", {})
    token = destination["token"]
    receipt = ancestors.get(token)
    if not isinstance(receipt, dict) or receipt.get("to") != destination.get("to"):
        raise ConflictError("Child handoff receipt is missing or mismatched")
    if rows != receipt.get("rows") or counters != receipt.get("next_runs"):
        raise ConflictError("Departed child namespace changed after handoff")
    seen = set()
    while token != source["token"]:
        if token in seen or token not in ancestors:
            raise ConflictError("Broken child ownership successor chain")
        seen.add(token)
        token = ancestors[token].get("to")
    if not rows.keys() <= payload["rows"].keys() or any(
        payload["counters"].get(key, 0) < value for key, value in counters.items()
    ):
        raise ConflictError("Child descendant lost history or reservations")


def depart_children(store, payload, successor):
    """Seal the exact source baseline before transferring namespace authority."""
    parent = payload["parent"]
    if (
        _snapshot(store, parent) != (payload["rows"], payload["counters"])
        or read_lineage(store, parent) != payload["lineage"]
    ):
        raise ConflictError("Child namespace changed after transfer preparation")
    record = deepcopy(payload["lineage"])
    if record is None:
        return
    record.update(phase="departed", to=successor)
    store.subagents.flush_cache()
    write_lineage(store, parent, record)


def import_children(store, payload, successor):
    """Install child data under a new tenure, publishing active authority last."""
    validate_children(store, payload)
    record = deepcopy(payload["lineage"])
    if record is None:
        return
    record["ancestors"][record["token"]] = {
        "to": successor,
        "rows": deepcopy(payload["rows"]),
        "next_runs": dict(payload["counters"]),
    }
    record.update(token=successor, phase="installing", to=None)
    parent = payload["parent"]
    write_lineage(store, parent, record)
    for key, value in payload["rows"].items():
        store.subagents[key] = value
    for key, value in payload["counters"].items():
        store._subagent_runs[key] = max(store._subagent_runs.get(key, 0), value)
    store.subagents.flush_cache()
    if _snapshot(store, parent) != (payload["rows"], payload["counters"]):
        raise ConflictError("Installed child namespace differs from prepared source")
    record.update(phase="active", next_runs=namespace_counters(store, parent))
    write_lineage(store, parent, record)


def preflight_child_merge(engine, graph_ids):
    """Validate participating stores before a merge mutates topology or events."""
    stores = {
        engine._session_stores[gid]
        for gid in graph_ids
        if gid in engine._session_stores
    }
    directory = getattr(engine, "_session_dir", None)
    if directory is not None and stores:
        target = (Path(directory) / f"{graph_ids[0]}.kohakutr").resolve()
        if target.exists() and all(
            Path(store.path).resolve() != target for store in stores
        ):
            raise ConflictError(
                "Merge target already exists outside the participating stores"
            )
    for creature in engine._creatures.values():
        if creature.graph_id not in graph_ids:
            continue
        source = getattr(creature.agent, "session_store", None)
        if source is None:
            continue
        payload = source.submit(
            capture_children, source, creature.agent.config.name
        ).result()
        for destination in stores:
            if destination is not source:
                destination.submit(validate_children, destination, payload).result()


def preflight_child_split(engine, projected, delta):
    """Reject child-bearing splits without a destination or onto unrelated files."""
    if delta.kind != "split":
        return
    parent = engine._session_stores.get(delta.old_graph_ids[0])
    if parent is None:
        return
    directory = getattr(engine, "_session_dir", None)
    for graph_id in delta.new_graph_ids:
        if directory is not None:
            path = (Path(directory) / f"{graph_id}.kohakutr").resolve()
            if path.exists() and path != Path(parent.path).resolve():
                raise ConflictError("Split destination exists outside its source store")
            continue
        if graph_id == delta.new_graph_ids[0]:
            continue
        for cid in projected.graphs[graph_id].creature_ids:
            creature = engine._creatures.get(cid)
            if creature is None:
                continue
            manager = getattr(creature.agent, "subagent_manager", None)
            active = any(
                not task.done() for task in getattr(manager, "_tasks", {}).values()
            )
            payload = parent.submit(
                capture_children, parent, creature.agent.config.name
            ).result()
            if active or payload["rows"] or payload["counters"]:
                raise ConflictError("Child-bearing split requires a session directory")


def transfer_graph_children(engine, destinations):
    """Preflight all child targets, seal sources, and install before loop rebinding."""
    transfers = []
    for graph_id, destination in destinations:
        graph = engine._topology.graphs.get(graph_id)
        if graph is None:
            continue
        for cid in graph.creature_ids:
            creature = engine._creatures.get(cid)
            if creature is None:
                continue
            source = getattr(creature.agent, "session_store", None)
            if source is None or source is destination:
                continue
            payload = source.submit(
                capture_children, source, creature.agent.config.name
            ).result()
            transfers.append((source, destination, payload, str(uuid4())))
    for _, destination, payload, _ in transfers:
        destination.submit(validate_children, destination, payload).result()
    for source, destination, payload, successor in transfers:
        source.submit(depart_children, source, payload, successor).result()
        destination.submit(import_children, destination, payload, successor).result()
