"""Store independent conversation checkpoints for diverging branch lineages."""

import math
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from typing import Any

from kohakuterrarium.session.history.ancestry import BranchAncestry, SharedPath
from uuid import uuid4

from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)
PAYLOAD_MARKER = ":branch_snapshot_messages:"


def snapshot_lock(store: Any) -> AbstractContextManager:
    """Serialize checkpoint reads and publication within an owning store."""
    return getattr(store, "_snapshot_lock", nullcontext())


def is_snapshot_payload_key(key: str) -> bool:
    """Whether a state key names an immutable branch message payload."""
    _, marker, suffix = key.rpartition(PAYLOAD_MARKER)
    return (
        bool(marker)
        and len(suffix) == 32
        and all(c in "0123456789abcdef" for c in suffix)
    )


def _own_payload_key(agent: str, key: Any) -> bool:
    prefix = f"{agent}{PAYLOAD_MARKER}"
    return (
        isinstance(key, str)
        and key.startswith(prefix)
        and len(key) == len(prefix) + 32
        and is_snapshot_payload_key(key)
    )


def _path(raw: Any) -> list[list[int]] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    out = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        if any(type(value) is not int or value <= 0 for value in pair):
            return None
        if out and pair[0] <= out[-1][0]:
            return None
        out.append(list(pair))
    return out


def branch_path(tag: Any) -> list[list[int]] | None:
    """Decode a snapshot's complete parent lineage and current turn."""
    if not isinstance(tag, dict):
        return None
    parents = _path(tag.get("parent_branch_path") or [])
    return (
        _path([*parents, [tag.get("turn_index"), tag.get("branch_id")]])
        if parents is not None
        else None
    )


def snapshot_tag(agent: Any) -> dict:
    """Capture the branch identity of a live conversation."""
    return {
        "turn_index": getattr(agent, "_turn_index", None),
        "branch_id": getattr(agent, "_branch_id", None),
        "parent_branch_path": deepcopy(getattr(agent, "_parent_branch_path", None)),
        "context_cutoff_event_id": getattr(agent, "_context_cutoff_event_id", 0),
    }


def compact_state(manager: Any) -> dict:
    """Capture the compact counters attached to a conversation."""
    return {
        "compact_count": getattr(manager, "_compact_count", 0),
        "last_compact_time": getattr(manager, "_last_compact_time", 0.0),
    }


def _valid_messages(messages: Any) -> bool:
    return isinstance(messages, list) and all(
        isinstance(msg, dict)
        and msg.get("role") in {"system", "user", "assistant", "tool"}
        and "content" in msg
        for msg in messages
    )


def matches_snapshot_view(path: Any, selected: dict[int, int]) -> bool:
    """Compare a checkpoint lineage with its selected view."""
    return (
        dict(path) == selected
        if isinstance(path, SharedPath)
        else path == sorted([t, b] for t, b in selected.items())
    )


def load_headers(store: Any, agent: str, *, shared_paths: bool = False) -> list[dict]:
    """Read checkpoint metadata without decoding separately stored messages.

    Legacy inline records remain readable without making detached payload copies.
    Callers that also load messages hold ``snapshot_lock`` across both operations.
    """
    try:
        raw = getattr(store, "state", {}).get(f"{agent}:branch_snapshots", [])
    except (KeyError, TypeError, AttributeError):
        return []
    if not isinstance(raw, list):
        return []
    records = []
    tree = BranchAncestry() if shared_paths else None
    for record in raw:
        if not isinstance(record, dict) or not _path(record.get("path")):
            continue
        watermark, state = (
            record.get("watermark"),
            record.get("compact_state"),
        )
        if type(watermark) is not int or watermark < 0:
            continue
        ref = record.get("messages_key")
        if ref is not None and record.get("version") not in (None, 2):
            continue
        if ref is not None:
            if not _own_payload_key(agent, ref):
                continue
        elif not _valid_messages(record.get("messages")):
            continue
        if (
            not isinstance(state, dict)
            or type(state.get("compact_count")) is not int
            or state["compact_count"] < 0
        ):
            continue
        cutoff = record.get("context_cutoff_event_id", 0)
        if type(cutoff) is not int or not 0 <= cutoff <= watermark:
            continue
        timestamp = state.get("last_compact_time")
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            continue
        if (
            tree is not None
            and type(record["path"]) is list
            and all(type(pair) is list for pair in record["path"])
        ):
            record = {**record, "path": tree.view(tree.intern(record["path"]))}
        records.append(record)
    return records


def load_record(store: Any, agent: str, header: dict) -> dict | None:
    """Load one valid checkpoint, detached from inline legacy records."""
    ref = header.get("messages_key")
    if ref is not None:
        if not _own_payload_key(agent, ref):
            return None
        messages = store.state.get(ref)
    else:
        messages = header.get("messages")
    if not _valid_messages(messages):
        return None
    record = deepcopy(
        {
            key: value
            for key, value in header.items()
            if key not in {"messages", "messages_key"}
        }
    )
    record["messages"] = deepcopy(messages) if ref is None else messages
    return record


def load_records(store: Any, agent: str) -> list[dict]:
    """Return all detached checkpoints for callers explicitly needing messages."""
    with snapshot_lock(store):
        return [
            record
            for header in load_headers(store, agent)
            if (record := load_record(store, agent, header)) is not None
        ]


def load_branch_snapshot(
    store: Any, agent: str, branch_view: dict[int, int]
) -> dict | None:
    """Select the deepest checkpoint on an explicitly resolved lineage."""
    target = _path(sorted(branch_view.items()))
    if not target:
        return None
    with snapshot_lock(store):
        candidates = sorted(
            (
                record
                for record in load_headers(store, agent)
                if target[: len(record["path"])] == record["path"]
            ),
            key=lambda record: (len(record["path"]), record["watermark"]),
            reverse=True,
        )
        for header in candidates:
            record = load_record(store, agent, header)
            if record is not None:
                return record
    return None


def _clean_payloads(store: Any, agent: str, retained: set[str]) -> None:
    prefix = f"{agent}{PAYLOAD_MARKER}"
    try:
        for raw_key in list(store.state.keys(prefix=prefix)):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            if key not in retained and _own_payload_key(agent, key):
                store.state.pop(key, None)
    except Exception:
        logger.warning(
            "Snapshot payload cleanup failed", namespace=agent, exc_info=True
        )


def clear_records(store: Any, agent: str) -> None:
    """Clear the checkpoint catalog before removing its message payloads."""
    with snapshot_lock(store):
        store.state[f"{agent}:branch_snapshots"] = []
        _clean_payloads(store, agent, set())


def save_branch_snapshot(
    store: Any,
    agent: str,
    messages: list[dict],
    *,
    branch: dict | None,
    watermark: int,
    compact_state: dict | None = None,
) -> list[dict]:
    """Write one coherent checkpoint and maintain the legacy latest mirrors."""
    with snapshot_lock(store):
        return _save_branch_snapshot(
            store,
            agent,
            messages,
            branch=branch,
            watermark=watermark,
            compact_state=compact_state,
        )


def _save_branch_snapshot(
    store: Any,
    agent: str,
    messages: list[dict],
    *,
    branch: dict | None,
    watermark: int,
    compact_state: dict | None,
) -> list[dict]:
    catalog = []
    path = branch_path(branch)
    if path and type(watermark) is int and watermark >= 0:
        records = [
            record
            for record in load_headers(store, agent)
            if record["path"][: min(len(path), len(record["path"]))]
            != path[: min(len(path), len(record["path"]))]
        ]
        state = compact_state or {"compact_count": 0, "last_compact_time": 0.0}
        rows = []
        for index, record in enumerate(records):
            if record.get("messages_key") is None:
                ref = f"{agent}{PAYLOAD_MARKER}{uuid4().hex}"
                rows.append((ref, record["messages"]))
                records[index] = {
                    key: value for key, value in record.items() if key != "messages"
                }
                records[index]["messages_key"] = ref
                records[index]["version"] = 2
        ref = f"{agent}{PAYLOAD_MARKER}{uuid4().hex}"
        records.append(
            {
                "path": path,
                "messages_key": ref,
                "version": 2,
                "watermark": watermark,
                "compact_state": deepcopy(state),
                "context_cutoff_event_id": branch.get("context_cutoff_event_id", 0),
            }
        )
        rows.extend(
            [
                (ref, messages),
                (f"{agent}:branch_snapshots", records),
                (f"{agent}:snapshot_branch", deepcopy(branch)),
                (f"{agent}:compact_count", state["compact_count"]),
                (f"{agent}:last_compact_time", state["last_compact_time"]),
            ]
        )
        store.state.put_many(rows)
        catalog = records
        _clean_payloads(store, agent, {record["messages_key"] for record in records})
    else:
        clear_records(store, agent)
        store.state.pop(f"{agent}:snapshot_branch", None)
    store.conversation[agent] = deepcopy(messages)
    store.state[f"{agent}:snapshot_event_id"] = watermark
    return catalog


def save_agent_snapshot(
    store: Any, namespace: str, agent: Any, messages: list[dict]
) -> None:
    """Persist a quiescent history mutation without discarding sibling checkpoints."""
    saver = getattr(store, "save_branch_snapshot", None)
    if not callable(saver):
        store.save_conversation(namespace, messages)
        return
    watermark = store.max_event_id(namespace)
    if not any(message["role"] != "system" for message in messages):
        agent._context_cutoff_event_id = watermark
    state = (
        compact_state(getattr(agent, "compact_manager", None))
        if any(message["role"] != "system" for message in messages)
        else {"compact_count": 0, "last_compact_time": 0.0}
    )
    saver(
        namespace,
        messages,
        branch=snapshot_tag(agent),
        watermark=watermark,
        compact_state=state,
    )
