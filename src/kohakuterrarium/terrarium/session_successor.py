"""Publish execution-resume successors after detached destination checkpoints."""

import asyncio
from copy import deepcopy
from pathlib import Path

from kohakuterrarium.core.compact_branch import snapshot_branch_view
from kohakuterrarium.errors import SessionNotResumableError
from kohakuterrarium.session.identity import new_conversation_id
from kohakuterrarium.session.output_snapshot import capture_turn_snapshot
from kohakuterrarium.session.resume_target import SUCCESSOR_KEY
from kohakuterrarium.session.snapshot_coverage import (
    capture_snapshot_boundary,
    require_snapshot_boundary,
    save_bound_snapshot,
)
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.session_resources import retain_store


def _target_ref(target):
    if not target.meta.get("conversation_id"):
        target.meta["conversation_id"] = new_conversation_id()
        target.checkpoint(strict=True)
    return {
        "path": str(Path(target.path).resolve()),
        "conversation_id": target.meta.get("conversation_id"),
    }


def _stage_marker(source, refs, kind):
    manifest = source.meta.get("live_graph_manifest")
    agents = (
        [member["name"] for member in manifest.get("creatures", [])]
        if isinstance(manifest, dict)
        else list(source.meta.get("agents") or [])
    )
    marker = {"kind": kind, "state": "pending", "targets": refs, "agents": agents}
    previous = source.meta.get(SUCCESSOR_KEY)
    if previous is not None and previous != marker:
        raise SessionNotResumableError("Session successor changed before retirement")
    source.meta[SUCCESSOR_KEY] = marker
    source.set_conversation_open(False)
    source.update_status("completed")
    source.checkpoint(strict=True)
    return marker


def _register_pending(engine, source, marker):
    pending = getattr(engine, "_pending_session_successors", None)
    if pending is None:
        pending = engine._pending_session_successors = []
    if not any(store is source and value == marker for store, value in pending):
        pending.append((source, marker))
    completed = getattr(engine, "_successor_checkpoint_paths", set())
    completed.difference_update(ref["path"] for ref in marker["targets"])
    retain_store(engine, source)


def stage_successor(engine, source, targets, *, kind="merge") -> None:
    """Block source execution and retain detached successor evidence; requires blocking."""
    refs = [target.submit(_target_ref, target).result() for target in targets]
    marker = source.submit(_stage_marker, source, refs, kind).result()
    _register_pending(engine, source, marker)


async def stage_successor_async(engine, source, targets, *, kind="merge") -> None:
    """Stage worker-owned retirement while keeping registry mutation on-loop."""
    refs = [await target.run(_target_ref, target) for target in targets]
    marker = await source.run(_stage_marker, source, refs, kind)
    _register_pending(engine, source, marker)


def _capture_message(agent):
    conversation = getattr(getattr(agent, "controller", None), "conversation", None)
    if conversation is None:
        return None
    messages = conversation.snapshot_messages(preserve_pending_tail=True)
    tag = None
    if getattr(agent, "_turn_index", 0) > 0 and getattr(agent, "_branch_id", 0) > 0:
        tag = {
            "turn_index": agent._turn_index,
            "branch_id": agent._branch_id,
            "parent_branch_path": getattr(agent, "_parent_branch_path", None),
            "branch_view": snapshot_branch_view(agent),
        }
    return deepcopy((agent.config.name, messages, tag))


def _persist_messages(destination, captures):
    for name, messages, branch_tag, envelope in captures:
        require_snapshot_boundary(destination, name, envelope)
        save_bound_snapshot(destination, name, messages, envelope)
        destination.state[f"{name}:snapshot_event_id"] = destination.max_event_id(name)
        if branch_tag is None:
            destination.state.pop(f"{name}:snapshot_branch", None)
        else:
            destination.state[f"{name}:snapshot_branch"] = branch_tag
    destination.checkpoint(strict=True)


def _commit_ready(store, marker):
    ready = deepcopy(marker)
    ready["state"] = "ready"
    if store.meta.get(SUCCESSOR_KEY) not in (marker, ready):
        raise SessionNotResumableError("Session successor changed during checkpoint")
    store.meta[SUCCESSOR_KEY] = ready
    store.checkpoint(strict=True)


def _publish_marker(source, marker):
    reopened = bool(getattr(source, "_closed", False))
    store = SessionStore(source.path, writer_lock=True) if reopened else source
    try:
        store.submit(_commit_ready, store, marker).result()
    finally:
        if reopened:
            store.close(update_status=False)


def _ready_sources(engine, destination, identity):
    completed = getattr(engine, "_successor_checkpoint_paths", None)
    if completed is None:
        completed = engine._successor_checkpoint_paths = set()
    path = str(Path(destination.path).resolve())
    for _, marker in getattr(engine, "_pending_session_successors", []):
        for ref in marker["targets"]:
            if ref["path"] == path and ref["conversation_id"] != identity:
                raise SessionNotResumableError(
                    "Successor destination identity changed before publication"
                )
    completed.add(path)
    result = []
    for source, marker in list(getattr(engine, "_pending_session_successors", [])):
        if not all(ref["path"] in completed for ref in marker["targets"]):
            continue
        if any(not ref["conversation_id"] for ref in marker["targets"]):
            raise SessionNotResumableError(
                "Cannot publish a successor without conversation identity"
            )
        result.append((source, marker))
    return result


def publish_successors(engine, graph_id: str) -> None:
    """Publish a synchronous graph checkpoint; live owners require the async entry."""
    if not getattr(engine, "_pending_session_successors", []):
        return
    destination = engine._session_stores[graph_id]
    graph = engine._topology.graphs[graph_id]
    captures = []
    for cid in graph.creature_ids:
        agent = engine._creatures[cid].agent
        if getattr(getattr(agent, "_session_state_owner", None), "is_running", False):
            raise SessionNotResumableError(
                "Live successor publication requires an owner capture"
            )
        output = getattr(agent, "_session_output", None)
        if output is not None:
            output.flush_sync()
        capture = _capture_message(agent)
        if capture is not None:
            name, messages, branch_tag = capture
            envelope = destination.submit(
                capture_snapshot_boundary, destination, name
            ).result()
            captures.append((name, messages, branch_tag, envelope))
    destination.submit(_persist_messages, destination, captures).result()
    identity = destination.submit(
        lambda: destination.meta.get("conversation_id")
    ).result()
    for source, marker in _ready_sources(engine, destination, identity):
        _publish_marker(source, marker)
        engine._pending_session_successors.remove((source, marker))


async def publish_successors_async(engine, graph_id: str) -> None:
    """Capture on owners, persist on workers, then publish confirmed source markers."""
    if not getattr(engine, "_pending_session_successors", []):
        return
    destination = engine._session_stores[graph_id]
    graph = engine._topology.graphs[graph_id]
    captures = []
    for cid in sorted(graph.creature_ids):
        agent = engine._creatures[cid].agent
        output = getattr(agent, "_session_output", None)
        if output is not None:
            await output.flush()
        if getattr(getattr(agent, "controller", None), "conversation", None) is None:
            continue
        name = agent.config.name
        envelope = await destination.run(capture_snapshot_boundary, destination, name)
        owner = getattr(agent, "_session_state_owner", None)
        if owner is not None and owner.is_running:
            messages, branch_tag, _ = await capture_turn_snapshot(agent)
        else:
            _, messages, branch_tag = _capture_message(agent)
        captures.append((name, messages, branch_tag, envelope))
    await destination.run(_persist_messages, destination, captures)
    if engine._session_stores.get(graph_id) is not destination:
        raise SessionNotResumableError(
            "Successor destination changed during checkpoint"
        )
    identity = await destination.run(lambda: destination.meta.get("conversation_id"))
    for source, marker in _ready_sources(engine, destination, identity):
        await asyncio.to_thread(_publish_marker, source, marker)
        engine._pending_session_successors.remove((source, marker))
