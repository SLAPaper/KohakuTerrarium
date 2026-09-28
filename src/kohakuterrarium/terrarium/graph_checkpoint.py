"""Serialized graph-manifest checkpoint boundaries."""

import asyncio
import weakref
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from kohakuterrarium.errors import (
    GraphManifestPersistenceError,
    SessionNotResumableError,
)

import kohakuterrarium.terrarium.graph_manifest as _manifest
import kohakuterrarium.terrarium.topology_snapshot as _topo_snap

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from kohakuterrarium.terrarium.engine import Terrarium


_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_dirty: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_suppression: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _engine_locks(engine: "Terrarium") -> dict[str, asyncio.Lock]:
    return _locks.setdefault(engine, {})


def dirty_graphs(engine: "Terrarium") -> set[str]:
    """Return the live dirty-graph set for an engine."""
    return _dirty.setdefault(engine, set())


@contextmanager
def suppress(engine: "Terrarium") -> "Iterator[None]":
    """Delay intermediate checkpoints until a compound mutation completes."""
    _suppression[engine] = _suppression.get(engine, 0) + 1
    try:
        yield
    finally:
        remaining = _suppression.get(engine, 1) - 1
        if remaining:
            _suppression[engine] = remaining
        else:
            _suppression.pop(engine, None)


async def preflight_add(
    engine: "Terrarium", graph_id: str, creature, *, will_persist: bool = False
) -> None:
    """Reject a duplicate state namespace before adding to a persisted graph."""
    if graph_id not in engine._session_stores and not will_persist:
        return
    existing_names = {
        existing.name
        for cid in engine._topology.graphs[graph_id].creature_ids
        if (existing := engine._creatures.get(cid)) is not None
        and hasattr(existing, "name")
    }
    if creature.name not in existing_names:
        return
    agent = getattr(creature, "agent", None)
    if agent is not None and hasattr(agent, "stop"):
        try:
            await agent.stop()
        except Exception:
            pass
    raise SessionNotResumableError(
        f"Persisted graph {graph_id!r} already contains creature name "
        f"{creature.name!r}"
    )


def _persist_checkpoint(store, manifest, snapshot):
    """Save detached graph captures and allocate revisions on the store worker."""
    if manifest is None:
        store.meta[_manifest.MANIFEST_KEY] = None
    else:
        previous = _manifest.load_manifest(store)
        manifest = replace(manifest, revision=previous.revision + 1 if previous else 1)
        _manifest.save_manifest(store, manifest)
    if snapshot is not None:
        _topo_snap.write_snapshot(store, snapshot)
    flush = getattr(store, "flush", None)
    if flush is not None:
        flush()


async def checkpoint(engine: "Terrarium", graph_id: str) -> bool:
    """Capture live graph state on-loop and persist detached values on its worker."""
    dirty = dirty_graphs(engine)
    if _suppression.get(engine, 0):
        dirty.add(graph_id)
        return False
    lock = _engine_locks(engine).setdefault(graph_id, asyncio.Lock())
    async with lock:
        store = engine._session_stores.get(graph_id)
        graph = engine._topology.graphs.get(graph_id)
        if store is None:
            dirty.discard(graph_id)
            return False
        if not hasattr(store, "meta"):
            _topo_snap.snapshot(engine, graph_id)
            dirty.discard(graph_id)
            return False
        missing = graph is None or not graph.creature_ids
        invalid = False
        try:
            snapshot = _topo_snap.capture_snapshot(engine, graph_id)
            try:
                manifest = (
                    None if missing else _manifest.capture_manifest(engine, graph_id)
                )
            except (KeyError, SessionNotResumableError):
                manifest, invalid = None, True
            manifest, snapshot = deepcopy((manifest, snapshot))
            run = getattr(store, "run", None)
            if callable(run):
                await run(_persist_checkpoint, store, manifest, snapshot)
            else:
                _persist_checkpoint(store, manifest, snapshot)
        except asyncio.CancelledError:
            dirty.add(graph_id)
            raise
        except GraphManifestPersistenceError as exc:
            dirty.add(graph_id)
            raise GraphManifestPersistenceError(str(exc), applied=True) from exc
        except Exception as exc:
            dirty.add(graph_id)
            raise GraphManifestPersistenceError(
                f"Graph {graph_id!r} mutated but its manifest checkpoint failed: {exc}",
                applied=True,
            ) from exc
        if invalid or engine._session_stores.get(graph_id) is not store:
            dirty.add(graph_id)
            return False
        dirty.discard(graph_id)
        return not missing


async def checkpoint_many(engine: "Terrarium", graph_ids: "Iterable[str]") -> None:
    """Checkpoint every existing persisted graph in deterministic order."""
    for graph_id in sorted(set(graph_ids)):
        await checkpoint(engine, graph_id)


def discard(engine: "Terrarium", graph_id: str) -> None:
    """Discard checkpoint bookkeeping for a graph that no longer exists."""
    dirty_graphs(engine).discard(graph_id)
    _engine_locks(engine).pop(graph_id, None)
