"""Detached creature-scoped plugin state transfer across store workers."""

from contextlib import ExitStack
from copy import deepcopy
from functools import wraps

from kohakuterrarium.modules.plugin.state_io import plugin_state_binding

from kohakuterrarium.modules.plugin.state_scope import (
    migrate_legacy_state,
    state_keys,
    state_prefix,
    validate_legacy_state,
)


def guarded_plugin_state_handoff(operation):
    """Reserve all affected host bindings through source retirement and attach."""

    @wraps(operation)
    def guarded(engine, delta):
        if delta.kind not in ("merge", "split"):
            return operation(engine, delta)
        hosts = {
            id(creature.agent): creature.agent
            for cid in delta.affected_creatures
            if (creature := engine._creatures.get(cid)) is not None
        }
        with ExitStack() as stack:
            for _, host in sorted(hosts.items()):
                stack.enter_context(plugin_state_binding(host))
            stores = {
                host.session_store
                for host in hosts.values()
                if getattr(host, "session_store", None) is not None
            }
            for store in stores:
                store.submit(validate_legacy_state, store).result()
            for store in stores:
                store.submit(migrate_legacy_state, store).result()
            return operation(engine, delta)

    return guarded


def validate_graph_plugin_state(engine, graph_ids):
    """Reject ambiguous legacy ownership before graph identities change."""
    stores = {
        engine._session_stores[gid]
        for gid in graph_ids
        if gid in engine._session_stores
    }
    for store in stores:
        store.submit(validate_legacy_state, store).result()


def _capture(store, prefix):
    migrate_legacy_state(store)
    return {key: deepcopy(store.state[key]) for key in state_keys(store.state, prefix)}


def _replace(store, prefix, values):
    migrate_legacy_state(store)
    for key, value in values.items():
        store.state[key] = value
    for key in state_keys(store.state, prefix):
        if key not in values:
            del store.state[key]


def transfer_graph_plugin_state(engine, graph_id, destination):
    """Transfer moving creatures' state before source retirement or rebinding."""
    graph = engine._topology.graphs.get(graph_id)
    if graph is None:
        return
    captured = []
    for cid in graph.creature_ids:
        creature = engine._creatures.get(cid)
        if creature is None:
            continue
        source = getattr(creature.agent, "session_store", None)
        if source is None or source is destination:
            continue
        prefix = state_prefix(creature.agent.config.name)
        values = source.submit(_capture, source, prefix).result()
        captured.append((prefix, values))
    for prefix, values in captured:
        destination.submit(_replace, destination, prefix, values).result()
