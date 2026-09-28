"""Acknowledged plugin state operations on the session store affinity worker."""

import asyncio
import threading
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from kohakuterrarium.errors import InvalidRequestError
from kohakuterrarium.core.model_switch_guard import require_topology_mutation_available
from kohakuterrarium.modules.plugin.state_scope import (
    context_state_key,
    migrate_legacy_state,
)


def read_plugin_state(store, key: str) -> Any:
    """Migrate legacy keys, then copy one scoped value off the store thread."""
    migrate_legacy_state(store)
    value = store.state.get(key)
    return None if value is None else deepcopy(value)


def write_plugin_state(store, key: str, value: Any) -> None:
    """Migrate legacy keys, then store an already-copied scoped value."""
    migrate_legacy_state(store)
    store.state[key] = value


_binding_creation_lock = threading.Lock()


def plugin_state_binding(host):
    """Serialize external state calls with replacement of their host's store."""
    if host is None:
        return nullcontext()
    require_topology_mutation_available(host)
    with _binding_creation_lock:
        lock = getattr(host, "_plugin_state_binding_lock", None)
        if lock is None:
            lock = host._plugin_state_binding_lock = threading.RLock()
        return lock


def _run_sync(store, operation, *args):
    """Execute within an accepted worker call or wait for a newly queued call."""
    if threading.get_ident() == store._affinity_worker_ident:
        return operation(store, *args)
    return store.submit(operation, store, *args).result()


def _external_loop(context):
    owner = getattr(context.host_agent, "_session_state_owner", None)
    loop = getattr(owner, "_loop", None)
    if loop is None:
        return None
    try:
        if asyncio.get_running_loop() is loop:
            return None
    except RuntimeError:
        pass
    if not loop.is_running() or not owner.is_running:
        raise InvalidRequestError("Plugin state owner is not running")
    return loop


async def _after_handoff(context, operation, *args):
    host = context.host_agent
    while getattr(host, "_topology_handoff_task", None) is not None:
        await host._topology_handoff_ready.wait()
    return await operation(context, *args)


def get_plugin_state_sync(context, key: str) -> Any:
    """Return a detached state value, blocking the caller until completion."""
    store = context.session_store
    if store is not None and threading.get_ident() == store._affinity_worker_ident:
        return read_plugin_state(store, context_state_key(context, key))
    loop = _external_loop(context)
    if loop is not None:
        return asyncio.run_coroutine_threadsafe(
            _after_handoff(context, get_plugin_state, key), loop
        ).result()
    with plugin_state_binding(context.host_agent):
        store = context.session_store
        if store is None:
            return None
        return _run_sync(store, read_plugin_state, context_state_key(context, key))


def set_plugin_state_sync(context, key: str, value: Any) -> None:
    """Confirm a copied state write synchronously on the store worker."""
    store = context.session_store
    if store is not None and threading.get_ident() == store._affinity_worker_ident:
        write_plugin_state(store, context_state_key(context, key), deepcopy(value))
        return
    loop = _external_loop(context)
    if loop is not None:
        asyncio.run_coroutine_threadsafe(
            _after_handoff(context, set_plugin_state, key, deepcopy(value)), loop
        ).result()
        return
    with plugin_state_binding(context.host_agent):
        store = context.session_store
        if store is None:
            return
        _run_sync(
            store, write_plugin_state, context_state_key(context, key), deepcopy(value)
        )


async def get_plugin_state(context, key: str) -> Any:
    """Read creature-scoped state on the store worker, or None when detached.

    The storage key is captured on the caller loop. Table IO and legacy
    migration run inside ``store.run``. A deepcopy of the stored value is made
    on the worker before the result is handed back. Worker exceptions propagate.
    """
    with plugin_state_binding(context.host_agent):
        store = context.session_store
        if store is None:
            return None
        scoped_key = context_state_key(context, key)
        future = store.submit(read_plugin_state, store, scoped_key)
    return await asyncio.shield(asyncio.wrap_future(future))


async def set_plugin_state(context, key: str, value: Any) -> None:
    """Write creature-scoped state on the store worker, or no-op when detached.

    The input is deep-copied before submission so later caller mutation cannot
    change the queued value. Accepted work follows the ``store.run`` contract:
    cancelling the caller stops waiting and does not roll the write back.
    ``store.close`` drains that accepted write before disposing tables.
    """
    with plugin_state_binding(context.host_agent):
        store = context.session_store
        if store is None:
            return None
        scoped_key = context_state_key(context, key)
        copied = deepcopy(value)
        future = store.submit(write_plugin_state, store, scoped_key, copied)
    await asyncio.shield(asyncio.wrap_future(future))
