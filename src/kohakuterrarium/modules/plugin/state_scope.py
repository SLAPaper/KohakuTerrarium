"""Creature-scoped plugin state and unambiguous legacy ownership."""

import json
from copy import deepcopy

from kohakuterrarium.errors import ConflictError


def state_prefix(namespace: str) -> str:
    """Encode the storage namespace independently of plugin/key delimiters."""
    if not isinstance(namespace, str) or not namespace:
        raise ConflictError("Plugin state requires a creature namespace")
    return f"plugin_state:{json.dumps(namespace, ensure_ascii=False)}:"


def state_keys(state, prefix: str) -> list[str]:
    keys = (
        state.keys()
        if isinstance(state, dict)
        else state.keys(prefix=prefix, limit=2**31 - 1)
    )
    decoded = [key.decode() if isinstance(key, bytes) else key for key in keys]
    return [key for key in decoded if isinstance(key, str) and key.startswith(prefix)]


def _legacy_plan(store):
    state = getattr(store, "state", None)
    keys = state_keys(state, "plugin:") if state is not None else []
    if not keys:
        return [], {}
    names = set()
    meta = getattr(store, "meta", {})
    agents = meta.get("agents", [])
    if not isinstance(agents, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in agents
    ):
        raise ConflictError("Invalid agent inventory for legacy plugin state")
    names.update(agents)
    discover = getattr(store, "discover_agents_from_events", None)
    if callable(discover):
        names.update(discover())
    if len(names) != 1:
        raise ConflictError(
            "Cannot determine creature ownership of legacy plugin state"
        )
    prefix = state_prefix(next(iter(names)))
    values = {}
    for key in keys:
        target = prefix + key.removeprefix("plugin:")
        value = deepcopy(store.state[key])
        if target in store.state and store.state[target] != value:
            raise ConflictError("Conflicting legacy and creature-scoped plugin state")
        values[target] = value
    return keys, values


def validate_legacy_state(store) -> None:
    """Reject ambiguous legacy ownership without changing stored state."""
    _legacy_plan(store)


def prepare_legacy_stores(stores) -> None:
    """Validate all legacy ownership before migrating any participating store."""
    plans = [(store, *_legacy_plan(store)) for store in stores]
    for store, keys, values in plans:
        for key, value in values.items():
            store.state[key] = value
        for key in keys:
            del store.state[key]


def migrate_legacy_state(store) -> None:
    """Move legacy plugin keys only when persisted agent identity is unambiguous."""
    prepare_legacy_stores([store])


def context_state_key(context, key: str) -> str:
    """Resolve the host's current storage namespace, not a stale display name."""
    config = getattr(context.host_agent, "config", None)
    namespace = getattr(config, "name", None) or context.agent_name
    return state_prefix(namespace) + f"{context._plugin_name}:{key}"


def transfer_plugin_state(source, destination, namespace: str) -> None:
    """Replace only the moving creature's plugin state in the destination."""
    if source is None or source is destination:
        return
    prepare_legacy_stores([source, destination])
    prefix = state_prefix(namespace)
    values = {
        key: deepcopy(source.state[key]) for key in state_keys(source.state, prefix)
    }
    for key, value in values.items():
        destination.state[key] = value
    for key in state_keys(destination.state, prefix):
        if key not in values:
            del destination.state[key]
