"""Engine mutation guards for graph-local creature identities."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from kohakuterrarium.core.model_switch_guard import require_model_mutation_available

from kohakuterrarium.terrarium.graph_identity import (
    GraphNameConflictError,
    creature_name_aliases,
    ensure_graph_name_available,
)
from kohakuterrarium.terrarium.session_plugin_state import validate_graph_plugin_state
from kohakuterrarium.terrarium.session_subagent_state import preflight_child_merge
from kohakuterrarium.terrarium.topology_admission import require_topology_available

if TYPE_CHECKING:
    from kohakuterrarium.terrarium.creature_host import Creature, CreatureRef
    from kohakuterrarium.terrarium.engine import Terrarium


def guard_graph_model_bindings(
    engine: "Terrarium", *graph_ids: str, store_preflight=True
) -> None:
    """Reject graph changes that would disturb a pending model publication."""
    require_topology_available(engine)
    for graph_id in set(graph_ids):
        graph = engine._topology.graphs.get(graph_id)
        if graph is not None:
            for cid in graph.creature_ids:
                creature = engine._creatures.get(cid)
                if creature is not None:
                    require_model_mutation_available(creature.agent)
    if store_preflight:
        validate_graph_plugin_state(engine, graph_ids)


@contextmanager
def reserve_graph_model_bindings(
    engine: "Terrarium", graph_id: str, *, store_preflight=True
):
    """Keep model replacement out of an asynchronous graph-removal window."""
    guard_graph_model_bindings(engine, graph_id, store_preflight=store_preflight)
    graph = engine._topology.graphs.get(graph_id)
    agents = (
        [
            engine._creatures[cid].agent
            for cid in graph.creature_ids
            if cid in engine._creatures
        ]
        if graph is not None
        else []
    )
    for agent in agents:
        agent._model_sensitive_extension_active = (
            getattr(agent, "_model_sensitive_extension_active", 0) + 1
        )
    try:
        yield
    finally:
        for agent in agents:
            agent._model_sensitive_extension_active -= 1


def bind_runtime_creature_id(creature: "Creature") -> None:
    """Expose the final engine runtime ID to tool execution contexts."""
    executor = getattr(creature.agent, "executor", None)
    if executor is not None:
        executor._creature_id = creature.creature_id


def guard_add_name(
    engine: "Terrarium", creature: "Creature", graph_id: str | None
) -> None:
    """Reject a duplicate role name when adding into an existing graph."""
    require_topology_available(engine)
    if graph_id is None:
        return
    for name in sorted(creature_name_aliases(creature)):
        ensure_graph_name_available(
            engine._topology,
            engine._creatures,
            graph_id=graph_id,
            name=name,
        )


def _graph_aliases(engine: "Terrarium", graph_id: str) -> dict[str, set[str]]:
    graph = engine._topology.graphs.get(graph_id)
    aliases: dict[str, set[str]] = {}
    if graph is None:
        return aliases
    for creature_id in graph.creature_ids:
        creature = engine._creatures.get(creature_id)
        if creature is None:
            continue
        for alias in creature_name_aliases(creature):
            aliases.setdefault(alias, set()).add(creature_id)
    return aliases


def resolve_and_guard_connect(
    engine: "Terrarium",
    sender: "CreatureRef",
    receiver: "CreatureRef",
    *,
    store_preflight=True,
) -> tuple[Any, Any]:
    """Resolve endpoints and reject merges that create duplicate names."""
    sender_creature = engine.get_creature(engine._resolve_creature_id(sender))
    receiver_creature = engine.get_creature(engine._resolve_creature_id(receiver))
    sender_graph_id = engine._topology.creature_to_graph.get(
        sender_creature.creature_id
    )
    receiver_graph_id = engine._topology.creature_to_graph.get(
        receiver_creature.creature_id
    )
    guard_graph_model_bindings(
        engine, sender_graph_id, receiver_graph_id, store_preflight=store_preflight
    )
    if (
        sender_graph_id != receiver_graph_id
        and sender_graph_id is not None
        and receiver_graph_id is not None
    ):
        sender_aliases = _graph_aliases(engine, sender_graph_id)
        receiver_aliases = _graph_aliases(engine, receiver_graph_id)
        conflicts = sorted(set(sender_aliases) & set(receiver_aliases))
        if conflicts:
            alias = conflicts[0]
            creature_ids = tuple(
                sorted(sender_aliases[alias] | receiver_aliases[alias])
            )
            raise GraphNameConflictError(
                graph_id=f"{sender_graph_id}+{receiver_graph_id}",
                name=alias,
                creature_ids=creature_ids,
            )
        if store_preflight:
            preflight_child_merge(engine, (sender_graph_id, receiver_graph_id))
    return sender_creature, receiver_creature
