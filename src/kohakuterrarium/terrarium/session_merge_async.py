"""Prepare merged session data before publishing graph membership."""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.terrarium import (
    session_coord,
    session_plugin_state,
    session_subagent_state,
)
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.session_attachment import relocate_topology_selectors
from kohakuterrarium.session.output import SessionOutput
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.version import FORMAT_VERSION
from kohakuterrarium.terrarium.session_resources import (
    archive_unpublished_store,
    retain_store,
)
from kohakuterrarium.terrarium.session_successor import stage_successor_async
from kohakuterrarium.terrarium.topology_admission import (
    fence_topology_members,
    record_topology_install,
    record_topology_transfer,
)
from kohakuterrarium.utils.logging import get_logger
from kohakuterrarium.modules.plugin.state_scope import (
    state_prefix,
    validate_legacy_state,
)

logger = get_logger(__name__)


def _retire_metadata(store):
    store.set_conversation_open(False)
    store.update_status("completed")
    store.checkpoint()


def _metadata(store, inherited, parents, names, kind, fresh):
    store.meta["session_id"] = store.session_id
    session_coord._inherit_resumable_meta(inherited, store)
    session_coord._inherit_conversation_lifecycle(
        inherited, store, new_identity=kind == "split" and fresh
    )
    store.meta["format_version"] = FORMAT_VERSION
    store.meta["parent_session_ids"] = parents
    store.meta["split_at" if kind == "split" else "merged_at"] = time.time()
    store.meta["agents"] = names
    store.meta["config_type"] = "agent" if len(names) <= 1 else "terrarium"
    store.meta["config_path"] = None
    session_coord._delete_meta(store, "config_snapshot")
    session_coord._delete_meta(store, "runtime_topology")
    store.flush()


def _capture_member(agent, coordinator):
    if getattr(agent, "_topology_handoff_task", None) is not coordinator:
        raise ConflictError("Merge member has no task-owned reservation")
    manager = agent.subagent_manager
    children = tuple(
        (job.subagent, job.subagent._run_index, job.subagent._run_receipt)
        for job in manager._jobs.values()
    )
    return (
        agent.session_store,
        agent.controller,
        agent.controller.conversation,
        agent._session_output,
        children,
    )


def _install_member(
    agent,
    coordinator,
    source,
    controller,
    conversation,
    old_sink,
    children,
    sink,
    destination,
    mapping,
):
    if (
        getattr(agent, "_topology_handoff_task", None) is not coordinator
        or agent.session_store is not source
        or agent.controller is not controller
        or controller.conversation is not conversation
        or agent._session_output is not old_sink
    ):
        raise ConflictError("Merge member binding changed before install")
    current = tuple(job.subagent for job in agent.subagent_manager._jobs.values())
    if current != tuple(child for child, _, _ in children):
        raise ConflictError("Merge child membership changed before install")
    agent.session_store = controller.session_store = destination
    outputs = agent.output_router._secondary_outputs
    if old_sink in outputs:
        outputs[outputs.index(old_sink)] = sink
    elif sink not in outputs:
        outputs.append(sink)
    agent._session_output = sink
    manager = agent.subagent_manager
    manager._session_store, manager._parent_name = destination, agent.config.name
    for child, run, receipt in children:
        child._session_store, child._parent_name = destination, agent.config.name
        if child._run_index is None and child._run_receipt is None:
            child._run_index, child._run_receipt = run, receipt
    agent.trigger_manager._session_store = destination
    agent.trigger_manager._agent_name = agent.config.name
    if agent.compact_manager is not None:
        agent.compact_manager._session_store = destination
    if mapping is not None:
        relocate_topology_selectors(agent, destination, coordinator, *mapping, old_sink)


@dataclass
class PreparedMerge:
    engine: object
    delta: object
    destination: object
    sources: dict
    owned: set
    fresh: bool
    coordinator: object
    members: list = field(default_factory=list)
    installed: bool = False
    authority_changed: bool = False
    target_changed: bool = False
    agents: list = field(default_factory=list)
    baseline: object = None

    def validate(self):
        if self.engine._topology != self.baseline or any(
            not any(
                creature.agent is agent for creature in self.engine._creatures.values()
            )
            for agent in self.agents
        ):
            raise ConflictError("Prepared topology changed before member publication")

    async def install(self):
        self.validate()
        self.installed = True
        if self.fresh:
            retain_store(self.engine, self.destination)
        for graph, store in self.sources.items():
            if graph in self.owned:
                retain_store(self.engine, store)
        if self.delta.kind == "merge":
            for store in dict.fromkeys(self.sources.values()):
                if store is not self.destination:
                    await stage_successor_async(self.engine, store, [self.destination])
        for agent, capture, sink, mapping in self.members:
            await submit_internal_session_state(
                agent,
                lambda: _install_member(
                    agent, self.coordinator, *capture, sink, self.destination, mapping
                ),
            )
            record_topology_install(agent, capture[0], self.destination)
        self.validate()

    def publish(self):
        keep = self.delta.new_graph_ids[0]
        self.engine._session_stores[keep] = self.destination
        for graph in self.delta.old_graph_ids:
            if graph != keep:
                self.engine._session_stores.pop(graph, None)
            self.engine._owned_sessions.discard(graph)
        if self.fresh or any(
            store is self.destination and graph in self.owned
            for graph, store in self.sources.items()
        ):
            self.engine._owned_sessions.add(keep)
            getattr(self.engine, "_retained_session_stores", set()).discard(
                self.destination
            )

    async def retire(self):
        retired = set()
        for graph, store in self.sources.items():
            if (
                store is not self.destination
                and graph in self.owned
                and store not in retired
            ):
                retired.add(store)
                try:
                    await store.run(_retire_metadata, store)
                    await asyncio.to_thread(store.close, update_status=False)
                    getattr(self.engine, "_retained_session_stores", set()).discard(
                        store
                    )
                except Exception:
                    retain_store(self.engine, store)
                    logger.error(
                        "Committed merge retained an unretired source", exc_info=True
                    )

    async def abort(self):
        if (
            self.installed
            or self.authority_changed
            or (self.target_changed and not self.fresh)
        ):
            if self.fresh and self.installed:
                retain_store(self.engine, self.destination)
            for graph, store in self.sources.items():
                if graph in self.owned:
                    retain_store(self.engine, store)
            await fence_topology_members(self.agents)
        if self.fresh and not self.installed:
            try:
                await asyncio.to_thread(self.destination.close, update_status=False)
            except Exception:
                retain_store(self.engine, self.destination)
                logger.error("Unpublished handoff target close failed", exc_info=True)
                return
            if not self.authority_changed:
                try:
                    await asyncio.to_thread(archive_unpublished_store, self.destination)
                except Exception:
                    logger.error(
                        "Unpublished handoff target archive failed",
                        path=str(self.destination.path),
                        exc_info=True,
                    )


async def prepare_merge(engine, projected, delta):
    """Build detached target state on workers and stage owner-only binding commits."""
    keep = delta.new_graph_ids[0]
    source_order = [keep, *(graph for graph in delta.old_graph_ids if graph != keep)]
    sources = {
        graph: engine._session_stores[graph]
        for graph in source_order
        if graph in engine._session_stores
    }
    if not sources or not all(
        isinstance(store, SessionStore) for store in sources.values()
    ):
        return None
    for source in dict.fromkeys(sources.values()):
        await source.run(validate_legacy_state, source)
    keep = delta.new_graph_ids[0]
    members = [
        engine._creatures[cid].agent
        for cid in sorted(projected.graphs[keep].creature_ids)
        if cid in engine._creatures
    ]
    if any(getattr(agent, "_session_state_owner", None) is None for agent in members):
        return None
    if any(cid not in engine._creatures for cid in projected.graphs[keep].creature_ids):
        raise ConflictError("Projected graph contains an unavailable creature")
    if any(
        agent.session_store is not None and agent.session_store not in sources.values()
        for agent in members
    ):
        raise ConflictError("Merge member store is outside its graph source bindings")
    names = {}
    for agent in members:
        names.setdefault(agent.session_store, set()).add(agent.config.name)
    if delta.kind == "merge":
        for source, owned_names in names.items():
            if source is None:
                continue
            for name in owned_names:
                payload = await source.run(
                    session_subagent_state.capture_children, source, name
                )
                for other in dict.fromkeys(sources.values()):
                    if other is not source:
                        await other.run(
                            session_subagent_state.validate_children, other, payload
                        )
    path = session_coord._store_path_for(engine, keep)
    destination = sources.get(keep)
    if path is None:
        destination = destination or next(iter(sources.values()))
    elif destination is None or Path(destination.path).resolve() != path.resolve():
        if path.exists():
            raise ConflictError("Merge destination exists outside its selected source")
        destination = await asyncio.to_thread(SessionStore, path)
    fresh = destination not in sources.values()
    plan = PreparedMerge(
        engine,
        delta,
        destination,
        sources,
        set(engine._owned_sessions),
        fresh,
        asyncio.current_task(),
    )
    plan.agents = members
    plan.baseline = deepcopy(engine._topology)
    try:
        metadata, captures, mappings, child_transfers, plugin_baselines = (
            [],
            {},
            {},
            [],
            [],
        )
        for source in dict.fromkeys(sources.values()):
            metadata.append(await source.run(source.load_meta))
            if source is destination:
                continue
            if Path(source.path).resolve() == Path(destination.path).resolve():
                raise ConflictError("Transfer store handles share the same path")
            payload = await source.run(
                session_coord.capture_runtime_transfer, source, names.get(source, set())
            )
            captures[source] = payload
            for name in names.get(source, ()):
                child = await source.run(
                    session_subagent_state.capture_children, source, name
                )
                await destination.run(
                    session_subagent_state.validate_children, destination, child
                )
                child_transfers.append((source, child, str(uuid4())))
            plan.target_changed = True
            receipts = await destination.run(
                session_coord.install_runtime_transfer, destination, payload
            )
            record_topology_transfer(source, destination, payload[0][0], receipts)
            mappings[source] = (
                payload[0][0],
                {tuple(row["origin"]): row["destination"][3] for row in receipts},
                {
                    (row["source"][1], row["source"][3]): row["destination"][3]
                    for row in receipts
                },
            )
            for name in names.get(source, ()):
                prefix = state_prefix(name)
                values = await source.run(session_plugin_state._capture, source, prefix)
                plugin_baselines.append((source, prefix, values))
                await destination.run(
                    session_plugin_state._replace, destination, prefix, values
                )
        for agent in members:
            if agent.session_store is destination:
                continue
            capture = await submit_internal_session_state(
                agent, lambda: _capture_member(agent, plan.coordinator)
            )
            children = []
            for child, run, receipt in capture[4]:
                if run is None and receipt is None:
                    receipt = destination.submit(
                        destination.next_subagent_run,
                        agent.config.name,
                        child.config.name,
                    )
                    await asyncio.shield(asyncio.wrap_future(receipt))
                children.append((child, run, receipt))
            capture = (*capture[:4], tuple(children))
            sink = SessionOutput(
                agent.config.name,
                destination,
                agent,
                capture_activity=getattr(capture[3], "_capture_activity", True),
                prepared_recovery=("", None, None, None),
            )
            plan.members.append((agent, capture, sink, mappings.get(capture[0])))
        for source, captured in captures.items():
            current = await source.run(
                session_coord.capture_runtime_transfer, source, names.get(source, set())
            )
            if current != captured:
                raise ConflictError(
                    "Merge source changed during asynchronous preparation"
                )
        for source, prefix, values in plugin_baselines:
            if (
                await source.run(session_plugin_state._capture, source, prefix)
                != values
            ):
                raise ConflictError(
                    "Plugin state changed during asynchronous preparation"
                )
        for source, payload, successor in child_transfers:
            plan.authority_changed = (
                plan.authority_changed or payload["lineage"] is not None
            )
            await source.run(
                session_subagent_state.depart_children, source, payload, successor
            )
            await destination.run(
                session_subagent_state.import_children, destination, payload, successor
            )
        plan.target_changed = True
        await destination.run(
            _metadata,
            destination,
            deepcopy(metadata[0]),
            [item["session_id"] for item in metadata if item.get("session_id")],
            [agent.config.name for agent in members],
            delta.kind,
            fresh,
        )
        return plan
    except BaseException:
        await plan.abort()
        raise
