"""Task-owned graph mutation admission and prepublication source drains."""

import asyncio
from contextvars import ContextVar
from copy import deepcopy

from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core.agent_state import submit_internal_session_state
from kohakuterrarium.core.branch_selection import _join_accepted
from kohakuterrarium.core.session_attachment import relocate_topology_selectors
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.terrarium.drive.handoff import drive_handoff
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)

_TRANSFERS = ContextVar("kt_topology_transfer_receipts", default=None)


async def fence_topology_members(agents):
    """Attempt every owner fence without replacing the transition's primary failure."""
    for agent in agents:
        try:
            await submit_internal_session_state(
                agent, lambda: setattr(agent, "_session_attach_uncertain", True)
            )
        except Exception:
            logger.error(
                "Topology recovery fence was not confirmed",
                agent_name=agent.config.name,
                exc_info=True,
            )


def record_topology_transfer(source, destination, source_uid, receipts):
    captured = _TRANSFERS.get()
    if captured is None or captured[0] is not asyncio.current_task():
        return
    canonical = {
        tuple(receipt["origin"]): receipt["destination"][3] for receipt in receipts
    }
    local = {
        (receipt["source"][1], receipt["source"][3]): receipt["destination"][3]
        for receipt in receipts
    }
    captured[1][source, destination] = source_uid, canonical, local


def record_topology_install(agent, source, destination):
    """Record an awaited owner install in the current coordinator's receipt scope."""
    captured = _TRANSFERS.get()
    if captured is None or captured[0] is not asyncio.current_task():
        raise ConflictError("Topology install has no task-owned receipt scope")
    captured[2].add((agent, source, destination))


async def _relocate_selectors(sources, outputs, transfers, installed, coordinator):
    for (agent, source), output in zip(sources, outputs):
        destination = agent.session_store
        if (
            source is None
            or destination is source
            or (agent, source, destination) in installed
        ):
            continue
        transfer = transfers.get((source, destination))
        if transfer is None:
            raise ConflictError("Live selector handoff has no confirmed transfer")
        await submit_internal_session_state(
            agent,
            lambda: relocate_topology_selectors(
                agent, destination, coordinator, *transfer, output
            ),
        )


def require_topology_available(engine):
    task = getattr(engine, "_topology_handoff_task", None)
    if task is None:
        return
    try:
        caller = asyncio.current_task()
    except RuntimeError:
        caller = None
    if caller is not task:
        raise ConflictError("Graph handoff is awaiting completion")


def _participants(engine, graph_ids):
    stores = {
        engine._session_stores[gid]
        for gid in graph_ids
        if gid in engine._session_stores
    }
    return tuple(
        creature.agent
        for creature in engine._creatures.values()
        if creature.graph_id in graph_ids
        or getattr(creature.agent, "session_store", None) in stores
    )


async def run_topology_operation(engine, graph_ids, operation, *, quiesce=None):
    """Reserve source bindings and join accepted draining before a graph mutation."""
    require_topology_available(engine)
    if getattr(engine, "_topology_handoff_task", None) is asyncio.current_task():
        if quiesce is not None:
            raise ConflictError(
                "Cannot stop a creature inside an admitted topology mutation"
            )
        return await operation()

    topology = deepcopy(engine._topology)
    agents = _participants(engine, graph_ids)
    sources = [(agent, getattr(agent, "session_store", None)) for agent in agents]
    outputs = [getattr(agent, "_session_output", None) for agent in agents]

    def validate():
        if engine._topology != topology or _participants(engine, graph_ids) != agents:
            raise ConflictError("Graph handoff topology changed before publication")
        if any(
            getattr(agent, "_session_output", None) is not output
            for agent, output in zip(agents, outputs)
        ):
            raise ConflictError("Graph handoff output binding changed while draining")
        if any(agent.session_store is not store for agent, store in sources):
            raise ConflictError("Graph handoff source binding changed while draining")

    async def accepted():
        require_topology_available(engine)
        validate()
        for agent in agents:
            require_model_mutation_available(agent, allow_compact_pending=True)
        task = asyncio.current_task()
        if quiesce is not None:
            engine._topology_handoff_task = task
            try:
                await quiesce()
                validate()
            finally:
                engine._topology_handoff_task = None
        ready = asyncio.Event()
        publications = tuple(
            pending
            for agent in agents
            for pending in getattr(
                getattr(agent, "output_router", None), "_terminal_publications", ()
            )
            if not getattr(pending, "_topology_deferred", False)
        )
        output_calls = tuple(
            receipt
            for output in outputs
            if output is not None
            for receipt in getattr(output, "_inflight_output_calls", ())
        )
        child_writes = tuple(
            receipt
            for agent in agents
            for receipt in getattr(agent, "_inflight_child_writes", ())
        )
        transfers, installed = {}, set()
        token = _TRANSFERS.set((task, transfers, installed))
        engine._topology_handoff_task = task
        for agent in agents:
            agent._topology_handoff_task = task
            agent._topology_handoff_ready = ready
        try:
            if publications or output_calls or child_writes:
                await asyncio.gather(
                    *(
                        asyncio.shield(pending)
                        for pending in (*publications, *output_calls, *child_writes)
                    )
                )
            for output in outputs:
                if output is not None:
                    output._flush_text_segment()
            for output in outputs:
                if output is not None:
                    await output.drain()
            for store in {store for _, store in sources if store is not None}:
                await store.run(store.flush)
            validate()
            for agent in agents:
                require_model_mutation_available(agent)
            try:
                async with drive_handoff(engine, graph_ids):
                    result = await operation()
                    await _relocate_selectors(
                        sources, outputs, transfers, installed, task
                    )
            except BaseException:
                if installed or any(
                    agent.session_store is not store for agent, store in sources
                ):
                    await fence_topology_members(agents)
                raise
            return result
        finally:
            _TRANSFERS.reset(token)
            for agent in agents:
                if getattr(agent, "_topology_handoff_task", None) is task:
                    agent._topology_handoff_task = None
            if getattr(engine, "_topology_handoff_task", None) is task:
                engine._topology_handoff_task = None
            ready.set()

    return await _join_accepted(asyncio.create_task(accepted()))
