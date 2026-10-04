"""Memory-only explicit attachment admission and reserved binding operations."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy

from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.session_generation_handoff import validate_generation_handoff
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.event_origin import validate_origin

_TOPOLOGY_BINDINGS = ContextVar("kt_topology_attachment", default=None)


@contextmanager
def reserve_topology_attachment(agents):
    """Scope synchronous handoff binding to its affected source hosts."""
    bindings = {}
    for agent in agents:
        require_model_mutation_available(agent)
        bindings[id(agent)] = (
            agent,
            getattr(agent, "session_store", None),
            agent.config.name,
        )
    token = _TOPOLOGY_BINDINGS.set(bindings)
    try:
        yield
    finally:
        _TOPOLOGY_BINDINGS.reset(token)


def topology_source_names(store):
    """Return namespaces owned by a source store in the current synchronous handoff."""
    bindings = _TOPOLOGY_BINDINGS.get()
    if bindings is None:
        return None
    return {name for _, source, name in bindings.values() if source is store}


def relocate_topology_selectors(
    agent, destination, coordinator, source_uid, canonical, local, source_output=None
):
    """Commit receipt-proven local selector changes on the creature's owner."""
    if (
        agent.session_store is not destination
        or getattr(agent, "_topology_handoff_task", None) is not coordinator
    ):
        raise ConflictError("Live selector handoff binding changed")
    messages = list(agent.controller.conversation.get_messages())
    for field in ("_runtime_context_bundle", "_installed_runtime_context_bundle"):
        bundle = getattr(agent, field, None)
        messages.extend(getattr(bundle, "provider_conversation", ()))
    seen, updates = set(), []
    for message in messages:
        if id(message) in seen:
            continue
        seen.add(id(message))
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, dict) or not (
            "event_id" in metadata or "event_origin" in metadata
        ):
            continue
        if "event_origin" in metadata:
            try:
                origin = validate_origin(metadata["event_origin"], agent.config.name)
            except (TypeError, ValueError) as exc:
                raise ConflictError(
                    "Invalid live selector origin during handoff"
                ) from exc
            event_id = canonical.get(origin)
            if event_id is None and origin[0] == source_uid:
                raise ConflictError("Live selector missing from confirmed transfer")
        else:
            event_id = local.get((agent.config.name, metadata["event_id"]))
            if event_id is None:
                raise ConflictError("Unproven local selector during handoff")
        updates.append((metadata, event_id))
    target_output = getattr(agent, "_session_output", None)
    if source_output is not None and target_output is not source_output:
        if target_output is None or target_output._store is not destination:
            raise ConflictError("Output binding changed before generation handoff")
        validate_generation_handoff(source_output, target_output)
        for name in (
            "_generation_id",
            "_generation_coordinates",
            "_generation_payload",
            "_chunk_seq",
        ):
            setattr(target_output, name, deepcopy(getattr(source_output, name)))
        source_output._handoff_successor = target_output
    for metadata, event_id in updates:
        if event_id is None:
            metadata.pop("event_id", None)
        else:
            metadata["event_id"] = event_id


def require_attachment_idle(agent):
    """Reject busy attachment before reading or mutating either session store."""
    require_model_mutation_available(agent)
    for name in ("_processing_lock", "_stop_lock"):
        lock = getattr(agent, name, None)
        if lock is not None and lock.locked():
            raise ConflictError("Cannot attach a session while the agent is busy")
    if getattr(agent, "_model_sensitive_extension_active", 0) or getattr(
        agent, "_turn_dispatched_bg", None
    ):
        raise ConflictError(
            "Cannot attach a session with unfinished extension or delivery work"
        )
    if getattr(agent, "_starting", False) or getattr(agent, "_event_inbox", None):
        raise ConflictError("Cannot attach a session with pending agent work")
    processing = getattr(agent, "_processing_task", None)
    if processing is not None and not processing.done():
        raise ConflictError("Cannot attach a session during a controller turn")
    if getattr(getattr(agent, "compact_manager", None), "is_compacting", False):
        raise ConflictError("Cannot attach a session during compaction")
    for manager_name in ("executor", "subagent_manager"):
        manager = getattr(agent, manager_name, None)
        if manager is None:
            continue
        tasks = getattr(manager, "_tasks", {})
        if any(not task.done() for task in tasks.values()):
            raise ConflictError("Cannot attach a session with unfinished jobs or hooks")
        running = getattr(manager, "get_running_jobs", None)
        if callable(running) and running():
            raise ConflictError("Cannot attach a session with running jobs")
    publications = getattr(
        getattr(agent, "output_router", None), "_terminal_publications", ()
    )
    writes = getattr(getattr(agent, "_session_output", None), "_pending_writes", ())
    if any(not task.done() for task in (*publications, *writes)):
        raise ConflictError("Cannot attach a session with pending publication")
    if any(
        entry.get("state") != "published"
        for entry in getattr(agent, "_background_terminals", {}).values()
    ):
        raise ConflictError(
            "Cannot attach a session with unresolved terminal publication"
        )


def attach_explicit(agent, store, *, capture_activity=True):
    require_attachment_idle(agent)
    agent._bind_session_store(store, capture_activity=capture_activity)


def attach_topology_member(agent, store):
    """Bind one member inside the synchronous topology handoff coordinator."""
    require_model_mutation_available(agent)
    binder = getattr(agent, "_bind_session_store", None)
    if binder is None:
        agent.attach_session_store(store)
    else:
        bindings = _TOPOLOGY_BINDINGS.get()
        source = bindings.get(id(agent)) if bindings is not None else None
        if (
            source is None
            or source[0] is not agent
            or source[1] is not agent.session_store
        ):
            raise ConflictError(
                "Topology attachment has no matching source reservation"
            )
        bindings.pop(id(agent))
        binder(store)


async def run_explicit_attachment(agents, operation):
    """Join an accepted graph attachment before reporting caller cancellation."""
    for agent in agents:
        require_attachment_idle(agent)

    async def run():
        async with reserve_explicit_attachment(agents):
            await operation()

    pending = asyncio.create_task(run())
    cancelled = False
    while True:
        try:
            await asyncio.shield(pending)
            break
        except asyncio.CancelledError:
            if pending.cancelled():
                raise
            cancelled = True
        except BaseException as exc:
            if cancelled:
                raise asyncio.CancelledError() from exc
            raise
    if cancelled:
        raise asyncio.CancelledError()


def bind_reserved_member(agent, store):
    """Install options synchronously while retaining the reserved turn mutex."""
    if getattr(agent, "_session_attach_task", None) is not asyncio.current_task():
        raise ConflictError("Session attachment reservation does not belong to caller")
    agent._session_attach_pending = False
    try:
        require_model_mutation_available(agent)
        agent._bind_session_store(store)
    finally:
        agent._session_attach_pending = True


@asynccontextmanager
async def reserve_explicit_attachment(agents):
    """Keep graph members idle across Drive detach/bind awaits during attachment."""
    for agent in agents:
        require_attachment_idle(agent)
    locks, marked = [], []
    try:
        for agent in agents:
            agent._session_attach_pending = True
            agent._session_attach_task = asyncio.current_task()
            marked.append(agent)
            lock = getattr(agent, "_processing_lock", None)
            if lock is not None:
                await lock.acquire()
                locks.append(lock)
        yield
    finally:
        for agent in marked:
            agent._session_attach_pending = False
            agent._session_attach_task = None
        for lock in reversed(locks):
            lock.release()
