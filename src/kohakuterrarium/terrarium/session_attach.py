"""Reserved explicit graph attachment with delayed source retirement."""

import asyncio
from pathlib import Path

import kohakuterrarium.terrarium.autosession as autosession
import kohakuterrarium.terrarium.channels as channels
import kohakuterrarium.terrarium.graph_checkpoint as checkpoint
from kohakuterrarium.core.session_attachment import bind_reserved_member
from kohakuterrarium.core.session_attachment_prepare import (
    prepare_attachment,
    install_attachment,
)
from kohakuterrarium.core.session_attachment_state import capture_attachment_state
from kohakuterrarium.terrarium.session_resources import (
    defer_store_close,
    release_failed_store,
)
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


def _bind(agent, store):
    if hasattr(agent, "_bind_session_store"):
        bind_reserved_member(agent, store)
    elif hasattr(agent, "attach_session_store"):
        agent.attach_session_store(store)
    elif hasattr(agent, "session_store"):
        agent.session_store = store


async def attach_reserved(engine, gid, store):
    """Keep the prior writer live and re-arm settled graph deliveries."""
    try:
        await _attach_reserved(engine, gid, store)
    finally:
        runtime = engine._drive_runtime
        if runtime is not None:
            for creature in engine._creatures.values():
                if (
                    creature.graph_id == gid
                    and creature.is_running
                    and not getattr(creature.agent, "_session_attach_uncertain", False)
                ):
                    runtime.schedule_reconcile(creature)


async def _attach_reserved(engine, gid, store):
    """Prepare, bind, checkpoint, and retire one reserved store replacement."""
    previous = engine._session_stores.get(gid)
    was_owned = gid in engine._owned_sessions
    minted = False
    if isinstance(store, (str, Path)):
        if (
            previous is not None
            and not getattr(previous, "_closed", False)
            and autosession.recipe_session_reuses_store(previous, store)
        ):
            store = previous
        else:
            names = [c.name for c in engine._creatures.values() if c.graph_id == gid]
            store = autosession.mint_store(engine, gid, path=store, agents=names)
            minted = True
    members = [c.agent for c in engine._creatures.values() if c.graph_id == gid]
    attempted = []
    prepared = {}
    runtime = engine._drive_runtime
    runtime_touched = False
    try:
        journals = [(agent, await capture_attachment_state(agent)) for agent in members]
        for agent in members:
            if hasattr(agent, "_bind_session_store"):
                prepared[id(agent)] = await prepare_attachment(agent, store)
        if previous is not None and previous is not store and runtime is not None:
            runtime_touched = True
            await runtime.detach_graph(gid)
        engine._session_stores[gid] = store
        if runtime is not None:
            runtime_touched = True
            await runtime.bind_graph_store(gid, store)
        env = engine._environments.get(gid)
        if env is not None:
            for channel in env.shared_channels._channels.values():
                channels._ensure_channel_persistence(channel, engine, gid)
        for agent, journal in journals:
            attempted.append((agent, journal))
            if id(agent) in prepared:
                await install_attachment(prepared[id(agent)])
            else:
                _bind(agent, store)
        await checkpoint.checkpoint(engine, gid)
    except BaseException:
        if previous is None:
            engine._session_stores.pop(gid, None)
        else:
            engine._session_stores[gid] = previous
        for agent, journal in reversed(attempted):
            try:
                await journal.restore()
            except (Exception, asyncio.CancelledError):
                defer_store_close(engine, store)
                agent._session_attach_uncertain = True
                logger.error("Session attachment rollback failed", exc_info=True)
        if runtime is not None and runtime_touched:
            try:
                await runtime.detach_graph(gid)
                if previous is not None:
                    await runtime.bind_graph_store(gid, previous)
            except (Exception, asyncio.CancelledError):
                defer_store_close(engine, store)
                for agent in members:
                    agent._session_attach_uncertain = True
                logger.error("Drive attachment rollback failed", exc_info=True)
        for agent, _ in attempted:
            item = prepared.get(id(agent))
            if item is not None:
                try:
                    await item.drain()
                except (Exception, asyncio.CancelledError):
                    defer_store_close(engine, store)
                    agent._session_attach_uncertain = True
                    logger.error("Attachment receipt settlement failed", exc_info=True)
        if minted:
            release_failed_store(engine, store)
        raise
    if minted or (was_owned and (previous is None or store is previous)):
        engine._owned_sessions.add(gid)
    else:
        engine._owned_sessions.discard(gid)
    if previous is not None and previous is not store:
        try:
            previous.close(update_status=False)
        except Exception:
            logger.warning("Replaced session store retirement failed", exc_info=True)
