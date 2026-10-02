"""Prepare every split destination before publishing any new graph."""

import asyncio
from dataclasses import dataclass, field, replace

from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.session_merge_async import (
    _retire_metadata,
    prepare_merge,
)
from kohakuterrarium.terrarium.session_resources import retain_store
from kohakuterrarium.terrarium.session_successor import stage_successor_async
from kohakuterrarium.terrarium.topology_admission import fence_topology_members
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedSplit:
    engine: object
    delta: object
    source: object
    owned: bool
    agents: list
    plans: list = field(default_factory=list)
    retirement_staged: bool = False

    async def install(self):
        try:
            for plan in self.plans:
                plan.validate()
            if all(plan.destination is not self.source for plan in self.plans):
                self.retirement_staged = True
                await stage_successor_async(
                    self.engine,
                    self.source,
                    [plan.destination for plan in self.plans],
                    kind="split",
                )
            for plan in self.plans:
                await plan.install()
        except BaseException:
            await self.abort()
            raise

    def publish(self):
        parent = self.delta.old_graph_ids[0]
        self.engine._session_stores.pop(parent, None)
        self.engine._owned_sessions.discard(parent)
        for plan in self.plans:
            gid = plan.delta.new_graph_ids[0]
            self.engine._session_stores[gid] = plan.destination
            if plan.fresh or self.owned:
                self.engine._owned_sessions.add(gid)
                getattr(self.engine, "_retained_session_stores", set()).discard(
                    plan.destination
                )

    async def retire(self):
        if self.owned and all(
            plan.destination is not self.source for plan in self.plans
        ):
            try:
                await self.source.run(_retire_metadata, self.source)
                await asyncio.to_thread(self.source.close, update_status=False)
                getattr(self.engine, "_retained_session_stores", set()).discard(
                    self.source
                )
            except Exception:
                retain_store(self.engine, self.source)
                logger.error(
                    "Committed split retained an unretired source", exc_info=True
                )

    async def abort(self):
        uncertain = self.retirement_staged or any(
            plan.installed
            or plan.authority_changed
            or (plan.target_changed and not plan.fresh)
            for plan in self.plans
        )
        if uncertain:
            await fence_topology_members(self.agents)
        for plan in reversed(self.plans):
            await plan.abort()


async def prepare_split(engine, projected, delta):
    """Prepare copied states and owner commits while original membership stays visible."""
    if delta.kind != "split" or getattr(engine, "_session_dir", None) is None:
        return None
    source = engine._session_stores.get(delta.old_graph_ids[0])
    if not isinstance(source, SessionStore):
        return None
    agents = [
        engine._creatures[cid].agent
        for cid in delta.affected_creatures
        if cid in engine._creatures
    ]
    if any(getattr(agent, "_session_state_owner", None) is None for agent in agents):
        return None
    result = PreparedSplit(
        engine, delta, source, delta.old_graph_ids[0] in engine._owned_sessions, agents
    )
    try:
        for gid in delta.new_graph_ids:
            component = replace(
                delta,
                new_graph_ids=[gid],
                affected_creatures=set(projected.graphs[gid].creature_ids),
            )
            result.plans.append(await prepare_merge(engine, projected, component))
        return result
    except BaseException:
        await result.abort()
        raise
