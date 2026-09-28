"""Coordinate finite Drive repository and delivery handoffs."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy

from kohakuterrarium.terrarium.drive.errors import DriveHandoffError
from kohakuterrarium.terrarium.drive.repository_base import BaseDriveRepository
from kohakuterrarium.terrarium.drive.wire import unpack_drive_delivery
from kohakuterrarium.utils.logging import get_logger

logger = get_logger(__name__)


class DriveTopologyHandoff:
    """Keep repositories sealed until bindings and live delivery routes agree."""

    def __init__(self, registry, graph_ids):
        self.registry = registry
        self.graph_ids = set(graph_ids)
        self.owner = asyncio.current_task()
        self.token = object()
        self.repositories = set()
        self.paused_repositories = set()
        self.destinations = {}
        self.written = False
        self.topology = deepcopy(registry._engine._topology)

    async def begin(self):
        managers = [self.registry.peek(gid) for gid in self.graph_ids]
        for manager in managers:
            if manager is not None:
                manager.dispatcher.pause_handoff()
        repos = {m.repository for m in managers if m is not None}
        await self.pause(repos)
        runtime = self.registry._engine._drive_runtime
        for graph_id in self.graph_ids:
            await runtime._drain_reconcile(graph_id)
        starts = list(self.registry._start_tasks)
        if starts:
            await asyncio.gather(*starts, return_exceptions=True)
        for repo in repos:
            await self.reserve(repo)

    async def pause(self, repos):
        book = self.registry._delivery_receipts
        self.paused_repositories.update(repos - book._paused)
        await book.pause(repos)

    async def reserve(self, repo):
        if repo in self.repositories:
            return
        if not isinstance(repo, BaseDriveRepository):
            raise DriveHandoffError(
                "Drive provider does not support transaction handoff"
            )
        await self.pause({repo})
        self.repositories.add(repo)
        await repo.reserve_handoff(self.token)

    def stage(self, payloads):
        routes = {}
        for gid, payload in payloads.items():
            dispatcher = self.registry.peek(gid).dispatcher
            for row in payload.get("deliveries", []):
                delivery = unpack_drive_delivery(row)
                if delivery.delivery_id in routes:
                    raise DriveHandoffError(
                        "Drive delivery appears in multiple successors"
                    )
                routes[delivery.delivery_id] = (dispatcher, delivery)
        self.destinations.update(routes)
        return self.registry._delivery_receipts.prepare(
            self.repositories,
            [
                (dispatcher, [delivery])
                for dispatcher, delivery in self.destinations.values()
            ],
        )

    async def finish(self):
        prepared = self.stage({})
        live_graphs = self.registry._engine._topology.graphs
        for gid in self.graph_ids - set(live_graphs):
            self.registry.drop_graph(gid)
        active = set(self.registry._repos.values())
        for repo in self.repositories:
            repo.finish_handoff(self.token, retired=repo not in active)
        self.registry._delivery_receipts.publish(self.repositories, prepared)
        for gid, manager in list(self.registry._managers.items()):
            if manager.repository in self.repositories:
                manager.dispatcher.resume_handoff()
                if gid in self.registry._ready_graphs:
                    await manager.start()

    async def fail(self):
        if self.written or self.registry._engine._topology != self.topology:
            for repo in self.repositories:
                repo.revoke_handoff(self.token)
            self.registry._delivery_receipts.abort(self.paused_repositories)
            return
        for repo in self.repositories:
            if repo._handoff_token is self.token:
                repo.finish_handoff(self.token)
        book = self.registry._delivery_receipts
        prepared = [
            (r, r.dispatcher)
            for r in book._receipts.values()
            if r.dispatcher._repo in self.paused_repositories
        ]
        book.publish(self.paused_repositories, prepared)
        for gid, manager in list(self.registry._managers.items()):
            if manager.repository in self.paused_repositories:
                manager.dispatcher.resume_handoff()
                if gid in self.registry._ready_graphs:
                    try:
                        await manager.start()
                    except (Exception, asyncio.CancelledError):
                        logger.error(
                            "Restored Drive dispatcher could not restart", exc_info=True
                        )


@asynccontextmanager
async def drive_handoff(engine, graph_ids):
    runtime = getattr(engine, "_drive_runtime", None)
    if runtime is None:
        yield None
        return
    registry = runtime.registry
    existing = registry._handoff
    if existing is not None:
        if existing.owner is not asyncio.current_task():
            raise DriveHandoffError("Drive handoff has another coordinator")
        yield existing
        return
    handoff = DriveTopologyHandoff(registry, graph_ids)
    registry._handoff = handoff
    try:
        await handoff.begin()
        yield handoff
        await handoff.finish()
    except BaseException:
        await handoff.fail()
        raise
    finally:
        if registry._handoff is handoff:
            registry._handoff = None
