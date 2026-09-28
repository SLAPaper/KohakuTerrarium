"""Drive movement resolves session sidecars without blocking the engine loop."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.drive import partition, topology
from kohakuterrarium.terrarium.drive.config import DriveRuntimeConfig
from kohakuterrarium.terrarium.drive.memory import MemoryDriveRepository
from kohakuterrarium.terrarium.drive.models import ActorRef
from kohakuterrarium.terrarium.drive.errors import DriveConflictError
from kohakuterrarium.terrarium.drive.requests import CreateDriveRequest
from kohakuterrarium.terrarium.drive.store import (
    SqliteDriveRepository,
    DriveRepositoryClosedError,
)
from kohakuterrarium.terrarium.drive.sink import (
    DeliveryOutcome,
    Settlement,
    SettlementStatus,
)
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("operation", ["merge", "split"])
async def test_borrowed_durable_provider_without_sqlite_path_retains_rows(operation):
    class ProviderRepository(MemoryDriveRepository):
        closed = False

        @property
        def durability(self):
            return "persistent"

        async def close(self):
            self.closed = True

    provided = []

    def provider(graph):
        repo = ProviderRepository()
        provided.append(repo)
        return repo

    async with Terrarium(
        drive_config=DriveRuntimeConfig(enabled=True), drive_store=provider
    ) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=False
        )
        if operation == "split":
            await engine.connect(alice, bob, channel="bridge")
        manager = engine.drives.manager_for(bob.graph_id)
        actor = ActorRef("user", "tester")
        records = []
        for creature in ([alice, bob] if operation == "split" else [bob]):
            record = await manager.create_drive(
                CreateDriveRequest(
                    kind="generic",
                    title="provider row",
                    scope_type="creature",
                    scope_id=creature.creature_id,
                    owner=actor,
                    owner_scope="graph",
                    created_by=actor,
                ),
                actor=actor,
                graph_id=bob.graph_id,
                is_privileged=True,
            )
            records.append((creature, record))
        if operation == "merge":
            await engine.connect(alice, bob, channel="bridge")
        else:
            await engine.disconnect(alice, bob, channel="bridge")
        for creature, record in records:
            target = engine.drives.manager_for(creature.graph_id).repository
            restored = await target.get(record.drive_id)
            assert restored is not None and restored.title == "provider row"
            if operation == "split":
                sibling = alice if creature is bob else bob
                other = engine.drives.manager_for(sibling.graph_id).repository
                assert await other.get(record.drive_id) is None
        assert not any(repo.closed for repo in provided)


@pytest.fixture
def promotion_store(tmp_path):
    store = SessionStore(tmp_path / "promoted.kohakutr")
    try:
        yield store
    finally:
        store.close()


@pytest.mark.parametrize("failure", ["before", "after", "rollback"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_promotion_failure_preserves_source_and_confirms_rollback(
    monkeypatch, promotion_store, failure, cancel
):
    opened = []
    error = asyncio.CancelledError if cancel else OSError
    imported, replaced = (
        SqliteDriveRepository.import_rows,
        SqliteDriveRepository.replace_rows,
    )

    async def rejected(repo, payload):
        opened.append(repo)
        if failure != "before":
            await imported(repo, payload)
        raise error("primary import failure")

    async def rollback(repo, payload):
        if failure == "rollback":
            raise OSError("secondary rollback failure")
        await replaced(repo, payload)

    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        manager = engine.drives.manager_for(creature.graph_id)
        actor = ActorRef("user", "tester")
        record = await manager.create_drive(
            CreateDriveRequest(
                kind="generic",
                title="retained source",
                scope_type="creature",
                scope_id=creature.creature_id,
                owner=actor,
                owner_scope="graph",
                created_by=actor,
            ),
            actor=actor,
            graph_id=creature.graph_id,
            is_privileged=True,
        )
        with monkeypatch.context() as patch:
            patch.setattr(SqliteDriveRepository, "import_rows", rejected)
            patch.setattr(SqliteDriveRepository, "replace_rows", rollback)
            with pytest.raises(error, match="primary import failure"):
                await engine.drives.registry.bind_store(
                    creature.graph_id, promotion_store
                )
        assert engine.drives.manager_for(creature.graph_id) is manager
        assert len(opened) == 1
        if failure == "rollback":
            with pytest.raises(DriveConflictError):
                await manager.repository.get(record.drive_id)
        else:
            assert (
                await manager.repository.get(record.drive_id)
            ).title == "retained source"
            assert (await opened[0].export_rows())["drives"] == []
            assert not manager.dispatcher._handoff_paused


@pytest.mark.parametrize("operation", ["merge", "split", "merge_split", "promotion"])
@pytest.mark.parametrize("waiting", ["admission", "settlement"])
@pytest.mark.parametrize("outcome", ["ok", "error", "interrupt"])
async def test_handoff_does_not_join_external_delivery_and_settles_in_successor(
    monkeypatch, operation, waiting, outcome, promotion_store
):
    settlement = asyncio.get_running_loop().create_future()
    entered, release = asyncio.Event(), asyncio.Event()
    delivered = []
    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=False
        )
        if operation == "split":
            await engine.connect(alice, bob, channel="bridge")
        manager = engine.drives.manager_for(bob.graph_id)
        await manager.stop()

        async def deliver(cid, event, *, delivery_id):
            delivered.append(delivery_id)
            entered.set()
            if waiting == "admission":
                await release.wait()
            return DeliveryOutcome.accepted(settlement)

        monkeypatch.setattr(engine.drives.registry._sink, "deliver", deliver)
        actor = ActorRef("user", "tester")
        record = await manager.create_drive(
            CreateDriveRequest(
                kind="generic",
                title="turn invokes connect",
                scope_type="creature",
                scope_id=bob.creature_id,
                owner=actor,
                owner_scope="graph",
                created_by=actor,
            ),
            actor=actor,
            graph_id=bob.graph_id,
            is_privileged=True,
        )
        dispatching = asyncio.create_task(manager.dispatcher.dispatch_once())
        moving = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if waiting == "settlement":
                await asyncio.wait_for(dispatching, 3)
            assert len(delivered) == 1
            operation_call = (
                engine.disconnect if operation == "split" else engine.connect
            )
            moving = asyncio.create_task(
                engine.drives.registry.bind_store(bob.graph_id, promotion_store)
                if operation == "promotion"
                else operation_call(alice, bob, channel="bridge")
            )
            await asyncio.wait_for(asyncio.shield(moving), 3)
            if operation == "merge_split":
                await asyncio.wait_for(
                    engine.disconnect(alice, bob, channel="bridge"), 3
                )
            assert not settlement.done()
            release.set()
            await asyncio.wait_for(dispatching, 3)
            target_manager = engine.drives.manager_for(bob.graph_id)
            await target_manager.reconcile(creature_id=bob.creature_id)
            assert len(delivered) == 1
            status = {
                "ok": SettlementStatus.OK,
                "error": SettlementStatus.ERROR,
                "interrupt": SettlementStatus.INTERRUPTED,
            }[outcome]
            detail = (
                {"interrupted_by_user": True}
                if outcome == "interrupt"
                else {"error": "expected failure"}
            )
            settlement.set_result(Settlement(status, detail))
            await asyncio.wait_for(manager.dispatcher.drain(), 3)
            rows = await target_manager.repository.list_deliveries(record.drive_id)
            expected = "retry_wait" if outcome == "error" else "acknowledged"
            assert [(row.delivery_id, row.state) for row in rows] == [
                (delivered[0], expected)
            ]
            assert len(delivered) == 1
            assert not engine.drives.registry._delivery_receipts._receipts
        finally:
            release.set()
            if not settlement.done():
                settlement.set_result(Settlement(SettlementStatus.OK))
            await asyncio.gather(
                dispatching,
                *([moving] if moving is not None else []),
                return_exceptions=True,
            )


@pytest.mark.parametrize("persistent", [False, True])
async def test_retained_manager_cannot_lose_successful_create_during_merge(
    tmp_path, monkeypatch, persistent
):
    prepared, release = asyncio.Event(), asyncio.Event()
    async with Terrarium(
        session_dir=tmp_path / "sessions" if persistent else None,
        drive_config=DriveRuntimeConfig(enabled=True),
    ) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=False
        )
        manager = engine.drives.manager_for(bob.graph_id)
        original_graph, source_repo = bob.graph_id, manager.repository
        actor = ActorRef("user", "tester")
        resolve = engine.drives.registry.resolve_destination_repo_async

        async def held(graph):
            result = await resolve(graph)
            assert result[0] is not source_repo
            if persistent:
                assert result[0]._path != source_repo._path
            prepared.set()
            await release.wait()
            return result

        monkeypatch.setattr(
            engine.drives.registry, "resolve_destination_repo_async", held
        )
        moving = asyncio.create_task(engine.connect(alice, bob, channel="bridge"))
        record = None
        try:
            await asyncio.wait_for(prepared.wait(), 5)
            try:
                record = await manager.create_drive(
                    CreateDriveRequest(
                        kind="generic",
                        title="accepted during export",
                        scope_type="creature",
                        scope_id=bob.creature_id,
                        owner=actor,
                        owner_scope="graph",
                        created_by=actor,
                    ),
                    actor=actor,
                    graph_id=original_graph,
                    is_privileged=True,
                )
            except (DriveConflictError, DriveRepositoryClosedError):
                pass
            release.set()
            await asyncio.wait_for(moving, 8)
            if record is not None:
                target = engine.drives.manager_for(bob.graph_id).repository
                restored = await target.get(record.drive_id)
                assert (
                    restored is not None and restored.title == "accepted during export"
                )
        finally:
            release.set()
            await asyncio.gather(moving, return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_drive_drain_preserves_failed_and_later_operations(
    tmp_path, monkeypatch, cancel
):
    async with Terrarium(
        session_dir=tmp_path / "sessions", drive_config=DriveRuntimeConfig(enabled=True)
    ) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=False
        )
        first_path = engine.drives.manager_for(alice.graph_id).repository._path
        topology.stash_merge(engine, alice.graph_id, [alice.graph_id])
        topology.stash_merge(engine, bob.graph_id, [bob.graph_id])
        queued = list(engine._pending_drive_topology)
        export = SqliteDriveRepository.export_rows
        visited = []

        async def failed(repo):
            visited.append(repo._path)
            if repo._path == first_path:
                topology.stash_merge(engine, bob.graph_id, [bob.graph_id])
                queued.extend(engine._pending_drive_topology)
                if cancel:
                    raise asyncio.CancelledError()
                raise OSError("source temporarily unavailable")
            return await export(repo)

        with monkeypatch.context() as patch:
            patch.setattr(SqliteDriveRepository, "export_rows", failed)
            with pytest.raises(asyncio.CancelledError if cancel else OSError):
                await engine.drives.drain_topology()
        assert engine._pending_drive_topology == queued
        assert visited == [first_path]
        await engine.drives.drain_topology()
        assert not engine._pending_drive_topology
        for creature in (alice, bob):
            repo = engine.drives.manager_for(creature.graph_id).repository
            assert (await repo.export_rows())["drives"] == []


@pytest.mark.parametrize("failure", ["open", "journal"])
async def test_split_preparation_failure_closes_unpublished_repositories(
    tmp_path, monkeypatch, failure
):
    async with Terrarium(
        session_dir=tmp_path / "sessions", drive_config=DriveRuntimeConfig(enabled=True)
    ) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=False
        )
        await engine.connect(alice, bob, channel="bridge")
        actor = ActorRef("user", "tester")
        manager = engine.drives.manager_for(alice.graph_id)
        source_path = manager.repository._path
        record = await manager.create_drive(
            CreateDriveRequest(
                kind="generic",
                title="retained source",
                scope_type="creature",
                scope_id=alice.creature_id,
                owner=actor,
                owner_scope="graph",
                created_by=actor,
            ),
            actor=actor,
            graph_id=alice.graph_id,
            is_privileged=True,
        )
        opened = []
        opener = partition.open_session_drive_repository

        def failed_open(store):
            if failure == "open" and opened:
                raise OSError("sidecar preparation failed")
            repo = opener(store)
            opened.append(repo)
            return repo

        def failed_journal(*args):
            raise OSError("journal preparation failed")

        with monkeypatch.context() as patch:
            patch.setattr(partition, "open_session_drive_repository", failed_open)
            if failure == "journal":
                patch.setattr(topology, "write_split_intent", failed_journal)
            with pytest.raises(OSError, match="preparation failed"):
                await engine.disconnect(alice, bob, channel="bridge")
        assert len(opened) == (1 if failure == "open" else 2)
        assert all(repo._closed for repo in opened)
        assert (
            alice.agent._session_attach_uncertain
            and bob.agent._session_attach_uncertain
        )
        source = SqliteDriveRepository(source_path)
        try:
            assert (await source.get(record.drive_id)).title == "retained source"
        finally:
            await topology._close_if_persistent(source)


@pytest.mark.parametrize("operation", ["merge", "split"])
async def test_drive_topology_opens_sidecars_on_session_workers(
    tmp_path, monkeypatch, operation
):
    async with Terrarium(
        session_dir=tmp_path / "sessions", drive_config=DriveRuntimeConfig(enabled=True)
    ) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=False,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=False
        )
        if operation == "split":
            await engine.connect(alice, bob, channel="bridge")
        actor = ActorRef("user", "tester")
        record = await engine.drives.manager_for(alice.graph_id).create_drive(
            CreateDriveRequest(
                kind="generic",
                title="survive topology",
                scope_type="creature",
                scope_id=alice.creature_id,
                owner=actor,
                owner_scope="graph",
                created_by=actor,
            ),
            actor=actor,
            graph_id=alice.graph_id,
            is_privileged=True,
        )
        opened, journal_threads = [], []
        opener = partition.open_session_drive_repository
        journal = topology.write_split_intent
        loop_thread = threading.get_ident()
        close = SqliteDriveRepository.close_blocking

        def checked_close(repo):
            assert threading.get_ident() != loop_thread
            return close(repo)

        def checked_open(store, **kwargs):
            assert threading.get_ident() == store._affinity_worker_ident
            opened.append(store)
            return opener(store, **kwargs)

        def checked_journal(*args):
            journal_threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread
            return journal(*args)

        with monkeypatch.context() as patch:
            patch.setattr(partition, "open_session_drive_repository", checked_open)
            patch.setattr(topology, "write_split_intent", checked_journal)
            patch.setattr(SqliteDriveRepository, "close_blocking", checked_close)
            if operation == "merge":
                await engine.connect(alice, bob, channel="bridge")
            else:
                await engine.disconnect(alice, bob, channel="bridge")
        assert len(opened) == (1 if operation == "merge" else 2)
        assert len({id(store) for store in opened}) == len(opened)
        if operation == "split":
            assert len(journal_threads) == 1
        restored = await engine.drives.manager_for(alice.graph_id).get_drive(
            record.drive_id
        )
        assert restored.title == "survive topology"
        assert restored.scope_id == alice.creature_id
