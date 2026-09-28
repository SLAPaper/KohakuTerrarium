"""Drive handoff failure releases or rejects exactly its affected bindings."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.terrarium.drive import topology
from kohakuterrarium.terrarium.drive.config import DriveRuntimeConfig
from kohakuterrarium.terrarium.drive.errors import DriveConflictError, DriveHandoffError
from kohakuterrarium.terrarium.drive.handoff import drive_handoff
from kohakuterrarium.terrarium.drive.memory import MemoryDriveRepository
from kohakuterrarium.terrarium.drive.models import ActorRef
from kohakuterrarium.terrarium.drive.requests import CreateDriveRequest
from kohakuterrarium.terrarium.drive.sink import (
    DeliveryOutcome,
    Settlement,
    SettlementStatus,
)
from kohakuterrarium.testing.llm import ScriptedLLM


async def seed(engine, name):
    creature = await engine.add_creature(
        AgentConfig(name=name), llm=ScriptedLLM(["ok"]), io="headless", start=False
    )
    manager = engine.drives.manager_for(creature.graph_id)
    actor = ActorRef("user", "tester")
    record = await manager.create_drive(
        CreateDriveRequest(
            kind="generic",
            title=name,
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
    return creature, manager, record


async def test_topology_reconciles_unowned_admission_before_restarting_dispatch(
    monkeypatch,
):
    recovered = asyncio.get_running_loop().create_future()

    async def deliver(cid, event, *, delivery_id):
        if event.context["drive_id"] == record.drive_id and not recovered.done():
            recovered.set_result(event.context["delivery_reason"])
        return DeliveryOutcome.accepted()

    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        alice, _, _ = await seed(engine, "alice")
        bob, manager, record = await seed(engine, "bob")
        await manager.stop()
        delivery = (await manager.repository.list_deliveries(record.drive_id))[0]
        await manager.repository.mark_delivery(delivery.delivery_id, "admitted")
        await bob.start()
        await engine.drives._drain_reconcile(bob.graph_id)
        await manager.stop()
        monkeypatch.setattr(engine.drives.registry._sink, "deliver", deliver)
        await engine.connect(alice, bob, channel="recover")
        assert await asyncio.wait_for(recovered, 3) == "recovery"
        target = engine.drives.manager_for(bob.graph_id).repository
        rows = await target.list_deliveries(record.drive_id)
        assert (
            sum(
                row.delivery_id == delivery.delivery_id and row.state == "superseded"
                for row in rows
            )
            == 1
        )
        assert len(rows) == 2


async def test_failed_delivery_binding_does_not_claim_or_recover_more_work():
    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        creature, manager, record = await seed(engine, "alice")
        repo, book = manager.repository, manager.dispatcher._receipts
        delivery = (await repo.list_deliveries(record.drive_id))[0]
        receipt = book.register(manager.dispatcher, delivery)
        book.fail(receipt, OSError("write confirmation missing"))
        await manager.dispatcher.dispatch_once()
        await manager.reconcile(creature_id=creature.creature_id)
        rows = await repo.list_deliveries(record.drive_id)
        assert [(row.delivery_id, row.state) for row in rows] == [
            (delivery.delivery_id, "pending")
        ]
        assert not book._receipts
        assert book._failed[delivery.delivery_id] is receipt


async def test_uncertain_topology_entry_is_retained_without_starving_later_graph(
    monkeypatch,
):
    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        alice, old, _ = await seed(engine, "alice")
        bob, _, record = await seed(engine, "bob")
        topology.stash_merge(engine, alice.graph_id, [alice.graph_id])
        topology.stash_merge(engine, bob.graph_id, [bob.graph_id])
        failed_entry, later_entry = engine._pending_drive_topology
        imported = MemoryDriveRepository.import_rows

        async def failed_import(repo, payload):
            await imported(repo, payload)
            raise OSError("uncertain import")

        async def failed_rollback(repo, payload):
            raise OSError("unconfirmed rollback")

        with monkeypatch.context() as patch:
            patch.setattr(MemoryDriveRepository, "import_rows", failed_import)
            patch.setattr(MemoryDriveRepository, "replace_rows", failed_rollback)
            with pytest.raises(OSError, match="uncertain import"):
                await engine.drives.drain_topology()
        assert engine._pending_drive_topology == [later_entry]
        failed = engine._failed_drive_topology
        assert len(failed) == 1 and failed[0][0] == failed_entry
        with pytest.raises(DriveHandoffError):
            await old.repository.export_rows()
        await engine.drives.drain_topology()
        assert not engine._pending_drive_topology
        assert (
            await engine.drives.manager_for(bob.graph_id).get_drive(record.drive_id)
        ).title == "bob"


async def test_cancellation_during_finite_drain_restores_only_its_pause_scope(
    monkeypatch,
):
    entered, release, pausing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        creature, manager, record = await seed(engine, "alice")
        repo = manager.repository
        book = engine.drives.registry._delivery_receipts
        delivery = (await repo.list_deliveries(record.drive_id))[0]
        receipt = book.register(manager.dispatcher, delivery)
        pause = book.pause

        async def observed_pause(repos):
            asyncio.get_running_loop().call_soon(pausing.set)
            await pause(repos)

        monkeypatch.setattr(book, "pause", observed_pause)

        async def accepted():
            async with book.phase(receipt):
                entered.set()
                await release.wait()

        async def moving():
            async with drive_handoff(engine, {creature.graph_id}):
                pytest.fail("cancelled drain must not publish")

        applying = asyncio.create_task(accepted())
        move = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            move = asyncio.create_task(moving())
            await asyncio.wait_for(pausing.wait(), 3)
            assert repo in book._paused
            assert repo._handoff_token is None
            move.cancel()
            with pytest.raises(asyncio.CancelledError):
                await move
            assert repo not in book._paused
            assert receipt.ready.is_set()
            assert not manager.dispatcher._handoff_paused
            release.set()
            await asyncio.wait_for(applying, 3)
            async with book.phase(receipt) as target:
                await target._repo.mark_delivery(delivery.delivery_id, "admitted")
            book.complete(receipt)
            assert (await repo.list_deliveries(record.drive_id))[0].state == "admitted"
        finally:
            release.set()
            await asyncio.gather(
                applying, *([move] if move else []), return_exceptions=True
            )


async def test_handoff_lease_covers_dispatcher_restart(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        creature, manager, _ = await seed(engine, "alice")
        await engine.drives.registry.ensure_started(creature.graph_id)
        start = manager.start

        async def held_start():
            entered.set()
            await release.wait()
            await start()

        async def moving():
            async with drive_handoff(engine, {creature.graph_id}):
                pass

        monkeypatch.setattr(manager, "start", held_start)
        first = asyncio.create_task(moving())
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            second = asyncio.create_task(moving())
            with pytest.raises(DriveHandoffError, match="another coordinator"):
                await asyncio.wait_for(asyncio.shield(second), 3)
        finally:
            release.set()
            await asyncio.gather(
                first, *([second] if second else []), return_exceptions=True
            )


@pytest.mark.parametrize("unsupported", [False, True])
async def test_handoff_restores_live_dispatch_scanning(monkeypatch, unsupported):
    class Provider:
        def __init__(self):
            self.backend = MemoryDriveRepository()

        def __getattr__(self, name):
            return getattr(self.backend, name)

    admitted = asyncio.Event()
    expected = None

    async def deliver(cid, event, *, delivery_id):
        if event.context["drive_id"] == expected:
            admitted.set()
        return DeliveryOutcome.accepted()

    factory = (
        (lambda gid: Provider())
        if unsupported
        else (lambda gid: MemoryDriveRepository())
    )
    async with Terrarium(
        drive_config=DriveRuntimeConfig(enabled=True), drive_store=factory
    ) as engine:
        alice, _, _ = await seed(engine, "alice")
        bob, _, _ = await seed(engine, "bob")
        monkeypatch.setattr(engine.drives.registry._sink, "deliver", deliver)
        for creature in (alice, bob):
            await engine.drives.registry.ensure_started(creature.graph_id)
        if unsupported:
            with pytest.raises(DriveHandoffError, match="does not support"):
                await engine.connect(alice, bob, channel="rejected")
        else:
            await engine.connect(alice, bob, channel="moved")
        manager = engine.drives.manager_for(alice.graph_id)
        actor = ActorRef("user", "tester")
        record = await manager.create_drive(
            CreateDriveRequest(
                kind="generic",
                title="after handoff",
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
        expected = record.drive_id
        await asyncio.wait_for(admitted.wait(), 3)
        assert (await manager.repository.list_deliveries(record.drive_id))[
            0
        ].state == "admitted"


async def test_unsupported_provider_does_not_leave_paused_reservations(monkeypatch):
    class Provider:
        def __init__(self):
            self.backend = MemoryDriveRepository()

        def __getattr__(self, name):
            return getattr(self.backend, name)

    admitted = []

    async def deliver(cid, event, *, delivery_id):
        admitted.append(delivery_id)
        return DeliveryOutcome.accepted()

    async with Terrarium(
        drive_config=DriveRuntimeConfig(enabled=True),
        drive_store=lambda gid: Provider(),
    ) as engine:
        alice, _, _ = await seed(engine, "alice")
        bob, manager, record = await seed(engine, "bob")
        monkeypatch.setattr(engine.drives.registry._sink, "deliver", deliver)
        with pytest.raises(DriveHandoffError, match="does not support"):
            await engine.connect(alice, bob, channel="rejected")
        assert alice.graph_id != bob.graph_id
        await manager.dispatcher.dispatch_once()
        rows = await manager.repository.list_deliveries(record.drive_id)
        assert len(admitted) == 1
        assert rows[0].state == "admitted"


@pytest.mark.parametrize("waiting", ["admission", "settlement"])
async def test_written_failure_releases_settlement_waiter_as_error(
    monkeypatch, waiting
):
    settlement = asyncio.get_running_loop().create_future()
    entered, release = asyncio.Event(), asyncio.Event()

    async def deliver(cid, event, *, delivery_id):
        entered.set()
        if waiting == "admission":
            await release.wait()
        return DeliveryOutcome.accepted(settlement)

    async with Terrarium(drive_config=DriveRuntimeConfig(enabled=True)) as engine:
        creature, manager, _ = await seed(engine, "alice")
        monkeypatch.setattr(engine.drives.registry._sink, "deliver", deliver)
        dispatching = asyncio.create_task(manager.dispatcher.dispatch_once())
        await asyncio.wait_for(entered.wait(), 3)
        if waiting == "settlement":
            await asyncio.wait_for(dispatching, 3)
        tasks = (
            list(manager.dispatcher._settle_tasks)
            if waiting == "settlement"
            else [dispatching]
        )
        target = MemoryDriveRepository()
        try:
            with pytest.raises(OSError, match="publication failed"):
                async with drive_handoff(engine, {creature.graph_id}) as handoff:
                    await handoff.reserve(target)
                    payload = await manager.repository.export_rows()
                    handoff.written = True
                    await target.import_rows(payload)
                    raise OSError("publication failed")
            release.set()
            settlement.set_result(Settlement(SettlementStatus.OK))
            with pytest.raises(DriveConflictError, match="recovery"):
                await asyncio.wait_for(asyncio.shield(tasks[0]), 3)
            with pytest.raises(DriveConflictError):
                await manager.repository.export_rows()
            assert not manager.dispatcher._inflight
            book = engine.drives.registry._delivery_receipts
            assert not book._receipts
            failed = list(book._failed.values())
            assert len(failed) == 1
            if waiting == "admission":
                assert failed[0].admission.admitted
            else:
                assert failed[0].settlement.status is SettlementStatus.OK
        finally:
            release.set()
            if not settlement.done():
                settlement.set_result(Settlement(SettlementStatus.OK))
            engine.drives.registry._delivery_receipts.abort()
            await asyncio.gather(*tasks, return_exceptions=True)
