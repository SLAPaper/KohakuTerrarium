"""Child-state handoff uses detached per-store operations and preflights conflicts."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium import session_subagent_state as transfer
from kohakuterrarium.terrarium.group_tool_context import GroupContext
from kohakuterrarium.terrarium.tools_group_channel import _unwire


@pytest.mark.parametrize("conflict", [False, "row", "reservation"])
async def test_all_child_destinations_preflight_before_import(
    tmp_path, monkeypatch, conflict
):
    source_a = SessionStore(tmp_path / "a.kohakutr")
    source_b = SessionStore(tmp_path / "b.kohakutr")
    destination = SessionStore(tmp_path / "destination.kohakutr")
    stores = [source_a, source_b, destination]
    try:
        for store, parent in ((source_a, "alice"), (source_b, "bob")):
            await store.run(store.next_subagent_run, parent, "child")
            if conflict != "reservation" or parent != "bob":
                await store.run(
                    store.save_subagent,
                    parent,
                    "child",
                    0,
                    {"job_id": parent},
                    parent + " history",
                )
        if conflict == "row":
            await destination.run(
                destination.save_subagent,
                "bob",
                "child",
                0,
                {"job_id": "different"},
                "different history",
            )
        elif conflict == "reservation":
            await destination.run(destination.next_subagent_run, "bob", "child")
        creatures = {
            name: SimpleNamespace(
                agent=SimpleNamespace(
                    config=SimpleNamespace(name=name), session_store=store
                )
            )
            for name, store in (("alice", source_a), ("bob", source_b))
        }
        engine = SimpleNamespace(
            _creatures=creatures,
            _topology=SimpleNamespace(
                graphs={"graph": SimpleNamespace(creature_ids=["alice", "bob"])}
            ),
        )
        seen = []
        loop_thread = threading.get_ident()
        for operation in ("capture_children", "validate_children", "import_children"):
            original = getattr(transfer, operation)

            def check(store, *args, _original=original):
                assert threading.get_ident() != loop_thread
                assert threading.get_ident() == store._affinity_worker_ident
                seen.append((_original.__name__, store))
                return _original(store, *args)

            monkeypatch.setattr(transfer, operation, check)
        if conflict:
            with pytest.raises(ConflictError):
                transfer.transfer_graph_children(engine, [("graph", destination)])
            assert (
                await destination.run(
                    destination.load_subagent_meta, "alice", "child", 0
                )
                is None
            )
            assert not any(name == "import_children" for name, _ in seen)
        else:
            transfer.transfer_graph_children(engine, [("graph", destination)])
            assert (
                await destination.run(
                    destination.load_subagent_conversation, "alice", "child", 0
                )
                == "alice history"
            )
            assert (
                await destination.run(destination.next_subagent_run, "bob", "child")
                == 1
            )
        assert all(store in stores for _, store in seen)
    finally:
        for store in stores:
            await asyncio.to_thread(store.close)


@pytest.mark.parametrize("tamper", [None, "extra_row", "counter", "missing_receipt"])
async def test_child_namespace_successor_chain_does_not_bless_diverged_archive(
    tmp_path, tamper
):
    stores = [SessionStore(tmp_path / f"{name}.kohakutr") for name in ("a", "b", "c")]
    a, b, c = stores
    host = SimpleNamespace(config=SimpleNamespace(name="parent"), session_store=a)
    engine = SimpleNamespace(
        _creatures={"parent": SimpleNamespace(agent=host)},
        _topology=SimpleNamespace(
            graphs={"g": SimpleNamespace(creature_ids=["parent"])}
        ),
    )
    try:
        await a.run(a.next_subagent_run, "parent", "child")
        await a.run(
            a.save_subagent, "parent", "child", 0, {"job_id": "first"}, "history A"
        )
        for target in (b, c):
            transfer.transfer_graph_children(engine, [("g", target)])
            host.session_store = target
            await target.run(
                target.save_subagent,
                "parent",
                "child",
                0,
                {"job_id": "first", "latest": str(target.path)},
                "history updated",
            )
        await asyncio.to_thread(c.close)
        c = SessionStore(c.path)
        stores[-1] = host.session_store = c
        if tamper == "extra_row":
            await a.run(
                lambda: a.subagents.__setitem__(
                    "parent:child:9:meta", {"job_id": "foreign"}
                )
            )
        elif tamper == "counter":
            await a.run(lambda: a._subagent_runs.update({"parent:child": 9}))
        elif tamper == "missing_receipt":

            def corrupt():
                key = 'child_lineage:"parent"'
                record = c.state[key]
                record["ancestors"] = {}
                c.state[key] = record

            await c.run(corrupt)
        if tamper:
            with pytest.raises(ConflictError):
                transfer.transfer_graph_children(engine, [("g", a)])
            assert (
                await a.run(a.load_subagent_conversation, "parent", "child", 0)
                == "history A"
            )
            assert await c.run(c.next_subagent_run, "parent", "child") == 1
        else:
            transfer.transfer_graph_children(engine, [("g", a)])
            assert (
                await a.run(a.load_subagent_conversation, "parent", "child", 0)
                == "history updated"
            )
            assert await a.run(a.next_subagent_run, "parent", "child") == 1
            with pytest.raises(ConflictError):
                await c.run(c.next_subagent_run, "parent", "child")
    finally:
        for store in stores:
            await asyncio.to_thread(store.close)


async def test_independent_identical_child_rows_reject_before_graph_merge(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        a, b = alice.agent.session_store, bob.agent.session_store
        for store in (a, b):
            await store.run(
                store.save_subagent,
                "alice",
                "child",
                0,
                {"job_id": "identical", "ts": 1},
                "identical history",
            )
        before = (
            alice.graph_id,
            bob.graph_id,
            await a.run(a.get_events, "alice"),
            await b.run(b.get_events, "bob"),
        )
        with pytest.raises(ConflictError):
            await engine.connect(alice, bob, channel="rejected")
        assert (alice.graph_id, bob.graph_id) == before[:2]
        assert alice.agent.session_store is a and bob.agent.session_store is b
        assert not a._closed and not b._closed
        assert await a.run(a.get_events, "alice") == before[2]
        assert await b.run(b.get_events, "bob") == before[3]


async def test_existing_unattached_merge_target_is_not_modified_before_rejection(
    tmp_path,
):
    directory = tmp_path / "sessions"
    async with Terrarium(session_dir=directory) as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        original_ids = (alice.graph_id, bob.graph_id)
        source = alice.agent.session_store
        await source.run(
            source.save_subagent,
            "alice",
            "child",
            0,
            {"job_id": "source"},
            "source history",
        )
        target_path = directory / f"{alice.graph_id}.kohakutr"
        target = SessionStore(target_path)
        await target.run(
            target.save_subagent,
            "alice",
            "child",
            0,
            {"job_id": "independent"},
            "untouched",
        )
        await asyncio.to_thread(target.close)
        with pytest.raises(ConflictError):
            await engine.connect(alice, bob, channel="must-not-merge")
        assert (alice.graph_id, bob.graph_id) == original_ids
        assert alice.agent.session_store is source and not source._closed
        target = SessionStore(target_path)
        try:
            assert await target.run(target.get_events, "alice") == []
            assert (
                await target.run(target.load_subagent_conversation, "alice", "child", 0)
                == "untouched"
            )
        finally:
            await asyncio.to_thread(target.close)


@pytest.mark.parametrize("damage", ["phase", "foreign_row", "counter"])
async def test_malformed_transfer_payload_cannot_claim_empty_namespace(
    tmp_path, damage
):
    source, destination = SessionStore(tmp_path / "source.kohakutr"), SessionStore(
        tmp_path / "destination.kohakutr"
    )
    try:
        await source.run(source.next_subagent_run, "parent", "child")
        payload = await source.run(transfer.capture_children, source, "parent")
        if damage == "phase":
            payload["lineage"]["phase"] = "departed"
        elif damage == "foreign_row":
            payload["rows"]["other:child:0:meta"] = {"job_id": "foreign"}
        else:
            payload["counters"]["parent:child"] = 99
        with pytest.raises(ConflictError):
            await destination.run(
                transfer.import_children, destination, payload, "successor"
            )
        assert await destination.run(destination.list_subagent_runs) == []
        assert (
            await destination.run(destination.next_subagent_run, "parent", "child") == 0
        )
    finally:
        await asyncio.to_thread(source.close)
        await asyncio.to_thread(destination.close)


@pytest.mark.parametrize("operation", ["disconnect", "remove_channel", "unwire"])
async def test_split_without_destination_rejects_before_child_namespace_is_orphaned(
    tmp_path, operation
):
    async with Terrarium() as engine:
        bob = await engine.add_creature(
            AgentConfig(name="bob"),
            llm=ScriptedLLM(["b"]),
            session=tmp_path / "graph.kohakutr",
            io="headless",
            start=True,
        )
        carol = await engine.add_creature(
            AgentConfig(name="carol"),
            graph=bob.graph_id,
            llm=ScriptedLLM(["c"]),
            io="headless",
            start=True,
        )
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            graph=bob.graph_id,
            llm=ScriptedLLM(["a"]),
            io="headless",
            start=True,
        )
        await engine.connect(bob, carol, channel="stay")
        await engine.connect(bob, alice, channel="bridge")
        await engine.connect(bob, alice, channel="extra")
        store = alice.agent.session_store
        await store.run(store.next_subagent_run, "alice", "child")
        await engine.disconnect(bob, alice, channel="extra")
        graph_id = alice.graph_id
        with pytest.raises(ConflictError):
            if operation == "disconnect":
                await engine.disconnect(bob, alice, channel="bridge")
            elif operation == "remove_channel":
                await engine.remove_channel(graph_id, "bridge")
            else:
                await _unwire(
                    GroupContext(engine, bob, engine.get_graph(graph_id)),
                    alice,
                    "bridge",
                    "listen",
                )
        assert alice.graph_id == bob.graph_id == graph_id
        assert "bridge" in engine.get_graph(graph_id).channels
        assert alice.agent.session_store is store and not store._closed


async def test_bridge_removal_rejects_before_stopping_or_orphaning_child_history(
    tmp_path,
):
    async with Terrarium() as engine:
        bob = await engine.add_creature(
            AgentConfig(name="bob"),
            llm=ScriptedLLM(["b"]),
            session=tmp_path / "graph.kohakutr",
            io="headless",
            start=True,
        )
        others = []
        for name in ("carol", "bridge", "alice"):
            others.append(
                await engine.add_creature(
                    AgentConfig(name=name),
                    graph=bob.graph_id,
                    llm=ScriptedLLM([name]),
                    io="headless",
                    start=True,
                )
            )
        carol, bridge, alice = others
        await engine.connect(bob, carol, channel="stay")
        await engine.connect(bob, bridge, channel="left")
        await engine.connect(bridge, alice, channel="right")
        store, graph_id = alice.agent.session_store, alice.graph_id
        await store.run(store.next_subagent_run, "alice", "child")
        with pytest.raises(ConflictError):
            await engine.remove_creature(bridge)
        assert engine.get_creature(bridge.creature_id) is bridge and bridge.is_running
        assert alice.graph_id == bob.graph_id == graph_id
        assert not store._closed


async def test_partial_child_installation_fences_parent_execution(
    tmp_path, monkeypatch
):
    captured = []
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["must not run"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"),
            llm=ScriptedLLM(["must not run"]),
            io="headless",
            start=True,
        )
        source = alice.agent.session_store
        await source.run(source.next_subagent_run, "alice", "child")
        import_children = transfer.import_children

        def fail(store, payload, successor):
            captured.append(store)
            if payload["parent"] == "alice":
                raise RuntimeError("child import failed after departure")
            return import_children(store, payload, successor)

        monkeypatch.setattr(transfer, "import_children", fail)
        try:
            with pytest.raises(RuntimeError, match="child import failed"):
                await engine.connect(alice, bob, channel="bridge")
            outcome = await alice.run(
                "reject after failed handoff", raise_on_error=False
            )
            assert outcome.status == "error"
            assert alice.agent.llm.call_count == 0
            with pytest.raises(ConflictError):
                await source.run(source.next_subagent_run, "alice", "child")
            assert captured and all(store._closed for store in captured)
        finally:
            for store in set(captured):
                if store not in engine._session_stores.values() and not store._closed:
                    await asyncio.to_thread(store.close)
