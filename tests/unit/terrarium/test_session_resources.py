"""Failed session transitions preserve cleanup responsibility and primary errors."""

import asyncio
from types import SimpleNamespace

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.session_attachment_state import AttachmentJournal
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import autosession, session_attach, session_merge_async
from kohakuterrarium.terrarium.autosession import close_attached_stores
from kohakuterrarium.terrarium.session_resources import (
    track_opened_store,
    transition_stores,
)


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("after_bind", [False, True])
async def test_failed_merge_keeps_retirement_responsibility_for_old_source(
    tmp_path, monkeypatch, owned, after_bind
):
    sources = []
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            a = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM(["no"]),
                io="headless",
                start=True,
            )
            b = await engine.add_creature(
                AgentConfig(name="bob"),
                llm=ScriptedLLM(["no"]),
                io="headless",
                start=True,
            )
            sources = [a.agent.session_store, b.agent.session_store]
            if not owned:
                engine._owned_sessions.clear()
            original_graphs = a.graph_id, b.graph_id
            bind = session_merge_async._install_member

            def fail_binding(agent, *args):
                if agent is b.agent:
                    if after_bind:
                        bind(agent, *args)
                    raise RuntimeError("merge member bind failed")
                return bind(agent, *args)

            monkeypatch.setattr(session_merge_async, "_install_member", fail_binding)
            with pytest.raises(RuntimeError, match="merge member bind failed"):
                await engine.connect(a, b, channel="merge")
            assert (a.graph_id, b.graph_id) == original_graphs
            assert all(store in engine._session_stores.values() for store in sources)
            assert all(not store._closed for store in sources)
            assert all(
                (store in getattr(engine, "_retained_session_stores", ()))
                == (owned or store.meta.get("resume_successor") is not None)
                for store in sources
            )
            assert (
                a.agent._session_attach_uncertain and b.agent._session_attach_uncertain
            )
        assert all(store._closed for store in sources)
    finally:
        for store in sources:
            store.close(update_status=False)


@pytest.mark.parametrize("mode", ["unbound", "bound", "close_fails"])
def test_failed_transition_closes_or_retains_every_opened_handle(
    tmp_path, monkeypatch, mode
):
    engine = SimpleNamespace(_session_stores={}, _owned_sessions=set(), _creatures={})
    store = SessionStore(tmp_path / "partial.kohakutr")
    close = store.close
    calls = []

    def fail_once(**kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("cleanup failed")
        return close(**kwargs)

    if mode == "close_fails":
        monkeypatch.setattr(store, "close", fail_once)
    if mode == "bound":
        engine._creatures["parent"] = SimpleNamespace(
            agent=SimpleNamespace(session_store=store)
        )
    try:
        with pytest.raises(RuntimeError, match="primary import error"):
            with transition_stores(engine):
                track_opened_store(store)
                raise RuntimeError("primary import error")
        if mode == "unbound":
            assert store._closed
        else:
            assert not store._closed
            assert store in engine._retained_session_stores
        close_attached_stores(engine)
        assert store._closed
        assert not getattr(engine, "_retained_session_stores", set())
    finally:
        close(update_status=False)


@pytest.mark.parametrize("route", ["explicit", "autosession"])
@pytest.mark.parametrize("fault", ["cleanup", "rollback", "rollback_cancel"])
async def test_attach_double_fault_retains_handle_and_primary_error(
    tmp_path, monkeypatch, request, route, fault
):
    captured = []
    mint = autosession.mint_store
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["must not run"]),
            io="headless",
            start=True,
        )
        agent, original = creature.agent, creature.agent.session_store

        def observe_mint(*args, **kwargs):
            store = mint(*args, **kwargs)
            captured.append(store)
            request.addfinalizer(lambda: SessionStore.close(store, update_status=False))
            if fault == "cleanup":
                close = store.close
                calls = []

                def fail_once(**kwargs):
                    calls.append(True)
                    if len(calls) == 1:
                        raise OSError("cleanup failed")
                    return close(**kwargs)

                monkeypatch.setattr(store, "close", fail_once)
            return store

        monkeypatch.setattr(autosession, "mint_store", observe_mint)
        restore = AttachmentJournal.restore_bindings

        def broken_restore(journal):
            if journal.agent is agent:
                error = (
                    asyncio.CancelledError
                    if fault == "rollback_cancel"
                    else RuntimeError
                )
                raise error("rollback failed")
            return restore(journal)

        if fault.startswith("rollback"):
            monkeypatch.setattr(AttachmentJournal, "restore_bindings", broken_restore)

        async def fail_checkpoint(*args):
            raise RuntimeError("primary checkpoint failure")

        with monkeypatch.context() as patch:
            patch.setattr(session_attach.checkpoint, "checkpoint", fail_checkpoint)
            with pytest.raises(RuntimeError, match="primary checkpoint failure"):
                if route == "explicit":
                    await engine.attach_session(
                        creature.graph_id, tmp_path / "replacement.kohakutr"
                    )
                else:
                    target = autosession.mint_store(
                        engine,
                        creature.graph_id,
                        path=tmp_path / "replacement.kohakutr",
                        agents=["alice"],
                    )
                    await autosession._attach_minted_store(
                        engine, creature.graph_id, target
                    )
        assert len(captured) == 1
        assert captured[0] in getattr(engine, "_retained_session_stores", set())
        assert not captured[0]._closed
        if fault.startswith("rollback"):
            assert agent._session_attach_uncertain
            with pytest.raises(ConflictError):
                agent.attach_session_store(original)
    assert captured[0]._closed


@pytest.mark.parametrize("route", ["explicit", "autosession", "borrowed"])
@pytest.mark.parametrize("rollback_error", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("primary_error", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("rollback_step", ["detach", "bind"])
async def test_drive_rollback_failure_keeps_referenced_store_open(
    tmp_path, monkeypatch, request, route, rollback_error, primary_error, rollback_step
):
    captured = []
    mint = autosession.mint_store
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["must not run"]),
            io="headless",
            start=True,
        )
        agent, original = creature.agent, creature.agent.session_store
        runtime, gid = engine._drive_runtime, creature.graph_id
        detach, bind = runtime.detach_graph, runtime.bind_graph_store
        detach_calls = []

        def observe_mint(*args, **kwargs):
            store = mint(*args, **kwargs)
            captured.append(store)
            request.addfinalizer(lambda: SessionStore.close(store, update_status=False))
            return store

        async def fail_rollback_detach(graph_id):
            detach_calls.append(graph_id)
            if len(detach_calls) == 2 and rollback_step == "detach":
                raise rollback_error("rollback detach interrupted")
            await detach(graph_id)

        async def fail_rollback_bind(graph_id, store):
            if store is original and rollback_step == "bind":
                raise rollback_error("rollback bind interrupted")
            await bind(graph_id, store)

        async def fail_checkpoint(*args):
            assert runtime._registry._bound_stores[gid] is captured[0]
            raise primary_error("primary checkpoint failure")

        with monkeypatch.context() as patch:
            patch.setattr(autosession, "mint_store", observe_mint)
            patch.setattr(runtime, "detach_graph", fail_rollback_detach)
            patch.setattr(runtime, "bind_graph_store", fail_rollback_bind)
            patch.setattr(session_attach.checkpoint, "checkpoint", fail_checkpoint)
            with pytest.raises(primary_error) as raised:
                if route == "explicit":
                    await engine.attach_session(gid, tmp_path / "replacement.kohakutr")
                else:
                    target = autosession.mint_store(
                        engine,
                        gid,
                        path=tmp_path / "replacement.kohakutr",
                        agents=["alice"],
                    )
                    if route == "autosession":
                        await autosession._attach_minted_store(engine, gid, target)
                    else:
                        await engine.attach_session(gid, target)
        if primary_error is RuntimeError:
            assert str(raised.value) == "primary checkpoint failure"
        if rollback_step == "detach":
            assert runtime._registry._bound_stores[gid] is captured[0]
        assert agent.session_store is original
        assert not captured[0]._closed
        assert (captured[0] in getattr(engine, "_retained_session_stores", ())) == (
            route != "borrowed"
        )
        assert agent._session_attach_uncertain
        with pytest.raises(ConflictError):
            agent.attach_session_store(original)
        await engine.start(creature)
        pending = tuple(runtime._reconcile_tasks.get(gid, ()))
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), 5)
        assert gid not in runtime._registry._ready_graphs
        outcome = await creature.run(
            "must stay fenced", timeout=5, raise_on_error=False
        )
        assert outcome.status == "error"
        assert agent.llm.call_count == 0
    assert captured[0]._closed == (route != "borrowed")
    assert not engine._session_store_close_holds
