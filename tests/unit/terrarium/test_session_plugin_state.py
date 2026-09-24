"""Plugin state handoff validates ownership and uses each store's worker."""

import threading
from types import SimpleNamespace

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import PluginContext
from kohakuterrarium.modules.plugin.state_scope import state_prefix
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.terrarium.session_plugin_state import transfer_graph_plugin_state
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_transfer_replaces_only_moving_namespace_on_correct_workers(
    tmp_path, monkeypatch
):
    source = SessionStore(tmp_path / "source.kohakutr")
    target = SessionStore(tmp_path / "target.kohakutr")
    namespace = state_prefix("alice")
    other = state_prefix("bob")
    try:
        await source.run(
            lambda: source.state.__setitem__(namespace + "p:keep", {"v": [2]})
        )
        await target.run(
            lambda: target.state.update({namespace + "p:stale": 1, other + "p:keep": 9})
        )
        creature = SimpleNamespace(
            agent=SimpleNamespace(
                config=SimpleNamespace(name="alice"), session_store=source
            )
        )
        engine = SimpleNamespace(
            _topology=SimpleNamespace(
                graphs={"g": SimpleNamespace(creature_ids={"a"})}
            ),
            _creatures={"a": creature},
        )
        accesses = []
        put = type(target.state).__setitem__

        def observed(table, key, value):
            if table is target.state:
                accesses.append(threading.get_ident())
            return put(table, key, value)

        monkeypatch.setattr(type(target.state), "__setitem__", observed)
        transfer_graph_plugin_state(engine, "g", target)
        assert accesses == [target._affinity_worker_ident]
        assert accesses[0] != threading.get_ident()
        assert await target.run(lambda: target.state[namespace + "p:keep"]) == {
            "v": [2]
        }
        assert await target.run(lambda: target.state.get(namespace + "p:stale")) is None
        assert await target.run(lambda: target.state[other + "p:keep"]) == 9
        await source.run(lambda: source.state.__delitem__(namespace + "p:keep"))
        transfer_graph_plugin_state(engine, "g", target)
        assert await target.run(lambda: target.state.get(namespace + "p:keep")) is None
        assert await target.run(lambda: target.state[other + "p:keep"]) == 9
    finally:
        source.close()
        target.close()


async def test_legacy_ambiguity_rejects_merge_before_graph_mutation(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["ok"]), io="headless", start=True
        )
        store = alice.agent.session_store

        def legacy():
            store.meta["agents"] = ["alice", "ambiguous"]
            store.state["plugin:p:value"] = 7

        await store.run(legacy)
        membership = dict(engine._topology.creature_to_graph)
        stores = dict(engine._session_stores)
        with pytest.raises(ConflictError, match="legacy"):
            await engine.connect(alice, bob, channel="bridge")
        assert engine._topology.creature_to_graph == membership
        assert engine._session_stores == stores
        assert await store.run(lambda: store.state["plugin:p:value"]) == 7
        await store.run(lambda: store.meta.__setitem__("agents", ["alice"]))
        bob_store = bob.agent.session_store
        await bob_store.run(lambda: bob_store.state.__setitem__("plugin:p:value", 9))
        await engine.connect(alice, bob, channel="bridge")
        context = PluginContext(_host_agent=alice.agent, _plugin_name="p")
        assert await context.get_state_async("value") == 7
        other_context = PluginContext(_host_agent=bob.agent, _plugin_name="p")
        assert other_context.get_state("value") == 9
