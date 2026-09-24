"""Topology changes must reject model conflicts before touching graphs or stores."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize(
    "operation", ["connect", "disconnect", "remove_channel", "remove_creature"]
)
async def test_model_switch_preflight_leaves_topology_and_stores_unchanged(
    tmp_path, operation
):
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
        if operation != "connect":
            await engine.connect(alice, bob, channel="bridge")
        membership = dict(engine._topology.creature_to_graph)
        stores = dict(engine._session_stores)
        environments = dict(engine._environments)
        alice.agent._model_switch_pending = True
        try:
            with pytest.raises(ConflictError, match="Model switch"):
                match operation:
                    case "connect":
                        await engine.connect(alice, bob, channel="bridge")
                    case "disconnect":
                        await engine.disconnect(alice, bob, channel="bridge")
                    case "remove_channel":
                        await engine.remove_channel(alice.graph_id, "bridge")
                    case "remove_creature":
                        await engine.remove_creature(bob)
            assert engine._topology.creature_to_graph == membership
            assert engine._session_stores == stores
            assert engine._environments == environments
            assert engine.get_creature(bob.creature_id) is bob
            for store in stores.values():
                assert not store._closed
                await store.run(store.flush)
        finally:
            alice.agent._model_switch_pending = False


async def test_removal_reserves_other_creature_model_across_stop_await(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class StopGate(BasePlugin):
        name = "stop_gate"

        async def on_agent_stop(self):
            entered.set()
            await release.wait()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"),
            llm=ScriptedLLM(["ok"]),
            plugins=[StopGate()],
            io="headless",
            start=True,
        )
        await engine.connect(alice, bob, channel="bridge")
        original = alice.agent.llm
        removal = asyncio.create_task(engine.remove_creature(bob))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            with pytest.raises(ConflictError):
                await alice.agent.switch_model_async("openai/gpt-5.4-nano")
            assert alice.agent.llm is original
            assert engine.get_creature(bob.creature_id) is bob
        finally:
            release.set()
            await removal
        assert alice.agent._model_sensitive_extension_active == 0
        assert (await alice.run("after removal", timeout=5)).text == "ok"
