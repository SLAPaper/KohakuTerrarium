"""Public lifecycle stops preserve accepted graph-handoff authority."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("target", ["self", "peer"])
async def test_coordinator_remove_authorizes_its_stop_but_not_plugin_child_stop(
    tmp_path,
    target,
):
    class StopProbe(BasePlugin):
        name = "stop_probe"
        creature = None
        observed = False

        async def on_agent_stop(self):
            async def foreign_stop():
                before = (
                    self.creature._running,
                    self.creature._stop_requested,
                    self.creature._restoration_state,
                )
                with pytest.raises(
                    ConflictError, match="callback" if target == "self" else "handoff"
                ):
                    await self.creature.stop(requested=False)
                assert (
                    self.creature._running,
                    self.creature._stop_requested,
                    self.creature._restoration_state,
                ) == before

            await asyncio.create_task(foreign_stop())
            self.observed = True

    probe = StopProbe()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["a", "after"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"),
            llm=ScriptedLLM(["b"]),
            plugins=[probe],
            io="headless",
            start=True,
        )
        probe.creature = bob if target == "self" else alice
        await alice.run("first", timeout=3)
        await bob.run("second", timeout=3)
        await engine.connect(alice, bob, channel="bridge")
        await asyncio.wait_for(engine.remove_creature(bob), 3)
        assert probe.observed and bob not in engine.list_creatures()
        assert (await alice.run("next", timeout=3)).text == "after"


@pytest.mark.parametrize("surface", ["agent", "creature", "creature_waiting"])
async def test_public_stop_rejects_before_host_or_agent_mutation_during_connect(
    tmp_path, monkeypatch, surface
):
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original_flush = SessionStore.flush
    blocked = False
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        await alice.run("first", timeout=3)
        await bob.run("second", timeout=3)
        store = alice.agent.session_store

        def held_flush(candidate, *args, **kwargs):
            nonlocal blocked
            if candidate is store and not blocked:
                blocked = True
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(10)
            return original_flush(candidate, *args, **kwargs)

        monkeypatch.setattr(SessionStore, "flush", held_flush)
        stop = None
        held_lifecycle = surface == "creature_waiting"
        before = (
            bob._running,
            bob._stop_requested,
            bob._restoration_state,
            bob._input_task,
            bob.agent._running,
            bob.agent._session_state_owner.is_running,
        )
        if held_lifecycle:
            await bob._lifecycle_lock.acquire()
            stop = asyncio.create_task(bob.stop())
            await asyncio.sleep(0)
        pending = asyncio.create_task(engine.connect(alice, bob, channel="bridge"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert engine._topology_handoff_task is not None
            if held_lifecycle:
                bob._lifecycle_lock.release()
                held_lifecycle = False
            else:
                stop = asyncio.create_task(
                    (bob.agent if surface == "agent" else bob).stop()
                )
            with pytest.raises(ConflictError, match="handoff"):
                await asyncio.wait_for(asyncio.shield(stop), 0.1)
            after = (
                bob._running,
                bob._stop_requested,
                bob._restoration_state,
                bob._input_task,
                bob.agent._running,
                bob.agent._session_state_owner.is_running,
            )
            assert after == before
            assert not getattr(bob.agent, "_model_switch_closing", False)
            assert not getattr(bob.agent, "_history_selection_closing", False)
            release.set()
            await asyncio.wait_for(pending, 3)
            assert len(engine.list_graphs()) == 1
            assert bob.is_running and alice.is_running
        finally:
            if held_lifecycle:
                bob._lifecycle_lock.release()
            release.set()
            await asyncio.gather(
                pending, *([stop] if stop is not None else []), return_exceptions=True
            )
