"""Compound regeneration reserves its source before accepting other mutations."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_cancelled_compound_waiter_keeps_reservation_until_accepted_rerun_settles(
    tmp_path,
):
    class RerunGate(BasePlugin):
        name = "rerun_cancel_probe"

        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def on_event(self, event):
            if event.context.get("rerun"):
                self.entered.set()
                await self.release.wait()

    gate = RerunGate()
    provider = ScriptedLLM(["original", "replacement"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[gate],
            io="headless",
            start=True,
        )
        await creature.run("question", timeout=3)
        agent = creature.agent
        pending = asyncio.create_task(agent.interrupt_and_regenerate())
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            pending.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not pending.done()
            assert agent._regeneration_reservation is not None
            with pytest.raises(ConflictError, match="regeneration"):
                await agent.select_branch({1: 1})
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 3)
            assert agent._regeneration_reservation is None
            assert provider.call_count == 2
            assert (
                agent.controller.conversation.get_messages()[-1].content
                == "replacement"
            )
        finally:
            gate.release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("surface", ["topology", "stop", "selection"])
async def test_foreign_mutation_cannot_replace_source_during_compound_scope_admission(
    tmp_path, monkeypatch, surface
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original", "rerun"]),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM([]), io="headless", start=True
        )
        await alice.run("question", timeout=3)
        agent = alice.agent
        assert getattr(agent, "_runtime_scope_id", None) is None
        owner = agent._session_state_owner
        execute = owner._execute
        entered = asyncio.Event()
        held = []

        def hold(callback, future):
            if getattr(callback, "__name__", None) == "initialize_scope" and not held:
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        source = agent.session_store
        pending = asyncio.create_task(agent.interrupt_and_regenerate())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            with pytest.raises(ConflictError, match="regeneration"):
                match surface:
                    case "topology":
                        await engine.connect(alice, bob, channel="must not merge")
                    case "stop":
                        await alice.stop()
                    case "selection":
                        await agent.select_branch({1: 1})
            assert alice.is_running and agent._session_state_owner.is_running
            assert alice.agent.session_store is source
            assert len(engine.list_graphs()) == 2
            await owner.submit_internal(lambda: execute(*held[0]))
            result = await asyncio.wait_for(pending, 3)
            assert result["status"] == "completed"
            assert alice.agent.llm.call_count == 2
        finally:
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(pending, return_exceptions=True)


async def test_already_queued_input_prevents_rerun_without_losing_input(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(["original", "queued answer"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        await creature.run("question", timeout=3)
        agent = creature.agent
        queued = asyncio.Event()
        put = agent._event_inbox.put

        def observe_put(envelope):
            if envelope.event.content == "queued question":
                agent._consumer_resume.clear()
                result = put(envelope)
                queued.set()
                return result
            return put(envelope)

        monkeypatch.setattr(agent._event_inbox, "put", observe_put)
        owner = agent._session_state_owner
        execute = owner._execute
        entered, held = asyncio.Event(), []

        def hold(callback, future):
            if getattr(callback, "__name__", None) == "initialize_scope" and not held:
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        waiting_input = asyncio.create_task(creature.run("queued question", timeout=5))
        request = None
        try:
            await asyncio.wait_for(queued.wait(), 2)
            request = asyncio.create_task(agent.interrupt_and_regenerate())
            await asyncio.wait_for(entered.wait(), 2)
            agent._consumer_resume.set()
            await owner.submit_internal(lambda: execute(*held[0]))
            with pytest.raises(ConflictError, match="turn is active"):
                await asyncio.wait_for(request, 3)
            assert (await waiting_input).text == "queued answer"
            assert provider.call_count == 2
            assert [
                m["content"] for m in provider.last_messages if m["role"] == "user"
            ] == ["question", "queued question"]
            assert getattr(agent, "_regeneration_reservation", None) is None
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert [
                (row["content"], row["turn_index"], row["branch_id"])
                for row in events
                if row["type"] == "user_message"
            ] == [("question", 1, 1), ("queued question", 2, 1)]
        finally:
            agent._consumer_resume.set()
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(
                waiting_input,
                *([request] if request is not None else []),
                return_exceptions=True,
            )
