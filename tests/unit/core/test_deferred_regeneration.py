"""Deferred requests distinguish accepted work from successful turn completion."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.output.base import BaseOutputModule
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


class RerunGate(BasePlugin):
    name = "rerun_gate"

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_event(self, event):
        if event.context.get("rerun"):
            self.entered.set()
            await self.release.wait()


class Activities(BaseOutputModule):
    def __init__(self):
        super().__init__()
        self.kinds = []

    async def write(self, content):
        pass

    def on_activity(self, kind, detail):
        self.kinds.append(kind)

    def on_activity_with_metadata(self, kind, detail, metadata):
        self.kinds.append(kind)


async def test_deferred_rerun_interrupted_outcome_is_not_success(tmp_path):
    gate = RerunGate()
    activities = Activities()
    provider = ScriptedLLM(["original", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[gate],
            io="headless",
            start=True,
        )
        agent = creature.agent
        agent.output_router.add_secondary(activities)
        await creature.run("question", timeout=3)
        token = agent.request_regeneration(request_id="retry-one")
        pending = agent._deferred_regeneration_task
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            await agent.interrupt_and_wait(timeout=1)
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.sleep(0)
            assert "regeneration_done" not in activities.kinds
            assert any(
                kind in activities.kinds
                for kind in ("regeneration_failed", "regeneration_cancelled")
            )
            assert provider.call_count == 1
            assert agent.request_regeneration(request_id=token) == token
            status = agent.get_regeneration_status(token)
            assert status["status"] == "interrupted"
            assert status["request_id"] == token
            assert provider.call_count == 1
        finally:
            gate.release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("terminal", ["completed", "error", "interrupted"])
async def test_cancelled_deferred_waiter_retains_actual_terminal_receipt(
    tmp_path, terminal
):
    class FailureProvider(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            if self.call_count and terminal == "error":
                self.call_count += 1
                raise RuntimeError("deferred provider failure")
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    provider, gate, activities = (
        FailureProvider(["original", "replacement"]),
        RerunGate(),
        Activities(),
    )
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[gate],
            io="headless",
            start=True,
        )
        agent = creature.agent
        agent.output_router.add_secondary(activities)
        await creature.run("question", timeout=3)
        token = agent.request_regeneration(request_id="cancelled-waiter")
        pending = agent._deferred_regeneration_task
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            if terminal == "interrupted":
                await agent.interrupt_and_wait(timeout=1)
            gate.release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.sleep(0)
            status = agent.get_regeneration_status(token)
            assert status["status"] == terminal
            assert status["waiter_cancelled"] is True
            assert status["applied"] is True
            assert (
                status["receipt"]["branch_id"] == 2
                and status["receipt"]["turn_index"] == 1
            )
            assert status["receipt"]["request_id"] == token
            expected = (
                "regeneration_done"
                if terminal == "completed"
                else (
                    "regeneration_cancelled"
                    if terminal == "interrupted"
                    else "regeneration_failed"
                )
            )
            assert activities.kinds.count(expected) == 1
            if terminal != "completed":
                assert "regeneration_done" not in activities.kinds
            assert provider.call_count == (1 if terminal == "interrupted" else 2)
            assert agent.request_regeneration(request_id=token) == token
            status["receipt"]["branch_id"] = 999
            assert agent.get_regeneration_status(token)["receipt"]["branch_id"] == 2
        finally:
            gate.release.set()
            await asyncio.gather(pending, return_exceptions=True)


async def test_deferred_cancel_before_first_step_has_no_applied_receipt(tmp_path):
    provider = ScriptedLLM(["original", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        await creature.run("question", timeout=3)
        agent = creature.agent
        token = agent.request_regeneration(request_id="pre-start-cancel")
        agent._deferred_regeneration_task.cancel()
        await asyncio.gather(agent._deferred_regeneration_task, return_exceptions=True)
        status = agent.get_regeneration_status(token)
        assert status["status"] == "cancelled" and status["waiter_cancelled"]
        assert not status["applied"] and status["receipt"] is None
        assert agent._branch_id == 1 and provider.call_count == 1
        assert not getattr(agent, "_regeneration_reservation", None)


async def test_pinned_provider_rechecked_at_actual_rerun_owner_admission(
    tmp_path, monkeypatch
):
    original = ScriptedLLM(["original", "must not rerun"])
    replacement = ScriptedLLM(["must not run on changed source"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=original, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("question", timeout=3)
        owner = agent._session_state_owner
        execute = owner._execute
        held, entered = [], asyncio.Event()

        def hold(callback, future):
            if not held and "_capture" in getattr(
                getattr(callback, "__code__", None), "co_names", ()
            ):
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        before = await agent.session_store.run(agent.session_store.get_events, "alice")
        pending = asyncio.create_task(agent.interrupt_and_regenerate(timeout=1))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            agent.llm = replacement
            await owner.submit_internal(lambda: execute(*held[0]))
            with pytest.raises(ConflictError, match="source changed"):
                await asyncio.wait_for(pending, 3)
            assert replacement.call_count == 0
            after = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert [row for row in after if row["type"] != "plugin_hook_timing"] == [
                row for row in before if row["type"] != "plugin_hook_timing"
            ]
            assert not getattr(agent, "_history_rerun_uncertain", False)
        finally:
            agent.llm = original
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(pending, return_exceptions=True)


async def test_completed_request_status_is_detached_and_replay_does_not_rerun(tmp_path):
    provider = ScriptedLLM(["original", "replacement"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        await creature.run("question", timeout=3)
        agent = creature.agent
        token = agent.request_regeneration(request_id="one")
        await asyncio.wait_for(asyncio.shield(agent._deferred_regeneration_task), 3)
        status = agent.get_regeneration_status(token)
        assert status["status"] == "completed"
        status["status"] = "forged"
        assert agent.get_regeneration_status(token)["status"] == "completed"
        assert agent.request_regeneration(request_id=token) == token
        assert provider.call_count == 2
        with pytest.raises(ConflictError, match="unknown"):
            agent.get_regeneration_status("not-that-request")
