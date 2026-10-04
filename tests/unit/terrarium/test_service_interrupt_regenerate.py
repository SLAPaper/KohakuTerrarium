"""Service regeneration settles interrupted execution before truthful completion."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.terrarium.service import LocalTerrariumService
from kohakuterrarium.testing.llm import ScriptedLLM


class FirstBeginGate(BasePlugin):
    name = "first_begin_gate"

    def __init__(self, rerun=False):
        self.rerun = rerun
        self.once = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_event(self, event):
        if self.once or bool(event.context.get("rerun")) != self.rerun:
            return
        self.once = True
        self.entered.set()
        await self.release.wait()


async def test_service_busy_regeneration_interrupts_pre_controller_and_runs_once(
    tmp_path,
):
    hook = FirstBeginGate()
    provider = ScriptedLLM(["replacement"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[hook],
            io="headless",
            start=True,
        )
        service = LocalTerrariumService(engine)
        original = asyncio.create_task(
            creature.run("question", timeout=5, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(hook.entered.wait(), 2)
            result = await asyncio.wait_for(
                service.regenerate(creature.creature_id, request_id="busy-rerun"), 3
            )
            assert (
                result["status"] == "completed" and result["request_id"] == "busy-rerun"
            )
            assert result["turn_index"] == 1 and result["branch_id"] == 2
            assert (await original).status == "interrupted"
            assert provider.call_count == 1 and provider.last_user_message == "question"
            assert (
                creature.agent.controller.conversation.get_messages()[-1].content
                == "replacement"
            )
        finally:
            hook.release.set()
            await asyncio.gather(original, return_exceptions=True)


async def test_service_receipt_is_not_relabelled_by_a_subsequent_turn(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(["original", "replacement", "later turn"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        await creature.run("question", timeout=3)
        operation = creature.agent.interrupt_and_regenerate

        async def settled_then_next(**kwargs):
            receipt = await operation(**kwargs)
            assert (
                await creature.run("later question", timeout=3)
            ).text == "later turn"
            return receipt

        monkeypatch.setattr(
            creature.agent, "interrupt_and_regenerate", settled_then_next
        )
        result = await LocalTerrariumService(engine).regenerate(
            creature.creature_id, request_id="stable"
        )
        assert result == {
            "status": "completed",
            "request_id": "stable",
            "turn_index": 1,
            "branch_id": 2,
            "parent_branch_path": [],
            "branch_view": {"1": 2},
        }
        assert creature.agent._turn_index == 2
        assert result["turn_index"] != creature.agent._turn_index
        assert provider.call_count == 3


async def test_service_does_not_report_completed_when_rerun_is_interrupted(tmp_path):
    hook = FirstBeginGate(rerun=True)
    provider = ScriptedLLM(["original", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[hook],
            io="headless",
            start=True,
        )
        service = LocalTerrariumService(engine)
        await creature.run("question", timeout=3)
        request = asyncio.create_task(
            service.regenerate(creature.creature_id, request_id="interrupted-rerun")
        )
        try:
            await asyncio.wait_for(hook.entered.wait(), 2)
            await service.interrupt(creature.creature_id)
            with pytest.raises(ConflictError, match="interrupted"):
                await asyncio.wait_for(request, 3)
            assert provider.call_count == 1
        finally:
            hook.release.set()
            await asyncio.gather(request, return_exceptions=True)
