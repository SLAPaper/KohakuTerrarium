"""Cancellation provenance remains explicit without Task.cancelling support."""

import asyncio

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


class AdmissionBarrier(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()

    async def wait(self):
        self.entered.set()
        return await super().wait()


async def test_interrupt_wait_settles_a_cancelled_pre_admission_task(tmp_path):
    provider = ScriptedLLM(["next turn"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        barrier = AdmissionBarrier()
        agent._topology_handoff_ready = barrier
        run = asyncio.create_task(
            creature.run("cancel admission", timeout=3, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(barrier.entered.wait(), 2)
            assert getattr(agent, "_turn_completion", None) is None
            await agent.interrupt_and_wait(timeout=1)
            assert (await run).status == "interrupted"
            assert (
                not agent._processing_lock.locked() and not agent._consumer_task.done()
            )
            assert provider.call_count == 0
            barrier.set()
            assert (await creature.run("next", timeout=3)).text == "next turn"
        finally:
            barrier.set()
            agent._topology_handoff_ready = None
            await asyncio.gather(run, return_exceptions=True)


class LegacyTask(asyncio.Task):
    def __getattribute__(self, name):
        if name == "cancelling":
            raise AttributeError(name)
        return super().__getattribute__(name)


class HookGate(BasePlugin):
    name = "legacy_gate"

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_event(self, event):
        self.entered.set()
        await self.release.wait()


async def test_consumer_cancellation_exits_without_newer_task_count_api(tmp_path):
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    def factory(current_loop, coroutine, **kwargs):
        if (
            getattr(getattr(coroutine, "cr_code", None), "co_name", None)
            == "_run_event_consumer"
        ):
            return LegacyTask(coroutine, loop=current_loop, **kwargs)
        if original_factory is not None:
            return original_factory(current_loop, coroutine, **kwargs)
        return asyncio.Task(coroutine, loop=current_loop, **kwargs)

    loop.set_task_factory(factory)
    gate = HookGate()
    try:
        async with Terrarium(session_dir=tmp_path / "sessions") as engine:
            creature = await engine.add_creature(
                AgentConfig(name="alice"),
                llm=ScriptedLLM([]),
                plugins=[gate],
                io="headless",
                start=True,
            )
            agent = creature.agent
            run = asyncio.create_task(
                creature.run("question", timeout=3, raise_on_error=False)
            )
            consumer = agent._consumer_task
            try:
                await asyncio.wait_for(gate.entered.wait(), 2)
                assert not hasattr(consumer, "cancelling")
                consumer.cancel()
                await asyncio.wait_for(asyncio.shield(consumer), 0.2)
                assert consumer.done()
                await run
            finally:
                gate.release.set()
                if not consumer.done():
                    agent._running = False
                    consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)
                await asyncio.gather(run, return_exceptions=True)
    finally:
        loop.set_task_factory(original_factory)
