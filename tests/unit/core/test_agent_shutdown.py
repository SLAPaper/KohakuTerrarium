"""Accepted agent shutdown joins cleanup without callback self-wait."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.operation_admission import require_stop_available
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.input.base import BaseInputModule
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


class ShutdownGate(BasePlugin):
    name = "shutdown_gate"

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.visits = 0

    async def on_agent_stop(self):
        self.visits += 1
        self.entered.set()
        await self.release.wait()


async def test_cancelled_creature_stop_reaps_host_input_and_restoration_before_return(
    tmp_path,
):
    gate = ShutdownGate()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM([]),
            plugins=[gate],
            io="headless",
            start=True,
        )
        input_task = creature._input_task
        restoration = creature._restoration_task
        pending = asyncio.create_task(creature.stop())
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
            assert creature._input_task is None and input_task.done()
            assert creature._restoration_task is None and restoration.done()
            assert not creature._running and creature.agent._stopped
            assert not creature._lifecycle_lock.locked()
            await creature.stop()
            assert gate.visits == 1
        finally:
            gate.release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await creature.stop()


async def test_stop_interrupts_early_hook_before_releasing_its_plugin_lifetime(
    tmp_path,
):
    class EarlyGate(BasePlugin):
        name = "early_shutdown_gate"

        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = False
            self.unloaded_before_settlement = False

        async def on_event(self, event):
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def on_unload(self):
            self.unloaded_before_settlement = not self.cancelled

    gate = EarlyGate()
    provider = ScriptedLLM(["must not execute"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[gate],
            io="headless",
            start=True,
        )
        running = asyncio.create_task(
            creature.run("question", timeout=20, raise_on_error=False)
        )
        stopping = None
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            stopping = asyncio.create_task(creature.stop())
            await asyncio.wait_for(asyncio.shield(stopping), 1)
            assert gate.cancelled and not gate.unloaded_before_settlement
            assert (await running).status == "interrupted"
            assert provider.call_count == 0
            assert creature.agent._stopped
            assert not creature.agent._processing_lock.locked()
        finally:
            gate.release.set()
            await asyncio.gather(
                running,
                *([stopping] if stopping is not None else []),
                return_exceptions=True,
            )


async def test_cancelled_stop_joins_cleanup_before_reporting_cancellation(tmp_path):
    gate = ShutdownGate()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            plugins=[gate],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question", timeout=3)
        pending = asyncio.create_task(agent.stop())
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            pending.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not pending.done()
            assert not agent._shutdown_task.cancelled()
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
            assert agent._stopped and not agent._running
            assert not agent._session_state_owner.is_running
            assert agent._consumer_task.done()
            await agent.stop()
            assert gate.visits == 1
        finally:
            gate.release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await agent.stop()


async def test_failed_text_finalization_during_stop_is_not_swallowed_as_provider_failure(
    tmp_path, monkeypatch
):
    class PausedProvider(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused"])
            self.entered = asyncio.Event()

        async def chat(self, messages, **kwargs):
            self.call_count += 1
            yield "P" * 600
            self.entered.set()
            await asyncio.Event().wait()

    class CleanupProbe(BasePlugin):
        name = "failed_finalization_cleanup"
        unloaded = False

        async def on_unload(self):
            self.unloaded = True

    provider, plugin = PausedProvider(), CleanupProbe()
    engine = Terrarium(session_dir=tmp_path / "sessions")
    await engine.__aenter__()
    creature = await engine.add_creature(
        AgentConfig(name="alice"),
        llm=provider,
        plugins=[plugin],
        io="headless",
        start=True,
    )
    agent, sink, store = (
        creature.agent,
        creature.agent._session_output,
        creature.agent.session_store,
    )
    running = asyncio.create_task(
        creature.run("question", timeout=5, raise_on_error=False)
    )
    append = store.append_event

    def fail_finalization(namespace, kind, data, **kwargs):
        if kind == "text_chunk":
            raise OSError("durable text finalization failed")
        return append(namespace, kind, data, **kwargs)

    try:
        await asyncio.wait_for(provider.entered.wait(), 2)
        await sink.drain()
        slot = await store.run(store.state.get, "alice:open_text")
        assert slot["text"] == "P" * 600
        with monkeypatch.context() as patch:
            patch.setattr(store, "append_event", fail_finalization)
            with pytest.raises(OSError, match="durable text finalization failed"):
                await asyncio.wait_for(agent.stop(), 2)
        assert (await running).status == "error"
        assert not agent._stopped and agent._session_state_owner.is_closing
        assert not agent._session_state_owner.task.done()
        assert (
            await agent._session_state_owner.submit_internal(
                lambda: "retained cleanup authority"
            )
            == "retained cleanup authority"
        )
        assert not plugin.unloaded
        assert await store.run(store.state.get, "alice:open_text") == slot
        assert not [
            r
            for r in await store.run(store.get_events, "alice")
            if r["type"] == "text_chunk"
        ]
    finally:
        agent.output_router.remove_secondary(sink)
        agent._session_output = None
        await asyncio.gather(running, return_exceptions=True)
        await engine.shutdown()


async def test_settled_provider_failure_during_stop_does_not_skip_dependency_cleanup(
    tmp_path,
):
    class CancelFailureProvider(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused"])
            self.entered = asyncio.Event()

        async def chat(self, messages, **kwargs):
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            yield "partial reply"
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("provider failed while cancellation settled")

    class CleanupProbe(BasePlugin):
        name = "cleanup_probe"
        unloaded = False

        async def on_unload(self):
            self.unloaded = True

    provider, plugin = CancelFailureProvider(), CleanupProbe()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[plugin],
            io="headless",
            start=True,
        )
        run = asyncio.create_task(
            creature.run("question", timeout=5, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(provider.entered.wait(), 2)
            await asyncio.wait_for(creature.stop(), 2)
            assert (await run).status == "error"
            assert plugin.unloaded
            assert (
                creature.agent._stopped
                and not creature.agent._session_state_owner.is_running
            )
            assert not creature.agent._processing_lock.locked()
            rows = await creature.agent.session_store.run(
                creature.agent.session_store.get_events, "alice"
            )
            assert (
                sum(
                    row.get("type") == "text_chunk"
                    and row.get("content") == "partial reply"
                    for row in rows
                )
                == 1
            )
        finally:
            await asyncio.gather(run, return_exceptions=True)


async def test_turn_callback_child_is_wait_ineligible_before_stop_mutation(tmp_path):
    class ChildAdmissionProbe(BasePlugin):
        name = "turn_child_stop_probe"
        agent = None
        failure = None

        async def on_event(self, event):
            async def child():
                try:
                    require_stop_available(self.agent)
                except ConflictError as exc:
                    self.failure = exc

            await asyncio.create_task(child())

    probe = ChildAdmissionProbe()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            plugins=[probe],
            io="headless",
            start=True,
        )
        probe.agent = creature.agent
        assert (await creature.run("question", timeout=3)).text == "answer"
        assert isinstance(probe.failure, ConflictError)
        assert creature.is_running and not creature.stop_requested
        assert creature.agent._session_state_owner.is_running


async def test_real_input_driver_is_wait_ineligible_before_stop_mutation(tmp_path):
    class InputStopProbe(BaseInputModule):
        def __init__(self):
            super().__init__()
            self.observed = asyncio.Event()
            self.release = asyncio.Event()
            self.failure = None
            self.creature = None

        async def get_input(self):
            assert asyncio.current_task() is self.creature._input_task
            try:
                require_stop_available(self.creature.agent)
            except ConflictError as exc:
                self.failure = exc
            self.observed.set()
            await self.release.wait()
            return None

        async def _on_stop(self):
            self.release.set()

    probe = InputStopProbe()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM([]), io="headless", start=False
        )
        probe.creature = creature
        creature.agent.input = probe
        await creature.start()
        try:
            await asyncio.wait_for(probe.observed.wait(), 2)
            assert isinstance(probe.failure, ConflictError)
            assert not creature.stop_requested and creature.is_running
            assert creature.agent._session_state_owner.is_running
            assert not getattr(creature.agent, "_model_switch_closing", False)
        finally:
            probe.release.set()


async def test_stop_hook_child_cannot_join_the_shutdown_awaiting_that_child(tmp_path):
    class ChildStop(BasePlugin):
        name = "stop_hook_child"

        def __init__(self):
            self.agent = None
            self.child = None
            self.failure = None

        async def on_agent_stop(self):
            self.child = asyncio.create_task(self.agent.stop())
            try:
                await asyncio.wait_for(asyncio.shield(self.child), 0.2)
            except Exception as exc:
                self.failure = exc

    plugin = ChildStop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM([]),
            plugins=[plugin],
            io="headless",
            start=True,
        )
        plugin.agent = creature.agent
        await creature.agent.stop()
        await asyncio.gather(plugin.child, return_exceptions=True)
        assert isinstance(plugin.failure, ConflictError)
        assert "callback" in str(plugin.failure)
        assert (
            creature.agent._stopped
            and not creature.agent._session_state_owner.is_running
        )


async def test_concurrent_stop_waiters_share_cleanup_and_preserve_other_waiter(
    tmp_path,
):
    gate = ShutdownGate()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM([]),
            plugins=[gate],
            io="headless",
            start=True,
        )
        agent = creature.agent
        first = asyncio.create_task(agent.stop())
        second = None
        try:
            await asyncio.wait_for(gate.entered.wait(), 2)
            second = asyncio.create_task(agent.stop())
            first.cancel()
            await asyncio.sleep(0)
            assert not first.done()
            assert not second.done()
            gate.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 2)
            await asyncio.wait_for(second, 2)
            assert gate.visits == 1 and agent._stopped
        finally:
            gate.release.set()
            await asyncio.gather(
                first, *([second] if second is not None else []), return_exceptions=True
            )
            await agent.stop()


async def test_stop_hook_self_stop_refuses_before_waiting_for_its_own_cleanup(tmp_path):
    class SelfStop(BasePlugin):
        name = "self_stop"

        def __init__(self):
            self.agent = None
            self.once = False
            self.error = None

        async def on_agent_stop(self):
            if self.once:
                return
            self.once = True
            task = asyncio.current_task()
            timer = asyncio.get_running_loop().call_later(0.1, task.cancel)
            try:
                await self.agent.stop()
            except asyncio.CancelledError:
                self.error = TimeoutError()
                uncancel = getattr(task, "uncancel", None)
                if callable(uncancel):
                    uncancel()
            except Exception as exc:
                self.error = exc
            finally:
                timer.cancel()

    plugin = SelfStop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM([]),
            plugins=[plugin],
            io="headless",
            start=True,
        )
        plugin.agent = creature.agent
        await creature.agent.stop()
        assert isinstance(plugin.error, ConflictError)
        assert "own" in str(plugin.error)
        assert creature.agent._stopped
