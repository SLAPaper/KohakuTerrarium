"""Interrupt and regeneration contracts across real controller and callback phases."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.interrupt_regeneration import _capture, _validate
from kohakuterrarium.core.session_state_owner import SessionStateOwner
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.output.base import BaseOutputModule
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("field", ["llm", "_session_state_owner"])
async def test_regeneration_binding_rejects_provider_or_owner_replacement(
    tmp_path, field
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original"]),
            io="headless",
            start=True,
        )
        await creature.run("question", timeout=3)
        agent = creature.agent
        binding = _capture(agent)
        old = getattr(agent, field)
        replacement = (
            ScriptedLLM([]) if field == "llm" else SessionStateOwner(name="other owner")
        )
        setattr(agent, field, replacement)
        try:
            with pytest.raises(ConflictError, match="source changed"):
                _validate(agent, binding)
            assert not agent._processing_lock.locked()
        finally:
            setattr(agent, field, old)


async def test_regeneration_binding_detaches_mutable_restored_parent_path(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["one", "two"]),
            io="headless",
            start=True,
        )
        await creature.run("first", timeout=3)
        await creature.run("second", timeout=3)
        agent = creature.agent
        original = agent._parent_branch_path
        agent._parent_branch_path = [list(pair) for pair in original]
        binding = _capture(agent)
        try:
            agent._parent_branch_path[0][1] = 999
            with pytest.raises(ConflictError, match="source changed"):
                _validate(agent, binding)
        finally:
            agent._parent_branch_path = original


async def test_regeneration_binding_rejects_changed_user_payload(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original"]),
            io="headless",
            start=True,
        )
        await creature.run("question", timeout=3)
        binding = _capture(creature.agent)
        message = binding.tail_message
        original = message.content
        message.content = "changed without a confirmed edit"
        try:
            with pytest.raises(ConflictError, match="source changed"):
                _validate(creature.agent, binding)
        finally:
            message.content = original


@pytest.mark.parametrize(
    "operation", ["capture_primary_user", "install_primary_user", "arm_primary_user"]
)
async def test_public_interrupt_joins_accepted_user_receipt_and_keeps_owner_usable(
    tmp_path, monkeypatch, operation
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        provider = ScriptedLLM(["next"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        owner = agent._session_state_owner
        held, entered = [], asyncio.Event()
        execute = owner._execute

        def hold(callback, future):
            if not held and operation in getattr(
                getattr(callback, "__code__", None), "co_names", ()
            ):
                held.append((callback, future))
                entered.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        run = asyncio.create_task(
            creature.run("question", timeout=5, raise_on_error=False)
        )
        wait = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            wait = asyncio.create_task(agent.interrupt_and_wait(timeout=1))
            await asyncio.sleep(0)
            assert not wait.done() and not held[0][1].cancelled()
            await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.wait_for(wait, 2)
            assert (await run).status == "interrupted"
            assert provider.call_count == 0
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert len([r for r in rows if r["type"] == "user_message"]) == (
                0 if operation == "capture_primary_user" else 1
            )
            assert getattr(agent, "_primary_user_receipt", None) is None
            assert (await creature.run("next question", timeout=3)).text == "next"
            assert provider.last_user_message == "next question"
        finally:
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(
                run, *([wait] if wait is not None else []), return_exceptions=True
            )


async def test_cancel_resistant_early_hook_cannot_restart_provider_after_interrupt(
    tmp_path,
):
    class ResistsCancellation(BasePlugin):
        name = "resists_cancel"

        def __init__(self):
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def on_event(self, event):
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()

    hook = ResistsCancellation()
    provider = ScriptedLLM(["must not generate"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[hook],
            io="headless",
            start=True,
        )
        run = asyncio.create_task(
            creature.run("question", timeout=5, raise_on_error=False)
        )
        pending = None
        try:
            await asyncio.wait_for(hook.entered.wait(), 2)
            pending = asyncio.create_task(creature.agent.interrupt_and_wait(timeout=1))
            await asyncio.wait_for(hook.cancelled.wait(), 2)
            assert not pending.done()
            hook.release.set()
            await asyncio.wait_for(pending, 2)
            assert (await run).status == "interrupted"
            assert provider.call_count == 0
            assert not creature.agent._consumer_task.done()
        finally:
            hook.release.set()
            await asyncio.gather(
                run, *([pending] if pending is not None else []), return_exceptions=True
            )


class BeginGate(BasePlugin):
    name = "begin_gate"

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.once = False

    async def on_event(self, event):
        if self.once:
            return
        self.once = True
        self.entered.set()
        await self.release.wait()


class SelfHistory(BasePlugin):
    name = "self_history"

    def __init__(self):
        self.agent = None
        self.outcome = None

    async def on_event(self, event):
        if not event.context.get("rerun"):
            return
        task = asyncio.current_task()
        timer = asyncio.get_running_loop().call_later(0.2, task.cancel)
        try:
            await self.agent.rewind_to(1)
        except asyncio.CancelledError:
            self.outcome = TimeoutError()
            uncancel = getattr(task, "uncancel", None)
            if callable(uncancel):
                uncancel()
        except Exception as exc:
            self.outcome = exc
        finally:
            timer.cancel()


async def test_interrupt_before_controller_keeps_consumer_and_next_turn_usable(
    tmp_path,
):
    gate = BeginGate()
    provider = ScriptedLLM(["after interrupt"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[gate],
            io="headless",
            start=True,
        )
        agent = creature.agent
        run = asyncio.create_task(
            creature.run("interrupted input", timeout=5, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            consumer = agent._consumer_task
            await agent.interrupt_and_wait(timeout=0.5)
            result = await asyncio.wait_for(run, timeout=2)
            assert result.status == "interrupted"
            assert provider.call_count == 0
            assert agent._consumer_task is consumer and not consumer.done()
            assert not agent._processing_lock.locked()
            assert (
                await creature.run("next input", timeout=3)
            ).text == "after interrupt"
            users = [
                m["content"] for m in provider.last_messages if m["role"] == "user"
            ]
            assert users[-1] == "next input"
            assert users.count("next input") == 1
        finally:
            gate.release.set()
            await asyncio.gather(run, return_exceptions=True)


async def test_same_turn_history_reentry_refuses_before_any_self_wait(tmp_path):
    plugin = SelfHistory()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original", "replacement"]),
            plugins=[plugin],
            io="headless",
            start=True,
        )
        plugin.agent = creature.agent
        await creature.run("question", timeout=3)
        await asyncio.wait_for(creature.agent.regenerate_last_response(), timeout=3)
        assert isinstance(plugin.outcome, ConflictError)
        assert "own" in str(plugin.outcome)
        assert not creature.agent._processing_lock.locked()
        assert (
            creature.agent.controller.conversation.get_messages()[-1].content
            == "replacement"
        )


async def test_external_interrupt_and_regenerate_replaces_early_hook_once(tmp_path):
    gate = BeginGate()
    provider = ScriptedLLM(["replacement"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[gate],
            io="headless",
            start=True,
        )
        agent = creature.agent
        run = asyncio.create_task(
            creature.run("question", timeout=5, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(gate.entered.wait(), timeout=2)
            await asyncio.wait_for(agent.interrupt_and_regenerate(timeout=1), timeout=3)
            assert (await run).status == "interrupted"
            assert provider.call_count == 1
            assert provider.last_messages[-1]["content"] == "question"
            assert (
                agent.controller.conversation.get_messages()[-1].content
                == "replacement"
            )
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert len([row for row in rows if row["type"] == "user_message"]) == 2
            assert (
                len(
                    [row for row in rows if row["type"] == "assistant_generation_start"]
                )
                == 1
            )
        finally:
            gate.release.set()
            await asyncio.gather(run, return_exceptions=True)


class DeferredRegeneration(BasePlugin):
    name = "deferred_regeneration"

    def __init__(self):
        self.agent = None
        self.tokens = []

    async def on_event(self, event):
        self.tokens.append(self.agent.request_regeneration())


async def test_callback_regeneration_returns_token_and_cannot_form_rerun_loop(tmp_path):
    plugin = DeferredRegeneration()
    provider = ScriptedLLM(["original", "replacement", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[plugin],
            io="headless",
            start=True,
        )
        plugin.agent = creature.agent
        await creature.run("question", timeout=3, raise_on_error=False)
        task = creature.agent._deferred_regeneration_task
        await asyncio.wait_for(asyncio.shield(task), timeout=3)
        assert plugin.tokens and all(isinstance(token, str) for token in plugin.tokens)
        assert len(set(plugin.tokens)) == 1
        assert provider.call_count == 1
        assert (
            creature.agent.controller.conversation.get_messages()[-1].content
            == "original"
        )
        assert task.done() and task.exception() is None


async def test_interrupt_wait_from_own_callback_is_rejected_without_interrupting_itself(
    tmp_path,
):
    class OwnInterrupt(BasePlugin):
        name = "own_interrupt"
        agent = None
        rejected = False

        async def on_event(self, event):
            with pytest.raises(ConflictError, match="own"):
                await self.agent.interrupt_and_wait()
            self.rejected = True

    plugin = OwnInterrupt()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["completed normally"]),
            plugins=[plugin],
            io="headless",
            start=True,
        )
        plugin.agent = creature.agent
        result = await creature.run("question", timeout=3)
        assert (
            plugin.rejected
            and result.text == "completed normally"
            and result.status == "ok"
        )


class PhaseOutputGate(BaseOutputModule):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.visits = 0

    async def write(self, content):
        pass

    async def on_processing_start(self):
        await self._visit("processing_start")

    async def on_processing_end(self):
        await self._visit("processing_end")

    async def _visit(self, phase):
        if phase == self.phase:
            self.visits += 1
            if self.visits == 1:
                self.entered.set()
                await self.release.wait()


async def test_interrupt_before_llm_withdraws_only_its_queued_controller_input(
    tmp_path,
):
    output = PhaseOutputGate("processing_start")
    provider = ScriptedLLM(["next response"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        agent.output_router.add_secondary(output)
        run = asyncio.create_task(
            creature.run("old input", timeout=5, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(output.entered.wait(), 2)
            assert agent.controller.has_pending_events()
            await agent.interrupt_and_wait(timeout=1)
            assert (await run).status == "interrupted"
            assert provider.call_count == 0
            assert not agent.controller.has_pending_events()
            assert (await creature.run("new input", timeout=3)).text == "next response"
            assert provider.last_user_message == "new input"
            assert provider.call_count == 1
        finally:
            output.release.set()
            await asyncio.gather(run, return_exceptions=True)


async def test_interrupt_during_finalization_waits_and_does_not_emit_end_twice(
    tmp_path,
):
    output = PhaseOutputGate("processing_end")
    provider = ScriptedLLM(["completed text"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        agent.output_router.add_secondary(output)
        run = asyncio.create_task(
            creature.run("input", timeout=5, raise_on_error=False)
        )
        try:
            await asyncio.wait_for(output.entered.wait(), 2)
            with pytest.raises(ConflictError, match="still finishing"):
                await agent.interrupt_and_wait(timeout=0.02)
            assert output.visits == 1 and not run.done()
            assert not agent._turn_finalization_task.cancelled()
            output.release.set()
            await agent.interrupt_and_wait(timeout=1)
            result = await run
            assert result.status == "interrupted" and result.text == "completed text"
            assert output.visits == 1 and provider.call_count == 1
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert len([r for r in rows if r["type"] == "text_chunk"]) == 1
        finally:
            output.release.set()
            await asyncio.gather(run, return_exceptions=True)


async def test_interrupt_regeneration_detaches_requested_view_before_waiting(tmp_path):
    output = PhaseOutputGate("processing_end")
    provider = ScriptedLLM(["original", "replacement"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        agent.output_router.add_secondary(output)
        run = asyncio.create_task(
            creature.run("question", timeout=5, raise_on_error=False)
        )
        request = None
        try:
            await asyncio.wait_for(output.entered.wait(), 2)
            view = {1: 1}
            request = asyncio.create_task(
                agent.interrupt_and_regenerate(
                    turn_index=1, branch_view=view, timeout=1
                )
            )
            await asyncio.sleep(0)
            assert not request.done()
            view[1] = 999
            output.release.set()
            await asyncio.wait_for(request, 3)
            assert (await run).status == "interrupted"
            assert provider.call_count == 2 and provider.last_user_message == "question"
            assert (
                agent.controller.conversation.get_messages()[-1].content
                == "replacement"
            )
        finally:
            output.release.set()
            await asyncio.gather(
                run, *([request] if request is not None else []), return_exceptions=True
            )
