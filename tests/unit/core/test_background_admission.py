"""Consumption gates keep classified completions out of unrelated lineages."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.background_admission import admit_background_run
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.event_inbox import EventEnvelope
from kohakuterrarium.core.events import (
    create_tool_complete_event,
    create_user_input_event,
)
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_cancelled_admission_does_not_burn_delivery(tmp_path, monkeypatch):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("first")
        agent.pause()
        event = create_tool_complete_event(
            job_id="read_job",
            content="retry",
            **capture_tool_context(agent, "read_job", "background"),
        )
        owner = agent._session_state_owner
        execute = owner._execute
        held, intercepted = [], asyncio.Event()

        def hold_once(callback, future):
            if not intercepted.is_set():
                held.append((callback, future))
                intercepted.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold_once)
        envelope = EventEnvelope(
            event, future=asyncio.get_running_loop().create_future()
        )
        async with agent._processing_lock:
            waiting = asyncio.create_task(admit_background_run(agent, [envelope]))
            await intercepted.wait()
            waiting.cancel("cancelled before consumption")
            await owner.submit_internal(lambda: execute(*held[0]))
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert envelope.future.done()
            retry = EventEnvelope(deepcopy(event))
            assert await admit_background_run(agent, [retry]) == [retry]


@pytest.mark.parametrize("after_commit", [False, True])
async def test_stopped_gate_does_not_claim_delivery(
    tmp_path, monkeypatch, after_commit
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("first")
        agent.pause()
        event = create_tool_complete_event(
            job_id="read_job",
            content="retry",
            **capture_tool_context(agent, "read_job", "background"),
        )
        async with agent._processing_lock:
            try:
                if after_commit:
                    owner = agent._session_state_owner
                    execute = owner._execute

                    def stop_after_commit(callback, future):
                        execute(callback, future)
                        agent._running = False

                    monkeypatch.setattr(owner, "_execute", stop_after_commit)
                else:
                    agent._running = False
                assert await admit_background_run(agent, [EventEnvelope(event)]) == []
            finally:
                if after_commit:
                    monkeypatch.setattr(owner, "_execute", execute)
                agent._running = True
            retry = EventEnvelope(deepcopy(event))
            assert await admit_background_run(agent, [retry]) == [retry]


@pytest.mark.parametrize("mid_turn", [False, True])
async def test_real_terminal_retained_when_queued_result_becomes_foreign(
    tmp_path, monkeypatch, mid_turn
):
    entered, release, enqueued = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class HeldTool(BaseTool):
        tool_name = "held"
        description = "Held background result"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            entered.set()
            await release.wait()
            return ToolResult(output="OLD BRANCH RESULT")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["[/held][held/]", "forbidden followup"])
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=llm,
            tools=[HeldTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("first")
        await entered.wait()
        agent.pause()
        original_process = agent._process_event

        async def observe(event):
            result = await original_process(event)
            enqueued.set()
            return result

        monkeypatch.setattr(agent, "_process_event", observe)
        try:
            release.set()
            await asyncio.wait_for(enqueued.wait(), 3)
            async with agent._processing_lock:
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent, "_branch_id", 2)
                )
                before = agent.controller.conversation.to_json()
                if mid_turn:
                    assert (
                        await agent._drain_mid_turn_pending_inputs(agent.controller)
                        == 0
                    )
                else:
                    claimed = agent._event_inbox.drain_all()
            if not mid_turn:
                await agent._run_turn_for_batch(claimed)
            await agent.output_router.flush()
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            terminal = next(e for e in events if e["type"] == "tool_result")
            assert (terminal["turn_index"], terminal["branch_id"]) == (1, 1)
            assert terminal["output"] == "OLD BRANCH RESULT"
            assert agent.controller.conversation.to_json() == before
            assert llm.call_count == 1
        finally:
            release.set()


async def test_admission_rechecks_after_processing_lock(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["first", "must not run"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("first")
        agent.pause()
        event = create_tool_complete_event(
            job_id="read_job",
            content="QUEUED",
            **capture_tool_context(agent, "read_job", "background"),
        )
        future = asyncio.get_running_loop().create_future()
        started = asyncio.Event()

        async def drive():
            started.set()
            await agent._run_turn_for_batch([EventEnvelope(event, future=future)])

        async with agent._processing_lock:
            task = asyncio.create_task(drive())
            await started.wait()
            await agent._session_state_owner.submit_internal(
                lambda: setattr(agent, "_branch_id", 2)
            )
        await task
        assert future.result().was_primary is False
        assert llm.call_count == 1


async def test_mixed_legacy_and_invalid_completion_keeps_user_identity(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("first")
        legacy = EventEnvelope(
            create_tool_complete_event(job_id="old", content="legacy")
        )
        user = EventEnvelope(create_user_input_event("user"))
        user.event.context["opaque"] = asyncio.get_running_loop().create_future()
        invalid = EventEnvelope(
            create_tool_complete_event(
                job_id="bad", content="no", execution_class="background"
            )
        )
        async with agent._processing_lock:
            accepted = await admit_background_run(agent, [invalid, user, legacy])
        assert accepted == [user, legacy]
        assert accepted[0].event is user.event
        assert not user.event.context["opaque"].done()


@pytest.mark.parametrize("mid_turn", [False, True])
@pytest.mark.parametrize("damage", ["sibling", "scope", "job", "boolean", "missing"])
async def test_unrelated_completion_is_rejected_before_conversation(
    tmp_path, mid_turn, damage
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["first answer", "must not be called"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("first")
        agent.pause()
        context = capture_tool_context(agent, "read_job", "background")
        if damage == "sibling":
            context["origin"]["branch_id"] += 1
            context["origin"]["origin_path"][-1][1] += 1
        elif damage == "scope":
            context["origin"]["runtime_scope_id"] = "other"
        elif damage == "job":
            context["origin"]["delivery_id"] += "forged"
        elif damage == "boolean":
            context["origin"]["turn_index"] = True
        else:
            context.pop("origin")
        event = create_tool_complete_event(
            job_id="read_job", content="FOREIGN RESULT", **context
        )
        future = asyncio.get_running_loop().create_future()
        envelope = EventEnvelope(event, future=future)
        before = deepcopy(agent.controller.conversation.to_json())
        if mid_turn:
            async with agent._processing_lock:
                agent._event_inbox.put(envelope)
                assert await agent._drain_mid_turn_pending_inputs(agent.controller) == 0
        else:
            await agent._run_turn_for_batch([envelope])
        assert future.done()
        assert future.result().was_primary is False
        assert agent.controller.conversation.to_json() == before
        assert llm.call_count == 1


@pytest.mark.parametrize("mid_turn", [False, True])
@pytest.mark.parametrize("descendant", [False, True])
async def test_compatible_delivery_is_consumed_once(
    tmp_path, mid_turn, descendant, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["first answer", "accepted", "duplicate must not run"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("first")
        context = capture_tool_context(agent, "read_job", "background")
        event = create_tool_complete_event(
            job_id="read_job", content="ONE RESULT", **context
        )
        if descendant:
            await creature.run("second")
        agent.pause()
        calls_before = llm.call_count
        owner = agent._session_state_owner
        original_submit = owner.submit_internal
        commits = []

        def observe(callback):
            def commit():
                commits.append(asyncio.current_task())
                return callback()

            return original_submit(commit)

        monkeypatch.setattr(owner, "submit_internal", observe)
        for attempt in range(2):
            envelope = EventEnvelope(deepcopy(event))
            if mid_turn:
                async with agent._processing_lock:
                    agent._event_inbox.put(envelope)
                    count = await agent._drain_mid_turn_pending_inputs(agent.controller)
                assert count == (1 if attempt == 0 else 0)
            else:
                await agent._run_turn_for_batch([envelope])
        messages = agent.controller.conversation.get_messages()
        assert sum("ONE RESULT" in str(m.content) for m in messages) == 1
        assert llm.call_count == calls_before + (0 if mid_turn else 1)
        assert commits and all(task is owner.task for task in commits)
