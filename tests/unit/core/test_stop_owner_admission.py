"""Stop and interruption pair the current native occurrence on its owner."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.events import create_tool_complete_event
from kohakuterrarium.core.backgroundify import BackgroundifyHandle
from kohakuterrarium.core.job import JobResult, JobStatus, JobState, JobType
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_completed_job_delivery_during_stop_is_persisted_without_new_turn(
    tmp_path, monkeypatch
):
    delivered, release = asyncio.Event(), asyncio.Event()

    class HeldBackground(BaseTool):
        tool_name = "held"
        description = "Held background"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            await release.wait()
            return ToolResult(output="completed before stop")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["[/held][held/]", "must not run"])
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=llm,
            tools=[HeldBackground()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        original = agent._on_bg_complete
        queued = []

        def hold(event):
            queued.append(event)
            delivered.set()

        agent.executor._on_complete = hold
        await creature.run("question")
        agent.pause()
        release.set()
        await asyncio.wait_for(delivered.wait(), 3)
        cancel = agent._cancel_executor_tasks

        async def finish_delivery():
            assert not agent._running
            original(queued[0])
            original(queued[0])
            await cancel()

        monkeypatch.setattr(agent, "_cancel_executor_tasks", finish_delivery)
        await agent.stop()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        terminals = [e for e in raw if e["type"] == "tool_result"]
        assert [e["output"] for e in terminals] == ["completed before stop"]
        assert terminals[0]["origin"] == queued[0].context["origin"]
        assert llm.call_count == 1 and not agent._event_inbox


async def test_concurrent_direct_finalizers_publish_once(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        context = capture_tool_context(agent, "read_job", "direct", "provider")
        agent.executor.job_store.register(
            JobStatus(
                job_id="read_job",
                job_type=JobType.TOOL,
                type_name="read",
                state=JobState.RUNNING,
                context=context,
            )
        )
        agent._register_direct_job(
            "read_job", kind="tool", name="read", tool_call_id="provider"
        )
        release = asyncio.Event()

        async def result():
            await release.wait()
            return JobResult(job_id="read_job", error="interrupted")

        task = asyncio.create_task(result())
        agent._active_handles["read_job"] = BackgroundifyHandle(
            job_id="read_job", task=task
        )
        a = asyncio.create_task(agent._finalize_interrupted_direct_job("read_job"))
        b = asyncio.create_task(agent._finalize_interrupted_direct_job("read_job"))
        release.set()
        await asyncio.gather(a, b)
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert (
            len(
                [
                    e
                    for e in raw
                    if e["type"] == "tool_result" and e.get("job_id") == "read_job"
                ]
            )
            == 1
        )


@pytest.mark.parametrize("fails", [False, True])
async def test_stop_joins_delayed_direct_finalizer_before_owner_close(
    tmp_path, monkeypatch, fails
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent._register_direct_job(
            "read_job", kind="tool", name="read", tool_call_id="provider"
        )

        async def held_tool():
            await asyncio.Event().wait()

        task = asyncio.create_task(held_tool())
        agent._active_handles["read_job"] = BackgroundifyHandle(
            job_id="read_job", task=task
        )
        entered, release = asyncio.Event(), asyncio.Event()
        finalize = agent._finalize_interrupted_direct_job

        async def held_finalize(job_id):
            entered.set()
            await release.wait()
            await finalize(job_id)
            if fails:
                raise RuntimeError("finalizer failed after commit")

        monkeypatch.setattr(agent, "_finalize_interrupted_direct_job", held_finalize)
        assert agent._interrupt_direct_job("read_job")
        await entered.wait()
        stopped = asyncio.create_task(agent.stop())
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopped), 0.1)
            release.set()
            await asyncio.wait_for(stopped, 3)
            assert not agent._direct_job_meta and not agent._active_handles
            assert agent._stopped and agent._session_state_owner.task is None
        finally:
            release.set()
            await stopped


async def test_invalid_stop_origin_does_not_publish_or_abort_other_jobs(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        for job_id in ("bad", "good"):
            context = capture_tool_context(agent, job_id, "background", "provider")
            if job_id == "bad":
                context["origin"]["turn_index"] = 0
                assert (
                    agent._on_bg_complete(
                        create_tool_complete_event(
                            job_id=job_id, content="bad", **context
                        )
                    )
                    is None
                )
            agent.executor.job_store.register(
                JobStatus(
                    job_id=job_id,
                    job_type=JobType.TOOL,
                    type_name="read",
                    state=JobState.RUNNING,
                    context=context,
                )
            )
        await agent.stop()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [e["job_id"] for e in raw if e["type"] == "tool_result"] == ["good"]
        assert not agent.is_running


async def test_accepted_background_receipt_during_stop_does_not_enqueue_turn(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        agent.pause()
        notified = asyncio.Event()

        async def observe(event):
            notified.set()

        monkeypatch.setattr(agent, "_process_event", observe)
        event = create_tool_complete_event(
            job_id="read_late",
            content="done",
            **capture_tool_context(agent, "read_late", "background", "provider"),
        )
        agent._on_bg_complete(event)
        await agent.stop()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [e["output"] for e in raw if e["type"] == "tool_result"] == ["done"]
        assert not notified.is_set()


class ReusingProvider(ScriptedLLM):
    def __init__(self):
        super().__init__(["unused", "first answer", "unused", "followup"])
        self.last_tool_calls = []

    async def chat(self, messages, **kwargs):
        self.last_tool_calls = []
        if self.call_count in (0, 2):
            name = "fast" if self.call_count == 0 else "held"
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            self.last_tool_calls = [NativeToolCall("reused", name, "{}")]
            yield ""
        else:
            async for chunk in super().chat(messages, **kwargs):
                yield chunk


@pytest.mark.parametrize("shutdown", [False, True])
async def test_reused_provider_id_pairs_only_current_pending_occurrence_on_owner(
    tmp_path, monkeypatch, shutdown
):
    entered = asyncio.Event()

    class Fast(BaseTool):
        tool_name = "fast"
        description = "Immediate"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="first result")

    class Held(BaseTool):
        tool_name = "held"
        description = "Held"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ReusingProvider(),
            tools=[Fast(), Held()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("first")
        turn = asyncio.create_task(creature.run("second", raise_on_error=False))
        await asyncio.wait_for(entered.wait(), 3)
        conversation = agent.controller.conversation
        append = type(conversation).append
        writers = []
        owner_task = agent._session_state_owner.task

        def observe(self, role, content, **kwargs):
            if self is conversation and role == "tool":
                writers.append(asyncio.current_task())
            return append(self, role, content, **kwargs)

        monkeypatch.setattr(type(conversation), "append", observe)
        if shutdown:
            await agent.stop()
        else:
            await agent.interrupt_and_wait()
            await asyncio.gather(
                *[h.task for h in agent._active_handles.values()],
                return_exceptions=True,
            )
            await agent.output_router.flush()
        await turn
        results = [
            m
            for m in conversation.get_messages()
            if m.role == "tool" and m.tool_call_id == "reused"
        ]
        assert len(results) == 2
        assert results[0].content == "first result"
        assert (
            "interrupt" in results[-1].content.lower()
            or "stopped" in results[-1].content.lower()
        )
        assert writers and all(task is owner_task for task in writers)
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert (
            len(
                [
                    event
                    for event in raw
                    if event["type"] == "tool_result" and event["turn_index"] == 2
                ]
            )
            == 1
        )
