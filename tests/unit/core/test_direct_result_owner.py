"""Ordinary direct results and interruptions share owner-owned terminal admission."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.backgroundify import BackgroundifyHandle
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.job import JobResult, JobState, JobStatus, JobType
from kohakuterrarium.core.tool_origin import capture_tool_context
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM


class CallingProvider(ScriptedLLM):
    def __init__(self):
        super().__init__(["unused", "answer"])
        self.last_tool_calls = []

    async def chat(self, messages, **kwargs):
        self.last_tool_calls = []
        if self.call_count == 0:
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            self.last_tool_calls = [
                NativeToolCall("provider", "work", '{"run_in_background":false}')
            ]
            yield ""
        else:
            async for chunk in super().chat(messages, **kwargs):
                yield chunk


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("child", [False, True])
async def test_direct_result_pair_is_owner_commit_but_activity_is_not(
    tmp_path, monkeypatch, failed, child
):
    class Work(BaseTool):
        tool_name = "work"
        description = "Immediate result"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="payload", error="failed" if failed else None)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=CallingProvider(),
            tools=[] if child else [Work()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        if child:
            config = SubAgentConfig(
                name="work", description="work child", tools=[], max_turns=1
            )
            agent.subagent_manager.register(config)
            agent.registry.register_subagent("work", config)
            agent.subagent_manager.llm = ScriptedLLM(
                [RuntimeError("failed") if failed else "payload"]
            )
        owner = agent._session_state_owner.task
        writes, activities = [], []
        append = type(agent.controller.conversation).append
        notify = agent.output_router.notify_activity

        def observe(self, role, content, **kwargs):
            if self is agent.controller.conversation and role == "tool":
                writes.append(asyncio.current_task())
            return append(self, role, content, **kwargs)

        def observe_activity(kind, detail, metadata=None):
            if kind in ("tool_done", "tool_error", "subagent_done", "subagent_error"):
                activities.append(asyncio.current_task())
            return notify(kind, detail, metadata)

        monkeypatch.setattr(type(agent.controller.conversation), "append", observe)
        monkeypatch.setattr(agent.output_router, "notify_activity", observe_activity)
        assert (await creature.run("question")).text == "answer"
        assert writes and all(task is owner for task in writes)
        assert activities and all(task is not owner for task in activities)
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert (
            len(
                [
                    event
                    for event in raw
                    if event["type"] == ("subagent_result" if child else "tool_result")
                ]
            )
            == 1
        )


async def test_cancelled_waiter_joins_accepted_direct_result_commit(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        context = capture_tool_context(agent, "work_job", "direct", "provider")
        agent.executor.job_store.register(
            JobStatus(
                job_id="work_job",
                job_type=JobType.TOOL,
                type_name="work",
                state=JobState.RUNNING,
                context=context,
            )
        )
        agent._register_direct_job(
            "work_job", kind="tool", name="work", tool_call_id="provider"
        )
        await agent._submit_session_state(
            lambda: agent.controller.conversation.append(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "provider",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )
        )

        async def completed():
            return JobResult(job_id="work_job", output="result")

        handle = BackgroundifyHandle(
            job_id="work_job", task=asyncio.create_task(completed())
        )
        agent._active_handles["work_job"] = handle
        intercepted, saved = asyncio.Event(), []
        owner, execute = agent._session_state_owner, agent._session_state_owner._execute

        def hold(callback, future):
            if not intercepted.is_set():
                saved.append((callback, future))
                intercepted.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        waiting = asyncio.create_task(
            agent._wait_handles(
                {"work_job": handle},
                ["work_job"],
                agent.controller,
                {"work_job": "provider"},
                True,
            )
        )
        await asyncio.wait_for(intercepted.wait(), 3)
        waiting.cancel()
        await owner.submit_internal(lambda: execute(*saved[0]))
        await asyncio.wait_for(waiting, 3)
        await agent.output_router.flush()
        assert [
            m.content
            for m in agent.controller.conversation.get_messages()
            if m.role == "tool"
        ] == ["result"]
        assert "work_job" not in agent._direct_job_meta
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [e["output"] for e in raw if e["type"] == "tool_result"] == ["result"]


@pytest.mark.parametrize("after_append", [False, True])
async def test_failed_pair_does_not_burn_direct_claim_and_can_retry(
    tmp_path, monkeypatch, after_append
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        context = capture_tool_context(agent, "work_job", "direct", "provider")
        agent.executor.job_store.register(
            JobStatus(
                job_id="work_job",
                job_type=JobType.TOOL,
                type_name="work",
                state=JobState.RUNNING,
                context=context,
            )
        )
        agent._register_direct_job(
            "work_job", kind="tool", name="work", tool_call_id="provider"
        )
        await agent._submit_session_state(
            lambda: agent.controller.conversation.append(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "provider",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )
        )

        async def completed():
            return JobResult(job_id="work_job", output="result")

        handle = BackgroundifyHandle(
            job_id="work_job", task=asyncio.create_task(completed())
        )
        agent._active_handles["work_job"] = handle
        conversation = agent.controller.conversation
        before = conversation.to_json()
        append = type(conversation).append
        failed = False

        def fail_once(self, role, content, **kwargs):
            nonlocal failed
            if self is conversation and role == "tool" and not failed:
                failed = True
                if after_append:
                    append(self, role, content, **kwargs)
                raise ValueError("pair failure")
            return append(self, role, content, **kwargs)

        monkeypatch.setattr(type(conversation), "append", fail_once)
        with pytest.raises(ValueError, match="pair failure"):
            await agent._wait_handles(
                {"work_job": handle},
                ["work_job"],
                agent.controller,
                {"work_job": "provider"},
                True,
            )
        assert "work_job" not in agent._job_terminal_claims
        assert conversation.to_json() == before
        results, _ = await agent._wait_handles(
            {"work_job": handle},
            ["work_job"],
            agent.controller,
            {"work_job": "provider"},
            True,
        )
        assert results["work_job"].output == "result"
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [e["output"] for e in raw if e["type"] == "tool_result"] == ["result"]


async def test_foreign_direct_origin_is_recorded_but_not_consumed(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["first"])
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=llm,
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        context = capture_tool_context(agent, "work_job", "direct", "provider")
        agent.executor.job_store.register(
            JobStatus(
                job_id="work_job",
                job_type=JobType.TOOL,
                type_name="work",
                state=JobState.RUNNING,
                context=context,
            )
        )
        agent._register_direct_job(
            "work_job", kind="tool", name="work", tool_call_id="provider"
        )

        def sibling():
            agent._branch_id = 2
            agent.controller.conversation.append(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "provider",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )

        await agent._submit_session_state(sibling)
        before = agent.controller.conversation.to_json()

        async def completed():
            return JobResult(job_id="work_job", output="foreign result")

        handle = BackgroundifyHandle(
            job_id="work_job", task=asyncio.create_task(completed())
        )
        agent._active_handles["work_job"] = handle
        results, _ = await agent._wait_handles(
            {"work_job": handle},
            ["work_job"],
            agent.controller,
            {"work_job": "provider"},
            True,
        )
        assert results == {}
        assert agent.controller.conversation.to_json() == before
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        terminals = [e for e in raw if e["type"] == "tool_result"]
        assert len(terminals) == 1 and terminals[0]["branch_id"] == 1
        assert terminals[0]["output"] == "foreign result"
        assert llm.call_count == 1


@pytest.mark.parametrize("interruption_first", [False, True])
async def test_normal_and_interrupted_finalizers_share_first_terminal(
    tmp_path, interruption_first
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        context = capture_tool_context(agent, "work_job", "direct", "provider")
        agent.executor.job_store.register(
            JobStatus(
                job_id="work_job",
                job_type=JobType.TOOL,
                type_name="work",
                state=JobState.RUNNING,
                context=context,
            )
        )
        agent._register_direct_job(
            "work_job", kind="tool", name="work", tool_call_id="provider"
        )
        await agent._submit_session_state(
            lambda: agent.controller.conversation.append(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "provider",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )
        )
        started, release = asyncio.Event(), asyncio.Event()

        async def completed():
            started.set()
            await release.wait()
            return JobResult(job_id="work_job", output="successful result")

        task = asyncio.create_task(completed())
        handle = BackgroundifyHandle(job_id="work_job", task=task)
        agent._active_handles["work_job"] = handle
        await started.wait()
        waiting = asyncio.create_task(
            agent._wait_handles(
                {"work_job": handle},
                ["work_job"],
                agent.controller,
                {"work_job": "provider"},
                True,
            )
        )
        if interruption_first:
            interrupted = asyncio.create_task(
                agent._finalize_interrupted_direct_job("work_job")
            )
            release.set()
            await interrupted
            await waiting
        else:
            release.set()
            await waiting
            await agent._finalize_interrupted_direct_job("work_job")
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert (
            len(
                [
                    event
                    for event in raw
                    if event["type"] == "tool_result"
                    and event.get("job_id") == "work_job"
                ]
            )
            == 1
        )
        results = [
            message
            for message in agent.controller.conversation.get_messages()
            if message.role == "tool" and message.tool_call_id == "provider"
        ]
        assert len(results) == 1
