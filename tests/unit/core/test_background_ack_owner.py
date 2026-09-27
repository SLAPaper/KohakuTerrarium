"""Background dispatch and promotion acknowledgements commit on the state owner."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core import agent_tools
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("hold_ack", [False, True, "stop", "interrupt"])
async def test_native_background_ack_is_owner_commit(
    tmp_path, monkeypatch, child, promote, hold_ack
):
    release, dispatched = asyncio.Event(), asyncio.Event()

    class Work(BaseTool):
        tool_name = "work"
        description = "Held work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            await release.wait()
            return ToolResult(output="terminal result")

    class Parent(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused", "acknowledged", "completed"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if self.call_count == 0:
                self.call_count += 1
                self.call_log.append(self._normalize_messages(messages))
                args = (
                    '{"task":"work","run_in_background":'
                    + ("false" if promote else "true")
                    + "}"
                )
                self.last_tool_calls = [NativeToolCall("provider", "work", args)]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    class Child(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Parent(),
            tools=[] if child else [Work()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        if child:
            config = SubAgentConfig(
                name="work", description="Held child", tools=[], max_turns=1
            )
            agent.subagent_manager.register(config)
            agent.registry.register_subagent("work", config)
            agent.subagent_manager.llm = Child(["terminal result"])
        conversation = agent.controller.conversation
        append, notify = type(conversation).append, agent.output_router.notify_activity
        writers, job_ids, observers = [], [], []

        def observe_append(self, role, content, **kwargs):
            if self is conversation and role == "tool":
                writers.append(asyncio.current_task())
            return append(self, role, content, **kwargs)

        def observe_activity(kind, detail, metadata=None):
            observers.append(asyncio.current_task())
            if kind in ("tool_start", "subagent_start"):
                job_ids.append(metadata["job_id"])
                dispatched.set()
            return notify(kind, detail, metadata)

        monkeypatch.setattr(type(conversation), "append", observe_append)
        monkeypatch.setattr(agent.output_router, "notify_activity", observe_activity)
        owner = agent._session_state_owner
        owner_task = owner.task
        execute = owner._execute
        held, ack_ready, terminal_written = [], asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        terminal = agent.output_router.notify_terminal_activity

        def hold_commit(callback, future):
            label = getattr(callback, "__qualname__", "")
            if (
                hold_ack
                and not held
                and (
                    "_append_background_ack" in label
                    or "_handle_promotion.<locals>.commit" in label
                )
            ):
                held.append((callback, future))
                ack_ready.set()
            else:
                execute(callback, future)

        def observe_terminal(*args, **kwargs):
            receipt = terminal(*args, **kwargs)
            assert receipt is not None
            receipt.add_done_callback(
                lambda done: loop.call_soon_threadsafe(terminal_written.set)
            )
            return receipt

        monkeypatch.setattr(owner, "_execute", hold_commit)
        monkeypatch.setattr(
            agent.output_router, "notify_terminal_activity", observe_terminal
        )
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        stopping = None
        try:
            await asyncio.wait_for(dispatched.wait(), 3)
            if promote:
                assert agent._promote_handle(job_ids[0])
            if hold_ack:
                await asyncio.wait_for(ack_ready.wait(), 3)
                if hold_ack == "stop":
                    stopping = asyncio.create_task(agent.stop())
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(asyncio.shield(stopping), 0.05)
                elif hold_ack == "interrupt":
                    agent.interrupt()
                else:
                    release.set()
                    await asyncio.wait_for(terminal_written.wait(), 3)
                    assert agent.llm.call_count == 1
                    assert not [
                        m for m in conversation.get_messages() if m.role == "tool"
                    ]
                    raw = await agent.session_store.run(
                        agent.session_store.get_events, "alice"
                    )
                    kind = "subagent" if child else "tool"
                    call = next(e for e in raw if e["type"] == f"{kind}_call")
                    result = next(e for e in raw if e["type"] == f"{kind}_result")
                    assert call["event_id"] < result["event_id"]
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.wait_for(turn, 3)
            responses = [
                m
                for m in conversation.get_messages()
                if m.role == "tool" and m.tool_call_id == "provider"
            ]
            assert len(responses) == 1
            if hold_ack != "stop":
                assert "background" in responses[0].content.lower()
            if hold_ack in ("stop", "interrupt"):
                assert agent.llm.call_count == 1
            if stopping is not None:
                await asyncio.wait_for(stopping, 3)
                assert agent._stopped
            assert writers and all(task is owner_task for task in writers)
            assert observers and all(task is not owner_task for task in observers)
        finally:
            release.set()
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(
                *[task for task in (turn, stopping) if task is not None],
                return_exceptions=True,
            )


@pytest.mark.parametrize("promote", [False, True])
@pytest.mark.parametrize("notify", [False, True])
async def test_rejected_ack_is_reported_without_losing_promotion_debt(
    tmp_path, monkeypatch, promote, notify
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        before = agent.controller.conversation.to_json()
        warnings = []

        def record_warning(message, **kwargs):
            warnings.append((message, kwargs, asyncio.current_task()))

        monkeypatch.setattr(agent_tools.logger, "warning", record_warning)
        agent._bg_controller_notify["work_test"] = notify
        if promote:
            admitted = await agent._handle_promotion(
                "work_test", agent.controller, {"work_test": "orphan-id"}, True
            )
            assert agent._has_unfinished_turn_bg_jobs() is notify
        else:
            admitted = await agent._append_background_ack(
                "work_test", "orphan-id", "work", "background"
            )
        assert not admitted
        assert agent.controller.conversation.to_json() == before
        assert warnings and warnings[-1][1]["job_id"] == "work_test"
        assert all(
            task is not agent._session_state_owner.task for _, _, task in warnings
        )
        assert agent.llm.call_count == 1


async def test_stop_during_first_ack_prevents_remaining_native_dispatch(
    tmp_path, monkeypatch
):
    class Work(BaseTool):
        tool_name = "work"
        description = "Held work"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            await asyncio.Event().wait()

    class Parent(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            self.last_tool_calls = [
                NativeToolCall(f"call-{i}", "work", "{}") for i in range(2)
            ]
            yield ""

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Parent(),
            tools=[Work()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        owner, held, ready = agent._session_state_owner, [], asyncio.Event()
        execute = owner._execute

        def hold(callback, future):
            if not held and "_append_background_ack" in callback.__qualname__:
                held.append((callback, future))
                ready.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        stopping = None
        try:
            await asyncio.wait_for(ready.wait(), 3)
            stopping = asyncio.create_task(agent.stop())
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopping), 0.05)
            await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.wait_for(stopping, 3)
            await turn
            jobs = agent.executor.job_store.get_all_statuses()
            assert len(jobs) == 1
            assert jobs[0].context["tool_call_id"] == "call-0"
            raw = await agent.session_store.run(agent.session_store.get_events, "alice")
            assert [
                event["tool_call_id"] for event in raw if event["type"] == "tool_call"
            ] == ["call-0"]
        finally:
            if held and not held[0][1].done():
                await owner.submit_internal(lambda: execute(*held[0]))
            await asyncio.gather(
                *[task for task in (turn, stopping) if task is not None],
                return_exceptions=True,
            )
