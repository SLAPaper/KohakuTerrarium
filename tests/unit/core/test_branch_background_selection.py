"""Public SELECT isolates late background deliveries from sibling branches."""

import asyncio
import threading

import pytest

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core import branch_selection
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.runtime_context import RuntimeContextPreparationError
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_stop_after_sibling_selection_does_not_pair_reused_provider_id(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class FastTool(BaseTool):
        tool_name = "fast"
        description = "Immediate"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="fast result")

    class HeldTool(BaseTool):
        tool_name = "held"
        description = "Promoted run"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            entered.set()
            await release.wait()
            return ToolResult(output="old result")

    class ReusedProvider(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused", "branch one", "unused", "unused"])
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

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ReusedProvider()
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=llm,
            tools=[FastTool(), HeldTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        rerun = asyncio.create_task(agent.regenerate_last_response())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            job_id = next(s.job_id for s in agent.executor.get_running_jobs())
            assert agent._active_handles[job_id].promote()
            await rerun
            agent.pause()
            await agent.select_branch({1: 1})
            before = agent.controller.conversation.to_json()
            await agent.stop()
            assert agent.controller.conversation.to_json() == before
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            terminal = next(
                e
                for e in events
                if e["type"] == "tool_result" and e.get("job_id") == job_id
            )
            assert terminal["branch_id"] == 2
            assert terminal["tool_call_id"] == "reused"
        finally:
            release.set()
            await asyncio.gather(rerun, return_exceptions=True)


class BranchProvider(ScriptedLLM):
    def __init__(self, native):
        super().__init__(["branch one", "[/held][held/]", "UNWANTED FOLLOWUP"])
        self.native = native
        self.last_tool_calls = []

    async def chat(self, messages, **kwargs):
        self.last_tool_calls = []
        if self.native and self.call_count == 1:
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            self.last_tool_calls = [NativeToolCall("provider-held", "held", "{}")]
            yield ""
        else:
            async for chunk in super().chat(messages, **kwargs):
                yield chunk


@pytest.mark.parametrize("queued", [False, True, "during_persist"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("selected", [1, 2])
@pytest.mark.parametrize("child", [False, True])
async def test_select_sibling_while_background_result_is_pending(
    tmp_path, monkeypatch, queued, native, selected, child
):
    entered, release, enqueued, consumed = (asyncio.Event() for _ in range(4))

    class HeldTool(BaseTool):
        tool_name = "held"
        description = "Wait for release"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            entered.set()
            await release.wait()
            return ToolResult(output="BRANCH TWO RESULT")

    class HeldChild(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            entered.set()
            await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = BranchProvider(native)
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=llm,
            tools=[] if child else [HeldTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        if child:
            cfg = SubAgentConfig(
                name="held", description="Held child", tools=[], max_turns=1
            )
            agent.subagent_manager.register(cfg)
            agent.registry.register_subagent("held", cfg)
            agent.subagent_manager.llm = HeldChild(["BRANCH TWO RESULT"])
        await creature.run("question")
        await agent.regenerate_last_response()
        await asyncio.wait_for(entered.wait(), 3)
        assert (agent._turn_index, agent._branch_id) == (1, 2)
        agent.pause()
        process = agent._process_event
        consume = agent._run_turn_for_batch

        async def observe(event):
            result = await process(event)
            enqueued.set()
            return result

        async def observe_consumption(run):
            await consume(run)
            consumed.set()

        monkeypatch.setattr(agent, "_process_event", observe)
        monkeypatch.setattr(agent, "_run_turn_for_batch", observe_consumption)
        persist_entered, persist_release = asyncio.Event(), threading.Event()
        selection_task = None
        try:
            if queued is True:
                release.set()
                await asyncio.wait_for(enqueued.wait(), 3)
            if selected == 2 and queued is not True:
                unchanged = agent.controller.conversation.to_json()
                with pytest.raises(
                    RuntimeContextPreparationError, match="incomplete results"
                ):
                    await agent.select_branch({1: selected})
                assert agent.controller.conversation.to_json() == unchanged
                assert llm.call_count == 2
                release.set()
                await asyncio.wait_for(enqueued.wait(), 3)
                await agent.output_router.flush()
            if queued == "during_persist":
                persist = branch_selection._persist_on_worker
                loop = asyncio.get_running_loop()

                def held_persist(request, prepared):
                    loop.call_soon_threadsafe(persist_entered.set)
                    assert persist_release.wait(5)
                    return persist(request, prepared)

                monkeypatch.setattr(
                    branch_selection, "_persist_on_worker", held_persist
                )
                terminal_submitted = asyncio.Event()
                submit = agent._session_output._submit_store_write

                def observe_submit(fn, *args, **kwargs):
                    receipt = submit(fn, *args, **kwargs)
                    if getattr(fn, "__name__", "") == "_persist_terminal":
                        terminal_submitted.set()
                    return receipt

                monkeypatch.setattr(
                    agent._session_output, "_submit_store_write", observe_submit
                )
                selection_task = asyncio.create_task(agent.select_branch({1: selected}))
                await asyncio.wait_for(persist_entered.wait(), 3)
                release.set()
                if selected == 1:
                    await asyncio.wait_for(terminal_submitted.wait(), 3)
                    assert not enqueued.is_set()
                persist_release.set()
                await selection_task
                await asyncio.wait_for(enqueued.wait(), 3)
            else:
                await agent.select_branch({1: selected})
            assert not agent._history_selection_uncertain
            assert (agent._turn_index, agent._branch_id) == (1, selected)
            before = agent.controller.conversation.to_json()
            if selected == 2 and child:
                messages = agent.controller.conversation.get_messages()
                assert "BRANCH TWO RESULT" in before
                if native:
                    calls = [
                        call
                        for message in messages
                        for call in (message.tool_calls or [])
                    ]
                    assert [call["id"] for call in calls] == ["provider-held"]
                    assert calls[0]["function"]["arguments"] == "{}"
                else:
                    assert not any(message.tool_calls for message in messages)
            if not queued:
                release.set()
                await asyncio.wait_for(enqueued.wait(), 3)
            agent.resume()
            await asyncio.wait_for(consumed.wait(), 3)
            await agent.output_router.flush()
            assert llm.call_count == (2 if selected == 1 else 3)
            if selected == 1:
                assert agent.controller.conversation.to_json() == before
            else:
                assert "BRANCH TWO RESULT" in agent.controller.conversation.to_json()
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            terminal = next(
                e
                for e in events
                if e["type"] == ("subagent_result" if child else "tool_result")
            )
            assert (terminal["turn_index"], terminal["branch_id"]) == (1, 2)
            assert terminal["output"] == "BRANCH TWO RESULT"
            restored = await Agent.build(
                AgentConfig(
                    name="alice", tool_format="native" if native else "bracket"
                ),
                llm=ScriptedLLM(["continued"]),
                io="headless",
            )
            await inject_saved_state_async(restored, agent.session_store, "alice")
            assert (restored._turn_index, restored._branch_id) == (1, selected)
            if selected == 1:
                assert (
                    "BRANCH TWO RESULT"
                    not in restored.controller.conversation.to_json()
                )
                assert "branch one" in restored.controller.conversation.to_json()
            await agent.stop()
        finally:
            release.set()
            persist_release.set()
            if selection_task is not None:
                await asyncio.gather(selection_task, return_exceptions=True)
