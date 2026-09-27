"""A real background tool completes against its creature's post-handoff store."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.plugin.base import BasePlugin, PluginContext
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.modules.plugin.state_scope import state_prefix
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("operation", ["merge", "split"])
async def test_live_tool_terminal_and_plugin_state_follow_handoff(tmp_path, operation):
    entered, release = asyncio.Event(), asyncio.Event()
    context = None

    class GatedTool(BaseTool):
        tool_name = "bgjob"
        description = "Complete after the graph handoff"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            entered.set()
            await release.wait()
            before = await context.get_state_async("dispatch")
            await context.set_state_async("terminal", "finished")
            return ToolResult(output=f"completed-from-{before}")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(["launched\n[/bgjob]\n[bgjob/]", "handled", "done"]),
            tools=[GatedTool()],
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"),
            llm=ScriptedLLM(["bob"]),
            io="headless",
            start=True,
        )
        if operation == "split":
            await engine.connect(alice, bob, channel="bridge")
        context = PluginContext(
            agent_name="alice", _host_agent=alice.agent, _plugin_name="stateful"
        )
        bob_context = PluginContext(
            agent_name="bob", _host_agent=bob.agent, _plugin_name="stateful"
        )
        await context.set_state_async("dispatch", "alice")
        await bob_context.set_state_async("dispatch", "bob")
        original = alice.agent.session_store
        terminal = asyncio.Event()
        loop = asyncio.get_running_loop()
        current = None

        def recorded(key, event):
            if key.startswith("alice:") and event.get("type") == "tool_result":
                loop.call_soon_threadsafe(terminal.set)

        try:
            assert (await alice.run("start background job", timeout=5)).ok
            await asyncio.wait_for(entered.wait(), 3)
            assert alice.agent.executor.get_running_jobs()
            if operation == "merge":
                await engine.connect(alice, bob, channel="bridge")
            else:
                await engine.disconnect(alice, bob, channel="bridge")
            current = alice.agent.session_store
            current.subscribe(recorded)
            release.set()
            await asyncio.wait_for(terminal.wait(), 5)
            records = await current.run(current.get_events, "alice")
            terminals = [event for event in records if event["type"] == "tool_result"]
            assert len(terminals) == 1
            assert "completed-from-alice" in str(terminals[0])
            call = next(event for event in records if event["type"] == "tool_call")
            assert terminals[0]["origin"] == call["origin"]
            assert terminals[0]["execution_class"] == "background"
            assert terminals[0]["job_id"] == call["job_id"]
            assert (terminals[0]["turn_index"], terminals[0]["branch_id"]) == (
                call["turn_index"],
                call["branch_id"],
            )
            assert await context.get_state_async("terminal") == "finished"
            assert await bob_context.get_state_async("dispatch") == "bob"
            assert await bob_context.get_state_async("terminal") is None
            if original in engine._session_stores.values():
                assert not original._closed
            else:
                assert original._closed
            persisted_path = current.path
        finally:
            release.set()
            if current is not None:
                current.unsubscribe(recorded)
    reopened = SessionStore(persisted_path)
    try:
        records = await reopened.run(reopened.get_events, "alice")
        assert sum(event["type"] == "tool_result" for event in records) == 1
        assert (
            await reopened.run(
                lambda: reopened.state[state_prefix("alice") + "stateful:terminal"]
            )
            == "finished"
        )
    finally:
        await asyncio.to_thread(reopened.close)


@pytest.mark.parametrize("operation", ["merge", "split"])
@pytest.mark.parametrize("phase", ["child", "posthook"])
async def test_live_child_history_and_reserved_run_follow_handoff(
    tmp_path, operation, phase
):
    entered, release = asyncio.Event(), asyncio.Event()

    class Parent(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused", "handled"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if not self.call_count:
                self.call_count += 1
                self.last_tool_calls = [
                    NativeToolCall(
                        "child-native",
                        "work",
                        '{"task":"preserve child history", "run_in_background":true}',
                    )
                ]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    class Child(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            if phase == "child":
                entered.set()
                await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    class Posthook(BasePlugin):
        name = "rewrite_child"

        async def post_subagent_run(self, result, **kwargs):
            if phase == "posthook":
                entered.set()
                await release.wait()
            result.success = False
            result.error = "posthook recorded"
            return result

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["bob"]), io="headless", start=True
        )
        alice = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Parent(),
            plugins=[Posthook()],
            io="headless",
            start=True,
        )
        if operation == "split":
            carol = await engine.add_creature(
                AgentConfig(name="carol"),
                llm=ScriptedLLM(["carol"]),
                io="headless",
                start=True,
            )
            await engine.connect(bob, carol, channel="stay")
            await engine.connect(bob, alice, channel="bridge")
        manager = alice.agent.subagent_manager
        manager._parent_plugins = alice.agent.plugins
        manager.llm = Child(["child output persisted"])
        config = SubAgentConfig(
            name="work", description="Persist child", tools=[], max_turns=1
        )
        manager.register(config)
        alice.agent.registry.register_subagent("work", config)
        original = alice.agent.session_store
        try:
            assert (await alice.run("launch", timeout=5)).ok
            await asyncio.wait_for(entered.wait(), 3)
            job_id, job = next(iter(manager._jobs.items()))
            run_index = job.subagent._run_index
            if operation == "merge":
                await engine.connect(bob, alice, channel="bridge")
            else:
                await engine.disconnect(bob, alice, channel="bridge")
            current = alice.agent.session_store
            assert current is not original
            second_job = await manager.spawn(
                "work", "second reserved task", background=True
            )
            release.set()
            await manager.wait_all(timeout=5)
            assert manager._jobs[second_job].subagent._run_index > run_index
            row = await current.run(current.find_subagent_run, job_id, parent="alice")
            assert row is not None, "handoff lost the canonical child run"
            assert row["run"] == run_index
            assert row["meta"]["success"] is False
            assert row["meta"]["error"] == "posthook recorded"
            history = await current.run(
                current.load_subagent_conversation, "alice", "work", run_index
            )
            assert (
                "preserve child history" in history
                and "child output persisted" in history
            )
            next_run = await current.run(current.next_subagent_run, "alice", "work")
            assert next_run > run_index
            if operation == "split":
                await engine.connect(bob, alice, channel="reunited")
                current = alice.agent.session_store
                assert (
                    await current.run(current.find_subagent_run, job_id, parent="alice")
                )["meta"]["error"] == "posthook recorded"
                assert (
                    await current.run(
                        current.find_subagent_run, second_job, parent="alice"
                    )
                    is not None
                )
            persisted_path = current.path
        finally:
            release.set()
            await manager.wait_all(timeout=5)
    reopened = SessionStore(persisted_path)
    try:
        row = await reopened.run(reopened.find_subagent_run, job_id, parent="alice")
        assert row["meta"]["error"] == "posthook recorded"
        assert "child output persisted" in await reopened.run(
            reopened.load_subagent_conversation, "alice", "work", run_index
        )
    finally:
        await asyncio.to_thread(reopened.close)


async def test_storeless_child_gets_reserved_identity_when_graph_gains_store(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Child(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            entered.set()
            await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        persisted = await engine.add_creature(
            AgentConfig(name="persisted"),
            llm=ScriptedLLM(["ok"]),
            io="headless",
            start=True,
        )
        moving = await engine.add_creature(
            AgentConfig(name="moving"),
            llm=ScriptedLLM(["ok"]),
            session=False,
            io="headless",
            start=True,
        )
        manager = moving.agent.subagent_manager
        manager.register(SubAgentConfig(name="child", max_turns=1))
        manager.llm = Child(["first child"])
        first = await manager.spawn("child", "first task", background=True)
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await engine.connect(persisted, moving, channel="bridge")
            store = moving.agent.session_store
            release.set()
            await manager.wait_for(first, timeout=3)
            manager.llm = ScriptedLLM(["second child"])
            second = await manager.spawn("child", "second task", background=False)
            first_row = await store.run(store.find_subagent_run, first, parent="moving")
            second_row = await store.run(
                store.find_subagent_run, second, parent="moving"
            )
            assert first_row is not None and second_row is not None
            assert first_row["run"] != second_row["run"]
            assert "first child" in await store.run(
                store.load_subagent_conversation, "moving", "child", first_row["run"]
            )
        finally:
            release.set()
            await manager.wait_all(timeout=3)
