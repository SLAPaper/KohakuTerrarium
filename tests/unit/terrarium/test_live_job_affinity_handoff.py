"""A real background tool completes against its creature's post-handoff store."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.modules.plugin.base import PluginContext
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
