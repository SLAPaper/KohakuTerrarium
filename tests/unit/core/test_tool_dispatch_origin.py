"""Actual Agent dispatch stamps immutable tool origin before execution."""

import asyncio
import json

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


class OriginProvider(ScriptedLLM):
    def __init__(self, native, requested_background=False):
        args = {"run_in_background": True} if requested_background else {}
        body = "@@run_in_background=true\n" if requested_background else ""
        super().__init__([f"[/originjob]{body}[originjob/]", "ack", "after"])
        self.native = native
        self.arguments = json.dumps(args)
        self.last_tool_calls = []

    async def chat(self, messages, **kwargs):
        self.last_tool_calls = []
        if self.native and self.call_count == 0:
            self.call_log.append(self._normalize_messages(messages))
            self.call_count += 1
            self.last_tool_calls = [
                NativeToolCall("provider-origin", "originjob", self.arguments)
            ]
            yield ""
        else:
            async for chunk in super().chat(messages, **kwargs):
                yield chunk


@pytest.mark.parametrize("native", [False, True])
async def test_stop_terminal_keeps_background_dispatch_origin(tmp_path, native):
    entered, release = asyncio.Event(), asyncio.Event()

    class BackgroundTool(BaseTool):
        tool_name = "originjob"
        description = "Wait until stop"
        execution_mode = ExecutionMode.BACKGROUND

        async def _execute(self, args, **kwargs):
            entered.set()
            await release.wait()
            return ToolResult(output="done")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=OriginProvider(native),
            tools=[BackgroundTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        try:
            assert (await creature.run("first", timeout=5)).ok
            await asyncio.wait_for(entered.wait(), 3)
            assert (await creature.run("second", timeout=5)).ok
            if native:
                agent.controller.conversation.clear(keep_system=True)
            before = agent.controller.conversation.get_messages()[:]
            await agent.stop()
            assert agent.controller.conversation.get_messages() == before
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            start = next(e for e in events if e["type"] == "tool_call")
            terminals = [e for e in events if e["type"] == "tool_result"]
            assert len(terminals) == 1
            assert terminals[0]["origin"] == start["origin"]
            assert terminals[0]["turn_index"] == 1
            assert terminals[0]["execution_class"] == "background"
            if native:
                assert (
                    terminals[0]["tool_call_id"]
                    == start["tool_call_id"]
                    == "provider-origin"
                )
        finally:
            release.set()


@pytest.mark.parametrize(
    "background", [False, True, "promoted", "promoted_immediate", "requested"]
)
@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("retention", [0, 100])
@pytest.mark.parametrize("native", [False, True])
async def test_tool_dispatch_and_terminal_keep_origin(
    tmp_path, background, fails, retention, native
):
    entered, release, recorded = asyncio.Event(), asyncio.Event(), asyncio.Event()
    statuses = []
    agent = None

    class GatedTool(BaseTool):
        tool_name = "originjob"
        description = "Origin test"
        execution_mode = (
            ExecutionMode.BACKGROUND if background is True else ExecutionMode.DIRECT
        )

        async def _execute(self, args, **kwargs):
            statuses.extend(agent.executor.get_running_jobs())
            entered.set()
            await release.wait()
            return ToolResult(output="done", error="failed" if fails else None)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=OriginProvider(native, requested_background=background == "requested"),
            tools=[GatedTool()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        store = agent.session_store
        agent.executor.job_store._max_completed = retention
        loop = asyncio.get_running_loop()

        def observer(key, event):
            if key.startswith("alice:") and event.get("type") == "tool_result":
                loop.call_soon_threadsafe(recorded.set)

        store.subscribe(observer)
        running = asyncio.create_task(creature.run("first", timeout=8))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            origin = statuses[0].context["origin"]
            assert origin["turn_index"] == 1
            assert origin["branch_id"] == 1
            assert origin["execution_class"] == (
                "background"
                if background is True or background == "requested"
                else "direct"
            )
            if background in ("promoted", "promoted_immediate"):
                assert agent._active_handles[statuses[0].job_id].promote()
            if background == "promoted_immediate":
                release.set()
            if background and background != "promoted_immediate":
                await running
                assert (await creature.run("second", timeout=5)).ok
                assert agent._turn_index == 2
            release.set()
            await running
            await asyncio.wait_for(recorded.wait(), 4)
            events = await store.run(store.get_events, "alice")
            start = next(e for e in events if e["type"] == "tool_call")
            terminal = next(e for e in events if e["type"] == "tool_result")
            assert start["origin"] == origin
            assert terminal["origin"] == {
                **origin,
                "execution_class": "background" if background else "direct",
            }
            assert start["job_id"] == terminal["job_id"] == statuses[0].job_id
            assert (terminal["turn_index"], terminal["branch_id"]) == (1, 1)
            assert terminal["execution_class"] == (
                "background" if background else "direct"
            )
            assert terminal["output"] == "done"
            if native:
                assert (
                    terminal["tool_call_id"]
                    == start["tool_call_id"]
                    == "provider-origin"
                )
            assert sum(e["type"] == "tool_result" for e in events) == 1
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)
            store.unsubscribe(observer)
