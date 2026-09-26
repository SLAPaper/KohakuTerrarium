"""Real child-agent runs retain dispatch lineage across parent turns."""

import asyncio
import json

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM


class ParentProvider(ScriptedLLM):
    def __init__(self, native, background):
        super().__init__(
            [
                f"[/child]@@run_in_background={'true' if background else 'false'}\n@@task=work\n[child/]",
                "ack",
                "after",
            ]
        )
        self.native = native
        self.background = background
        self.last_tool_calls = []

    async def chat(self, messages, **kwargs):
        self.last_tool_calls = []
        if self.native and not self.call_count:
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            self.last_tool_calls = [
                NativeToolCall(
                    "provider-child",
                    "child",
                    json.dumps({"task": "work", "run_in_background": self.background}),
                )
            ]
            yield ""
        else:
            async for chunk in super().chat(messages, **kwargs):
                yield chunk


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("rejection", ["depth", "unknown", "invalid_flag"])
async def test_depth_rejection_has_native_terminal(tmp_path, background, rejection):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ParentProvider(
                True, "maybe" if rejection == "invalid_flag" else background
            ),
            io="headless",
            start=True,
        )
        agent = creature.agent
        cfg = SubAgentConfig(name="child", description="child", tools=[])
        agent.subagent_manager.register(cfg)
        agent.registry.register_subagent("child", cfg)
        if rejection == "depth":
            agent.subagent_manager._max_depth = 1
            agent.subagent_manager._current_depth = 1
        elif rejection == "unknown":
            agent.subagent_manager._configs.pop("child")
        expected_error = {
            "depth": "depth limit",
            "unknown": "not registered",
            "invalid_flag": "boolean",
        }[rejection]
        result = await creature.run("first", timeout=5)
        assert result.ok
        await agent.output_router.flush()
        events = await agent.session_store.run(agent.session_store.get_events, "alice")
        terminals = [e for e in events if e["type"] == "subagent_result"]
        assert len(terminals) == 1
        assert terminals[0]["tool_call_id"] == "provider-child"
        assert terminals[0]["origin"]["turn_index"] == 1
        assert expected_error in terminals[0]["error"]
        results = [
            m
            for m in agent.controller.conversation.get_messages()
            if m.tool_call_id == "provider-child"
        ]
        assert len(results) == 1
        if not background:
            assert expected_error in str(results[0].content)


class ChildProvider(ScriptedLLM):
    def __init__(self, fails=False):
        super().__init__(["child done"])
        self.fails = fails
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def chat(self, messages, **kwargs):
        self.entered.set()
        await self.release.wait()
        if self.fails:
            raise RuntimeError("child failed")
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", ["direct", "background", "promoted_immediate"])
@pytest.mark.parametrize("outcome", ["success", "error", "stop"])
async def test_child_origin_survives_parent_turn_and_retention(
    tmp_path, native, mode, outcome
):
    child = ChildProvider(fails=outcome == "error")
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=ParentProvider(native, mode == "background"),
            io="headless",
            start=True,
        )
        agent = creature.agent
        cfg = SubAgentConfig(
            name="child",
            description="Child",
            tools=[],
            system_prompt="child",
            max_turns=1,
        )
        agent.subagent_manager.register(cfg)
        agent.registry.register_subagent("child", cfg)
        agent.subagent_manager.llm = child
        agent.subagent_manager.job_store._max_completed = 0
        recorded = asyncio.Event()
        loop = asyncio.get_running_loop()
        store = agent.session_store

        def observer(key, event):
            if key.startswith("alice:") and event.get("type") == "subagent_result":
                loop.call_soon_threadsafe(recorded.set)

        store.subscribe(observer)
        task = asyncio.create_task(creature.run("first", timeout=8))
        try:
            await asyncio.wait_for(child.entered.wait(), 3)
            status = agent.subagent_manager.get_running_jobs()[0]
            origin = status.context["origin"]
            assert origin["turn_index"] == 1
            assert origin["execution_class"] == (
                "background" if mode == "background" else "direct"
            )
            if mode == "background":
                await task
                assert (await creature.run("second", timeout=5)).ok
                assert agent._turn_index == 2
            elif mode == "promoted_immediate":
                assert agent._active_handles[status.job_id].promote()
            if outcome == "stop":
                await agent.stop()
            else:
                child.release.set()
            await task
            await asyncio.wait_for(recorded.wait(), 4)
            events = await store.run(store.get_events, "alice")
            start = next(e for e in events if e["type"] == "subagent_call")
            terminals = [e for e in events if e["type"] == "subagent_result"]
            assert len(terminals) == 1
            terminal = terminals[0]
            assert start["origin"] == origin
            assert terminal["origin"] == {
                **origin,
                "execution_class": "direct" if mode == "direct" else "background",
            }
            assert terminal["job_id"] == start["job_id"] == status.job_id
            assert terminal["turn_index"] == 1
            if outcome == "success":
                assert "child done" in terminal["output"]
            elif outcome == "error":
                assert "child failed" in terminal["error"]
            else:
                assert terminal["interrupted"] is True
                assert terminal["final_state"] == "interrupted"
            if native:
                assert (
                    terminal["tool_call_id"]
                    == start["tool_call_id"]
                    == "provider-child"
                )
        finally:
            child.release.set()
            await asyncio.gather(task, return_exceptions=True)
            store.unsubscribe(observer)
