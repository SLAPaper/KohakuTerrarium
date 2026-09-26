"""Hosted controller messages commit on the owner; external calls do not."""

import asyncio

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core import controller_history
from kohakuterrarium.core.controller_history import (
    append_assistant,
    capture_turn_messages,
)
from kohakuterrarium.core.controller_plugins import run_post_llm_call_chain
from kohakuterrarium.core.events import create_user_input_event
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.llm.message import AssistantMessage, TextPart
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("native", [False, True])
async def test_restarted_agent_runs_controller_and_direct_feedback(tmp_path, native):
    class Work(BaseTool):
        tool_name = "work"
        description = "Direct work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="restarted result")

    class Provider(ScriptedLLM):
        def __init__(self):
            super().__init__(["first", "[/work][work/]", "restarted answer"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if native and self.call_count == 1:
                self.call_count += 1
                self.call_log.append(self._normalize_messages(messages))
                self.last_tool_calls = [NativeToolCall("restarted-call", "work", "{}")]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = Provider()
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=llm,
            tools=[Work()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        assert (await creature.run("first question")).text == "first"
        first_owner = agent._session_state_owner.task
        await agent.stop()
        await agent.start()
        assert agent._session_state_owner.task is not first_owner
        assert (await creature.run("second question")).text.endswith("restarted answer")
        assert llm.call_count == 3
        messages = agent.controller.conversation.get_messages()
        assert sum("restarted result" in m.get_text_content() for m in messages) == 1
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [e["output"] for e in raw if e["type"] == "tool_result"] == [
            "restarted result"
        ]


async def test_post_hook_does_not_rewrite_replaced_conversation(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Rewriter(BasePlugin):
        name = "held-rewriter"

        async def post_llm_call(self, messages, response, usage, **kwargs):
            entered.set()
            await release.wait()
            return "stale rewrite"

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["unused"]),
            plugins=[Rewriter()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await append_assistant(agent.controller, "original")
        before = agent.controller.conversation.get_last_assistant_message()
        rewriting = asyncio.create_task(run_post_llm_call_chain(agent.controller, []))
        await entered.wait()
        try:
            await append_assistant(agent.controller, "newer")
        finally:
            release.set()
            await rewriting
        assert before.content == "original"
        assert (
            agent.controller.conversation.get_last_assistant_message().content
            == "newer"
        )


async def test_owner_queued_input_is_detached_and_retains_edit_identity(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["unused"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        owner, held, intercepted = agent._session_state_owner, [], asyncio.Event()
        execute = owner._execute

        def hold(callback, future):
            if not intercepted.is_set():
                held.append((callback, future))
                intercepted.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        event = create_user_input_event("edited")
        event.context.update(
            rerun=True,
            edited=True,
            _history_user_identity={"event_id": 19, "turn_index": 2, "branch_id": 3},
        )
        content = [TextPart(text="before")]
        capture = asyncio.create_task(
            capture_turn_messages(agent.controller, [event], content, False)
        )
        await intercepted.wait()
        content[0].text = "mutated"
        event.context["_history_user_identity"]["event_id"] = 999
        await owner.submit_internal(lambda: execute(*held[0]))
        messages = await capture
        messages[-1]["content"][0]["text"] = "provider mutation"
        last = agent.controller.conversation.get_messages()[-1]
        assert last.get_text_content() == "before"
        assert last.metadata["event_id"] == 19


@pytest.mark.parametrize("shutdown", [False, True])
async def test_interrupt_after_input_commit_does_not_start_new_provider_call(
    tmp_path, monkeypatch, shutdown
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["must not call"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        original = controller_history.commit_controller_state
        intercepted, release = asyncio.Event(), asyncio.Event()

        async def held(controller, callback):
            value = await original(controller, callback)
            if not intercepted.is_set():
                intercepted.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    if shutdown:
                        release.set()
                    await release.wait()
            return value

        monkeypatch.setattr(controller_history, "commit_controller_state", held)
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        await asyncio.wait_for(intercepted.wait(), 3)
        if shutdown:
            await asyncio.wait_for(agent.stop(), 3)
        else:
            agent.interrupt()
            release.set()
        await asyncio.wait_for(turn, 3)
        assert llm.call_count == 0
        assert [
            m.content
            for m in agent.controller.conversation.get_messages()
            if m.role == "user"
        ] == ["question"]


async def test_stop_after_assistant_commit_does_not_wait_for_direct_tool(
    tmp_path, monkeypatch
):
    entered, captured = asyncio.Event(), asyncio.Event()

    class Held(BaseTool):
        tool_name = "held"
        description = "Wait until cancelled"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(["[/held][held/]"]),
            tools=[Held()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        original = controller_history.commit_controller_state

        async def hold_assistant(controller, callback):
            value = await original(controller, callback)
            if entered.is_set() and not captured.is_set():
                captured.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            return value

        monkeypatch.setattr(
            controller_history, "commit_controller_state", hold_assistant
        )
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        await asyncio.wait_for(captured.wait(), 3)
        await asyncio.wait_for(agent.stop(), 3)
        await turn
        assert agent._stopped


@pytest.mark.parametrize("native", [False, True])
async def test_hosted_controller_appends_and_snapshots_on_owner(
    tmp_path, monkeypatch, native
):
    outside = []

    class Provider(ScriptedLLM):
        def __init__(self):
            super().__init__(["[/work][work/]", "answer"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            outside.append(asyncio.current_task())
            self.last_tool_calls = []
            if native and self.call_count == 0:
                self.call_count += 1
                self.call_log.append(self._normalize_messages(messages))
                self.last_tool_calls = [NativeToolCall("provider", "work", "{}")]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    class Work(BaseTool):
        tool_name = "work"
        description = "Direct work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            outside.append(asyncio.current_task())
            return ToolResult(output="result")

    class Plugin(BasePlugin):
        name = "observe"

        async def pre_llm_call(self, messages, **kwargs):
            outside.append(asyncio.current_task())
            return None

        async def post_llm_call(self, messages, response, usage, **kwargs):
            outside.append(asyncio.current_task())
            return response + " rewritten" if response == "answer" else None

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=Provider(),
            tools=[Work()],
            plugins=[Plugin()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        owner = agent._session_state_owner.task
        conversation = agent.controller.conversation
        append, serialize = type(conversation).append, type(conversation).to_messages
        writes, reads, rewrites = [], [], []
        set_attribute = AssistantMessage.__setattr__

        def observe_rewrite(message, key, value):
            if key == "content" and value == "answer rewritten":
                rewrites.append(asyncio.current_task())
            return set_attribute(message, key, value)

        monkeypatch.setattr(AssistantMessage, "__setattr__", observe_rewrite)

        def observe_append(self, role, content, **kwargs):
            if self is conversation:
                writes.append((role, asyncio.current_task()))
            return append(self, role, content, **kwargs)

        def observe_read(self, **kwargs):
            if self is conversation:
                reads.append(asyncio.current_task())
            return serialize(self, **kwargs)

        monkeypatch.setattr(type(conversation), "append", observe_append)
        monkeypatch.setattr(type(conversation), "to_messages", observe_read)
        await creature.run("question")
        assert writes and all(task is owner for _, task in writes)
        assert reads and all(task is owner for task in reads)
        assert outside and all(task is not owner for task in outside)
        assert rewrites and all(task is owner for task in rewrites)
        assert conversation.get_last_assistant_message().content == "answer rewritten"
        assert [role for role, _ in writes].count("assistant") == 2
