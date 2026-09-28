"""Native acknowledgement receipts gate continuation separately from job terminals."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core import agent_tools, native_ack
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("failure", [None, "before", "after", "replaced"])
async def test_ack_receipt_precedes_owner_append_and_provider_continuation(
    tmp_path, monkeypatch, failure, cancelled
):
    entered, release = asyncio.Event(), threading.Event()
    finish_tool = asyncio.Event()
    loop = asyncio.get_running_loop()

    class Work(BaseTool):
        tool_name = "work"
        description = "Pending work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            await finish_tool.wait()
            return ToolResult(output="late terminal")

    class Parent(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused", "acknowledged"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if self.call_count == 0:
                self.call_count += 1
                self.last_tool_calls = [
                    NativeToolCall("provider", "work", '{"run_in_background":true}')
                ]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Parent(),
            tools=[Work()],
            io="headless",
            start=True,
        )
        agent, store = creature.agent, creature.agent.session_store
        persist = native_ack.persist_ack
        append_ack, callers = agent_tools.append_ack, []

        async def observed(*args):
            callers.append(asyncio.current_task())
            return await append_ack(*args)

        monkeypatch.setattr(agent_tools, "append_ack", observed)

        def held(*args):
            assert threading.get_ident() == store._affinity_worker_ident
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(6)
            if failure == "before":
                raise OSError("ack not written")
            value = persist(*args)
            if failure == "after":
                raise OSError("ack written but not confirmed")
            return value

        monkeypatch.setattr(native_ack, "persist_ack", held)
        pending = asyncio.create_task(creature.run("question", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert agent.llm.call_count == 1
            assert not any(
                message.metadata.get("tool_delivery", {}).get("phase") == "ack"
                for message in agent.controller.conversation.get_messages()
            )
            if failure == "replaced":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(
                        agent.controller,
                        "conversation",
                        type(agent.controller.conversation)(),
                    )
                )
            if cancelled:
                callers[0].cancel()
                await agent._session_state_owner.submit_internal(lambda: None)
                assert not callers[0].done()
            release.set()
            result = await asyncio.wait_for(pending, 5)
            assert agent.llm.call_count == 1
            if failure is None:
                assert result.status == ("interrupted" if cancelled else "ok")
                require_model_mutation_available(agent)
                await creature.run("continue independently", timeout=5)
                assert agent.llm.call_count == 2
                assert any(
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == "provider"
                    for message in agent.llm.call_log[-1]
                )
            else:
                with pytest.raises(ConflictError, match="recovery"):
                    require_model_mutation_available(agent)
            messages = agent.controller.conversation.get_messages()
            assert sum(
                message.metadata.get("tool_delivery", {}).get("phase") == "ack"
                for message in messages
            ) == (1 if failure is None else 0)
            rows = await store.run(store.get_events, "alice")
            assert sum(row["type"] == "native_tool_ack" for row in rows) == (
                0 if failure == "before" else 1
            )
            assert not any(row["type"] == "tool_result" for row in rows)
        finally:
            agent.pause()
            release.set()
            finish_tool.set()
            await asyncio.gather(pending, return_exceptions=True)
