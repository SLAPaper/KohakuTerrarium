"""Native declaration persistence precedes dispatch and cannot cross a binding."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core import native_declaration
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.controller_history import append_assistant
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.session.resume_replay import replay_resume_history
from kohakuterrarium.terrarium import session_coord, topology_admission


class Provider(ScriptedLLM):
    def __init__(self):
        super().__init__(["unused", "done"])
        self.last_tool_calls = []

    async def chat(self, messages, **kwargs):
        self.last_tool_calls = []
        if self.call_count == 0:
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            self.last_tool_calls = [NativeToolCall("provider-call", "work", '{"x":1}')]
            yield ""
        else:
            async for chunk in super().chat(messages, **kwargs):
                yield chunk


async def test_accepted_native_install_finishes_before_topology_copy(
    tmp_path, monkeypatch
):
    installing, release, reserved = asyncio.Event(), asyncio.Event(), asyncio.Event()
    submit = native_declaration.submit_internal_session_state
    guard = topology_admission.require_model_mutation_available
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        agent = alice.agent
        sink = native_declaration._canonical_output(agent.controller)
        generation = {"schema": 1, "generation_id": "accepted-generation", "source": {}}
        sink._generation_coordinates = (1, 1, [])
        agent.controller._active_generation = generation

        async def hold_install(host, callback):
            if host is agent and callback.__name__ == "install":
                installing.set()
                await release.wait()
            return await submit(host, callback)

        def reserve(host, **kwargs):
            guard(host, **kwargs)
            reserved.set()

        captures = []
        capture = session_coord.capture_runtime_transfer

        def observed(*args):
            captures.append(args[0])
            return capture(*args)

        monkeypatch.setattr(
            native_declaration, "submit_internal_session_state", hold_install
        )
        monkeypatch.setattr(
            topology_admission, "require_model_mutation_available", reserve
        )
        monkeypatch.setattr(session_coord, "capture_runtime_transfer", observed)
        writing = asyncio.create_task(
            append_assistant(
                agent.controller,
                "",
                metadata={"generation": generation},
                tool_calls=[
                    {
                        "id": "accepted-call",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )
        )
        moving = None
        try:
            await asyncio.wait_for(installing.wait(), 3)
            moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
            await asyncio.wait_for(reserved.wait(), 3)
            assert agent._topology_handoff_task is not None
            assert not captures and not moving.done()
            release.set()
            await asyncio.wait_for(writing, 3)
            await asyncio.wait_for(moving, 8)
            assert captures
            assert (
                sum(
                    bool(message.metadata.get("native_declaration_origin"))
                    for message in agent.controller.conversation.get_messages()
                )
                == 1
            )
        finally:
            release.set()
            await asyncio.gather(
                writing, *([moving] if moving else []), return_exceptions=True
            )


async def test_declaration_arriving_during_copy_uses_rebound_store(
    tmp_path, monkeypatch
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice"), llm=ScriptedLLM(["a"]), io="headless", start=True
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        agent = alice.agent
        sink = native_declaration._canonical_output(agent.controller)
        generation = {"schema": 1, "generation_id": "handoff-generation", "source": {}}
        sink._generation_coordinates = (1, 1, [])
        agent.controller._active_generation = generation
        capture = session_coord.capture_runtime_transfer
        first = True
        source = agent.session_store

        def held(*args):
            nonlocal first
            result = capture(*args)
            if first and args[0] is source:
                first = False
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(6)
            return result

        monkeypatch.setattr(session_coord, "capture_runtime_transfer", held)
        moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
        writing = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            writing = asyncio.create_task(
                append_assistant(
                    agent.controller,
                    "",
                    metadata={"generation": generation},
                    tool_calls=[
                        {
                            "id": "handoff-call",
                            "type": "function",
                            "function": {"name": "work", "arguments": "{}"},
                        }
                    ],
                )
            )
            await agent._session_state_owner.submit_internal(lambda: None)
            assert not writing.done()
            release.set()
            await asyncio.wait_for(moving, 8)
            await asyncio.wait_for(writing, 3)
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            declarations = [
                row for row in rows if row["type"] == "assistant_tool_calls"
            ]
            assert len(declarations) == 1
            messages = agent.controller.conversation.get_messages()
            assert (
                sum(
                    bool(message.metadata.get("native_declaration_origin"))
                    for message in messages
                )
                == 1
            )
        finally:
            release.set()
            await asyncio.gather(
                moving, *([writing] if writing else []), return_exceptions=True
            )


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "failure",
        "replacement",
        "write_then_raise",
        "branch",
        "view",
        "namespace",
        "scope_done",
    ],
)
async def test_native_dispatch_waits_for_declaration_on_source_worker(
    tmp_path, monkeypatch, mode
):
    entered, release, dispatched = asyncio.Event(), threading.Event(), []
    loop = asyncio.get_running_loop()
    persist = native_declaration._persist

    def held(*args):
        assert threading.current_thread() is not threading.main_thread()
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(8)
        if mode == "failure":
            raise OSError("declaration write failed")
        result = persist(*args)
        if mode == "write_then_raise":
            raise OSError("declaration confirmed write then failure")
        return result

    class Work(BaseTool):
        tool_name = "work"
        description = "Work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            dispatched.append(args)
            return ToolResult(output="result")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Provider(),
            tools=[Work()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        monkeypatch.setattr(native_declaration, "_persist", held)
        turn = asyncio.create_task(creature.run("question", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not dispatched
            if mode == "replacement":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent.controller, "conversation", Conversation())
                )
            elif mode == "branch":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent, "_branch_id", agent._branch_id + 1)
                )
            elif mode == "view":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent, "_branch_view", {1: 99})
                )
            elif mode == "namespace":
                sink = native_declaration._canonical_output(agent.controller)
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(sink, "_event_key_prefix", "other")
                )
            elif mode == "scope_done":
                await agent._session_state_owner.submit_internal(
                    lambda: agent._turn_completion.set_result(None)
                )
            release.set()
            result = await asyncio.wait_for(turn, 8)
            assert bool(dispatched) == (mode == "success")
            assert result.ok == (mode == "success")
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            declarations = [
                row for row in rows if row["type"] == "assistant_tool_calls"
            ]
            assert len(declarations) == (0 if mode == "failure" else 1)
            if mode in ("replacement", "write_then_raise"):
                restored = replay_resume_history(rows)
                pairs = [
                    message
                    for message in restored
                    if message.get("tool_call_id") == "provider-call"
                ]
                assert len(pairs) == 1 and "Interrupted" in pairs[0]["content"]
                assert not any(row["type"] == "tool_result" for row in rows)
            if mode == "success":
                assert (
                    declarations[0]["tool_calls"][0]["function"]["arguments"]
                    == '{"x":1}'
                )
                assert rows.index(declarations[0]) < next(
                    i for i, row in enumerate(rows) if row["type"] == "tool_call"
                )
        finally:
            release.set()
            await asyncio.gather(turn, return_exceptions=True)


async def test_cancelled_declaration_joins_write_then_rethrows(tmp_path, monkeypatch):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Provider(),
            io="headless",
            start=True,
        )
        agent = creature.agent
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        persist = native_declaration._persist

        def held(*args):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(8)
            return persist(*args)

        from_generation = {
            "schema": 1,
            "generation_id": "test-generation",
            "source": {},
        }
        sink = native_declaration._canonical_output(agent.controller)
        sink._generation_coordinates = (1, 1, [])
        agent.controller._active_generation = from_generation
        monkeypatch.setattr(native_declaration, "_persist", held)
        pending = asyncio.create_task(
            append_assistant(
                agent.controller,
                "",
                metadata={"generation": from_generation},
                tool_calls=[
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
            assert not pending.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert len([row for row in rows if row["type"] == "assistant_tool_calls"]) == 1
        assert not any(row["type"] == "tool_call" for row in rows)


@pytest.mark.parametrize("coordinates", [(None, None, None), (True, 1, []), (1, 0, [])])
async def test_unstamped_native_generation_is_rejected_before_write(
    tmp_path, coordinates
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=Provider(),
            io="headless",
            start=True,
        )
        controller = creature.agent.controller
        generation = {"schema": 1, "generation_id": "missing-turn", "source": {}}
        native_declaration._canonical_output(controller)._generation_coordinates = (
            coordinates
        )
        controller._active_generation = generation
        with pytest.raises(ConflictError, match="coordinates"):
            await append_assistant(
                controller,
                "",
                metadata={"generation": generation},
                tool_calls=[
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "work", "arguments": "{}"},
                    }
                ],
            )
        rows = await creature.agent.session_store.run(
            creature.agent.session_store.get_events, "alice"
        )
        assert not any(row["type"] == "assistant_tool_calls" for row in rows)
