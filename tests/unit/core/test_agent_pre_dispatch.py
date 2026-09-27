"""Unit tests for :mod:`kohakuterrarium.core.agent_pre_dispatch`."""

import asyncio
import threading
import types
from concurrent.futures import Future
from pathlib import Path


import pytest

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.core.controller_history import append_assistant
from kohakuterrarium.llm.base import NativeToolCall
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.tool.base import BaseTool, ExecutionMode, ToolResult
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.session.history import replay_conversation
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.core.agent_pre_dispatch import (
    run_pre_subagent_dispatch,
    run_pre_tool_dispatch,
)
from kohakuterrarium.modules.plugin.base import BasePlugin, PluginBlockError
from kohakuterrarium.parsing import SubAgentCallEvent, ToolCallEvent

# ── stubs ────────────────────────────────────────────────────────


class _StubController:
    def __init__(self, tool_format="bracket"):
        self.events: list = []
        self.appended_messages: list = []
        self.config = types.SimpleNamespace(tool_format=tool_format)
        self.conversation = types.SimpleNamespace(
            append=lambda *args, **kwargs: self.appended_messages.append(
                {"args": args, "kwargs": kwargs}
            )
        )

    def push_event_sync(self, event):
        self.events.append(event)


class _StubPluginManager:
    def __init__(self, plugins=None, prehook_result=None, prehook_raise=None):
        self._plugins = plugins or [object()]  # truthy default
        self._app = plugins or []
        self._prehook_result = prehook_result
        self._prehook_raise = prehook_raise
        self.prehook_calls: list = []

    def _applicable_plugins(self):
        return list(self._app)

    async def run_pre_hooks(self, hook_name, value, **kwargs):
        self.prehook_calls.append((hook_name, value, kwargs))
        if self._prehook_raise is not None:
            raise self._prehook_raise
        return self._prehook_result if self._prehook_result is not None else value


class _PrePlugin(BasePlugin):
    """Plugin that rewrites the tool args."""

    def __init__(self, name, *, replacement=None, raises=None):
        self.name = name
        self._rep = replacement
        self._raises = raises

    async def pre_tool_dispatch(self, event, ctx):
        if self._raises is not None:
            raise self._raises
        return self._rep


class _NoOpPlugin(BasePlugin):
    def __init__(self):
        self.name = "noop"


def _agent(*, plugins=None, registry_tools=None):
    return types.SimpleNamespace(
        plugins=plugins,
        config=types.SimpleNamespace(name="alice"),
        executor=types.SimpleNamespace(_working_dir=Path(".")),
        llm=types.SimpleNamespace(model="m"),
        registry=types.SimpleNamespace(list_tools=lambda: list(registry_tools or [])),
    )


# ── run_pre_tool_dispatch ────────────────────────────────────────


class TestRunPreToolDispatch:
    async def test_no_plugins_passthrough(self):
        evt = ToolCallEvent(name="bash", args={}, raw="")
        c = _StubController()
        result = await run_pre_tool_dispatch(_agent(plugins=None), evt, c)
        assert result is evt

    async def test_empty_plugin_manager(self):
        evt = ToolCallEvent(name="bash", args={}, raw="")
        c = _StubController()
        mgr = _StubPluginManager(plugins=[])
        mgr._plugins = []  # falsy → returns evt verbatim
        result = await run_pre_tool_dispatch(_agent(plugins=mgr), evt, c)
        assert result is evt

    async def test_no_overriding_plugins(self):
        evt = ToolCallEvent(name="bash", args={}, raw="")
        c = _StubController()
        mgr = _StubPluginManager(plugins=[_NoOpPlugin()])
        result = await run_pre_tool_dispatch(_agent(plugins=mgr), evt, c)
        assert result is evt

    async def test_plugin_rewrites_event(self):
        original = ToolCallEvent(name="bash", args={"command": "ls"}, raw="")
        rewritten = ToolCallEvent(name="bash", args={"command": "ls -la"}, raw="")
        plug = _PrePlugin("rewriter", replacement=rewritten)
        mgr = _StubPluginManager(plugins=[plug])
        c = _StubController()
        a = _agent(plugins=mgr, registry_tools=["bash"])
        result = await run_pre_tool_dispatch(a, original, c)
        assert result is rewritten

    async def test_block_synthesises_tool_result(self):
        evt = ToolCallEvent(name="bash", args={"command": "rm -rf /"}, raw="")
        plug = _PrePlugin("guard", raises=PluginBlockError("dangerous"))
        mgr = _StubPluginManager(plugins=[plug])
        c = _StubController()
        a = _agent(plugins=mgr, registry_tools=["bash"])
        result = await run_pre_tool_dispatch(a, evt, c)
        assert result is None
        # Synthetic tool_complete event injected.
        assert len(c.events) == 1
        injected = c.events[0]
        assert "guard" in injected.content
        assert "dangerous" in injected.content

    async def test_block_native_mode_appends_tool_message(self):
        evt = ToolCallEvent(
            name="bash",
            args={"command": "rm", "_tool_call_id": "call_42"},
            raw="",
        )
        plug = _PrePlugin("guard", raises=PluginBlockError("nope"))
        mgr = _StubPluginManager(plugins=[plug])
        c = _StubController(tool_format="native")
        a = _agent(plugins=mgr, registry_tools=["bash"])
        result = await run_pre_tool_dispatch(a, evt, c)
        assert result is None
        # Tool message appended to conversation.
        assert c.appended_messages
        msg = c.appended_messages[0]
        assert msg["args"] == ("tool", "[guard] nope")
        assert msg["kwargs"]["tool_call_id"] == "call_42"

    async def test_plugin_exception_skipped(self):
        evt = ToolCallEvent(name="bash", args={}, raw="")
        plug = _PrePlugin("flaky", raises=RuntimeError("boom"))
        ok = _PrePlugin(
            "ok", replacement=ToolCallEvent(name="bash", args={"x": 1}, raw="")
        )
        mgr = _StubPluginManager(plugins=[plug, ok])
        c = _StubController()
        a = _agent(plugins=mgr, registry_tools=["bash"])
        result = await run_pre_tool_dispatch(a, evt, c)
        # Flaky's exception swallowed; ok still ran.
        assert result.args == {"x": 1}

    async def test_non_event_return_ignored(self):
        evt = ToolCallEvent(name="bash", args={}, raw="")
        plug = _PrePlugin("weird", replacement="not an event")  # bogus type
        mgr = _StubPluginManager(plugins=[plug])
        c = _StubController()
        a = _agent(plugins=mgr, registry_tools=["bash"])
        result = await run_pre_tool_dispatch(a, evt, c)
        # Bogus return ignored — original event survives.
        assert result is evt

    async def test_rewrite_to_unknown_tool_blocked(self):
        evt = ToolCallEvent(name="bash", args={}, raw="")
        new_evt = ToolCallEvent(name="ghost_tool", args={}, raw="")
        plug = _PrePlugin("renamer", replacement=new_evt)
        mgr = _StubPluginManager(plugins=[plug])
        c = _StubController()
        a = _agent(plugins=mgr, registry_tools=["bash"])  # ghost_tool not present
        result = await run_pre_tool_dispatch(a, evt, c)
        assert result is None
        # Veto produced a synthesised result.
        assert c.events

    async def test_none_return_does_not_overwrite_current(self):
        # When plugin returns None, the event is left as-is, so the
        # original passes through.
        evt = ToolCallEvent(name="bash", args={}, raw="")
        plug = _PrePlugin("nop", replacement=None)
        mgr = _StubPluginManager(plugins=[plug])
        c = _StubController()
        a = _agent(plugins=mgr, registry_tools=["bash"])
        result = await run_pre_tool_dispatch(a, evt, c)
        assert result is evt


# ── run_pre_subagent_dispatch ────────────────────────────────────


class TestRunPreSubagentDispatch:
    async def test_no_plugins_passthrough(self):
        evt = SubAgentCallEvent(name="explore", args={"task": "do it"}, raw="")
        c = _StubController()
        result = await run_pre_subagent_dispatch(_agent(plugins=None), evt, c)
        assert result is evt

    async def test_empty_plugins_passthrough(self):
        evt = SubAgentCallEvent(name="explore", args={"task": "do it"}, raw="")
        c = _StubController()
        mgr = _StubPluginManager(plugins=[])
        mgr._plugins = []
        result = await run_pre_subagent_dispatch(_agent(plugins=mgr), evt, c)
        assert result is evt

    async def test_task_unchanged_returns_original(self):
        evt = SubAgentCallEvent(name="explore", args={"task": "do it"}, raw="")
        mgr = _StubPluginManager(prehook_result="do it")  # unchanged
        c = _StubController()
        result = await run_pre_subagent_dispatch(_agent(plugins=mgr), evt, c)
        assert result is evt

    async def test_task_rewritten(self):
        evt = SubAgentCallEvent(name="explore", args={"task": "old"}, raw="")
        mgr = _StubPluginManager(prehook_result="new task")
        c = _StubController()
        result = await run_pre_subagent_dispatch(_agent(plugins=mgr), evt, c)
        # New event with updated task.
        assert result is not evt
        assert result.args["task"] == "new task"

    async def test_content_field_rewritten(self):
        evt = SubAgentCallEvent(name="explore", args={"content": "old"}, raw="")
        mgr = _StubPluginManager(prehook_result="new content")
        c = _StubController()
        result = await run_pre_subagent_dispatch(_agent(plugins=mgr), evt, c)
        # ``task`` not present originally and ``content`` is — rewrite goes to content.
        assert result.args["content"] == "new content"
        assert "task" not in result.args

    async def test_block_synthesises_subagent_result(self):
        evt = SubAgentCallEvent(name="explore", args={"task": "x"}, raw="")
        mgr = _StubPluginManager(prehook_raise=PluginBlockError("policy violation"))
        c = _StubController()
        result = await run_pre_subagent_dispatch(_agent(plugins=mgr), evt, c)
        assert result is None
        # Synthetic event injected.
        assert c.events
        injected = c.events[0]
        assert "explore" in injected.content
        assert "policy violation" in injected.content


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_veto_pairs_original_call_on_owner_before_continuation(
    tmp_path, monkeypatch, child, native
):
    hooks = []

    class Veto(BasePlugin):
        name = "veto"

        async def pre_tool_dispatch(self, event, ctx):
            hooks.append(asyncio.current_task())
            raise PluginBlockError("policy denied")

        async def pre_subagent_run(self, task, **kwargs):
            hooks.append(asyncio.current_task())
            raise PluginBlockError("policy denied")

    class Work(BaseTool):
        tool_name = "work"
        description = "Must not execute"

        async def _execute(self, args, **kwargs):
            raise AssertionError("vetoed tool executed")

    class Provider(ScriptedLLM):
        def __init__(self):
            super().__init__(
                [
                    "unused" if native else '[/work]{"task":"work"}[work/]',
                    "denial acknowledged",
                ]
            )
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if native and self.call_count == 0:
                self.call_count += 1
                self.call_log.append(self._normalize_messages(messages))
                self.last_tool_calls = [
                    NativeToolCall("original-call", "work", '{"task":"work"}')
                ]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = Provider()
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native" if native else "bracket"),
            llm=llm,
            tools=[] if child else [Work()],
            plugins=[Veto()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        if child:
            config = SubAgentConfig(name="work", description="Must not spawn", tools=[])
            agent.subagent_manager.register(config)
            agent.registry.register_subagent("work", config)
        append = type(agent.controller.conversation).append
        writers = []

        def observe(conversation, role, content, **kwargs):
            if conversation is agent.controller.conversation and role == "tool":
                writers.append(asyncio.current_task())
            return append(conversation, role, content, **kwargs)

        monkeypatch.setattr(type(agent.controller.conversation), "append", observe)
        result = await creature.run("question")
        assert result.text == ""
        assert llm.call_count == 1 and agent.controller.has_pending_events()
        assert (
            await creature.run("acknowledge the denial")
        ).text == "denial acknowledged"
        assert llm.call_count == 2
        pairs = [m for m in llm.call_log[1] if m["role"] == "tool"]
        if native:
            assert len(pairs) == 1 and pairs[0]["tool_call_id"] == "original-call"
            assert "policy denied" in pairs[0]["content"]
            assert writers and all(
                task is agent._session_state_owner.task for task in writers
            )
        else:
            assert not pairs
            assert "policy denied" in str(llm.call_log[1])
        assert hooks and all(
            task is not agent._session_state_owner.task for task in hooks
        )
        assert not agent.executor.job_store.get_all_statuses()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        restored = replay_conversation(raw)
        restored_results = [
            message for message in restored if message["role"] == "tool"
        ]
        if native:
            assert (
                len(restored_results) == 1
                and restored_results[0]["tool_call_id"] == "original-call"
            )
            assert "policy denied" in restored_results[0]["content"]
        path = agent.session_store.path
    if native:
        for use_snapshot in (True, False):
            store = SessionStore(path)
            resumed_llm = ScriptedLLM(["resumed"])
            resumed = await Agent.build(
                AgentConfig(name="alice", tool_format="native"),
                llm=resumed_llm,
                io="headless",
            )
            try:
                if not use_snapshot:
                    await store.run(lambda: store.conversation.pop("alice", None))
                try:
                    await inject_saved_state_async(resumed, store, "alice")
                except Exception as exc:
                    raise AssertionError(
                        f"Veto cold restore failed: use_snapshot={use_snapshot}"
                    ) from exc
                resumed.attach_session_store(store)
                await resumed.start()
                assert resumed_llm.call_count == 0
                assert (await resumed.run("continue")).text == "resumed"
                pairs = [
                    message
                    for message in resumed_llm.call_log[0]
                    if message.get("tool_call_id") == "original-call"
                ]
                assert len(pairs) == 1 and "policy denied" in pairs[0]["content"]
            finally:
                await resumed.stop()
                store.close()


@pytest.mark.parametrize("replace_context", [False, True])
async def test_veto_uses_original_identity_and_rejects_replaced_target(
    tmp_path, replace_context
):
    class Veto(BasePlugin):
        name = "veto"

        async def pre_tool_dispatch(self, event, ctx):
            event.name = "rewritten"
            event.args["_tool_call_id"] = "mutated-id"
            if replace_context:

                def replace():
                    agent.controller.conversation = Conversation.from_json(
                        agent.controller.conversation.to_json()
                    )

                await agent._submit_session_state(replace)
            raise PluginBlockError("identity denial")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            plugins=[Veto()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        await append_assistant(
            agent.controller,
            "",
            tool_calls=[
                {
                    "id": "original-call",
                    "type": "function",
                    "function": {"name": "work", "arguments": "{}"},
                }
            ],
        )
        original = agent.controller.conversation
        before = original.to_json()
        event = ToolCallEvent(
            name="work", args={"_tool_call_id": "original-call"}, raw=""
        )
        assert await run_pre_tool_dispatch(agent, event, agent.controller) is None
        await agent.output_router.flush()
        raw = await agent.session_store.run(agent.session_store.get_events, "alice")
        results = [e for e in raw if e["type"] == "tool_result"]
        if replace_context:
            assert not results
            assert agent.controller.conversation.to_json() == before
            assert original.to_json() == before
            assert not agent.controller.has_pending_events()
        else:
            assert len(results) == 1
            assert results[0]["tool_call_id"] == "original-call"
            assert results[0]["name"] == "work"
            assert agent.controller.has_pending_events()
            duplicate = ToolCallEvent(
                name="work", args={"_tool_call_id": "original-call"}, raw=""
            )
            assert (
                await run_pre_tool_dispatch(agent, duplicate, agent.controller) is None
            )
            await agent.output_router.flush()
            raw = await agent.session_store.run(agent.session_store.get_events, "alice")
            assert len([e for e in raw if e["type"] == "tool_result"]) == 1
            assert agent.controller._event_queue.qsize() == 1


async def test_cancelled_capture_does_not_continue_to_plugin_dispatch(
    tmp_path, monkeypatch
):
    calls = []

    class Observe(BasePlugin):
        name = "observe"

        async def pre_tool_dispatch(self, event, ctx):
            calls.append(event.name)
            return None

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["unused"]),
            plugins=[Observe()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        owner = agent._session_state_owner
        execute = owner._execute
        intercepted, held = asyncio.Event(), []

        def hold(callback, future):
            if not intercepted.is_set():
                held.append((callback, future))
                intercepted.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        event = ToolCallEvent(
            name="work", args={"_tool_call_id": "original-call"}, raw=""
        )
        dispatch = asyncio.create_task(
            run_pre_tool_dispatch(agent, event, agent.controller)
        )
        await asyncio.wait_for(intercepted.wait(), 3)
        dispatch.cancel()
        await owner.submit_internal(lambda: execute(*held[0]))
        with pytest.raises(asyncio.CancelledError):
            await dispatch
        assert not calls


@pytest.mark.parametrize("after_append", [False, True, "cancelled_receipt"])
async def test_veto_write_failure_preserves_live_and_retries_once(
    tmp_path, monkeypatch, after_append
):
    class Veto(BasePlugin):
        name = "veto"

        async def pre_tool_dispatch(self, event, ctx):
            raise PluginBlockError("first denial")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            plugins=[Veto()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        await append_assistant(
            agent.controller,
            "",
            tool_calls=[
                {
                    "id": "original-call",
                    "type": "function",
                    "function": {"name": "work", "arguments": "{}"},
                }
            ],
        )
        before = agent.controller.conversation.to_json()
        store = agent.session_store
        append = type(store).append_event
        failed = False

        def fail_once(self, namespace, event_type, data, **kwargs):
            nonlocal failed
            if self is store and event_type == "tool_result" and not failed:
                failed = True
                if after_append:
                    append(self, namespace, event_type, data, **kwargs)
                raise OSError("veto write failed")
            return append(self, namespace, event_type, data, **kwargs)

        if after_append == "cancelled_receipt":
            submit = agent._session_output._submit_store_write

            def lost_receipt(fn, *args, **kwargs):
                nonlocal failed
                receipt = submit(fn, *args, **kwargs)
                if fn.__name__ == "_persist_veto" and not failed:
                    failed = True
                    lost = Future()
                    lost.cancel()
                    return lost
                return receipt

            monkeypatch.setattr(
                agent._session_output, "_submit_store_write", lost_receipt
            )
        else:
            monkeypatch.setattr(type(store), "append_event", fail_once)
        event = ToolCallEvent(
            name="work", args={"_tool_call_id": "original-call"}, raw=""
        )
        with pytest.raises(
            RuntimeError if after_append == "cancelled_receipt" else OSError
        ):
            await run_pre_tool_dispatch(agent, event, agent.controller)
        assert agent.controller.conversation.to_json() == before
        assert not agent.controller.has_pending_events()
        await run_pre_tool_dispatch(agent, event, agent.controller)
        await agent.output_router.flush()
        raw = await store.run(store.get_events, "alice")
        assert len([e for e in raw if e["type"] == "assistant_tool_calls"]) == 1
        assert len([e for e in raw if e["type"] == "tool_result"]) == 1
        assert agent.controller._event_queue.qsize() == 1


@pytest.mark.parametrize("replace_target", [False, True])
@pytest.mark.parametrize("shutdown", [False, True])
async def test_veto_cancel_joins_receipt_and_never_publishes_into_replacement(
    tmp_path, monkeypatch, replace_target, shutdown
):
    class Veto(BasePlugin):
        name = "veto"

        async def pre_tool_dispatch(self, event, ctx):
            raise PluginBlockError("denied")

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            plugins=[Veto()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        await append_assistant(
            agent.controller,
            "",
            tool_calls=[
                {
                    "id": "original-call",
                    "type": "function",
                    "function": {"name": "work", "arguments": "{}"},
                }
            ],
        )
        original = agent.controller.conversation
        before = original.to_json()
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        store = agent.session_store
        append = type(store).append_event

        def held(self, namespace, kind, data, **kwargs):
            if self is store and kind == "tool_result":
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return append(self, namespace, kind, data, **kwargs)

        monkeypatch.setattr(type(store), "append_event", held)
        event = ToolCallEvent(
            name="work", args={"_tool_call_id": "original-call"}, raw=""
        )
        dispatch = asyncio.create_task(
            run_pre_tool_dispatch(agent, event, agent.controller)
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert original.to_json() == before
            if replace_target:
                await agent._submit_session_state(
                    lambda: setattr(
                        agent.controller, "conversation", Conversation.from_json(before)
                    )
                )
            stopped = asyncio.create_task(agent.stop()) if shutdown else None
            dispatch.cancel()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(stopped or dispatch), 0.05)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
            if stopped is not None:
                await asyncio.wait_for(stopped, 3)
                assert agent._stopped and agent._session_state_owner.task is None
            await agent.output_router.flush()
            assert agent.controller.has_pending_events() is (not replace_target)
            if replace_target:
                assert agent.controller.conversation.to_json() == before
                assert original.to_json() == before
            raw = await store.run(store.get_events, "alice")
            assert len([e for e in raw if e["type"] == "tool_result"]) == 1
        finally:
            release.set()
            await asyncio.gather(dispatch, return_exceptions=True)


@pytest.mark.parametrize(
    "calls", [("deny", "deny"), ("allow", "deny"), ("deny", "allow")]
)
@pytest.mark.parametrize("write_failure", [False, True])
async def test_native_mixed_veto_batch_survives_cold_replay(
    tmp_path, monkeypatch, calls, write_failure
):
    class Gate(BasePlugin):
        name = "gate"

        async def pre_tool_dispatch(self, event, ctx):
            if event.args.get("decision") == "deny":
                raise PluginBlockError("denied")

    class Work(BaseTool):
        tool_name = "work"
        description = "Direct work"
        execution_mode = ExecutionMode.DIRECT

        async def _execute(self, args, **kwargs):
            return ToolResult(output="allowed result")

    class Provider(ScriptedLLM):
        def __init__(self):
            super().__init__(["unused", "finished"])
            self.last_tool_calls = []

        async def chat(self, messages, **kwargs):
            self.last_tool_calls = []
            if self.call_count == 0:
                self.call_count += 1
                self.call_log.append(self._normalize_messages(messages))
                self.last_tool_calls = [
                    NativeToolCall(
                        f"call-{index}", "work", '{"decision":"' + decision + '"}'
                    )
                    for index, decision in enumerate(calls)
                ]
                yield ""
            else:
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = Provider()
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=llm,
            plugins=[Gate()],
            tools=[Work()],
            io="headless",
            start=True,
        )
        if write_failure:
            append = type(creature.agent.session_store).append_event

            def fail_terminal(self, namespace, kind, data, **kwargs):
                if kind == "tool_result" and data.get("error"):
                    raise OSError("denial archive unavailable")
                return append(self, namespace, kind, data, **kwargs)

            monkeypatch.setattr(
                type(creature.agent.session_store), "append_event", fail_terminal
            )
            failed = await creature.run("question", raise_on_error=False)
            assert failed.status == "error"
            assert "denial archive unavailable" in str(failed.error)
            assert llm.call_count == 1
            assert not creature.agent.controller.has_pending_events()
            return
        first = await creature.run("question")
        if "allow" in calls:
            assert first.text == "finished"
        else:
            assert first.text == ""
            assert (await creature.run("acknowledge")).text == "finished"
        results = [message for message in llm.call_log[1] if message["role"] == "tool"]
        assert {message["tool_call_id"] for message in results} == {"call-0", "call-1"}
        path = creature.agent.session_store.path
    store = SessionStore(path)
    fresh_llm = ScriptedLLM(["restored"])
    fresh = await Agent.build(
        AgentConfig(name="alice", tool_format="native"), llm=fresh_llm, io="headless"
    )
    try:
        await store.run(lambda: store.conversation.pop("alice", None))
        await inject_saved_state_async(fresh, store, "alice")
        fresh.attach_session_store(store)
        await fresh.start()
        assert (await fresh.run("continue")).text == "restored"
        results = [m for m in fresh_llm.call_log[0] if m["role"] == "tool"]
        assert len(results) == 2
        assert {m["tool_call_id"]: m["content"] for m in results} == {
            f"call-{index}": "[gate] denied" if decision == "deny" else "allowed result"
            for index, decision in enumerate(calls)
        }
    finally:
        await fresh.stop()
        store.close()
