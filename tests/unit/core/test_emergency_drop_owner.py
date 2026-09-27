"""Provider recovery must not replace unrelated host history or its identities."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core import controller_recovery
from kohakuterrarium.core.compact import CompactConfig, CompactManager
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.llm.base import BaseLLMProvider, OverflowRecoveryState
from kohakuterrarium.modules.plugin.base import BasePlugin
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium.runtime_prompt import apply_managed_section


class RecoveringLLM(ScriptedLLM, BaseLLMProvider):
    def __init__(self):
        BaseLLMProvider.__init__(self)
        ScriptedLLM.__init__(self, ["recovered answer"])

    async def chat(self, messages, **kwargs):
        current = messages
        if any(m.get("tool_calls") for m in current):
            recovered = await self._recover_from_overflow(
                current, OverflowRecoveryState()
            )
            if recovered is not None:
                current = recovered
        async for chunk in ScriptedLLM.chat(self, current, **kwargs):
            yield chunk


def tool_round():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "overflow-call",
                    "type": "function",
                    "function": {"name": "scratchpad", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "large tool output",
            "tool_call_id": "overflow-call",
        },
    ]


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize(
    "change", ["graph", "user", "system_role", "system_object", "provider", "branch"]
)
async def test_prompt_refresh_before_dispatch_keeps_request_snapshot_and_recovery_stamp(
    overflow, change
):
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldHook(BasePlugin):
        name = "hold_before_dispatch"

        async def pre_llm_call(self, messages, **kwargs):
            entered.set()
            await release.wait()
            return messages

    async with Terrarium() as engine:
        provider = RecoveringLLM() if overflow else ScriptedLLM(["normal answer"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[HeldHook()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await agent._submit_session_state(
            lambda: apply_managed_section(agent, "old graph facts")
        )
        if overflow:
            await agent._submit_session_state(
                lambda: [
                    agent.controller.conversation.append(**m) for m in tool_round()
                ]
            )
        running = asyncio.create_task(creature.run("question", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 3)

            def mutate():
                conversation = agent.controller.conversation
                if change == "graph":
                    apply_managed_section(agent, "new graph facts")
                elif change == "user":
                    conversation._messages[-1].content = "changed user input"
                elif change == "system_role":
                    conversation._messages[0].role = "user"
                elif change == "system_object":
                    conversation._messages[0] = deepcopy(conversation._messages[0])
                elif change == "provider":
                    agent.controller.llm = ScriptedLLM(
                        ["must not dispatch replacement"]
                    )
                else:
                    agent._branch_id += 1

            await agent._submit_session_state(mutate)
            release.set()
            result = await running
            if overflow or change != "graph":
                assert result.status == "error"
                assert provider.call_count == 0
                if overflow:
                    assert any(
                        m.tool_calls
                        for m in agent.controller.conversation.get_messages()
                    )
            else:
                assert result.status == "ok"
                assert result.text == "normal answer"
                assert "old graph facts" in provider.last_messages[0]["content"]
                assert "new graph facts" not in provider.last_messages[0]["content"]
            if change == "graph":
                assert (
                    "new graph facts"
                    in agent.controller.conversation.get_system_message().content
                )
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)


async def test_unscoped_provider_recovery_cannot_replace_host_conversation():
    async with Terrarium() as engine:
        provider = RecoveringLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("host question")
        before = deepcopy(agent.controller.conversation.to_messages())
        foreign = [
            {"role": "system", "content": "child system"},
            {"role": "user", "content": "child question"},
            *tool_round(),
        ]
        recovered = await provider._recover_from_overflow(
            foreign, OverflowRecoveryState()
        )
        assert recovered is not None
        assert not any(message.get("tool_calls") for message in recovered)
        assert agent.controller.conversation.to_messages() == before


async def test_host_recovery_preserves_surviving_message_metadata():
    async with Terrarium() as engine:
        provider = RecoveringLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent

        def seed():
            feedback = agent.controller.conversation.append(
                "user", "older tool feedback"
            )
            feedback.metadata.update(
                {
                    "kind": "tool_results",
                    "event_id": 42,
                    "turn_index": 1,
                    "branch_id": 1,
                }
            )
            for message in tool_round():
                agent.controller.conversation.append(**message)

        await agent._submit_session_state(seed)
        result = await creature.run("new question")
        assert result.text == "recovered answer"
        messages = agent.controller.conversation.get_messages()
        survivor = next(m for m in messages if m.content == "older tool feedback")
        assert survivor.metadata == {
            "kind": "tool_results",
            "event_id": 42,
            "turn_index": 1,
            "branch_id": 1,
        }
        assert not any(m.tool_calls for m in messages)
        assert any(m.content == "new question" for m in messages)
        placeholders = [
            m for m in messages if "scratchpad" in str(m.content) and m.role == "user"
        ]
        assert (
            len(placeholders) == 1
            and placeholders[0].metadata["kind"] == "tool_results"
        )


@pytest.mark.parametrize("change", ["provider", "conversation", "prefix", "branch"])
async def test_stale_request_cannot_drop_from_new_host(change):
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldLLM(RecoveringLLM):
        async def chat(self, messages, **kwargs):
            entered.set()
            await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium() as engine:
        provider = HeldLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await agent._submit_session_state(
            lambda: [agent.controller.conversation.append(**m) for m in tool_round()]
        )
        running = asyncio.create_task(creature.run("question", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 3)

            def mutate():
                if change == "provider":
                    agent.controller.llm = ScriptedLLM(["replacement"])
                elif change == "conversation":
                    agent.controller.conversation = deepcopy(
                        agent.controller.conversation
                    )
                elif change == "prefix":
                    agent.controller.conversation._messages[0].content = (
                        "changed system"
                    )
                else:
                    agent._branch_id += 1

            await agent._submit_session_state(mutate)
            expected = agent.controller.conversation.to_messages()
            release.set()
            result = await running
            assert result.status == "error"
            assert provider.call_count == 0
            assert agent.controller.conversation.to_messages() == expected
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)


async def test_drop_commits_on_owner_before_observer_and_retry(monkeypatch):
    async with Terrarium() as engine:
        provider = RecoveringLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await agent._submit_session_state(
            lambda: [agent.controller.conversation.append(**m) for m in tool_round()]
        )
        seen = []
        apply = controller_recovery.apply_drop

        def record(*args):
            seen.append(("commit", asyncio.current_task()))
            return apply(*args)

        def observe(messages):
            assert not any(
                m.tool_calls for m in agent.controller.conversation.get_messages()
            )
            seen.append(("observe", asyncio.current_task()))

        monkeypatch.setattr(controller_recovery, "apply_drop", record)
        provider.on_emergency_drop(observe)
        assert (await creature.run("question")).text == "recovered answer"
        assert [name for name, task in seen] == ["commit", "observe"]
        assert seen[0][1] is agent._session_state_owner.task
        assert seen[1][1] is not agent._session_state_owner.task


async def test_child_task_inherits_no_host_recovery_authority():
    class ChildLLM(RecoveringLLM):
        async def chat(self, messages, **kwargs):
            foreign = [{"role": "user", "content": "child input"}, *tool_round()]
            child = await asyncio.create_task(
                self._recover_from_overflow(foreign, OverflowRecoveryState())
            )
            assert child[0]["content"] == "child input"
            async for chunk in ScriptedLLM.chat(self, messages, **kwargs):
                yield chunk

    async with Terrarium() as engine:
        provider = ChildLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        assert (await creature.run("host input")).text == "recovered answer"
        messages = creature.agent.controller.conversation.to_messages()
        assert any(m.get("content") == "host input" for m in messages)
        assert not any(m.get("content") == "child input" for m in messages)


async def test_scoped_rescue_then_drop_uses_post_compact_capture():
    class TwiceOverflow(RecoveringLLM):
        async def chat(self, messages, **kwargs):
            assert manager.trigger_compact()
            state = OverflowRecoveryState()
            rescued = await self._recover_from_overflow(messages, state)
            assert rescued is not None and len(rescued) < len(messages)
            assert any("rescued summary" in str(m.get("content")) for m in rescued)
            dropped = await self._recover_from_overflow(rescued, state)
            assert dropped is not None
            async for chunk in ScriptedLLM.chat(self, dropped, **kwargs):
                yield chunk

    async with Terrarium() as engine:
        provider = TwiceOverflow()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        manager = CompactManager(CompactConfig(keep_recent_turns=2, cooldown_seconds=0))
        manager._controller, manager._agent = agent.controller, agent
        manager._llm = ScriptedLLM(["rescued summary"])
        agent.compact_manager = manager

        def seed():
            for index in range(6):
                agent.controller.conversation.append("user", f"old user {index}")
                agent.controller.conversation.append("assistant", f"old answer {index}")
            agent.controller.conversation.append("user", "current tool request")
            for m in tool_round():
                agent.controller.conversation.append(**m)

        await agent._submit_session_state(seed)
        result = await asyncio.wait_for(creature.run("question"), 5)
        assert result.text == "recovered answer"
        await manager.wait_for_current()
        messages = agent.controller.conversation.get_messages()
        assert any("rescued summary" in str(m.content) for m in messages)
        assert not any(m.tool_calls for m in messages)


@pytest.mark.parametrize("failure", ["raise", "cancel"])
async def test_failed_or_cancelled_commit_never_notifies_or_retries(
    monkeypatch, failure
):
    async with Terrarium() as engine:
        provider = RecoveringLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await agent._submit_session_state(
            lambda: [agent.controller.conversation.append(**m) for m in tool_round()]
        )
        observed = []
        provider.on_emergency_drop(lambda messages: observed.append(messages))
        apply = controller_recovery.apply_drop

        def fail(*args):
            if failure == "raise":
                raise RuntimeError("owner commit failed")
            agent._processing_task.cancel()
            return apply(*args)

        monkeypatch.setattr(controller_recovery, "apply_drop", fail)
        result = await creature.run("question", raise_on_error=False)
        if failure == "raise":
            assert result.status == "error"
        assert result.text == ""
        assert provider.call_count == 0
        assert observed == []
        assert any(
            m.tool_calls for m in agent.controller.conversation.get_messages()
        ) == (failure == "raise")


async def test_transformed_wire_and_appended_pending_tail_preserve_canonical_objects():
    entered, release = asyncio.Event(), asyncio.Event()

    class WirePlugin(BasePlugin):
        name = "wire_transform"

        async def pre_llm_call(self, messages, **kwargs):
            wire = deepcopy(messages)
            wire.insert(1, {"role": "user", "content": "wire-only injected"})
            for m in wire:
                if m.get("role") == "tool":
                    m["content"] = "resolved attachment bytes"
            return wire

    class HeldLLM(RecoveringLLM):
        async def chat(self, messages, **kwargs):
            entered.set()
            await release.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk

    async with Terrarium() as engine:
        provider = HeldLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[WirePlugin()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await agent._submit_session_state(
            lambda: [agent.controller.conversation.append(**m) for m in tool_round()]
        )
        running = asyncio.create_task(creature.run("question"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            objects = []

            def append():
                objects.append(
                    agent.controller.conversation.append("user", "new queued input")
                )
                pending = deepcopy(tool_round()[0])
                pending["tool_calls"][0]["id"] = "still-pending"
                objects.append(agent.controller.conversation.append(**pending))

            await agent._submit_session_state(append)
            release.set()
            assert (await running).text == "recovered answer"
            live = agent.controller.conversation.get_messages()
            assert all(any(m is original for m in live) for original in objects)
            assert objects[1].tool_calls[0]["id"] == "still-pending"
            assert not any(m.content == "wire-only injected" for m in live)
            assert any(
                m.get("content") == "wire-only injected" for m in provider.last_messages
            )
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize("malformed", ["foreign", "partial", "duplicate"])
async def test_unproven_wire_round_is_rejected_not_imported(malformed):
    class BadWire(BasePlugin):
        name = "bad_wire"

        async def pre_llm_call(self, messages, **kwargs):
            wire = deepcopy(messages)
            if malformed == "foreign":
                wire.extend(tool_round())
                wire[-2]["tool_calls"][0]["id"] = "foreign-id"
                wire[-1]["tool_call_id"] = "foreign-id"
            elif malformed == "partial":
                wire = [m for m in wire if m.get("role") != "tool"]
            else:
                wire.extend(deepcopy(tool_round()))
            return wire

    async with Terrarium() as engine:
        provider = RecoveringLLM()
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            plugins=[BadWire()],
            io="headless",
            start=True,
        )
        agent = creature.agent
        await agent._submit_session_state(
            lambda: [agent.controller.conversation.append(**m) for m in tool_round()]
        )
        result = await creature.run("question", raise_on_error=False)
        assert result.status == "error"
        assert provider.call_count == 0
        assert any(m.tool_calls for m in agent.controller.conversation.get_messages())
