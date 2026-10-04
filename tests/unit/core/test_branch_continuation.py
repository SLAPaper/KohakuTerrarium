"""Resolved selected paths govern continuation without replacing exact live context."""

import pytest

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core.branch_continuation import normalize_branch_view
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.session.history_support import InvalidBranchViewError
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize(
    "value",
    [
        [],
        "latest",
        {True: 1},
        {0: 1},
        {1: False},
        {1: 0},
        {1: 1.5},
        {"bad": 1},
        {"１": 1},
        {1: 1, "1": 2},
    ],
)
def test_invalid_coordinates_are_not_silently_dropped(value):
    with pytest.raises(ValueError, match="branch_view"):
        normalize_branch_view(value)


def test_coordinates_are_detached_and_normalized():
    original = {"1": 2, 3: 4}
    normalized = normalize_branch_view(original)
    original["1"] = 99
    assert normalized == {1: 2, 3: 4}
    assert normalize_branch_view(None) is None
    assert normalize_branch_view({}) == {}


async def test_same_branch_input_uses_live_context_and_uncertainty_fence(tmp_path):
    provider = ScriptedLLM(["first", "second", "recovered"])
    agent = await Agent.build(
        AgentConfig(name="continuation", agent_path=tmp_path),
        llm=provider,
        io="headless",
    )
    await agent.start()
    try:
        await agent.run("ROOT")
        assert await agent.inject_input("SAME", branch_view={"1": 1})
        assert [
            m["content"] for m in provider.last_messages if m["role"] == "user"
        ] == ["ROOT", "SAME"]
        before = agent.controller.conversation.to_json()
        with pytest.raises(ValueError, match="branch_view"):
            await agent.inject_input("INVALID", branch_view={"1": False})
        agent._history_selection_uncertain = True
        with pytest.raises(ConflictError, match="uncertain"):
            await agent.inject_input("UNCONFIRMED", branch_view={1: 1})
        agent._history_selection_uncertain = False
        agent._paused = True
        with pytest.raises(ConflictError, match="paused"):
            await agent.inject_input("PAUSED", branch_view={1: 2})
        agent._paused = False
        assert provider.call_count == 2
        assert agent.controller.conversation.to_json() == before
        assert agent._event_inbox.empty()
        assert await agent.inject_input("RECOVERED", branch_view={1: 1})
        assert [
            m["content"] for m in provider.last_messages if m["role"] == "user"
        ] == ["ROOT", "SAME", "RECOVERED"]
    finally:
        agent._paused = False
        agent._history_selection_uncertain = False
        await agent.stop()


async def test_sparse_ancestor_continuation_uses_latest_compatible_child(tmp_path):
    provider = ScriptedLLM(["root", "child", "edited", "continued"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("ROOT", timeout=3)
        await creature.run("CHILD", timeout=3)
        index = next(
            i
            for i, message in enumerate(agent.controller.conversation.get_messages())
            if message.role == "user" and message.content == "CHILD"
        )
        await agent.edit_and_rerun(index, "EDITED-CHILD")
        await agent.select_branch({1: 1, 2: 1})
        assert await agent.inject_input("CONTINUE", branch_view={1: 1})
        assert agent.controller.conversation.get_messages()[-1].content == "continued"
        assert [
            m["content"] for m in provider.last_messages if m["role"] == "user"
        ] == ["ROOT", "EDITED-CHILD", "CONTINUE"]
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [
            list(pair)
            for pair in [r for r in rows if r["type"] == "user_message"][-1][
                "parent_branch_path"
            ]
        ] == [[1, 1], [2, 2]]


async def test_exact_resolved_selection_keeps_live_system_and_compact_summary(tmp_path):
    provider = ScriptedLLM(["first", "continued"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", system_prompt="baseline"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("ROOT", timeout=3)
        conversation = agent.controller.conversation
        system = conversation.get_messages()[0]
        conversation.append(
            "system", "retained live summary", metadata={"is_compact_summary": True}
        )
        assert await agent.inject_input("CONTINUE", branch_view={1: 1})
        assert agent.controller.conversation.get_messages()[-1].content == "continued"
        assert agent.controller.conversation is conversation
        assert agent.controller.conversation.get_messages()[0] is system
        assert any(
            m["content"] == "retained live summary" for m in provider.last_messages
        )


async def test_invalid_explicit_branch_does_not_run_or_change_live_binding(tmp_path):
    provider = ScriptedLLM(["root", "child"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("ROOT", timeout=3)
        await creature.run("CHILD", timeout=3)
        conversation = agent.controller.conversation
        with pytest.raises((ConflictError, InvalidBranchViewError)):
            await agent.inject_input("MUST NOT RUN", branch_view={1: 999})
        assert provider.call_count == 2
        assert agent.controller.conversation is conversation
        assert not agent._processing_lock.locked()
