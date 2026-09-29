"""Explicit continuation coordinates are detached and rejected before ingress."""

import pytest

from kohakuterrarium import Agent
from kohakuterrarium.core.branch_continuation import normalize_branch_view
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
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
