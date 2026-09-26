"""History admission must not publish a branch rejected before persistence."""

from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.builtins.user_commands.branch import BranchCommand
from kohakuterrarium.modules.user_command.base import UserCommandContext
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.session.raw_history import UserMessageSelector
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("operation", ["tail", "edit", "turn", "raw", "view_edit"])
async def test_rejected_regeneration_preserves_live_conversation_and_coordinates(
    tmp_path, monkeypatch, operation
):
    provider = ScriptedLLM(["original answer", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        conversation = agent.controller.conversation
        messages = list(conversation.get_messages())
        before = deepcopy(conversation.to_messages())
        coordinates = (
            agent._turn_index,
            agent._branch_id,
            deepcopy(agent._parent_branch_path),
        )
        store = agent.session_store
        raw_before = await store.run(store.get_events, "alice")

        def reject(*args, **kwargs):
            raise OSError("branch admission rejected before write")

        with monkeypatch.context() as patch:
            patch.setattr(
                "kohakuterrarium.core.agent_messages.append_user_event_pair", reject
            )
            with pytest.raises(OSError, match="branch admission rejected"):
                match operation:
                    case "tail":
                        await agent.regenerate_last_response()
                    case "edit":
                        await agent.edit_and_rerun(-1, "edited", turn_index=1)
                    case "turn":
                        await agent.regenerate_last_response(turn_index=1)
                    case "raw":
                        raw_user = next(
                            e for e in raw_before if e["type"] == "user_message"
                        )
                        await agent.edit_and_rerun(
                            -1,
                            "edited",
                            target=UserMessageSelector(raw_user["event_id"], 1, 1),
                        )
                    case "view_edit":
                        await agent.edit_and_rerun(
                            -1, "edited", turn_index=1, branch_view={1: 1}
                        )
        assert provider.call_count == 1
        assert await store.run(store.get_events, "alice") == raw_before
        assert agent.controller.conversation is conversation
        assert conversation.get_messages() == messages
        assert conversation.to_messages() == before
        assert (
            agent._turn_index,
            agent._branch_id,
            agent._parent_branch_path,
        ) == coordinates


async def test_invalid_edit_target_preserves_live_selection(tmp_path):
    provider = ScriptedLLM(["original answer", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        conversation = agent.controller.conversation
        messages = conversation.get_messages()[:]
        events = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert not await agent.edit_and_rerun(999, "invalid", branch_view={1: 1})
        assert agent.controller.conversation is conversation
        assert conversation.get_messages() == messages
        assert (
            await agent.session_store.run(agent.session_store.get_events, "alice")
            == events
        )
        assert provider.call_count == 1


async def test_branch_latest_keeps_live_system_context_without_calling_llm(tmp_path):
    provider = ScriptedLLM(["original answer"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", system_prompt="Preserve this instruction"),
            llm=provider,
            io="headless",
            start=True,
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        system_before = agent.controller.conversation.get_system_message().content
        result = await BranchCommand().execute(
            "latest", UserCommandContext(agent=agent)
        )
        assert result.success, result.error
        assert provider.call_count == 1
        system_after = agent.controller.conversation.get_system_message()
        assert system_after is not None
        assert system_after.content == system_before
        assert [
            m.content
            for m in agent.controller.conversation.get_messages()
            if m.role == "assistant"
        ] == ["original answer"]
