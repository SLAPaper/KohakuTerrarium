"""Unit tests for the /branch command helpers."""

from kohakuterrarium import Terrarium
from kohakuterrarium.builtins.user_commands.branch import BranchCommand, _rebuild_conv
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.conversation import Conversation
from kohakuterrarium.modules.user_command.base import UserCommandContext
from kohakuterrarium.testing.llm import ScriptedLLM


def test_rebuild_conv_preserves_user_turn_metadata():
    # /branch rebuilds the live conversation from replay; the replayed user
    # messages carry turn identity (include_metadata=True) that must survive
    # the rebuild so edit targeting stays deterministic after a branch switch.
    msgs = [
        {
            "role": "user",
            "content": "U1",
            "metadata": {"event_id": 1, "turn_index": 1, "branch_id": 1},
        },
        {"role": "assistant", "content": "R1"},
    ]
    conv = _rebuild_conv(msgs, Conversation)
    out = conv.to_messages(include_metadata=True)
    assert out[0]["metadata"] == {"event_id": 1, "turn_index": 1, "branch_id": 1}
    assert out[1] == {"role": "assistant", "content": "R1"}


async def test_ancestor_switch_roundtrip_drops_only_stale_descendant_choices(tmp_path):
    provider = ScriptedLLM(
        ["root answer", "child answer", "edited child", "edited root", "continued"]
    )
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
        child_index = next(
            i
            for i, message in enumerate(agent.controller.conversation.get_messages())
            if message.role == "user" and message.content == "CHILD"
        )
        await agent.edit_and_rerun(child_index, "EDITED-CHILD")
        await agent.select_branch({1: 1, 2: 1})
        command = BranchCommand()
        context = UserCommandContext(agent=agent)
        assert (await command.execute("1 1", context)).success
        assert [
            m.content
            for m in agent.controller.conversation.get_messages()
            if m.role == "user"
        ] == ["ROOT", "CHILD"]
        root_index = next(
            i
            for i, message in enumerate(agent.controller.conversation.get_messages())
            if message.role == "user" and message.content == "ROOT"
        )
        await agent.edit_and_rerun(root_index, "EDITED-ROOT")
        assert (await command.execute("1 1", context)).success
        assert [
            m.content
            for m in agent.controller.conversation.get_messages()
            if m.role == "user"
        ] == ["ROOT", "EDITED-CHILD"]
        switched = await command.execute("1 2", context)
        assert switched.success, switched.error
        assert [
            m.content
            for m in agent.controller.conversation.get_messages()
            if m.role == "user"
        ] == ["EDITED-ROOT"]
        assert (await creature.run("CONTINUE", timeout=3)).text == "continued"
        assert [
            m["content"] for m in provider.last_messages if m["role"] == "user"
        ] == ["EDITED-ROOT", "CONTINUE"]
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        last_user = [row for row in rows if row["type"] == "user_message"][-1]
        assert [list(pair) for pair in last_user["parent_branch_path"]] == [[1, 2]]
        assert (
            await command.execute("1 99", context)
        ).error == "Turn 1 has no branch 99. Available: [1, 2]"
