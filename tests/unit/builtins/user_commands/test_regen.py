"""The regen command surfaces settled results rather than legacy admission-only success."""

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.builtins.user_commands.regen import RegenCommand
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.modules.user_command.base import UserCommandContext
from kohakuterrarium.testing.llm import ScriptedLLM


class CommandProvider(ScriptedLLM):
    def __init__(self, fail):
        super().__init__(["original", "regenerated"])
        self.fail = fail

    async def chat(self, messages, **kwargs):
        if self.call_count and self.fail:
            self.call_count += 1
            raise RuntimeError("command provider failure")
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("fail", [False, True])
async def test_regen_command_exposes_completed_or_applied_failed_receipt(
    tmp_path, fail
):
    provider = CommandProvider(fail)
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        await creature.run("question", timeout=3)
        result = await RegenCommand().execute(
            "", UserCommandContext(agent=creature.agent)
        )
        if fail:
            assert not result.success and "error" in result.error
            assert not result.output
        else:
            assert result.success and result.output == "Regenerated last response."
        assert result.data["operation"]["status"] == ("error" if fail else "completed")
        assert (
            result.data["operation"]["turn_index"],
            result.data["operation"]["branch_id"],
        ) == (1, 2)
        assert result.data["applied"] is True and result.data["operation"]["request_id"]
        assert provider.call_count == 2


async def test_regen_command_no_target_does_not_report_generation(tmp_path):
    provider = ScriptedLLM([])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        result = await RegenCommand().execute(
            "", UserCommandContext(agent=creature.agent)
        )
        assert not result.success and not result.output
        assert result.data["applied"] is False
        assert provider.call_count == 0
