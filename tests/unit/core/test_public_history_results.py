"""Explicit public result APIs report settled outcomes without changing legacy SDK returns."""

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.interrupt_regeneration import RegenerationTurnError
from kohakuterrarium.testing.llm import ScriptedLLM


class ResultProvider(ScriptedLLM):
    def __init__(self, fail):
        super().__init__(["original", "replacement", "legacy rerun", "legacy edit"])
        self.fail = fail

    async def chat(self, messages, **kwargs):
        if self.call_count and self.fail:
            self.call_count += 1
            raise RuntimeError("public result provider failed")
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("kind", ["edit", "rerun"])
@pytest.mark.parametrize("fail", [False, True])
async def test_public_result_api_reports_settlement_and_preserves_legacy_returns(
    tmp_path, kind, fail
):
    provider = ResultProvider(fail)
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        await creature.run("question", timeout=3)
        index = next(
            i
            for i, message in enumerate(agent.controller.conversation.get_messages())
            if message.role == "user"
        )
        if kind == "edit":
            operation = agent.edit_and_rerun_result(
                index, "edited", request_id="result-token"
            )
        else:
            operation = agent.regenerate_last_response_result(request_id="result-token")
        if fail:
            with pytest.raises(RegenerationTurnError) as failed:
                await operation
            receipt = failed.value.receipt
            assert receipt["status"] == "error" and failed.value.may_have_run
        else:
            receipt = await operation
            assert receipt["status"] == "completed"
            assert (
                agent.controller.conversation.get_messages()[-1].content
                == "replacement"
            )
        assert receipt["request_id"] == "result-token"
        assert (receipt["turn_index"], receipt["branch_id"]) == (1, 2)
        assert receipt["branch_view"] == {"1": 2}
        assert provider.call_count == 2
        provider.fail = False
        assert await agent.regenerate_last_response() is None
        assert await agent.edit_and_rerun(index, "legacy edited") is True
        assert provider.call_count == 4


async def test_public_result_api_rejects_no_target_without_calling_provider(tmp_path):
    provider = ScriptedLLM([])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        with pytest.raises(RegenerationTurnError) as error:
            await creature.agent.regenerate_last_response_result()
        assert error.value.status == "rejected" and not error.value.may_have_run
        assert provider.call_count == 0
        assert await creature.agent.edit_and_rerun_result(999, "invalid") is None
        assert creature.agent._branch_id == 0
