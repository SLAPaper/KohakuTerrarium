"""Rerun completion reflects processing errors and internal compaction admission."""

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.interrupt_regeneration import RegenerationTurnError
from kohakuterrarium.terrarium.service import LocalTerrariumService
from kohakuterrarium.testing.llm import ScriptedLLM


class FailingRerunProvider(ScriptedLLM):
    async def chat(self, messages, **kwargs):
        if self.call_count:
            self.call_count += 1
            self.call_log.append(self._normalize_messages(messages))
            raise RuntimeError("rerun provider unavailable")
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("surface", ["compound", "service", "edit_service"])
async def test_provider_failure_is_not_a_completed_regeneration(tmp_path, surface):
    provider = FailingRerunProvider(["original answer"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=provider,
            io="headless",
            start=True,
        )
        await creature.run("question", timeout=3)
        if surface == "compound":
            operation = creature.agent.interrupt_and_regenerate
        elif surface == "service":

            async def operation(**kwargs):
                return await LocalTerrariumService(engine).regenerate(
                    creature.creature_id, **kwargs
                )

        else:
            index = next(
                i
                for i, message in enumerate(
                    creature.agent.controller.conversation.get_messages()
                )
                if message.role == "user"
            )

            async def operation(**kwargs):
                return await LocalTerrariumService(engine).edit_message(
                    creature.creature_id, index, "edited question", **kwargs
                )

        with pytest.raises(RegenerationTurnError, match="error") as failed:
            await operation(request_id="failed-rerun")
        assert failed.value.status == "error"
        assert failed.value.receipt["branch_id"] == 2
        assert failed.value.receipt["turn_index"] == 1
        assert failed.value.receipt["request_id"] == "failed-rerun"
        assert failed.value.may_have_run is True
        assert provider.call_count == 2
        agent = creature.agent
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert [
            (r["turn_index"], r["branch_id"])
            for r in rows
            if r["type"] == "user_message"
        ] == [(1, 1), (1, 2)]
        assert agent.controller.conversation.get_messages()[-1].role == "user"
        assert getattr(agent, "_regeneration_reservation", None) is None


async def test_compound_rerun_auto_compact_does_not_reject_its_own_finalization(
    tmp_path,
):
    provider = ScriptedLLM(["first", "second"])
    provider.last_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "total_tokens": 102,
    }
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", compact={"max_tokens": 10}),
            llm=provider,
            io="headless",
            start=True,
        )
        await creature.run("question", timeout=3)
        result = await creature.agent.interrupt_and_regenerate(
            request_id="compact-rerun"
        )
        assert result["status"] == "completed"
        assert result["branch_id"] == 2
        assert (
            creature.agent.controller.conversation.get_messages()[-1].content
            == "second"
        )
        assert provider.call_count == 2
        assert creature.agent._regeneration_reservation is None
