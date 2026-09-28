"""Named-output effects are distinct from confirmed continuation receipts."""

import asyncio
import threading

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.core import agent_handlers, output_feedback
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.modules.output.base import BaseOutputModule
from kohakuterrarium.testing.llm import ScriptedLLM


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("failure", [None, "before", "after", "provider"])
async def test_output_receipt_controls_continuation_after_external_effect(
    tmp_path, monkeypatch, failure, cancel
):
    entered, release = asyncio.Event(), threading.Event()
    loop, effects, callers = asyncio.get_running_loop(), [], []

    class Target(BaseOutputModule):
        async def write(self, content):
            effects.append(content)

        async def write_stream(self, chunk):
            pass

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(["[/output_notice]payload[output_notice/]", "received"]),
            io="headless",
            start=True,
        )
        agent, store = creature.agent, creature.agent.session_store
        agent.output_router.named_outputs["notice"] = Target()
        agent.controller.config.known_outputs.add("notice")
        original_provider = agent.controller.llm
        persist, route = (
            output_feedback.persist_output_delivery,
            agent_handlers.route_output_feedback,
        )

        def held(*args):
            assert threading.get_ident() == store._affinity_worker_ident
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(6)
            if failure == "before":
                raise OSError("output receipt not written")
            proof = persist(*args)
            if failure == "after":
                raise OSError("output receipt unconfirmed")
            return proof

        async def observed(*args):
            callers.append(asyncio.current_task())
            return await route(*args)

        monkeypatch.setattr(output_feedback, "persist_output_delivery", held)
        monkeypatch.setattr(agent_handlers, "route_output_feedback", observed)
        turn = asyncio.create_task(creature.run("send", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert effects == ["payload"]
            assert original_provider.call_count == 1
            assert not getattr(agent, "_output_feedback_receipts", None)
            if failure == "provider":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(
                        agent.controller, "llm", ScriptedLLM(["replacement"])
                    )
                )
            if cancel:
                callers[0].cancel()
                await agent._session_state_owner.submit_internal(lambda: None)
                assert not callers[0].done()
            release.set()
            outcome = await asyncio.wait_for(turn, 6)
            assert outcome.ok == (failure is None and not cancel)
            assert original_provider.call_count == (
                2 if failure is None and not cancel else 1
            )
            if failure:
                with pytest.raises(ConflictError, match="recovery"):
                    require_model_mutation_available(agent)
            rows = await store.run(store.get_events, "alice")
            assert sum(row["type"] == "output_delivery" for row in rows) == (
                0 if failure == "before" else 1
            )
            assert sum(row["type"] == "tool_feedback_admitted" for row in rows) == (
                1 if failure is None and not cancel else 0
            )
            assert not getattr(agent, "_output_feedback_receipts", None)
        finally:
            release.set()
            await asyncio.gather(turn, return_exceptions=True)


@pytest.mark.parametrize("binding", ["handoff", "detached", "missing_sink"])
async def test_output_target_await_does_not_block_handoff(tmp_path, binding):
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    class Target(BaseOutputModule):
        async def write(self, content):
            entered.set()
            await release.wait()
            effects.append(content)

        async def write_stream(self, chunk):
            pass

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        alice = await engine.add_creature(
            AgentConfig(name="alice", tool_format="bracket"),
            llm=ScriptedLLM(
                ["[/output_notice]late output[output_notice/]", "received"]
            ),
            io="headless",
            start=True,
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        agent, source = alice.agent, alice.agent.session_store
        agent.output_router.named_outputs["notice"] = Target()
        agent.controller.config.known_outputs.add("notice")
        turn = asyncio.create_task(alice.run("send", raise_on_error=False))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if binding == "handoff":
                await asyncio.wait_for(engine.connect(bob, alice, channel="bridge"), 8)
                assert agent.session_store is not source
            elif binding == "detached":
                await agent._session_state_owner.submit_internal(
                    lambda: setattr(agent.controller, "session_store", None)
                )
            else:
                await agent._session_state_owner.submit_internal(
                    lambda: agent.output_router.remove_secondary(agent._session_output)
                )
            assert not turn.done() and effects == []
            release.set()
            outcome = await asyncio.wait_for(turn, 5)
            if binding != "handoff":
                with pytest.raises(ConflictError, match="recovery"):
                    require_model_mutation_available(agent)
                assert not outcome.ok
                assert effects == ["late output"]
                assert agent.llm.call_count == 1
                return
            assert outcome.ok
            assert effects == ["late output"]
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert sum(row["type"] == "output_delivery" for row in rows) == 1
            assert sum(row["type"] == "tool_feedback_admitted" for row in rows) == 1
            with pytest.raises(RuntimeError, match="closed"):
                await source.run(source.get_events, "alice")
        finally:
            release.set()
            await asyncio.gather(turn, return_exceptions=True)
