"""Tail regeneration admission across store, owner, and inbox boundaries."""

import asyncio
import threading

import pytest

import kohakuterrarium.core.agent_messages as message_api

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError, InvalidRequestError
from kohakuterrarium.testing.llm import ScriptedLLM


class _HeldSecondResponse(ScriptedLLM):
    def __init__(self):
        super().__init__(["original", "replacement"])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def chat(self, messages, **kwargs):
        if self.call_count:
            self.entered.set()
            await self.release.wait()
        async for chunk in super().chat(messages, **kwargs):
            yield chunk


@pytest.mark.parametrize("cancel", [False, True])
async def test_waiting_regeneration_keeps_live_state_and_joins_admission(
    tmp_path, monkeypatch, cancel
):
    provider = ScriptedLLM(["original", "replacement"])
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    append = message_api.append_user_event_pair
    seen = []

    def held(*args, **kwargs):
        seen.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return append(*args, **kwargs)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        conv = agent.controller.conversation
        messages = list(conv.get_messages())
        coordinates = (agent._turn_index, agent._branch_id)
        monkeypatch.setattr(message_api, "append_user_event_pair", held)
        task = asyncio.create_task(agent.regenerate_last_response(request_id="retry-1"))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert agent.controller.conversation is conv
            assert conv.get_messages() == messages
            assert (agent._turn_index, agent._branch_id) == coordinates
            assert provider.call_count == 1
            with pytest.raises(ConflictError):
                agent.update_system_prompt("conflict", replace=True)
            if cancel:
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                await asyncio.wait_for(agent._branch_rerun_completion, 5)
            else:
                await asyncio.wait_for(task, 5)
            assert agent.controller.conversation is conv
            assert messages[0] is conv.get_messages()[0]
            assert [
                m.content for m in conv.get_messages() if m.role == "assistant"
            ] == ["replacement"]
            assert agent._branch_id == 2
            assert seen == [agent.session_store._affinity_worker_ident]
            assert not agent._processing_lock.locked()
            assert not agent._history_rerun_pending
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_new_input_waits_behind_admitted_regeneration(tmp_path, monkeypatch):
    provider = ScriptedLLM(["original", "replacement", "next answer"])
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    append = message_api.append_user_event_pair

    def held(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return append(*args, **kwargs)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        monkeypatch.setattr(message_api, "append_user_event_pair", held)
        retry = asyncio.create_task(agent.regenerate_last_response())
        following = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            following = asyncio.create_task(creature.run("next question", timeout=5))
            for _ in range(6):
                await asyncio.sleep(0)
            assert provider.call_count == 1
            release.set()
            await asyncio.wait_for(retry, 5)
            assert (await following).text == "next answer"
            assert [
                m.content
                for m in agent.controller.conversation.get_messages()
                if m.role != "system"
            ] == ["question", "replacement", "next question", "next answer"]
        finally:
            release.set()
            await asyncio.gather(
                *[t for t in (retry, following) if t is not None],
                return_exceptions=True,
            )


async def test_stop_does_not_wait_for_regenerated_llm(tmp_path):
    provider = _HeldSecondResponse()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        retry = asyncio.create_task(agent.regenerate_last_response())
        try:
            await asyncio.wait_for(provider.entered.wait(), 3)
            assert agent._history_rerun_task.done()
            assert not agent._get_message_mutation_lock().locked()
            await asyncio.wait_for(agent.stop(), 3)
            await asyncio.wait_for(retry, 3)
            assert agent._session_state_owner.task is None
        finally:
            provider.release.set()
            await asyncio.gather(retry, return_exceptions=True)


async def test_stop_joins_pending_regeneration_commit(tmp_path, monkeypatch):
    provider = ScriptedLLM(["original", "replacement"])
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    append = message_api.append_user_event_pair

    def held(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return append(*args, **kwargs)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        monkeypatch.setattr(message_api, "append_user_event_pair", held)
        retry = asyncio.create_task(agent.regenerate_last_response())
        stop = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            stop = asyncio.create_task(agent.stop())
            await asyncio.sleep(0)
            assert not stop.done()
            release.set()
            await asyncio.wait_for(asyncio.gather(retry, stop), 5)
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert any(
                e["type"] == "user_message" and e["branch_id"] == 2 for e in events
            )
            assert agent._branch_id == 2
            assert not agent._history_rerun_pending
        finally:
            release.set()
            await asyncio.gather(
                *[t for t in (retry, stop) if t is not None], return_exceptions=True
            )


async def test_partial_pair_failure_fences_select_and_preserves_context(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(["original", "must not run"])
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        before = agent.controller.conversation.get_messages()[:]

        def partial(store, name, payload, turn, branch, path):
            store.append_event(
                name,
                "user_input",
                payload,
                turn_index=turn,
                branch_id=branch,
                parent_branch_path=path,
            )
            raise OSError("second append rejected")

        monkeypatch.setattr(message_api, "append_user_event_pair", partial)
        with pytest.raises(OSError, match="second append"):
            await agent.regenerate_last_response()
        assert agent.controller.conversation.get_messages() == before
        assert agent._branch_id == 1
        assert provider.call_count == 1
        with pytest.raises(ConflictError, match="uncertain"):
            await agent.select_branch({1: 1})


async def test_owner_closes_after_persistence_still_fences_regeneration(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["original"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        before = agent.controller.conversation.get_messages()[:]
        append = message_api.append_user_event_pair
        loop = asyncio.get_running_loop()

        def close_owner_after_write(*args, **kwargs):
            append(*args, **kwargs)
            asyncio.run_coroutine_threadsafe(
                agent._session_state_owner.close(), loop
            ).result(timeout=3)

        monkeypatch.setattr(
            message_api, "append_user_event_pair", close_owner_after_write
        )
        with pytest.raises(InvalidRequestError):
            await agent.regenerate_last_response()
        assert agent._history_rerun_uncertain
        assert agent.controller.conversation.get_messages() == before
        with pytest.raises(ConflictError, match="uncertain"):
            agent.update_system_prompt("must reject", replace=True)


async def test_select_then_rerun_publishes_new_context_and_user_identity(tmp_path):
    provider = _HeldSecondResponse()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=provider, io="headless", start=True
        )
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original"
        await agent.select_branch({1: 1})
        previous = agent._runtime_context_bundle
        user = next(
            m for m in agent.controller.conversation.get_messages() if m.role == "user"
        )
        retry = asyncio.create_task(agent.regenerate_last_response())
        try:
            await asyncio.wait_for(provider.entered.wait(), 3)
            bundle = agent._runtime_context_bundle
            assert bundle.ref.context_id != previous.ref.context_id
            assert bundle.branch_id == agent._branch_id == 2
            assert (
                tuple(agent.controller.conversation.get_messages())
                == bundle.provider_conversation
            )
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            persisted = next(
                e for e in events if e["type"] == "user_message" and e["branch_id"] == 2
            )
            assert user.metadata["event_id"] == persisted["event_id"]
            assert user.metadata["branch_id"] == 2
            assert (
                next(
                    m
                    for m in agent.controller.conversation.get_messages()
                    if m.role == "user"
                )
                is user
            )
        finally:
            provider.release.set()
            await asyncio.gather(retry, return_exceptions=True)


async def test_tail_regeneration_without_store_preserves_prefix_identity():
    agent = await Agent.build(
        AgentConfig(name="alice"),
        llm=ScriptedLLM(["original", "replacement"]),
        io="headless",
    )
    await agent.start()
    try:
        assert (await agent.run("question", timeout=5)).text == "original"
        conv = agent.controller.conversation
        user = next(m for m in conv.get_messages() if m.role == "user")
        await agent.regenerate_last_response()
        assert agent.controller.conversation is conv
        assert next(m for m in conv.get_messages() if m.role == "user") is user
        assert [m.content for m in conv.get_messages() if m.role == "assistant"] == [
            "replacement"
        ]
    finally:
        await agent.stop()
