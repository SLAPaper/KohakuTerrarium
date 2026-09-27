"""Compact preparation and live mutations belong to the session-state owner."""

import asyncio
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core import compact as compact_module
from kohakuterrarium.core.compact import CompactConfig, CompactManager
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM


async def prepare(engine):
    creature = await engine.add_creature(
        AgentConfig(name="alice"),
        llm=ScriptedLLM(["one", "two", "three"]),
        io="headless",
        start=True,
    )
    for question in ("first", "second", "third"):
        await creature.run(question)
    agent = creature.agent
    manager = CompactManager(CompactConfig(keep_recent_turns=1, cooldown_seconds=0))
    manager._controller, manager._agent = agent.controller, agent
    manager._llm = ScriptedLLM(["summary retained"])
    manager._agent_name = "alice"
    agent.compact_manager = manager
    return agent, manager


async def test_compact_reads_and_splices_on_owner_but_summarizes_outside(monkeypatch):
    async with Terrarium() as engine:
        agent, manager = await prepare(engine)
        owner = agent._session_state_owner.task
        conversation = agent.controller.conversation
        reads, writes, summaries = [], [], []
        get_messages = type(conversation).get_messages
        splice, summarize, elide = (
            manager._splice_conversation,
            manager._summarize,
            compact_module.elide_after_compact,
        )

        def read(self):
            if self is conversation:
                reads.append(asyncio.current_task())
            return get_messages(self)

        def commit(*args, **kwargs):
            writes.append(asyncio.current_task())
            return splice(*args, **kwargs)

        def elide_live(controller):
            writes.append(asyncio.current_task())
            return elide(controller)

        async def summary(text):
            summaries.append(asyncio.current_task())
            return await summarize(text)

        monkeypatch.setattr(type(conversation), "get_messages", read)
        monkeypatch.setattr(manager, "_splice_conversation", commit)
        monkeypatch.setattr(manager, "_summarize", summary)
        monkeypatch.setattr(compact_module, "elide_after_compact", elide_live)
        await manager._run_compact()
        assert reads and all(task is owner for task in reads)
        assert writes and all(task is owner for task in writes)
        assert summaries and all(task is not owner for task in summaries)
        assert manager._compact_count == 1


@pytest.mark.parametrize(
    "target",
    [
        "provider",
        "summary_provider",
        "tool_format",
        "branch",
        "branch_view",
        "conversation",
    ],
)
async def test_summary_rejects_replaced_owner_target(target):
    async with Terrarium() as engine:
        agent, manager = await prepare(engine)
        entered, release = asyncio.Event(), asyncio.Event()

        class HeldLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                entered.set()
                await release.wait()
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

        manager._llm = HeldLLM(["stale summary"])
        task = asyncio.create_task(manager._run_compact())
        try:
            await asyncio.wait_for(entered.wait(), 3)

            def replace():
                if target == "provider":
                    agent.controller.llm = ScriptedLLM(["replacement"])
                elif target == "summary_provider":
                    manager._llm = ScriptedLLM(["replacement summary"])
                elif target == "tool_format":
                    agent.controller.config.tool_format = (
                        "bracket"
                        if agent.controller.config.tool_format == "native"
                        else "native"
                    )
                elif target == "branch":
                    agent._branch_id += 1
                elif target == "branch_view":
                    agent._branch_view = {1: 2, 2: 1, 3: 1}
                else:
                    agent.controller.conversation = deepcopy(
                        agent.controller.conversation
                    )

            await agent._submit_session_state(replace)
            before = agent.controller.conversation.to_messages()
            release.set()
            await asyncio.wait_for(task, 3)
            assert manager._compact_count == 0
            assert agent.controller.conversation.to_messages() == before
            assert manager._last_compact_time > 0
            manager.config.cooldown_seconds = 60
            assert not manager.should_compact(manager.config.max_tokens)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_plugin_edit_does_not_change_captured_summary_input():
    async with Terrarium() as engine:
        agent, manager = await prepare(engine)

        class EditingPlugin:
            async def should_proceed(self, *args, **kwargs):
                await agent._submit_session_state(
                    lambda: setattr(
                        agent.controller.conversation._messages[1],
                        "content",
                        "edited during plugin",
                    )
                )
                return True

            async def notify(self, *args, **kwargs):
                pass

        manager._plugins = EditingPlugin()
        await manager._run_compact()
        assert "edited during plugin" not in str(manager._llm.last_messages)
        assert "first" in str(manager._llm.last_messages)
        assert manager._compact_count == 0
        assert (
            agent.controller.conversation._messages[1].content == "edited during plugin"
        )


@pytest.mark.parametrize("stage", ["splice", "elide"])
async def test_failed_compact_commit_restores_live_state(monkeypatch, stage):
    async with Terrarium() as engine:
        agent, manager = await prepare(engine)
        conversation = agent.controller.conversation
        before = conversation.to_json()
        old_messages = conversation.get_messages()
        agent.controller._last_usage = {"prompt_tokens": 456}
        splice, elide = manager._splice_conversation, compact_module.elide_after_compact

        def broken_splice(*args, **kwargs):
            splice(*args, **kwargs)
            raise RuntimeError("failed after splice")

        def broken_elide(controller):
            elide(controller)
            raise RuntimeError("failed after elide")

        if stage == "splice":
            monkeypatch.setattr(manager, "_splice_conversation", broken_splice)
        else:
            monkeypatch.setattr(compact_module, "elide_after_compact", broken_elide)
        await manager._run_compact()
        assert conversation.to_json() == before
        assert all(a is b for a, b in zip(conversation.get_messages(), old_messages))
        assert manager._compact_count == 0
        assert manager._last_compact_time == 0
        assert agent.controller._last_usage == {"prompt_tokens": 456}


async def test_cancel_during_owner_capture_does_not_start_summary(monkeypatch):
    async with Terrarium() as engine:
        agent, manager = await prepare(engine)
        getter = type(agent.controller.conversation).get_messages
        running = None

        def cancel_capture(self):
            if self is agent.controller.conversation:
                running.cancel()
            return getter(self)

        monkeypatch.setattr(
            type(agent.controller.conversation), "get_messages", cancel_capture
        )
        running = asyncio.create_task(manager._run_compact())
        await running
        assert not manager._llm.last_messages
        assert manager._compact_count == 0


async def test_appended_turn_survives_compact_and_does_not_expand_summary_coverage(
    tmp_path,
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        agent, manager = await prepare(engine)
        manager._session_store = agent.session_store
        manager._output_router = agent.output_router
        agent.llm = agent.controller.llm = ScriptedLLM(["fourth answer"])
        entered, release = asyncio.Event(), asyncio.Event()

        class HeldLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                entered.set()
                await release.wait()
                async for chunk in super().chat(messages, **kwargs):
                    yield chunk

        manager._llm = HeldLLM(["summary of first two"])
        task = asyncio.create_task(manager._run_compact())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert (await agent.run("fourth")).text == "fourth answer"
            release.set()
            await asyncio.wait_for(task, 3)
            assert manager._compact_count == 1
            contents = [
                m.get("content") for m in agent.controller.conversation.to_messages()
            ]
            assert (
                "third" in contents
                and "fourth" in contents
                and "fourth answer" in contents
            )
            assert "first" not in contents and "second" not in contents
            events = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            completed = [e for e in events if e.get("type") == "compact_complete"]
            assert len(completed) == 1
            metadata = completed[0].get("metadata", completed[0])
            assert [4, 1] not in metadata["compact_path"]
            replaced = [
                e
                for e in events
                if metadata["replaced_from_event_id"]
                <= e["event_id"]
                <= metadata["replaced_to_event_id"]
            ]
            assert {e.get("turn_index") for e in replaced} <= {1, 2}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
