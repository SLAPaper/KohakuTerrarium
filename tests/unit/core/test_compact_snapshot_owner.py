"""Compact snapshots cross the store boundary as detached owner-captured values."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.compact import CompactConfig, CompactManager
from kohakuterrarium.core.compact_branch import persist_compacted_async
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.events import create_tool_complete_event
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_compact_snapshot_reads_owner_and_writes_store_worker(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        conversation, store = agent.controller.conversation, agent.session_store
        snapshot, save = (
            type(conversation).snapshot_messages,
            type(store).save_conversation,
        )
        readers, writers = [], []
        loop_thread = threading.get_ident()

        def capture(self, **kwargs):
            if self is conversation:
                readers.append(
                    (
                        threading.get_ident(),
                        (
                            asyncio.current_task()
                            if threading.get_ident() == loop_thread
                            else None
                        ),
                    )
                )
            return snapshot(self, **kwargs)

        def persist(self, *args, **kwargs):
            if self is store:
                writers.append(threading.get_ident())
            return save(self, *args, **kwargs)

        monkeypatch.setattr(type(conversation), "snapshot_messages", capture)
        monkeypatch.setattr(type(store), "save_conversation", persist)
        await persist_compacted_async(store, "alice", agent, conversation, 2, 123.0)
        assert readers == [(loop_thread, agent._session_state_owner.task)]
        assert writers and all(tid != loop_thread for tid in writers)
        assert await store.run(lambda: store.state["alice:compact_count"]) == 2
        assert await store.run(lambda: store.state["alice:last_compact_time"]) == 123.0


async def test_compact_snapshot_is_frozen_before_affinity_queue(tmp_path, monkeypatch):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first", "second"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("one")
        await creature.run("two")
        store = agent.session_store
        before = agent.controller.conversation.snapshot_messages(
            preserve_pending_tail=True
        )
        ancestry = deepcopy(agent._parent_branch_path)
        captured, release = asyncio.Event(), threading.Event()
        barrier = None
        run = type(store).run

        def held():
            assert release.wait(5)

        async def intercept(self, fn, *args, **kwargs):
            nonlocal barrier
            if self is store and getattr(fn, "__name__", "").startswith(
                "persist_compacted"
            ):
                captured.set()
            if (
                self is store
                and getattr(fn, "__name__", "") == "_persist_compacted_snapshot"
            ):
                barrier = store.submit(held)
                captured.set()
            return await run(self, fn, *args, **kwargs)

        monkeypatch.setattr(type(store), "run", intercept)
        saving = asyncio.create_task(
            persist_compacted_async(
                store, "alice", agent, agent.controller.conversation, 3, 456.0
            )
        )
        try:
            await asyncio.wait_for(captured.wait(), 3)

            def mutate():
                agent.controller.conversation.append("user", "later input")
                agent._parent_branch_path.append((99, 99))

            await agent._submit_session_state(mutate)
            release.set()
            await saving
            assert await store.run(store.load_conversation, "alice") == before
            tag = await store.run(lambda: store.state["alice:snapshot_branch"])
            assert [list(pair) for pair in tag["parent_branch_path"]] == [
                list(pair) for pair in ancestry
            ]
        finally:
            release.set()
            if barrier is not None:
                await asyncio.wrap_future(barrier)
            await asyncio.gather(saving, return_exceptions=True)
            agent._parent_branch_path[:] = ancestry


async def test_compact_snapshot_does_not_cover_later_same_turn_completion(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first", "late answer"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        captured, release = asyncio.Event(), asyncio.Event()
        store = agent.session_store
        run = type(store).run

        async def hold_submission(self, fn, *args, **kwargs):
            if self is store and fn.__name__ == "_persist_compacted_snapshot":
                captured.set()
                await release.wait()
            return await run(self, fn, *args, **kwargs)

        monkeypatch.setattr(type(store), "run", hold_submission)
        saving = asyncio.create_task(
            persist_compacted_async(
                store, "alice", agent, agent.controller.conversation, 1, 123.0
            )
        )
        later = None
        try:
            await asyncio.wait_for(captured.wait(), 3)
            later = asyncio.create_task(
                agent.run_event(
                    create_tool_complete_event(
                        job_id="legacy_job", content="late result"
                    )
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(later), 0.05)
            release.set()
            await saving
            assert (await later).text == "late answer"
            assert agent._turn_index == 1
            saved = await store.run(store.load_conversation, "alice")
            watermark = await store.run(lambda: store.state["alice:snapshot_event_id"])
            raw = await store.run(store.get_events, "alice")
            late = [e["event_id"] for e in raw if e.get("content") == "late answer"]
            assert late
            assert any(
                m.get("content") == "late answer" for m in saved
            ) or watermark < min(late)
        finally:
            release.set()
            await asyncio.gather(
                *[task for task in (saving, later) if task is not None],
                return_exceptions=True,
            )


@pytest.mark.parametrize("overflow", [False, True])
async def test_turn_held_compact_wait_returns_splice_before_publication(
    tmp_path, overflow
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["one", "two", "three"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        for question in ("first", "second", "third"):
            await creature.run(question)
        manager = CompactManager(CompactConfig(keep_recent_turns=1, cooldown_seconds=0))
        manager._controller, manager._agent = agent.controller, agent
        manager._llm = ScriptedLLM(["summary kept"])
        manager._session_store, manager._agent_name = agent.session_store, "alice"
        manager._output_router = agent.output_router
        agent.compact_manager = manager

        async def turn_held_wait():
            async with agent._processing_lock:
                agent._turn_lock_holder = asyncio.current_task()
                try:
                    assert manager.trigger_compact()
                    if overflow:
                        assert await agent._compact_overflow_rescue() is None
                        await manager.wait_for_splice()
                        messages = agent.controller.conversation.to_messages()
                    else:
                        await manager.wait_for_current()
                        messages = agent.controller.conversation.to_messages()
                    assert any(
                        "summary kept" in str(message.get("content"))
                        for message in messages
                    )
                    assert manager.is_compacting
                finally:
                    agent._turn_lock_holder = None

        await asyncio.wait_for(turn_held_wait(), 3)
        await asyncio.wait_for(manager.wait_for_current(), 3)
        saved = await agent.session_store.run(
            agent.session_store.load_conversation, "alice"
        )
        assert any("summary kept" in str(message.get("content")) for message in saved)
        assert not manager.is_compacting


@pytest.mark.parametrize("phase", ["capture", "worker"])
async def test_accepted_snapshot_cancel_joins_before_unlock(
    tmp_path, monkeypatch, phase
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first", "later"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        store, conversation = agent.session_store, agent.controller.conversation
        entered, release = asyncio.Event(), asyncio.Event()
        run, snapshot = type(store).run, type(conversation).snapshot_messages
        saving = None

        def capture(self, **kwargs):
            if self is conversation and phase == "capture":
                saving.cancel()
            return snapshot(self, **kwargs)

        async def hold(self, fn, *args, **kwargs):
            if self is store and fn.__name__ == "_persist_compacted_snapshot":
                entered.set()
                await release.wait()
            return await run(self, fn, *args, **kwargs)

        monkeypatch.setattr(type(conversation), "snapshot_messages", capture)
        monkeypatch.setattr(type(store), "run", hold)
        saving = asyncio.create_task(
            persist_compacted_async(store, "alice", agent, conversation, 4, 789.0)
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if phase == "worker":
                saving.cancel()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(saving), 0.05)
            assert agent._processing_lock.locked()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await saving
            assert not agent._processing_lock.locked()
            assert await store.run(lambda: store.state["alice:compact_count"]) == 4
        finally:
            release.set()
            await asyncio.gather(saving, return_exceptions=True)
            monkeypatch.setattr(type(conversation), "snapshot_messages", snapshot)


@pytest.mark.parametrize("replacement", ["conversation", "store"])
async def test_snapshot_rejects_replaced_target_while_waiting_for_turn(
    tmp_path, replacement
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        store, conversation = agent.session_store, agent.controller.conversation
        before = await store.run(store.load_conversation, "alice")
        await agent._processing_lock.acquire()
        saving = asyncio.create_task(
            persist_compacted_async(store, "alice", agent, conversation, 7, 100.0)
        )
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(saving), 0.05)

            def replace():
                if replacement == "conversation":
                    agent.controller.conversation = deepcopy(conversation)
                else:
                    agent.session_store = object()

            await agent._submit_session_state(replace)
            agent._processing_lock.release()
            with pytest.raises(ValueError, match="replaced"):
                await saving
            assert await store.run(store.load_conversation, "alice") == before
            assert (
                await store.run(lambda: store.state.get("alice:compact_count", 0)) != 7
            )
        finally:
            if agent._processing_lock.locked():
                agent._processing_lock.release()
            await asyncio.gather(saving, return_exceptions=True)
            agent.session_store, agent.controller.conversation = store, conversation


async def test_cancel_before_compact_starts_releases_splice_waiters(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["one", "two", "three"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        for question in ("first", "second", "third"):
            await creature.run(question)
        manager = CompactManager(CompactConfig(keep_recent_turns=1, cooldown_seconds=0))
        manager._controller, manager._agent = agent.controller, agent
        manager._llm = ScriptedLLM(["must not summarize"])
        waiting = asyncio.create_task(manager.wait_for_splice())
        try:
            assert manager.trigger_compact()
            await manager.cancel()
            await asyncio.wait_for(waiting, 1)
            assert manager._compact_count == 0
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            await manager.cancel()


async def test_snapshot_cancel_before_turn_admission_does_not_publish(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        store = agent.session_store
        before = await store.run(store.load_conversation, "alice")
        async with agent._processing_lock:
            saving = asyncio.create_task(
                persist_compacted_async(
                    store, "alice", agent, agent.controller.conversation, 9, 99.0
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(saving), 0.05)
            saving.cancel()
            with pytest.raises(asyncio.CancelledError):
                await saving
            assert agent._processing_lock.locked()
        assert await store.run(store.load_conversation, "alice") == before
        assert await store.run(lambda: store.state.get("alice:compact_count", 0)) != 9


async def test_compact_snapshot_preserves_pending_native_tail(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        call = {
            "id": "pending-call",
            "type": "function",
            "function": {"name": "scratchpad", "arguments": "{}"},
        }
        await agent._submit_session_state(
            lambda: agent.controller.conversation.append(
                "assistant", "", tool_calls=[call]
            )
        )
        await persist_compacted_async(
            agent.session_store, "alice", agent, agent.controller.conversation, 1, 12.0
        )
        saved = await agent.session_store.run(
            agent.session_store.load_conversation, "alice"
        )
        assert saved[-1]["tool_calls"] == [call]
        assert not any(
            message.get("tool_call_id") == "pending-call" for message in saved
        )
