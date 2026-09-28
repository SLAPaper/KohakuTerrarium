"""Turn snapshots capture detached conversation and branch state on the owner."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_turn_snapshot_reads_on_owner_and_persists_on_worker(
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
        owner_task = agent._session_state_owner.task
        snapshot = type(agent.controller.conversation).snapshot_messages
        readers = []

        def read_snapshot(self, **kwargs):
            if self is agent.controller.conversation:
                assert agent._processing_lock.locked()
                readers.append(asyncio.current_task())
            return snapshot(self, **kwargs)

        monkeypatch.setattr(
            type(agent.controller.conversation), "snapshot_messages", read_snapshot
        )
        writer_threads = []
        persist = agent._session_output._persist_turn_snapshot

        def record_persist(*args):
            writer_threads.append(threading.get_ident())
            return persist(*args)

        monkeypatch.setattr(
            agent._session_output, "_persist_turn_snapshot", record_persist
        )
        await creature.run("question")
        assert writer_threads and all(
            identity != threading.get_ident() for identity in writer_threads
        )
        assert readers and all(task is owner_task for task in readers)
        saved = await agent.session_store.run(
            agent.session_store.load_conversation, "alice"
        )
        assert any(message["content"] == "answer" for message in saved)


async def test_snapshot_capture_is_detached_before_worker_wait(tmp_path, monkeypatch):
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
        sink = agent._session_output
        captured, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        persist = sink._persist_turn_snapshot
        original_path = deepcopy(agent._parent_branch_path)

        def held(messages, branch_tag, state_kwargs, envelope):
            loop.call_soon_threadsafe(captured.set)
            assert release.wait(5)
            persist(messages, branch_tag, state_kwargs, envelope)

        monkeypatch.setattr(sink, "_persist_turn_snapshot", held)
        saving = asyncio.create_task(sink.on_processing_end())
        try:
            await asyncio.wait_for(captured.wait(), 3)
            await agent._submit_session_state(
                lambda: agent._parent_branch_path.append((99, 99))
            )
            release.set()
            await saving
            saved_tag = await agent.session_store.run(
                lambda: agent.session_store.state["alice:snapshot_branch"]
            )
            assert [list(pair) for pair in saved_tag["parent_branch_path"]] == [
                list(pair) for pair in original_path
            ]
        finally:
            release.set()
            await saving
            agent._parent_branch_path[:] = original_path


async def test_queued_input_cannot_advance_snapshot_watermark_before_capture_persists(
    tmp_path, monkeypatch
):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        llm = ScriptedLLM(["first answer", "second answer"])
        creature = await engine.add_creature(
            AgentConfig(name="alice"), llm=llm, io="headless", start=True
        )
        agent = creature.agent
        store = agent.session_store
        sink = agent._session_output
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        persist = sink._persist_turn_snapshot
        saved = []

        def held(*args):
            if not saved:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            persist(*args)
            saved.append(
                (
                    store.load_conversation("alice"),
                    store.state["alice:snapshot_event_id"],
                    store.get_events("alice"),
                )
            )

        monkeypatch.setattr(sink, "_persist_turn_snapshot", held)
        first = asyncio.create_task(creature.run("first question"))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert agent._processing_lock.locked()
            second = asyncio.create_task(creature.run("second question"))
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(second), 0.05)
            assert llm.call_count == 1
            release.set()
            assert (await first).text == "first answer"
            assert (await second).text == "second answer"
            messages, watermark, raw = saved[0]
            assert [m["content"] for m in messages if m["role"] == "user"] == [
                "first question"
            ]
            assert not any(
                e.get("content") == "second question"
                for e in raw
                if e["event_id"] <= watermark
            )
            final = await store.run(store.load_conversation, "alice")
            assert [m["content"] for m in final if m["role"] == "user"] == [
                "first question",
                "second question",
            ]
        finally:
            release.set()
            await asyncio.gather(
                *[task for task in (first, second) if task is not None],
                return_exceptions=True,
            )


async def test_cancelled_capture_waiter_joins_owner_before_persisting(
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
        owner = agent._session_state_owner
        execute = owner._execute
        intercepted, held = asyncio.Event(), []

        def hold(callback, future):
            if not intercepted.is_set():
                held.append((callback, future))
                intercepted.set()
            else:
                execute(callback, future)

        monkeypatch.setattr(owner, "_execute", hold)
        saving = asyncio.create_task(agent._session_output.on_processing_end())
        await asyncio.wait_for(intercepted.wait(), 3)
        saving.cancel()
        await owner.submit_internal(lambda: execute(*held[0]))
        await asyncio.wait_for(saving, 3)
        saved = await agent.session_store.run(
            agent.session_store.load_conversation, "alice"
        )
        assert any(m["content"] == "answer" for m in saved)
        assert not agent._session_output._pending_writes


async def test_turn_snapshot_keeps_genuine_pending_native_tail(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        creature = await engine.add_creature(
            AgentConfig(name="alice", tool_format="native"),
            llm=ScriptedLLM(["first"]),
            io="headless",
            start=True,
        )
        agent = creature.agent
        await creature.run("question")
        await agent._submit_session_state(
            lambda: agent.controller.conversation.append(
                "assistant",
                "pending",
                tool_calls=[
                    {
                        "id": "pending-id",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            )
        )
        await agent._session_output.on_processing_end()
        saved = await agent.session_store.run(
            agent.session_store.load_conversation, "alice"
        )
        assert saved[-1]["tool_calls"][0]["id"] == "pending-id"
        assert not any(message["role"] == "tool" for message in saved)
