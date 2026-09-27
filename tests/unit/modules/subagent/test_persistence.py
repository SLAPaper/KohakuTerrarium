"""Child allocation and history receipts use store affinity and survive cancellation."""

import asyncio
import threading

import pytest

from kohakuterrarium.core.job import JobState
from kohakuterrarium.core.registry import Registry
from kohakuterrarium.modules.subagent.config import SubAgentConfig
from kohakuterrarium.modules.subagent.manager import SubAgentManager
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


async def test_child_io_uses_store_worker_and_detached_conversation(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "child.kohakutr")
    manager = SubAgentManager(Registry(), ScriptedLLM(["child output"]))
    manager._session_store, manager._parent_name = store, "parent"
    manager.register(SubAgentConfig(name="child", max_turns=1))
    loop_thread = threading.get_ident()
    threads = []
    for name in ("next_subagent_run", "save_subagent", "find_subagent_run"):
        original = getattr(store, name)

        def observe(*args, _call=original, **kwargs):
            threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread
            return _call(*args, **kwargs)

        monkeypatch.setattr(store, name, observe)
    try:
        jid = await manager.spawn("child", "task", background=False)
        assert manager.get_result(jid).output == "child output"
        assert len(threads) >= 4
        assert len(set(threads)) == 1
        assert "child output" in await store.run(
            store.load_subagent_conversation, "parent", "child", 0
        )
    finally:
        await asyncio.to_thread(store.close)


async def test_failed_allocation_cannot_overwrite_existing_run_zero(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "allocation.kohakutr")
    manager = SubAgentManager(Registry(), ScriptedLLM(["must not run"]))
    manager._session_store, manager._parent_name = store, "parent"
    manager.register(SubAgentConfig(name="child", max_turns=1))
    await store.run(
        store.save_subagent,
        "parent",
        "child",
        0,
        {"job_id": "old", "success": True},
        "original history",
    )
    original_meta = await store.run(store.load_subagent_meta, "parent", "child", 0)

    def failed(*args):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(store, "next_subagent_run", failed)
    try:
        try:
            await manager.spawn("child", "task", background=False)
        except RuntimeError:
            pass
        assert (
            await store.run(store.load_subagent_meta, "parent", "child", 0)
            == original_meta
        )
        assert (
            await store.run(store.load_subagent_conversation, "parent", "child", 0)
            == "original history"
        )
        assert manager.llm.call_count == 0
    finally:
        await asyncio.to_thread(store.close)


async def test_cancel_waits_for_accepted_child_snapshot(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "cancel.kohakutr")
    manager = SubAgentManager(Registry(), ScriptedLLM(["completed child"]))
    manager._session_store, manager._parent_name = store, "parent"
    manager.register(SubAgentConfig(name="child", max_turns=1))
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    save = store.save_subagent
    calls = []

    def held(**kwargs):
        if not calls:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        calls.append(kwargs)
        return save(**kwargs)

    monkeypatch.setattr(store, "save_subagent", held)
    job_id = await manager.spawn("child", "task", background=True)
    runner = manager._tasks[job_id]
    try:
        await asyncio.wait_for(entered.wait(), 3)
        runner.cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(runner), 0.05)
        assert manager.get_result(job_id) is None
        release.set()
        result = await asyncio.wait_for(runner, 3)
        assert result.cancelled
        meta = await store.run(store.load_subagent_meta, "parent", "child", 0)
        assert meta["cancelled"] is True
        assert "completed child" in await store.run(
            store.load_subagent_conversation, "parent", "child", 0
        )
    finally:
        release.set()
        await asyncio.gather(runner, return_exceptions=True)
        await asyncio.to_thread(store.close)


async def test_persistent_save_failure_does_not_leave_finished_job_running(
    tmp_path, monkeypatch
):
    store = SessionStore(tmp_path / "failed-save.kohakutr")
    manager = SubAgentManager(Registry(), ScriptedLLM(["child answer"]))
    manager._session_store, manager._parent_name = store, "parent"
    manager.register(SubAgentConfig(name="child", max_turns=1))

    def fail(**kwargs):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(store, "save_subagent", fail)
    try:
        jid = await manager.spawn("child", "task", background=True)
        with pytest.raises(RuntimeError, match="disk unavailable"):
            await manager._tasks[jid]
        status = manager.job_store.get_status(jid)
        assert status.state == JobState.ERROR
        assert "disk unavailable" in status.error
        assert not manager.get_running_jobs()
    finally:
        await asyncio.to_thread(store.close)
