"""Compact preparation and live mutations belong to the session-state owner."""

import asyncio
import threading
from copy import deepcopy

import pytest

from kohakuterrarium import Terrarium
from kohakuterrarium.core import compact as compact_module, compact_commit
from kohakuterrarium.core.compact import CompactConfig, CompactManager
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.output.base import BaseOutputModule
from kohakuterrarium.core.model_switch_guard import require_model_mutation_available
from kohakuterrarium.session.compact_coverage import coordinate
from kohakuterrarium.session.resume_replay import replay_resume_history
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM
from kohakuterrarium.terrarium import session_coord


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


async def test_compact_rejects_unbound_controller_with_canonical_router(tmp_path):
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        agent, manager = await prepare(engine)
        manager._session_store = agent.session_store
        manager._output_router = agent.output_router
        await agent._session_state_owner.submit_internal(
            lambda: setattr(agent.controller, "session_store", None)
        )
        before = agent.controller.conversation.to_json()
        await manager._run_compact()
        assert manager._compact_count == 0
        assert agent.controller.conversation.to_json() == before
        rows = await agent.session_store.run(agent.session_store.get_events, "alice")
        assert not any(row["type"] == "compact_complete" for row in rows)


@pytest.mark.parametrize("fence_failure", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "append",
        "append_then_raise",
        "counters",
        "confirmation",
        "apply_ack",
        "cancel_write",
        "confirm_ack",
    ],
)
async def test_compact_persistence_failure_never_announces_success(
    tmp_path, monkeypatch, failure, fence_failure
):
    notifications = []

    class Observer(BaseOutputModule):
        async def write(self, content):
            pass

        async def write_stream(self, chunk):
            pass

        def on_activity(self, kind, detail):
            notifications.append(kind)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        agent, manager = await prepare(engine)
        store = agent.session_store
        manager._session_store = store
        manager._output_router = agent.output_router
        agent.output_router.add_secondary(Observer())
        append, save = store.append_event, store.save_state

        def failed_append(namespace, kind, *args, **kwargs):
            if kind == "compact_complete":
                if failure == "append_then_raise":
                    append(namespace, kind, *args, **kwargs)
                raise OSError("compact event unconfirmed")
            return append(namespace, kind, *args, **kwargs)

        def failed_counters(namespace, **kwargs):
            if "compact_count" in kwargs:
                raise OSError("compact counters unconfirmed")
            return save(namespace, **kwargs)

        submit, submissions = compact_commit.submit_internal_session_state, []

        async def fail_fence(host, callback):
            submissions.append(callback)
            if failure == "confirm_ack" and len(submissions) == 2:
                await submit(host, callback)
                raise OSError("confirmed owner acknowledgement unavailable")
            if (failure == "confirmation" and len(submissions) == 2) or (
                fence_failure and len(submissions) > 1
            ):
                raise RuntimeError("owner unavailable for confirmation or fence")
            result = await submit(host, callback)
            if failure == "apply_ack" and len(submissions) == 1:
                raise OSError("applied owner acknowledgement unavailable")
            return result

        run = store.run

        async def cancel_commit(func, *args, **kwargs):
            if func is compact_commit.persist_compact_counters:
                asyncio.current_task().cancel()
            return await run(func, *args, **kwargs)

        with monkeypatch.context() as patch:
            if fence_failure or failure in ("confirmation", "apply_ack", "confirm_ack"):
                patch.setattr(
                    compact_commit, "submit_internal_session_state", fail_fence
                )
            if failure == "counters":
                patch.setattr(store, "save_state", failed_counters)
            elif failure == "cancel_write":
                patch.setattr(store, "run", cancel_commit)
            elif failure not in ("confirmation", "apply_ack", "confirm_ack"):
                patch.setattr(store, "append_event", failed_append)
            await manager._run_compact()
        if failure == "confirm_ack":
            assert notifications.count("compact_complete") == 1
            assert "compact_skipped" not in notifications
            assert "processing_error" not in notifications
            require_model_mutation_available(agent)
            agent._init_compact_manager()
            await agent.compact_manager.wait_for_splice()
            rows = await store.run(store.get_events, "alice")
            assert sum(row["type"] == "compact_complete" for row in rows) == 1
            return
        assert "compact_complete" not in notifications
        assert "compact_skipped" not in notifications
        assert "processing_error" in notifications
        if fence_failure:
            assert not getattr(agent, "_session_attach_uncertain", False)
            with pytest.raises(ConflictError, match="recovery"):
                await manager.wait_for_splice()
        else:
            with pytest.raises(ConflictError, match="recovery"):
                require_model_mutation_available(agent)
        if fence_failure:
            agent._init_compact_manager()
            replacement = agent.compact_manager
            assert replacement is not manager
            with pytest.raises(ConflictError, match="recovery"):
                await replacement.wait_for_splice()
            with pytest.raises(ConflictError, match="recovery"):
                replacement.trigger_compact()
            with pytest.raises(
                (ConflictError, compact_commit.CompactCommitError), match="recovery"
            ):
                manager.trigger_compact()
            replacement.config.cooldown_seconds = 0
            replacement.config.max_tokens = 1
            assert not replacement.should_compact(1000000)
            alice = next(c for c in engine.list_creatures() if c.agent is agent)
            bob = await engine.add_creature(
                AgentConfig(name="bob"),
                llm=ScriptedLLM(["b"]),
                io="headless",
                start=False,
            )
            original_graphs = (alice.graph_id, bob.graph_id)
            with pytest.raises(ConflictError, match="recovery"):
                await engine.connect(alice, bob, channel="rejected")
            assert (alice.graph_id, bob.graph_id) == original_graphs
            assert agent.session_store is store
        rows = await store.run(store.get_events, "alice")
        assert sum(row["type"] == "compact_complete" for row in rows) == (
            0 if failure in ("append", "apply_ack") else 1
        )


@pytest.mark.parametrize("write_failure", [False, True])
async def test_cancelled_confirmed_summary_does_not_emit_skipped(
    tmp_path, monkeypatch, write_failure
):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    persist = compact_commit.persist_compact_counters
    notifications = []

    class Observer(BaseOutputModule):
        async def write(self, content):
            pass

        async def write_stream(self, chunk):
            pass

        def on_activity(self, kind, detail):
            notifications.append(kind)

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        agent, manager = await prepare(engine)
        manager._session_store = agent.session_store
        manager._output_router = agent.output_router
        store = agent.session_store
        snapshot_watermark = await store.run(store.state.get, "alice:snapshot_event_id")
        agent.output_router.add_secondary(Observer())

        def held(*args):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            if write_failure:
                raise OSError("cancelled compact counters failed")
            return persist(*args)

        monkeypatch.setattr(compact_commit, "persist_compact_counters", held)
        compacting = asyncio.create_task(manager._run_compact())
        manager._compact_task = compacting
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert "compact_complete" not in notifications
            assert not manager._splice_ready.is_set()
            compacting.cancel()
            await agent._session_state_owner.submit_internal(lambda: None)
            assert not compacting.done()
            release.set()
            await asyncio.wait_for(compacting, 5)
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert sum(row["type"] == "compact_complete" for row in rows) == 1
            assert not any(row["type"] == "compact_skipped" for row in rows)
            assert "compact_skipped" not in notifications
            assert ("processing_error" in notifications) == write_failure
            assert notifications.count("compact_complete") == (
                0 if write_failure else 1
            )
            assert manager._compact_count == 1
            if write_failure:
                with pytest.raises(ConflictError, match="recovery"):
                    require_model_mutation_available(agent)
                with pytest.raises(ConflictError, match="recovery"):
                    await manager.wait_for_splice()
            else:
                await manager.wait_for_splice()
            restored = replay_resume_history(rows)
            assert (
                sum(
                    "summary retained" in str(message.get("content", ""))
                    for message in restored
                )
                == 1
            )
            assert (
                await store.run(store.state.get, "alice:snapshot_event_id")
                == snapshot_watermark
            )
        finally:
            release.set()
            await asyncio.gather(compacting, return_exceptions=True)
    reopened = SessionStore(store.path)
    try:
        rows = await reopened.run(reopened.get_events, "alice")
        restored = replay_resume_history(rows)
        assert (
            sum(
                "summary retained" in str(message.get("content", ""))
                for message in restored
            )
            == 1
        )
        assert (
            await reopened.run(reopened.state.get, "alice:snapshot_event_id")
            == snapshot_watermark
        )
    finally:
        await asyncio.to_thread(reopened.close, update_status=False)


async def test_summary_returning_during_copy_waits_then_rejects_old_binding(
    tmp_path, monkeypatch
):
    summarizing, finish_summary, returned, copying = (asyncio.Event() for _ in range(4))
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class HeldSummary(ScriptedLLM):
        async def chat(self, messages, **kwargs):
            summarizing.set()
            await finish_summary.wait()
            async for chunk in super().chat(messages, **kwargs):
                yield chunk
            returned.set()

    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        agent, manager = await prepare(engine)
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        source = agent.session_store
        manager._session_store = source
        manager._output_router = agent.output_router
        manager._llm = HeldSummary(["must not install after handoff"])
        capture = session_coord.capture_runtime_transfer
        first = True

        def held(store, names):
            nonlocal first
            result = capture(store, names)
            if first and store is source:
                first = False
                loop.call_soon_threadsafe(copying.set)
                assert release.wait(6)
            return result

        monkeypatch.setattr(session_coord, "capture_runtime_transfer", held)
        compacting = asyncio.create_task(manager._run_compact())
        moving = None
        try:
            await asyncio.wait_for(summarizing.wait(), 3)
            alice = next(
                creature
                for creature in engine.list_creatures()
                if creature.agent is agent
            )
            moving = asyncio.create_task(engine.connect(bob, alice, channel="bridge"))
            await asyncio.wait_for(copying.wait(), 3)
            finish_summary.set()
            await asyncio.wait_for(returned.wait(), 3)
            await agent._session_state_owner.submit_internal(lambda: None)
            release.set()
            await asyncio.wait_for(moving, 8)
            await asyncio.wait_for(compacting, 5)
            assert manager._compact_count == 0
            assert all(
                "must not install" not in str(message.content)
                for message in agent.controller.conversation.get_messages()
            )
            rows = await agent.session_store.run(
                agent.session_store.get_events, "alice"
            )
            assert not any(row["type"] == "compact_complete" for row in rows)
            assert any(
                row["type"] == "compact_skipped"
                and row.get("reason") == "source_changed"
                for row in rows
            )
        finally:
            finish_summary.set()
            release.set()
            await asyncio.gather(
                compacting, *([moving] if moving else []), return_exceptions=True
            )


async def test_handoff_does_not_join_compact_waiting_for_initiator_turn_lock(
    tmp_path, monkeypatch
):
    committed = asyncio.Event()
    async with Terrarium(session_dir=tmp_path / "sessions") as engine:
        agent, manager = await prepare(engine)
        alice = next(
            creature for creature in engine.list_creatures() if creature.agent is agent
        )
        bob = await engine.add_creature(
            AgentConfig(name="bob"), llm=ScriptedLLM(["b"]), io="headless", start=True
        )
        manager._session_store = agent.session_store
        manager._output_router = agent.output_router
        splice = manager._splice_conversation

        def observed(*args, **kwargs):
            result = splice(*args, **kwargs)
            committed.set()
            return result

        monkeypatch.setattr(manager, "_splice_conversation", observed)
        compacting = None
        try:
            async with agent._processing_lock:
                compacting = asyncio.create_task(manager._run_compact())
                await asyncio.wait_for(committed.wait(), 3)
                await asyncio.wait_for(engine.connect(bob, alice, channel="bridge"), 5)
                assert not compacting.done()
            await asyncio.wait_for(compacting, 3)
            store = agent.session_store
            assert await store.run(store.state.get, "alice:compact_count") == 1
            rows = await store.run(store.get_events, "alice")
            assert sum(row["type"] == "compact_complete" for row in rows) == 1
            assert manager._compact_count == 1
        finally:
            if compacting is not None:
                await asyncio.gather(compacting, return_exceptions=True)


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
            assert metadata["coverage_mode"] == "exact"
            covered = {tuple(origin) for origin in metadata["replaced_event_origins"]}
            replaced = [e for e in events if coordinate(e) in covered]
            assert {e.get("turn_index") for e in replaced} == {1, 2}
            assert len(replaced) == len(covered)
            assert all(
                coordinate(e) not in covered
                for e in events
                if e.get("turn_index") in (3, 4)
            )
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
