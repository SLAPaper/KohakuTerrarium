"""SELECT publishes a prepared branch only after a durable store write."""

import asyncio
import threading

import pytest
from kohakuvault import KVault

from kohakuterrarium import Agent, Terrarium
from kohakuterrarium.builtins.user_commands.branch import BranchCommand
from kohakuterrarium.core import branch_selection
from kohakuterrarium.core.branch_selection import (
    require_history_selection_available,
    select_branch,
)
from kohakuterrarium.core.config_types import AgentConfig
from kohakuterrarium.core.job import JobState, JobStatus, JobType
from kohakuterrarium.errors import ConflictError
from kohakuterrarium.modules.user_command.base import UserCommandContext
from kohakuterrarium.session.resume_async import inject_saved_state_async
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.testing.llm import ScriptedLLM


async def _agent(tmp_path, provider, *, max_messages=0):
    engine = Terrarium(session_dir=tmp_path / "sessions")
    await engine.__aenter__()
    creature = await engine.add_creature(
        AgentConfig(
            name="alice",
            system_prompt="Preserve this instruction",
            max_messages=max_messages,
        ),
        llm=provider,
        io="headless",
        start=True,
    )
    return engine, creature


def _state(store, key):
    return store.state[key]


async def test_branch_input_continues_selected_history_and_restores_it(tmp_path):
    provider = ScriptedLLM(["original", "edited", "continued"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        await creature.run("ROOT")
        await agent.edit_and_rerun(1, "EDITED")
        assert await agent.inject_input(
            "CONTINUE", source="web", pending_id="selected-input", branch_view={"1": 1}
        )
        inputs = [m["content"] for m in provider.last_messages if m["role"] == "user"]
        assert inputs == ["ROOT", "CONTINUE"]
        restored = await Agent.build(
            AgentConfig(name="alice"), llm=ScriptedLLM(), io="headless"
        )
        await inject_saved_state_async(restored, agent.session_store, "alice")
        assert [
            m.content
            for m in restored.controller.conversation.get_messages()
            if m.role == "user"
        ] == inputs
        await restored.stop()
    finally:
        await engine.shutdown()


async def test_branch_input_save_failure_never_enqueues_or_changes_live(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(["original", "edited"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        await creature.run("ROOT")
        await agent.edit_and_rerun(1, "EDITED")
        before = agent.controller.conversation.to_json()

        def fail(*args):
            raise OSError("selection save failed")

        monkeypatch.setattr(branch_selection, "_persist_on_worker", fail)
        with pytest.raises(OSError, match="selection save failed"):
            await agent.inject_input("NEVER", branch_view={1: 1})
        assert agent.controller.conversation.to_json() == before
        assert agent._event_inbox.empty()
        assert provider.call_count == 2
    finally:
        await engine.shutdown()


async def test_branch_continuation_admission_precedes_later_input(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(
        ["original", "edited", "selected answer", "later answer", "final"]
    )
    engine, creature = await _agent(tmp_path, provider)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    selected = None
    try:
        agent = creature.agent
        await creature.run("ROOT")
        await agent.edit_and_rerun(1, "EDITED")
        persist = branch_selection._persist_on_worker

        def held(*args):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return persist(*args)

        monkeypatch.setattr(branch_selection, "_persist_on_worker", held)
        selected = asyncio.create_task(
            agent.inject_input("SELECTED", branch_view={1: 1})
        )
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(ConflictError):
            await agent.inject_input("WRONG", branch_view={1: 2})
        assert await agent.inject_input("LATER") is False
        for _ in range(8):
            await asyncio.sleep(0)
        release.set()
        assert await selected
        await creature.run("FINAL")
        requests = [
            [m["content"] for m in call if m["role"] == "user"]
            for call in provider.call_log[2:]
        ]
        assert requests[0] == ["ROOT", "SELECTED"]
        assert all(
            "WRONG" not in str(call) and "EDITED" not in str(call) for call in requests
        )
    finally:
        release.set()
        if selected is not None:
            await asyncio.gather(selected, return_exceptions=True)
        await engine.shutdown()


async def test_branch_latest_keeps_live_system_and_persists_full_context(tmp_path):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider, max_messages=1)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        config = agent.controller.conversation.config
        system = agent.controller.conversation.get_system_message()
        assert system is not None
        result = await BranchCommand().execute(
            "latest", UserCommandContext(agent=agent)
        )
        assert result.success, result.error
        assert provider.call_count == 1
        after = agent.controller.conversation.get_system_message()
        assert after is not None
        assert after.content == system.content
        assert after is system
        assert agent.controller.conversation.config is config
        assert [
            message.content
            for message in agent.controller.conversation.get_messages()
            if message.role == "assistant"
        ] == ["original answer"]
        store = agent.session_store
        watermark = await store.run(_state, store, "alice:snapshot_event_id")
        branch = await store.run(_state, store, "alice:snapshot_branch")
        assert isinstance(watermark, int) and watermark > 0
        assert branch["turn_index"] == agent._turn_index
        assert branch["branch_id"] == agent._branch_id
        assert branch["branch_view"] == {1: 1}
        bundle = agent._runtime_context_bundle
        assert isinstance(bundle.provider_conversation, tuple)
        assert (
            list(bundle.provider_conversation)
            == agent.controller.conversation.get_messages()
        )
        assert bundle.ref.context_id.ordinal == 1
    finally:
        await engine.__aexit__(None, None, None)


async def test_non_latest_selection_uses_full_context_without_llm(tmp_path):
    provider = ScriptedLLM(["first", "second"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("one", timeout=5)).text == "first"
        await agent.regenerate_last_response()
        assert provider.call_count == 2
        count = await select_branch(agent, {1: 1})
        assert provider.call_count == 2
        assert count >= 2
        assert [
            message.content
            for message in agent.controller.conversation.get_messages()
            if message.role == "assistant"
        ] == ["first"]
        assert agent.controller.conversation.get_system_message().content
    finally:
        await engine.__aexit__(None, None, None)


async def test_nonlatest_selection_survives_close_reopen_and_continuation(tmp_path):
    engine, creature = await _agent(tmp_path, ScriptedLLM(["first", "second"]))
    agent = creature.agent
    path = agent.session_store.path
    try:
        assert (await creature.run("question", timeout=5)).text == "first"
        await agent.regenerate_last_response()
        await agent.select_branch({1: 1})
    finally:
        await engine.__aexit__(None, None, None)
    store = SessionStore(path)
    resumed = await Agent.build(
        AgentConfig(name="alice"), llm=ScriptedLLM(["continued"]), io="headless"
    )
    try:
        await inject_saved_state_async(resumed, store, "alice")
        await store.run(resumed.attach_session_store, store)
        assert resumed._branch_id == 1
        assert [
            m.content
            for m in resumed.controller.conversation.get_messages()
            if m.role == "assistant"
        ] == ["first"]
        await resumed.start()
        assert (await resumed.run("next", timeout=5)).text == "continued"
        assert (1, 1) in resumed._parent_branch_path
        assert (1, 2) not in resumed._parent_branch_path
    finally:
        await resumed.stop()
        await asyncio.to_thread(store.close)


async def test_lineage_conflict_rejects_without_store_or_memory_change(tmp_path):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        conversation = agent.controller.conversation
        store = agent.session_store
        before = await store.run(_state, store, "alice:snapshot_branch")
        with pytest.raises(Exception):
            await select_branch(agent, {1: 1, 2: 9})
        assert agent.controller.conversation is conversation
        after = await store.run(_state, store, "alice:snapshot_branch")
        assert after == before
        assert provider.call_count == 1
    finally:
        await engine.__aexit__(None, None, None)


async def test_save_fault_before_write_keeps_memory_and_marks_uncertain(
    tmp_path, monkeypatch
):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        conversation = agent.controller.conversation
        messages = list(conversation.get_messages())
        original = type(agent.session_store).save_conversation

        def fail(self, namespace, payload):
            saved = payload.get("messages") if isinstance(payload, dict) else payload
            if isinstance(saved, list) and saved and saved[0].get("role") == "system":
                raise OSError("selection save rejected")
            return original(self, namespace, payload)

        monkeypatch.setattr(type(agent.session_store), "save_conversation", fail)
        with pytest.raises(OSError, match="selection save rejected"):
            await select_branch(agent, None)
        assert agent.controller.conversation is conversation
        assert list(conversation.get_messages()) == messages
        assert agent._history_selection_uncertain is True
        with pytest.raises(ConflictError, match="uncertain"):
            require_history_selection_available(agent)
    finally:
        await engine.__aexit__(None, None, None)


async def test_prepare_and_publish_run_on_worker_and_owner(tmp_path, monkeypatch):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        seen = {}
        owner_task = agent._session_state_owner.task
        prepare = branch_selection._prepare_on_worker
        bind = branch_selection._bind
        capture = branch_selection._capture

        def wrapped_capture(*args, **kwargs):
            seen["capture_task"] = asyncio.current_task()
            return capture(*args, **kwargs)

        def wrapped_prepare(request):
            seen["prepare"] = threading.get_ident()
            return prepare(request)

        def wrapped_bind(*args, **kwargs):
            seen["publish_task"] = asyncio.current_task()
            return bind(*args, **kwargs)

        monkeypatch.setattr(branch_selection, "_capture", wrapped_capture)
        monkeypatch.setattr(branch_selection, "_prepare_on_worker", wrapped_prepare)
        monkeypatch.setattr(branch_selection, "_bind", wrapped_bind)
        await select_branch(agent, None)
        assert seen["prepare"] == agent.session_store._affinity_worker_ident
        assert seen["prepare"] != threading.get_ident()
        assert seen["publish_task"] is owner_task
        assert seen["capture_task"] is owner_task
    finally:
        await engine.__aexit__(None, None, None)


async def test_repeated_cancel_joins_until_disk_and_live_agree(tmp_path, monkeypatch):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        put = KVault.put

        def write(table, key, value, **kwargs):
            text = (
                key.decode("utf-8", "replace") if isinstance(key, bytes) else str(key)
            )
            if text.endswith(":snapshot_branch"):
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5)
            return put(table, key, value, **kwargs)

        monkeypatch.setattr(KVault, "put", write)
        prior_bundle = getattr(agent, "_runtime_context_bundle", None)
        task = asyncio.create_task(select_branch(agent, None))
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert getattr(agent, "_runtime_context_bundle", None) is prior_bundle
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        store = agent.session_store
        state = await store.run(_state, store, "alice:snapshot_branch")
        assert state["turn_index"] == agent._turn_index
        assert state["branch_id"] == agent._branch_id
        assert agent.controller.conversation.get_system_message() is not None
    finally:
        release.set()
        await engine.__aexit__(None, None, None)


async def test_slow_store_read_lets_owner_heartbeat(tmp_path, monkeypatch):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        release = threading.Event()
        original = agent.session_store.get_events
        calls = {"n": 0}

        def slow(namespace, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                assert release.wait(5)
            return original(namespace, *args, **kwargs)

        monkeypatch.setattr(agent.session_store, "get_events", slow)
        task = asyncio.create_task(select_branch(agent, None))
        await asyncio.wait_for(asyncio.sleep(0), timeout=1)
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, timeout=5) >= 2
    finally:
        release.set()
        await engine.__aexit__(None, None, None)


async def test_stop_joins_selection_before_owner_and_store_close(tmp_path, monkeypatch):
    engine, creature = await _agent(tmp_path, ScriptedLLM(["original"]))
    agent = creature.agent
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    persist = branch_selection._persist_on_worker

    def held(request, prepared):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return persist(request, prepared)

    try:
        assert (await creature.run("question", timeout=5)).ok
        monkeypatch.setattr(branch_selection, "_persist_on_worker", held)
        selecting = asyncio.create_task(agent.select_branch({1: 1}))
        await asyncio.wait_for(entered.wait(), 3)
        stopping = asyncio.create_task(agent.stop())
        await asyncio.sleep(0)
        assert not stopping.done()
        assert not agent._session_state_owner.task.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(selecting, stopping), 5)
        assert agent._runtime_context_bundle.branch_view == ((1, 1),)
        assert agent._session_state_owner.task is None
    finally:
        release.set()
        await engine.__aexit__(None, None, None)


@pytest.mark.parametrize("action", ["prompt", "model", "compact", "attach"])
async def test_waiting_selection_rejects_conflicting_mutation(
    tmp_path, monkeypatch, action
):
    engine, creature = await _agent(tmp_path, ScriptedLLM(["original"]))
    agent = creature.agent
    release = threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    persist = branch_selection._persist_on_worker

    def held(request, prepared):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return persist(request, prepared)

    task = None
    try:
        assert (await creature.run("question", timeout=5)).ok
        monkeypatch.setattr(branch_selection, "_persist_on_worker", held)
        task = asyncio.create_task(agent.select_branch({1: 1}))
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(ConflictError):
            match action:
                case "prompt":
                    agent.update_system_prompt("changed", replace=True)
                case "model":
                    await agent.switch_model_async("openai/gpt-5.4-nano")
                case "compact":
                    agent.compact_manager.trigger_compact()
                case "attach":
                    agent.attach_session_store(agent.session_store)
        release.set()
        await asyncio.wait_for(task, 5)
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await engine.__aexit__(None, None, None)


async def test_active_jobs_and_compaction_reject_before_mutation(tmp_path):
    provider = ScriptedLLM(["original answer"])
    engine, creature = await _agent(tmp_path, provider)
    try:
        agent = creature.agent
        assert (await creature.run("question", timeout=5)).text == "original answer"
        conversation = agent.controller.conversation
        agent.executor.job_store.register(
            JobStatus(
                job_id="job-1",
                job_type=JobType.TOOL,
                type_name="demo",
                state=JobState.RUNNING,
            )
        )
        with pytest.raises(ConflictError, match="locally idle"):
            await select_branch(agent, None)
        agent.executor.job_store._statuses.clear()
        agent.compact_manager._dispatch._running = True
        with pytest.raises(ConflictError, match="locally idle"):
            await select_branch(agent, None)
        assert agent.controller.conversation is conversation
        assert not getattr(agent, "_history_selection_pending", False)
    finally:
        await engine.__aexit__(None, None, None)
